// sm2_context.c - Context creation and core inference functions

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "smollm2.h"
#include "sm2_utils.h"

// ============================================================================
// FLOAT16 UTILITIES (deprecated - use sm2_utils.h)
// ============================================================================

// Convert IEEE-754 float16 to float (deprecated - use sm2_f16_to_float)
static float f16_to_float(uint16_t h) {
    unsigned int bits = h;
    int sign = (bits >> 15) & 1;
    int exp = (bits >> 10) & 0x1F;
    int frac = bits & 0x3FF;
    float result;

    if (exp == 0) {
        result = (float)frac / 1024.0f;
    } else if (exp == 31) {
        result = (frac == 0) ? 1.0f / 0.0f : 0.0f / 0.0f;
    } else {
        result = (1.0f + (float)frac / 1024.0f) * powf(2.0f, (float)(exp - 15));
    }

    return sign ? -result : result;
}

// ============================================================================
// CONTEXT ALLOCATION (preallocated, no runtime allocation in hot path)
// ============================================================================

int sm2_create_context(sm2_model* model, sm2_context** out_ctx) {
    fprintf(stderr, "DEBUG: sm2_create_context ENTRY\n"); fflush(stderr);
    sm2_context* ctx = calloc(1, sizeof(sm2_context));
    if (!ctx) return -1;
    fprintf(stderr, "DEBUG: ctx allocated\n"); fflush(stderr);
    
    ctx->model = model;
    ctx->pos = 0;
    ctx->last_token = 1; // BOS
    ctx->rng_state = 123456789ULL; // Initial RNG state
    
    // Get spec for dimensions
    const sm2_spec* spec = sm2_get_spec(model->variant);
    
    // Preallocate scratch buffers (never freed during lifetime)
    size_t dim = model->dim;
    size_t hidden = model->hidden_dim;
    size_t vocab = model->vocab_size;
    size_t kv_size = (size_t)model->n_kv_heads * model->head_dim;
    fprintf(stderr, "DEBUG: model->n_layers=%d dim=%zu hidden=%zu vocab=%zu kv_size=%zu\n", 
            model->n_layers, dim, hidden, vocab, kv_size); fflush(stderr);
    
    ctx->scratch.x = calloc(dim, sizeof(float));
    ctx->scratch.xb = calloc(dim, sizeof(float));
    ctx->scratch.q = calloc(dim, sizeof(float));
    ctx->scratch.k = calloc(kv_size, sizeof(float));
    ctx->scratch.v = calloc(kv_size, sizeof(float));
    ctx->scratch.attn_out = calloc(dim, sizeof(float));
    ctx->scratch.logits = calloc(vocab, sizeof(float));
    fprintf(stderr, "DEBUG: scratch.x=%p xb=%p q=%p k=%p v=%p attn=%p logits=%p\n",
            (void*)ctx->scratch.x, (void*)ctx->scratch.xb, (void*)ctx->scratch.q,
            (void*)ctx->scratch.k, (void*)ctx->scratch.v, (void*)ctx->scratch.attn_out,
            (void*)ctx->scratch.logits); fflush(stderr);
    
    if (!ctx->scratch.x || !ctx->scratch.xb || !ctx->scratch.q ||
        !ctx->scratch.k || !ctx->scratch.v || !ctx->scratch.attn_out ||
        !ctx->scratch.logits) {
        sm2_free_context(ctx);
        return -1;
    }
    
    // Default generation params
    ctx->params.temperature = 0.8f;
    ctx->params.top_p = 90;
    ctx->params.top_k = 40;
    ctx->params.max_context = spec->max_seq_len;
    ctx->params.max_output = 256;
    
    // Initialize KV pool for model
    // For Phase 1, use contiguous F16 KV (not paged yet)
    ctx->kv_pool = calloc(1, sizeof(sm2_kv_pool));
    if (!ctx->kv_pool) {
        sm2_free_context(ctx);
        return -1;
    }
    fprintf(stderr, "DEBUG: kv_pool allocated\n"); fflush(stderr);
    
    // Simple contiguous KV for Phase 1
    int max_tokens = spec->max_seq_len;
    fprintf(stderr, "DEBUG: max_tokens=%d, n_layers=%d, n_kv_heads=%d, head_dim=%d\n",
            max_tokens, model->n_layers, model->n_kv_heads, model->head_dim); fflush(stderr);
    int kv_elements = (int)model->n_layers * (int)model->n_kv_heads * max_tokens * (int)model->head_dim * 2; // K+V
    ctx->kv_pool->pages = calloc(1, sizeof(sm2_kv_page)); // Single contiguous page
    ctx->kv_pool->n_layers = model->n_layers;
    ctx->kv_pool->n_kv_heads = model->n_kv_heads;
    ctx->kv_pool->head_dim = model->head_dim;
    ctx->kv_pool->page_tokens = max_tokens;
    ctx->kv_pool->max_pages = 1;
    ctx->kv_pool->dtype = SM2_KV_F16;
    
    // Allocate KV data (F16 = 2 bytes per element)
    size_t kv_bytes = (size_t)model->n_layers * model->n_kv_heads * max_tokens * model->head_dim * 2 * 2;
    fprintf(stderr, "DEBUG: kv_bytes=%zu\n", kv_bytes); fflush(stderr);
    uint8_t* kv_data = calloc(kv_bytes, 1);
    fprintf(stderr, "DEBUG: kv_data=%p\n", (void*)kv_data); fflush(stderr);
    
    ctx->kv.seq_len = 0;
    ctx->kv.n_pages = 1;
    ctx->kv.page_ids[0] = 0;
    
    *out_ctx = ctx;
    fprintf(stderr, "DEBUG: sm2_create_context DONE\n"); fflush(stderr);
    return 0;
}

int sm2_free_context(sm2_context* ctx) {
    if (!ctx) return 0;
    
    if (ctx->scratch.x) free(ctx->scratch.x);
    if (ctx->scratch.xb) free(ctx->scratch.xb);
    if (ctx->scratch.q) free(ctx->scratch.q);
    if (ctx->scratch.k) free(ctx->scratch.k);
    if (ctx->scratch.v) free(ctx->scratch.v);
    if (ctx->scratch.attn_out) free(ctx->scratch.attn_out);
    if (ctx->scratch.logits) free(ctx->scratch.logits);
    if (ctx->kv_pool) {
        if (ctx->kv_pool->pages) free(ctx->kv_pool->pages);
        free(ctx->kv_pool);
    }
    
    free(ctx);
    return 0;
}

// ============================================================================
// EMBEDDING LOOKUP
// ============================================================================

static void embedding_lookup(float* out, int token, const sm2_tensor_f16* embed) {
    // out = embed[token], where embed is [vocab, dim] stored as F16
    int dim = embed->cols;
    
    for (int i = 0; i < dim; i++) {
        uint16_t h = embed->data[token * dim + i];
        out[i] = f16_to_float(h);
    }
}

// ============================================================================
// ATTENTION (simplified GQA)
// ============================================================================

static void attention_forward(float* out, const float* q, const float* k, const float* v,
                              int n_heads, int n_kv_heads, int head_dim, int seq_len) {
    // Simplified attention with GQA support
    // Q: [n_heads, head_dim], K/V: [n_kv_heads, head_dim]
    
    int group_size = n_heads / n_kv_heads;
    
    // For each query head, attend over all KV heads
    for (int qh = 0; qh < n_heads; qh++) {
        int kv_head = qh / group_size;
        
        // Compute attention scores
        float scores[256]; // max seq_len
        float max_score = -1e9f;
        
        for (int pos = 0; pos < seq_len; pos++) {
            // Dot product q with K[pos]
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q[qh * head_dim + d] * k[kv_head * head_dim + pos * head_dim + d];
            }
            dot /= sqrtf((float)head_dim);
            scores[pos] = dot;
            if (dot > max_score) max_score = dot;
        }
        
        // Softmax
        float sum_exp = 0.0f;
        for (int pos = 0; pos < seq_len; pos++) {
            scores[pos] = expf(scores[pos] - max_score);
            sum_exp += scores[pos];
        }
        
        // Weighted sum of V
        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (int pos = 0; pos < seq_len; pos++) {
                sum += scores[pos] * v[kv_head * head_dim + pos * head_dim + d];
            }
            out[qh * head_dim + d] = sum / sum_exp;
        }
    }
}

// ============================================================================
// MATMUL HELPERS
// ============================================================================

// matmul: out[m] += a[m,k] @ b[k,n] (column-major storage)
static void matmul_f16(float* out, const float* a, const uint16_t* b, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        for (int j = 0; j < k; j++) {
            uint16_t h = b[i * k + j];
            sum += a[j] * sm2_f16_to_float(h);
        }
        out[i] += sum;
    }
}

// out[m,n] += a[m,k] @ b[k,n] (column-major, accumulates into out)
static void matmul_f16_2d(float* out, const float* a, const uint16_t* b, int m, int n, int k) {
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < m; row++) {
            float sum = 0.0f;
            for (int j = 0; j < k; j++) {
                uint16_t h = b[j * n + col];  // b[k,n] column-major -> b[j*n + col]
                sum += a[row * k + j] * sm2_f16_to_float(h);
            }
            out[row * n + col] += sum;
        }
    }
}

// ============================================================================
// SINGLE LAYER FORWARD
// ============================================================================

static void layer_forward(float* xb, const float* x, int layer,
                          sm2_model* model, sm2_context* ctx, int seq_pos) {
    // Check for NULL pointers
    if (!xb || !x || !model || !ctx) {
        fprintf(stderr, "FATAL: NULL pointer in layer_forward! xb=%p, x=%p, model=%p, ctx=%p\n",
                (void*)xb, (void*)x, (void*)model, (void*)ctx);
        fflush(stderr);
        return;
    }
    
    fprintf(stderr, "layer_forward ENTRY: layer=%d, x=%p, xb=%p\n", layer, (void*)x, (void*)xb);
    fflush(stderr);
    
    const sm2_spec* spec = sm2_get_spec(model->variant);
    int dim = model->dim;
    int kv_dim = model->n_kv_heads * model->head_dim;
    int hidden_dim = model->hidden_dim;
    
    // Offset into layer weight arrays
    size_t ln_off  = (size_t)layer * dim;
    size_t q_off   = (size_t)layer * dim * dim;
    size_t k_off   = (size_t)layer * kv_dim * dim;
    size_t v_off   = (size_t)layer * kv_dim * dim;
    size_t o_off   = (size_t)layer * dim * dim;  // o_proj is [dim, dim]
    size_t post_off = (size_t)layer * dim;
    size_t gate_off = (size_t)layer * hidden_dim * dim;
    size_t up_off   = (size_t)layer * hidden_dim * dim;
    size_t down_off = (size_t)layer * dim * hidden_dim;
    
    float* q = ctx->scratch.q;
    float* k = ctx->scratch.k;
    float* v = ctx->scratch.v;
    float* attn_out = ctx->scratch.attn_out;
    float* xb2 = ctx->scratch.xb;  // reuse xb buffer
    
// 1. RMSNorm (input layernorm): xb = rmsnorm(x, input_layernorm[layer])
    // Simplified: just scale x by layernorm weight (no mean removal for now)
    fprintf(stderr, "DEBUG: Layer %d: ln_off=%zu, dim=%d\n", layer, ln_off, dim);
    // Check for NaN in input x
    int x_has_nan = 0;
    for (int i = 0; i < 5; i++) {
        if (isnan(x[i])) { x_has_nan = 1; break; }
    }
    fprintf(stderr, "DEBUG: Layer %d: x[0..4]={%.4f,%.4f,%.4f,%.4f,%.4f}, x_nan=%d\n", 
            layer, x[0], x[1], x[2], x[3], x[4], x_has_nan);
    for (int i = 0; i < dim; i++) {
        uint16_t w = model->input_layernorm[ln_off + i];
        float wf = sm2_f16_to_float(w);
        xb[i] = x[i] * wf;
    }
    fprintf(stderr, "DEBUG: Layer %d: RMSNorm done, xb[0..4]={%.4f,%.4f,%.4f,%.4f,%.4f}\n", 
            layer, xb[0], xb[1], xb[2], xb[3], xb[4]);
    
    // 2. Q projection: q = xb @ q_proj.T  (q_proj is [dim, dim], stored row-major)
    // We want q[i] = sum_j xb[j] * q_proj[j][i]  =>  xb @ q_proj^T
    // q is [dim], q_proj is [dim x dim] row-major, so q_proj[j][i] = q_proj[j * dim + i]
    fprintf(stderr, "DEBUG: Layer %d: q_proj, q_off=%zu, dim=%d\n", layer, q_off, dim);
    memset(q, 0, dim * sizeof(float));
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            uint16_t h = model->q_proj[q_off + j * dim + i];  // q_proj[j][i] = q_proj[j*dim + i]
            float hf = sm2_f16_to_float(h);
            q[i] += xb[j] * hf;
        }
    }
    fprintf(stderr, "DEBUG: Layer %d: q_proj done, q[0]=%f\n", layer, q[0]);
    
    // 3. K projection: k = xb @ k_proj.T  (k_proj is [kv_dim, dim])
    memset(k, 0, kv_dim * sizeof(float));
    for (int i = 0; i < kv_dim; i++) {
        for (int j = 0; j < dim; j++) {
            uint16_t h = model->k_proj[k_off + j * kv_dim + i];
            k[i] += xb[j] * sm2_f16_to_float(h);
        }
    }
    
    // 4. V projection: v = xb @ v_proj.T  (v_proj is [kv_dim, dim])
    memset(v, 0, kv_dim * sizeof(float));
    for (int i = 0; i < kv_dim; i++) {
        for (int j = 0; j < dim; j++) {
            uint16_t h = model->v_proj[v_off + j * kv_dim + i];
            v[i] += xb[j] * sm2_f16_to_float(h);
        }
    }
    
    // 5. Apply RoPE to Q and K
    sm2_rope(q, k, model->head_dim, seq_pos, model->n_heads, model->n_kv_heads, spec->rope_theta);
    
    // 6. Attention (simplified - no KV cache yet)
    attention_forward(attn_out, q, k, v, model->n_heads, model->n_kv_heads, model->head_dim, seq_pos + 1);
    
    // 7. O projection: xb = attn_out @ o_proj.T  (o_proj is [dim, kv_dim])
    memset(xb, 0, dim * sizeof(float));
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < kv_dim; j++) {
            uint16_t h = model->o_proj[o_off + j * dim + i];
            xb[i] += attn_out[j] * sm2_f16_to_float(h);
        }
    }
    
    // 8. Residual: xb += residual (original input was xb before layernorm, now add)
    for (int i = 0; i < dim; i++) {
        xb[i] = x[i] + xb[i];
    }
    
    // 9. Post-attention RMSNorm
    for (int i = 0; i < dim; i++) {
        xb2[i] = xb[i] * sm2_f16_to_float(model->post_attention_layernorm[post_off + i]);
    }
    
    // 10. SwiGLU FFN: gate = xb2 @ gate_proj.T, up = xb2 @ up_proj.T, down = silu(gate) * up @ down_proj.T
    // gate_proj: [hidden_dim, dim], up_proj: [hidden_dim, dim], down_proj: [dim, hidden_dim]
    float* gate_out = ctx->scratch.q;   // reuse scratch
    float* up_out = ctx->scratch.k;    // reuse scratch
    
    // gate = xb2 @ gate_proj.T -> [hidden_dim]
    memset(gate_out, 0, hidden_dim * sizeof(float));
    for (int i = 0; i < hidden_dim; i++) {
        for (int j = 0; j < dim; j++) {
            uint16_t h = model->gate_proj[gate_off + j * hidden_dim + i];
            gate_out[i] += xb2[j] * sm2_f16_to_float(h);
        }
    }
    
    // up = xb2 @ up_proj.T -> [hidden_dim]
    memset(up_out, 0, hidden_dim * sizeof(float));
    for (int i = 0; i < hidden_dim; i++) {
        for (int j = 0; j < dim; j++) {
            uint16_t h = model->up_proj[up_off + j * hidden_dim + i];
            up_out[i] += xb2[j] * sm2_f16_to_float(h);
        }
    }
    
    // Apply SiLU to gate: silu(x) = x * sigmoid(x)
    for (int i = 0; i < hidden_dim; i++) {
        float s = 1.0f / (1.0f + expf(-gate_out[i]));
        gate_out[i] = gate_out[i] * s;
    }
    
    // Multiply: gate_out = silu(gate) * up
    for (int i = 0; i < hidden_dim; i++) {
        gate_out[i] *= up_out[i];
    }
    
    // down = gate_out @ down_proj.T -> [dim]
    memset(xb2, 0, dim * sizeof(float));
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < hidden_dim; j++) {
            uint16_t h = model->down_proj[down_off + j * dim + i];
            xb2[i] += gate_out[j] * sm2_f16_to_float(h);
        }
    }
    
    // 11. Final residual: xb += ffn_output
    for (int i = 0; i < dim; i++) {
        xb[i] = xb[i] + xb2[i];
    }
}

// ============================================================================
// PREFILL - Process entire prompt
// ============================================================================

int sm2_prefill(sm2_context* ctx, const int* tokens, int n_tokens) {
    sm2_model* model = ctx->model;
    
    fprintf(stderr, "sm2_prefill ENTRY: tokens=%p, n_tokens=%d, scratch.x=%p\n",
            (void*)tokens, n_tokens, (void*)ctx->scratch.x);
    fflush(stderr);
    
    // Embed first token
    float* x = ctx->scratch.x;
    
    fprintf(stderr, "sm2_prefill: tok_embeddings=%p, data=%p\n",
            (void*)model->tok_embeddings,
            (void*)(model->tok_embeddings ? model->tok_embeddings->data : NULL));
    
    if (model->tok_embeddings && model->tok_embeddings->data) {
        fprintf(stderr, "sm2_prefill: calling embedding_lookup for token %d\n", tokens[0]);
        embedding_lookup(x, tokens[0], model->tok_embeddings);
    } else {
        // Fallback: simple positional embedding
        fprintf(stderr, "sm2_prefill: using fallback embedding\n");
        for (int i = 0; i < model->dim; i++) {
            x[i] = (float)(tokens[0] % 256) / 256.0f;
        }
    }
    
    fprintf(stderr, "sm2_prefill: x[0]=%f, x[1]=%f\n", x[0], x[1]);
    
    // Forward through all layers
    for (int layer = 0; layer < model->n_layers; layer++) {
        fprintf(stderr, "sm2_prefill: calling layer_forward for layer %d\n", layer);
        layer_forward(ctx->scratch.xb, x, layer, model, ctx, n_tokens - 1);
        
        // Swap xb and x for next layer
        float* tmp = ctx->scratch.x;
        ctx->scratch.x = ctx->scratch.xb;
        ctx->scratch.xb = tmp;
    }
    
    // Final RMSNorm (apply final_norm weight)
    if (model->final_norm) {
        float mean = 0.0f;
        for (int i = 0; i < model->dim; i++) mean += ctx->scratch.x[i];
        mean /= model->dim;
        for (int i = 0; i < model->dim; i++) {
            float v = ctx->scratch.x[i] - mean;
            float w = sm2_f16_to_float(model->final_norm[i]);
            ctx->scratch.x[i] = v * w;
        }
    }
    
    ctx->pos = n_tokens;
    
    // Compute logits
    // logits = x @ lm_head (shared with embeddings)
    if (model->tok_embeddings && model->tok_embeddings->data) {
        for (int i = 0; i < model->vocab_size; i++) {
            float sum = 0.0f;
            for (int j = 0; j < model->dim; j++) {
                uint16_t h = model->tok_embeddings->data[i * model->dim + j];
                sum += ctx->scratch.x[j] * f16_to_float(h);
            }
            ctx->scratch.logits[i] = sum;
        }
    }
    
    return 0;
}

// ============================================================================
// DECODE NEXT - Generate single token (no allocation in hot path)
// ============================================================================

int sm2_decode_next(sm2_context* ctx, int* out_token) {
    // Sample from logits
    int token = sm2_sample_token(ctx->scratch.logits, &ctx->params, &ctx->rng_state);
    
    // Update position
    ctx->pos++;
    ctx->last_token = token;
    
    // For decode, we need to compute logits for next token
    // Get embedding for this token
    float* x = ctx->scratch.x;
    
    if (ctx->model->tok_embeddings && ctx->model->tok_embeddings->data) {
        embedding_lookup(x, token, ctx->model->tok_embeddings);
    } else {
        for (int i = 0; i < ctx->model->dim; i++) {
            x[i] = (float)(token % 256) / 256.0f;
        }
    }
    
    // Forward through layers
    for (int layer = 0; layer < ctx->model->n_layers; layer++) {
        layer_forward(ctx->scratch.xb, x, layer, ctx->model, ctx, ctx->pos);
        
        float* tmp = ctx->scratch.x;
        ctx->scratch.x = ctx->scratch.xb;
        ctx->scratch.xb = tmp;
    }
    
    // Final norm and logits
    // Apply final_norm weight (simplified RMSNorm)
    if (ctx->model->final_norm) {
        float mean = 0.0f;
        for (int i = 0; i < ctx->model->dim; i++) mean += ctx->scratch.x[i];
        mean /= ctx->model->dim;
        for (int i = 0; i < ctx->model->dim; i++) {
            float v = ctx->scratch.x[i] - mean;
            float w = sm2_f16_to_float(ctx->model->final_norm[i]);
            ctx->scratch.x[i] = v * w;
        }
    }
    
    // Compute logits for next token
    if (ctx->model->tok_embeddings && ctx->model->tok_embeddings->data) {
        for (int i = 0; i < ctx->model->vocab_size; i++) {
            float sum = 0.0f;
            for (int j = 0; j < ctx->model->dim; j++) {
                uint16_t h = ctx->model->tok_embeddings->data[i * ctx->model->dim + j];
                sum += ctx->scratch.x[j] * f16_to_float(h);
            }
            ctx->scratch.logits[i] = sum;
        }
    }
    
    *out_token = token;
    return 0;
}

// ============================================================================
// DEBUG: Print top N logits
// ============================================================================

void sm2_debug_print_logits(sm2_context* ctx, int top_n) {
    float* logits = ctx->scratch.logits;
    int vocab_size = ctx->model->vocab_size;
    
    // Simple approach: just print first 20 logits
    fprintf(stderr, "First 20 logits:");
    for (int i = 0; i < 20 && i < vocab_size; i++) {
        fprintf(stderr, " [%d]=%.2f", i, logits[i]);
    }
    fprintf(stderr, "\n");
    
    // Find top N
    fprintf(stderr, "Top %d logits:\n", top_n);
    for (int t = 0; t < top_n; t++) {
        int max_idx = 0;
        float max_val = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }
        fprintf(stderr, "  [%d] = %.4f\n", max_idx, max_val);
        logits[max_idx] = -1e9; // Mark as done
    }
}

// ============================================================================
// STREAMING DECODE
// ============================================================================

int sm2_decode_stream(sm2_context* ctx, int max_new_tokens, sm2_stream_cb cb, void* user_data) {
    int n = 0;
    int token;
    
    while (n < max_new_tokens) {
        int ok = sm2_decode_next(ctx, &token);
        if (ok != 0 || token == 2) break; // EOS
        
        if (cb) cb(token, user_data);
        n++;
    }
    
    return n;
}