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
    sm2_context* ctx = calloc(1, sizeof(sm2_context));
    if (!ctx) return -1;
    
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
    
    ctx->scratch.x = calloc(dim, sizeof(float));
    ctx->scratch.xb = calloc(dim, sizeof(float));
    ctx->scratch.q = calloc(dim, sizeof(float));         // Used for Q
    ctx->scratch.k = calloc(kv_size, sizeof(float));      // Used for K
    ctx->scratch.v = calloc(kv_size, sizeof(float));      // Used for V
    ctx->scratch.attn_out = calloc(dim, sizeof(float));
    ctx->scratch.logits = calloc(vocab, sizeof(float));
    
    // FFN scratch buffers (hidden_dim + dim for gate/up and down output)
    ctx->scratch.xb2 = calloc(hidden + dim, sizeof(float)); // FFN temp
    
    if (!ctx->scratch.x || !ctx->scratch.xb || !ctx->scratch.q ||
        !ctx->scratch.k || !ctx->scratch.v || !ctx->scratch.attn_out ||
        !ctx->scratch.logits || !ctx->scratch.xb2) {
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
    ctx->kv_pool = calloc(1, sizeof(sm2_kv_pool));
    if (!ctx->kv_pool) {
        sm2_free_context(ctx);
        return -1;
    }
    
    // Simple contiguous KV for Phase 1
    int max_tokens = spec->max_seq_len;
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
    uint8_t* kv_data = calloc(kv_bytes, 1);
    
    ctx->kv.seq_len = 0;
    ctx->kv.n_pages = 1;
    ctx->kv.page_ids[0] = 0;
    
    *out_ctx = ctx;
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
    const sm2_spec* spec = sm2_get_spec(model->variant);
    int dim = model->dim;
    int kv_dim = model->n_kv_heads * model->head_dim;
    int hidden_dim = model->hidden_dim;
    
    // ALWAYS print layer 28 entry
    // if (layer == 28) {
    //     // layer_forward: ENTER layer 28
    // }
    
    // 1. RMSNorm (input layernorm): xb = rmsnorm(x, input_layernorm[layer])
    size_t ln_off   = (size_t)layer * dim;
    size_t q_off    = (size_t)layer * dim * dim;
    size_t k_off    = (size_t)layer * kv_dim * dim;
    size_t v_off    = (size_t)layer * kv_dim * dim;
    size_t o_off    = (size_t)layer * dim * dim;
    size_t post_off = (size_t)layer * dim;
    size_t gate_off = (size_t)layer * hidden_dim * dim;
    size_t up_off   = (size_t)layer * hidden_dim * dim;
    size_t down_off = (size_t)layer * dim * hidden_dim;
    
    float* q = ctx->scratch.q;
    float* k = ctx->scratch.k;
    float* v = ctx->scratch.v;
    float* attn_out = ctx->scratch.attn_out;
    float* xb2 = ctx->scratch.xb;  // reuse xb buffer
    
    // Debug: print input x values for layer 0
    // if (layer == 0) {
    //     fprintf(stderr, "DEBUG layer 0 START: x[0]=%f, x[1]=%f\n", x[0], x[1]);
    // }
    
    // 1. RMSNorm (input layernorm): xb = rmsnorm(x, input_layernorm[layer])
    // Proper RMSNorm: xb = (x / rms(x)) * weight
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = sqrtf(sum_sq / (float)dim + 1e-5f);
    for (int i = 0; i < dim; i++) {
        uint16_t w = model->input_layernorm[ln_off + i];
        float wf = sm2_f16_to_float(w);
        xb[i] = (x[i] / rms) * wf;
    }
    // DEBUG: Layer 0 RMSNorm only
    // if (layer == 0) { fprintf(stderr, "DEBUG: Layer 0 RMSNorm: xb[0]=%f, xb[1]=%f, rms=%f\n", xb[0], xb[1], rms); }
    
    // 2. Q projection: q = xb @ q_proj.T  (q_proj is [dim, dim], stored row-major)
    // We want q[i] = sum_j xb[j] * q_proj[j][i]  =>  xb @ q_proj^T
    // q is [dim], q_proj is [dim x dim] row-major, so q_proj[j][i] = q_proj[j*dim + i]
    memset(q, 0, dim * sizeof(float));
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            uint16_t h = model->q_proj[q_off + j * dim + i];  // q_proj[j][i] = q_proj[j*dim + i]
            float hf = sm2_f16_to_float(h);
            q[i] += xb[j] * hf;
        }
    }
    
    // 3. K projection: k = xb @ k_proj.T  (k_proj is [kv_dim, dim], stored row-major)
    // k_proj[j][i] = k_proj[j * dim + i] (row-major with dim columns)
    memset(k, 0, kv_dim * sizeof(float));
    for (int i = 0; i < kv_dim; i++) {
        for (int j = 0; j < dim; j++) {
            uint16_t h = model->k_proj[k_off + j * kv_dim + i];
            k[i] += xb[j] * sm2_f16_to_float(h);
        }
    }
    
    // 4. V projection: v = xb @ v_proj.T  (v_proj is [kv_dim, dim], stored row-major)
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
    // For prefill with single token, seq_len=1 (attend only to current position)
    // K and V are computed for current position only, so only attend to pos 0
    attention_forward(attn_out, q, k, v, model->n_heads, model->n_kv_heads, model->head_dim, 1);
    
    // 7. O projection: xb = attn_out @ o_proj.T  (o_proj is [dim, dim])
    memset(xb, 0, dim * sizeof(float));
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            uint16_t h = model->o_proj[o_off + j * dim + i];
            xb[i] += attn_out[j] * sm2_f16_to_float(h);
        }
    }
    
    // 8. Residual: xb += residual (original input was xb before layernorm, now add)
    for (int i = 0; i < dim; i++) {
        xb[i] = x[i] + xb[i];
    }
    
    // 9. Post-attention RMSNorm: xb2 = (xb / rms(xb)) * post_attention_layernorm
    // Compute rms of xb (the residual output after attention)
    {
        float sum_sq2 = 0.0f;
        for (int i = 0; i < dim; i++) {
            sum_sq2 += xb[i] * xb[i];
        }
        float rms2 = sqrtf(sum_sq2 / (float)dim + 1e-5f);
        for (int i = 0; i < dim; i++) {
            uint16_t w = model->post_attention_layernorm[post_off + i];
            float wf = sm2_f16_to_float(w);
            xb2[i] = (xb[i] / rms2) * wf;
        }
    }
    
    // 10. SwiGLU FFN: gate = xb2 @ gate_proj.T, up = xb2 @ up_proj.T, down = silu(gate) * up @ down_proj.T
    // gate_proj: [hidden_dim, dim] stored row-major, gate_proj[i][j] = gate_proj[i * dim + j]
    // up_proj: [hidden_dim, dim] stored row-major, up_proj[i][j] = up_proj[i * dim + j]
    // down_proj: [dim, hidden_dim] stored row-major, down_proj[i][j] = down_proj[i * hidden_dim + j]
    // Use xb2[0..hidden_dim-1] for gate_out, xb2[hidden_dim..2*hidden_dim-1] for up_out
    float* gate_out = ctx->scratch.xb2;
    float* up_out = ctx->scratch.xb2 + hidden_dim;
    
    // gate = xb2 @ gate_proj.T -> [hidden_dim]
    // gate_proj: [hidden_dim, dim] row-major, flat[i*dim + j] = element[i][j]
    // gate_proj.T[i][j] = gate_proj[j][i] = flat[j*dim + i]
    memset(gate_out, 0, hidden_dim * sizeof(float));
    for (int i = 0; i < hidden_dim; i++) {
        for (int j = 0; j < dim; j++) {
            uint16_t h = model->gate_proj[gate_off + j * dim + i];
            gate_out[i] += xb2[j] * sm2_f16_to_float(h);
        }
    }
    
    // up = xb2 @ up_proj.T -> [hidden_dim]
    // up_proj: [hidden_dim, dim] row-major, flat[i*dim + j] = element[i][j]
    // up_proj.T[i][j] = up_proj[j][i] = flat[j*dim + i]
    memset(up_out, 0, hidden_dim * sizeof(float));
    for (int i = 0; i < hidden_dim; i++) {
        for (int j = 0; j < dim; j++) {
            uint16_t h = model->up_proj[up_off + j * dim + i];
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
    // down_proj: [dim, hidden_dim] row-major, flat[i*hidden_dim + j] = element[i][j]
    // down_proj.T[i][j] = down_proj[j][i] = flat[j*hidden_dim + i]
    // Compute directly into xb (which is free since we already saved residual in step 8)
    for (int i = 0; i < dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < hidden_dim; j++) {
            uint16_t h = model->down_proj[down_off + i * hidden_dim + j];
            sum += gate_out[j] * sm2_f16_to_float(h);
        }
        xb[i] += sum;  // Add to residual (xb already has residual from step 8)
    }
    
    // Debug: after FFN + residual
    // if (layer == 1) {
    //     fprintf(stderr, "DEBUG layer 1: after FFN + residual, xb[0]=%f, xb[1]=%f\n", xb[0], xb[1]);
    // }
}

// ============================================================================
// PREFILL - Process entire prompt
// ============================================================================

int sm2_prefill(sm2_context* ctx, const int* tokens, int n_tokens) {
    sm2_model* model = ctx->model;
    
    // Embed first token
    float* x = ctx->scratch.x;
    
    if (model->tok_embeddings && model->tok_embeddings->data) {
        embedding_lookup(x, tokens[0], model->tok_embeddings);
    } else {
        for (int i = 0; i < model->dim; i++) {
            x[i] = (float)(tokens[0] % 256) / 256.0f;
        }
    }
    
    // Forward through all layers
    for (int layer = 0; layer < model->n_layers; layer++) {
        layer_forward(ctx->scratch.xb, x, layer, model, ctx, n_tokens - 1);
        
        // Swap xb and x for next layer
        float* tmp = ctx->scratch.x;
        ctx->scratch.x = ctx->scratch.xb;
        ctx->scratch.xb = tmp;
    }
    
    // Compute logits from final hidden state
    // logits = tok_embeddings @ hidden_state.T  (vocab_size x dim)
    // Use scratch.x (which has the final hidden state after layer loop)
    if (model->tok_embeddings && model->tok_embeddings->data) {
        float* final_h = ctx->scratch.x;  // Final hidden state
        for (int i = 0; i < model->vocab_size; i++) {
            float sum = 0.0f;
            for (int j = 0; j < model->dim; j++) {
                uint16_t h = model->tok_embeddings->data[i * model->dim + j];
                sum += final_h[j] * sm2_f16_to_float(h);
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
    // Use current position for this token (don't increment yet)
    int seq_pos = ctx->pos;
    
    // Sample from logits (computed in previous iteration or prefill)
    int token = sm2_sample_token(ctx->scratch.logits, &ctx->params, &ctx->rng_state);
    
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
        layer_forward(ctx->scratch.xb, x, layer, ctx->model, ctx, seq_pos);
        
        float* tmp = ctx->scratch.x;
        ctx->scratch.x = ctx->scratch.xb;
        ctx->scratch.xb = tmp;
    }
    
    // Compute logits for next token
    if (ctx->model->tok_embeddings && ctx->model->tok_embeddings->data) {
        for (int i = 0; i < ctx->model->vocab_size; i++) {
            float sum = 0.0f;
            for (int j = 0; j < ctx->model->dim; j++) {
                uint16_t h = ctx->model->tok_embeddings->data[i * ctx->model->dim + j];
                sum += ctx->scratch.x[j] * sm2_f16_to_float(h);
            }
            ctx->scratch.logits[i] = sum;
        }
    }
    
    // Update position AFTER processing
    ctx->pos++;
    ctx->last_token = token;
    
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