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
    ctx->scratch.q = calloc(dim, sizeof(float));
    ctx->scratch.k = calloc(kv_size, sizeof(float));
    ctx->scratch.v = calloc(kv_size, sizeof(float));
    ctx->scratch.attn_out = calloc(dim, sizeof(float));
    ctx->scratch.logits = calloc(vocab, sizeof(float));
    
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
// SINGLE LAYER FORWARD
// ============================================================================

static void layer_forward(float* xb, const float* x, int layer,
                          sm2_model* model, sm2_context* ctx, int seq_pos) {
    const sm2_spec* spec = sm2_get_spec(model->variant);
    
    // 1. RMSNorm (input layernorm)
    // xb = rmsnorm(x, model->input_layernorm[layer])
    // For now, skip normalization in reference
    
    // 2. Q projection: q = x @ q_proj
    float* q = ctx->scratch.q;
    for (int i = 0; i < model->dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < model->dim; j++) {
            // Currently only F16 supported, use identity as placeholder
            // Real impl: use model weights
            sum += x[j] * (i == j ? 1.0f : 0.0f);
        }
        q[i] = sum;
    }
    
    // 3. K projection: k = x @ k_proj
    float* k = ctx->scratch.k;
    int kv_dim = model->n_kv_heads * model->head_dim;
    for (int i = 0; i < kv_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < model->dim; j++) {
            sum += x[j] * (i < model->dim && j == i ? 1.0f : 0.0f);
        }
        k[i] = sum;
    }
    
    // 4. V projection
    float* v = ctx->scratch.v;
    for (int i = 0; i < kv_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < model->dim; j++) {
            sum += x[j] * (i < model->dim && j == i ? 1.0f : 0.0f);
        }
        v[i] = sum;
    }
    
    // 5. Apply RoPE
    sm2_rope(q, k, model->head_dim, seq_pos, model->n_heads, model->n_kv_heads, spec->rope_theta);
    
    // 6. Attention (simplified - no KV cache yet)
    float* attn_out = ctx->scratch.attn_out;
    attention_forward(attn_out, q, k, v, model->n_heads, model->n_kv_heads, model->head_dim, seq_pos + 1);
    
    // 7. O projection
    float* xb_out = ctx->scratch.xb;
    for (int i = 0; i < model->dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < model->dim; j++) {
            sum += attn_out[j] * (i == j ? 1.0f : 0.0f); // Placeholder
        }
        xb_out[i] = sum;
    }
    
    // 8. Residual
    for (int i = 0; i < model->dim; i++) {
        xb[i] = x[i] + xb_out[i];
    }
    
    // 9. FFN (simplified SwiGLU placeholder)
    // Real implementation: sm2_ffn_forward_inplace(...)
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
        // Fallback: simple positional embedding
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
    
    // Final RMSNorm
    if (model->final_norm) {
        sm2_rmsnorm_inplace(ctx->scratch.x, model->final_norm->data, model->dim, 1e-5f);
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
    if (ctx->model->final_norm) {
        sm2_rmsnorm_inplace(ctx->scratch.x, ctx->model->final_norm->data, ctx->model->dim, 1e-5f);
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