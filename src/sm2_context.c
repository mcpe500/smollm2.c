// sm2_context.c - Context creation and core inference functions

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "smollm2.h"
#include "sm2_utils.h"

// ============================================================================
// F16 TO F32 LOOKUP TABLE (precomputed for fast weight access)
// ============================================================================

static float f16_to_f32_table[65536];
static int f16_table_initialized = 0;

static void init_f16_table(void) {
    if (f16_table_initialized) return;
    for (int i = 0; i < 65536; i++) {
        f16_to_f32_table[i] = sm2_f16_to_float((uint16_t)i);
    }
    f16_table_initialized = 1;
}

// Fast F16->F32 using lookup table (inline for speed)
static inline float f16_lookup(uint16_t h) {
    return f16_to_f32_table[h];
}

// ============================================================================
// CONTEXT ALLOCATION (preallocated, no runtime allocation in hot path)
// ============================================================================

int sm2_create_context(sm2_model* model, sm2_context** out_ctx) {
    sm2_context* ctx = calloc(1, sizeof(sm2_context));
    if (!ctx) return -1;

    // Initialize F16 lookup table (256KB, done once)
    init_f16_table();

    ctx->model = model;
    ctx->pos = 0;
    ctx->last_token = 1; // BOS
    ctx->rng_state = 123456789ULL; // Initial RNG state

    const sm2_spec* spec = sm2_get_spec(model->variant);
    int dim = model->dim;
    int hidden = model->hidden_dim;
    int vocab = model->vocab_size;
    int kv_dim = model->n_kv_heads * model->head_dim;
    int max_seq = spec->max_seq_len;
    int n_layers = model->n_layers;

    ctx->scratch.x = calloc(dim, sizeof(float));
    ctx->scratch.xb = calloc(dim, sizeof(float));
    ctx->scratch.q = calloc(dim, sizeof(float));
    ctx->scratch.k = calloc(kv_dim, sizeof(float));
    ctx->scratch.v = calloc(kv_dim, sizeof(float));
    ctx->scratch.attn_out = calloc(dim, sizeof(float));
    ctx->scratch.logits = calloc(vocab, sizeof(float));
    ctx->scratch.xb2 = calloc((size_t)hidden * 2, sizeof(float));

    if (!ctx->scratch.x || !ctx->scratch.xb || !ctx->scratch.q ||
        !ctx->scratch.k || !ctx->scratch.v || !ctx->scratch.attn_out ||
        !ctx->scratch.logits || !ctx->scratch.xb2) {
        sm2_free_context(ctx);
        return -1;
    }

    // Allocate KV cache: [n_layers][n_kv_heads][max_seq][head_dim] flat as float
    size_t kv_per_layer = (size_t)model->n_kv_heads * max_seq * model->head_dim;
    ctx->scratch.k_cache = calloc(n_layers, sizeof(float*));
    ctx->scratch.v_cache = calloc(n_layers, sizeof(float*));
    if (!ctx->scratch.k_cache || !ctx->scratch.v_cache) {
        sm2_free_context(ctx);
        return -1;
    }
    for (int l = 0; l < n_layers; l++) {
        ctx->scratch.k_cache[l] = calloc(kv_per_layer, sizeof(float));
        ctx->scratch.v_cache[l] = calloc(kv_per_layer, sizeof(float));
        if (!ctx->scratch.k_cache[l] || !ctx->scratch.v_cache[l]) {
            sm2_free_context(ctx);
            return -1;
        }
    }
    ctx->scratch.kv_cache_len = 0;

    ctx->params.temperature = 0.0f;  // Greedy by default
    ctx->params.top_p = 100;
    ctx->params.top_k = 0;
    ctx->params.max_context = max_seq;
    ctx->params.max_output = 256;
    ctx->params.repetition_penalty = 1.0f;  // Disabled by default
    ctx->params.penalty_window = 100;       // Check last 100 tokens

    // Ring buffer for repetition penalty tracking
    ctx->scratch.recent_max = max_seq;
    ctx->scratch.recent_tokens = calloc(max_seq, sizeof(int));
    ctx->scratch.recent_head = 0;
    if (!ctx->scratch.recent_tokens) {
        sm2_free_context(ctx);
        return -1;
    }

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
    if (ctx->scratch.xb2) free(ctx->scratch.xb2);
    if (ctx->scratch.recent_tokens) free(ctx->scratch.recent_tokens);

    if (ctx->scratch.k_cache) {
        for (int l = 0; l < ctx->model->n_layers; l++) {
            free(ctx->scratch.k_cache[l]);
            free(ctx->scratch.v_cache[l]);
        }
        free(ctx->scratch.k_cache);
        free(ctx->scratch.v_cache);
    }

    free(ctx);
    return 0;
}

// ============================================================================
// EMBEDDING LOOKUP
// ============================================================================

static void embedding_lookup(float* out, int token, const sm2_tensor_f16* embed) {
    int dim = embed->cols;
    uint16_t* data = embed->data + token * dim;
    int i = 0;
    // 4x unrolled
    for (; i + 3 < dim; i += 4) {
        out[i+0] = f16_lookup(data[i+0]);
        out[i+1] = f16_lookup(data[i+1]);
        out[i+2] = f16_lookup(data[i+2]);
        out[i+3] = f16_lookup(data[i+3]);
    }
    for (; i < dim; i++) {
        out[i] = f16_lookup(data[i]);
    }
}

// ============================================================================
// ATTENTION with KV cache
// ============================================================================

static void attention_with_cache(float* out, const float* q, int layer,
                              sm2_context* ctx, sm2_model* model) {
    int n_heads = model->n_heads;
    int n_kv_heads = model->n_kv_heads;
    int head_dim = model->head_dim;
    int group_size = n_heads / n_kv_heads;
    // K/V for the current position has already been stored by layer_forward
    // at position kv_cache_len. So we must attend over kv_cache_len + 1 positions.
    int kv_seq = ctx->scratch.kv_cache_len + 1;

    // Precompute scale factor (avoid sqrtf in hot loop)
    float scale = 1.0f / sqrtf((float)head_dim);
    int stride = ctx->params.max_context * head_dim;

    // For each query head, attend over all cached K/V positions
    for (int qh = 0; qh < n_heads; qh++) {
        int kv_head = qh / group_size;
        const float* q_head = q + qh * head_dim;

        // Compute attention scores for all cached positions
        float max_score = -1e9f;
        float scores[256];  // max 256 for typical decode

        for (int pos = 0; pos < kv_seq; pos++) {
            float* k_cache = ctx->scratch.k_cache[layer] + kv_head * stride + pos * head_dim;

            // 4x unrolled dot product
            float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
            int d = 0;
            for (; d + 3 < head_dim; d += 4) {
                dot0 += q_head[d+0] * k_cache[d+0];
                dot1 += q_head[d+1] * k_cache[d+1];
                dot2 += q_head[d+2] * k_cache[d+2];
                dot3 += q_head[d+3] * k_cache[d+3];
            }
            float dot = dot0 + dot1 + dot2 + dot3;
            for (; d < head_dim; d++) {
                dot += q_head[d] * k_cache[d];
            }
            dot *= scale;
            scores[pos] = dot;
            if (dot > max_score) max_score = dot;
        }

        // Softmax
        float sum_exp = 0.0f;
        for (int pos = 0; pos < kv_seq; pos++) {
            scores[pos] = expf(scores[pos] - max_score);
            sum_exp += scores[pos];
        }

        // Weighted sum
        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (int pos = 0; pos < kv_seq; pos++) {
                int cache_idx = kv_head * ctx->params.max_context * head_dim + pos * head_dim + d;
                float vv = ctx->scratch.v_cache[layer][cache_idx];
                sum += scores[pos] * vv;
            }
            out[qh * head_dim + d] = sum / sum_exp;
        }
    }
}

// ============================================================================
// SINGLE LAYER FORWARD
// Writes result to xb_out. Does NOT use ctx->scratch.x or ctx->scratch.xb
// as those are the caller's ping-pong buffers.
// ============================================================================

static void layer_forward(float* xb_out, const float* x_in, int layer,
                          sm2_model* model, sm2_context* ctx, int seq_pos) {
    const sm2_spec* spec = sm2_get_spec(model->variant);
    int dim = model->dim;
    int kv_dim = model->n_kv_heads * model->head_dim;
    int hidden_dim = model->hidden_dim;

    // xb_out is the output buffer, x_in is the input (do NOT modify)
    // Use scratch space for Q/K/V projections and attention output
    float* q = ctx->scratch.q;
    float* k = ctx->scratch.k;
    float* v = ctx->scratch.v;
    float* attn_out = ctx->scratch.attn_out;
    float* ffn_temp = ctx->scratch.xb2;  // [hidden_dim * 2]

    // Save x_in for residual connections (we'll overwrite xb_out with RMSNorm)
    float x_residual[DIM_MAX];
    for (int i = 0; i < dim; i++) x_residual[i] = x_in[i];

    size_t ln_off = (size_t)layer * dim;
    size_t q_off = (size_t)layer * dim * dim;
    size_t k_off = (size_t)layer * kv_dim * dim;
    size_t v_off = (size_t)layer * kv_dim * dim;
    size_t o_off = (size_t)layer * dim * dim;
    size_t post_off = (size_t)layer * dim;
    size_t gate_off = (size_t)layer * hidden_dim * dim;
    size_t up_off = (size_t)layer * hidden_dim * dim;
    size_t down_off = (size_t)layer * dim * hidden_dim;

    // 1. RMSNorm on x_in, write to xb_out
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += x_in[i] * x_in[i];
    float rms = sqrtf(sum_sq / (float)dim + 1e-5f);
    for (int i = 0; i < dim; i++) {
        uint16_t w = model->input_layernorm[ln_off + i];
        xb_out[i] = (x_in[i] / rms) * f16_lookup(w);
    }

    // 2. Q projection: q = xb_out @ q_proj.T (with 4x unrolling)
    for (int i = 0; i < dim; i++) {
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        int j = 0;
        for (; j + 3 < dim; j += 4) {
            sum0 += xb_out[j+0] * f16_lookup(model->q_proj[q_off + i * dim + j + 0]);
            sum1 += xb_out[j+1] * f16_lookup(model->q_proj[q_off + i * dim + j + 1]);
            sum2 += xb_out[j+2] * f16_lookup(model->q_proj[q_off + i * dim + j + 2]);
            sum3 += xb_out[j+3] * f16_lookup(model->q_proj[q_off + i * dim + j + 3]);
        }
        for (; j < dim; j++) {
            sum0 += xb_out[j] * f16_lookup(model->q_proj[q_off + i * dim + j]);
        }
        q[i] = sum0 + sum1 + sum2 + sum3;
    }

    // 3. K projection: k = xb_out @ k_proj.T (with 4x unrolling)
    for (int i = 0; i < kv_dim; i++) {
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        int j = 0;
        for (; j + 3 < dim; j += 4) {
            sum0 += xb_out[j+0] * f16_lookup(model->k_proj[k_off + i * dim + j + 0]);
            sum1 += xb_out[j+1] * f16_lookup(model->k_proj[k_off + i * dim + j + 1]);
            sum2 += xb_out[j+2] * f16_lookup(model->k_proj[k_off + i * dim + j + 2]);
            sum3 += xb_out[j+3] * f16_lookup(model->k_proj[k_off + i * dim + j + 3]);
        }
        for (; j < dim; j++) {
            sum0 += xb_out[j] * f16_lookup(model->k_proj[k_off + i * dim + j]);
        }
        k[i] = sum0 + sum1 + sum2 + sum3;
    }

    // 4. V projection: v = xb_out @ v_proj.T (with 4x unrolling)
    for (int i = 0; i < kv_dim; i++) {
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        int j = 0;
        for (; j + 3 < dim; j += 4) {
            sum0 += xb_out[j+0] * f16_lookup(model->v_proj[v_off + i * dim + j + 0]);
            sum1 += xb_out[j+1] * f16_lookup(model->v_proj[v_off + i * dim + j + 1]);
            sum2 += xb_out[j+2] * f16_lookup(model->v_proj[v_off + i * dim + j + 2]);
            sum3 += xb_out[j+3] * f16_lookup(model->v_proj[v_off + i * dim + j + 3]);
        }
        for (; j < dim; j++) {
            sum0 += xb_out[j] * f16_lookup(model->v_proj[v_off + i * dim + j]);
        }
        v[i] = sum0 + sum1 + sum2 + sum3;
    }

    // 5. RoPE on Q and K
    sm2_rope(q, k, model->head_dim, seq_pos, model->n_heads, model->n_kv_heads, spec->rope_theta);

    // 6. Store K/V to cache at position seq_pos
    int kv_head_dim = model->head_dim;
    for (int head = 0; head < model->n_kv_heads; head++) {
        for (int d = 0; d < kv_head_dim; d++) {
            int cache_idx = head * ctx->params.max_context * kv_head_dim + seq_pos * kv_head_dim + d;
            ctx->scratch.k_cache[layer][cache_idx] = k[head * kv_head_dim + d];
            ctx->scratch.v_cache[layer][cache_idx] = v[head * kv_head_dim + d];
        }
    }

    // 7. Attention with KV cache -> attn_out
    attention_with_cache(attn_out, q, layer, ctx, model);

    // 8. O projection: xb_out = attn_out @ o_proj.T (with 4x unrolling)
    for (int i = 0; i < dim; i++) {
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        int j = 0;
        for (; j + 3 < dim; j += 4) {
            sum0 += attn_out[j+0] * f16_lookup(model->o_proj[o_off + i * dim + j + 0]);
            sum1 += attn_out[j+1] * f16_lookup(model->o_proj[o_off + i * dim + j + 1]);
            sum2 += attn_out[j+2] * f16_lookup(model->o_proj[o_off + i * dim + j + 2]);
            sum3 += attn_out[j+3] * f16_lookup(model->o_proj[o_off + i * dim + j + 3]);
        }
        for (; j < dim; j++) {
            sum0 += attn_out[j] * f16_lookup(model->o_proj[o_off + i * dim + j]);
        }
        xb_out[i] = sum0 + sum1 + sum2 + sum3;
    }

    // 9. Residual: xb_out += original input (for attention)
    for (int i = 0; i < dim; i++) {
        xb_out[i] += x_residual[i];
    }

    // Save post-attention hidden state for FFN residual
    float x_post_attn[DIM_MAX];
    for (int i = 0; i < dim; i++) x_post_attn[i] = xb_out[i];

    // 10. Post-attention RMSNorm on xb_out, store back to xb_out
    sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += xb_out[i] * xb_out[i];
    rms = sqrtf(sum_sq / (float)dim + 1e-5f);
    for (int i = 0; i < dim; i++) {
        uint16_t w = model->post_attention_layernorm[post_off + i];
        xb_out[i] = (xb_out[i] / rms) * f16_lookup(w);
    }

    // 11. SwiGLU FFN using ffn_temp as workspace
    float* gate_out = ffn_temp;                  // first hidden_dim elements
    float* up_out = ffn_temp + hidden_dim;      // second hidden_dim elements

    // gate = xb_out @ gate_proj.T (with 4x unrolling)
    for (int i = 0; i < hidden_dim; i++) {
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        int j = 0;
        for (; j + 3 < dim; j += 4) {
            sum0 += xb_out[j+0] * f16_lookup(model->gate_proj[gate_off + i * dim + j + 0]);
            sum1 += xb_out[j+1] * f16_lookup(model->gate_proj[gate_off + i * dim + j + 1]);
            sum2 += xb_out[j+2] * f16_lookup(model->gate_proj[gate_off + i * dim + j + 2]);
            sum3 += xb_out[j+3] * f16_lookup(model->gate_proj[gate_off + i * dim + j + 3]);
        }
        for (; j < dim; j++) {
            sum0 += xb_out[j] * f16_lookup(model->gate_proj[gate_off + i * dim + j]);
        }
        gate_out[i] = sum0 + sum1 + sum2 + sum3;
    }

    // up = xb_out @ up_proj.T (with 4x unrolling)
    for (int i = 0; i < hidden_dim; i++) {
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        int j = 0;
        for (; j + 3 < dim; j += 4) {
            sum0 += xb_out[j+0] * f16_lookup(model->up_proj[up_off + i * dim + j + 0]);
            sum1 += xb_out[j+1] * f16_lookup(model->up_proj[up_off + i * dim + j + 1]);
            sum2 += xb_out[j+2] * f16_lookup(model->up_proj[up_off + i * dim + j + 2]);
            sum3 += xb_out[j+3] * f16_lookup(model->up_proj[up_off + i * dim + j + 3]);
        }
        for (; j < dim; j++) {
            sum0 += xb_out[j] * f16_lookup(model->up_proj[up_off + i * dim + j]);
        }
        up_out[i] = sum0 + sum1 + sum2 + sum3;
    }

    // SiLU: gate = gate * sigmoid(gate)
    for (int i = 0; i < hidden_dim; i++) {
        gate_out[i] *= 1.0f / (1.0f + expf(-gate_out[i]));
    }

    // Multiply: gate *= up
    for (int i = 0; i < hidden_dim; i++) gate_out[i] *= up_out[i];

    // down = gate_out @ down_proj.T (with 4x unrolling)
    for (int i = 0; i < dim; i++) {
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        int j = 0;
        for (; j + 3 < hidden_dim; j += 4) {
            sum0 += gate_out[j+0] * f16_lookup(model->down_proj[down_off + i * hidden_dim + j + 0]);
            sum1 += gate_out[j+1] * f16_lookup(model->down_proj[down_off + i * hidden_dim + j + 1]);
            sum2 += gate_out[j+2] * f16_lookup(model->down_proj[down_off + i * hidden_dim + j + 2]);
            sum3 += gate_out[j+3] * f16_lookup(model->down_proj[down_off + i * hidden_dim + j + 3]);
        }
        for (; j < hidden_dim; j++) {
            sum0 += gate_out[j] * f16_lookup(model->down_proj[down_off + i * hidden_dim + j]);
        }
        // FFN output + residual (post-attention hidden state)
        xb_out[i] = x_post_attn[i] + sum0 + sum1 + sum2 + sum3;
    }
    // xb_out now holds the final layer output
}

// ============================================================================
// PREFILL - Process entire prompt
// ============================================================================

int sm2_prefill(sm2_context* ctx, const int* tokens, int n_tokens) {
    sm2_model* model = ctx->model;

    // Process all layers for each token independently
    // This is correct for transformers: each token goes through all layers
    // The KV cache stores K/V so attention can attend to previous tokens
    for (int t = 0; t < n_tokens; t++) {
        int seq_pos = t;

        // Get embedding for current token into scratch.x
        if (model->tok_embeddings && model->tok_embeddings->data) {
            embedding_lookup(ctx->scratch.x, tokens[t], model->tok_embeddings);
        } else {
            for (int i = 0; i < model->dim; i++) ctx->scratch.x[i] = (float)(tokens[t] % 256) / 256.0f;
        }

        // Run this token through all layers
        // Each layer reads from scratch.x, writes to scratch.xb
        // After each layer, swap buffers so output becomes input for next layer
        for (int layer = 0; layer < model->n_layers; layer++) {
            layer_forward(ctx->scratch.xb, ctx->scratch.x, layer, model, ctx, seq_pos);
            // Swap input and output buffers
            float* tmp = ctx->scratch.x;
            ctx->scratch.x = ctx->scratch.xb;
            ctx->scratch.xb = tmp;
        }

        // After all layers, the final hidden state is in scratch.x
        // Update KV cache length and last_token
        ctx->scratch.kv_cache_len = t + 1;
        ctx->last_token = tokens[t];  // Track last prompt token for decode

        // Track prompt tokens for repetition penalty in decode
        if (ctx->scratch.recent_tokens && t < ctx->scratch.recent_max) {
            ctx->scratch.recent_tokens[t] = tokens[t];
        }
    }

    // Final RMSNorm
    if (model->final_norm) {
        float* final_h = ctx->scratch.x;
        float sum_sq = 0.0f;
        for (int i = 0; i < model->dim; i++) sum_sq += final_h[i] * final_h[i];
        float rms = sqrtf(sum_sq / (float)model->dim + 1e-5f);
        float scale = 1.0f / rms;
        for (int i = 0; i < model->dim; i++) {
            final_h[i] = final_h[i] * scale * f16_lookup(model->final_norm[i]);
        }
    }

    // Compute logits (with 4x unrolling)
    if (model->tok_embeddings && model->tok_embeddings->data) {
        float* final_h = ctx->scratch.x;
        int dim = model->dim;
        int vocab = model->vocab_size;
        for (int i = 0; i < vocab; i++) {
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
            int j = 0;
            for (; j + 3 < dim; j += 4) {
                sum0 += final_h[j+0] * f16_lookup(model->tok_embeddings->data[i * dim + j + 0]);
                sum1 += final_h[j+1] * f16_lookup(model->tok_embeddings->data[i * dim + j + 1]);
                sum2 += final_h[j+2] * f16_lookup(model->tok_embeddings->data[i * dim + j + 2]);
                sum3 += final_h[j+3] * f16_lookup(model->tok_embeddings->data[i * dim + j + 3]);
            }
            for (; j < dim; j++) {
                sum0 += final_h[j] * f16_lookup(model->tok_embeddings->data[i * dim + j]);
            }
            ctx->scratch.logits[i] = sum0 + sum1 + sum2 + sum3;
        }
    }

    return 0;
}

// ============================================================================
// DECODE NEXT - Generate single token
// ============================================================================

int sm2_decode_next(sm2_context* ctx, int* out_token) {
    // seq_pos is absolute position in the full sequence
    int seq_pos = ctx->scratch.kv_cache_len;  // Position right after all cached tokens

    // Get embedding for this token into ctx->scratch.x
    // The token to embed is the last generated token (ctx->last_token)
    if (ctx->model->tok_embeddings && ctx->model->tok_embeddings->data) {
        embedding_lookup(ctx->scratch.x, ctx->last_token, ctx->model->tok_embeddings);
    } else {
        for (int i = 0; i < ctx->model->dim; i++) ctx->scratch.x[i] = (float)(ctx->last_token % 256) / 256.0f;
    }

    // Forward through all layers
    // Note: After the loop, xb_out is in ctx->scratch.xb (last swap puts output there)
    for (int layer = 0; layer < ctx->model->n_layers; layer++) {
        layer_forward(ctx->scratch.xb, ctx->scratch.x, layer, ctx->model, ctx, seq_pos);
        // Swap input and output buffers
        float* tmp = ctx->scratch.x;
        ctx->scratch.x = ctx->scratch.xb;
        ctx->scratch.xb = tmp;
    }

    // Final RMSNorm - need to use the correct buffer (final hidden state is in scratch.x)
    if (ctx->model->final_norm) {
        float* final_h = ctx->scratch.x;
        float sum_sq = 0.0f;
        for (int i = 0; i < ctx->model->dim; i++) sum_sq += final_h[i] * final_h[i];
        float rms = sqrtf(sum_sq / (float)ctx->model->dim + 1e-5f);
        float scale = 1.0f / rms;
        for (int i = 0; i < ctx->model->dim; i++) {
            final_h[i] = final_h[i] * scale * f16_lookup(ctx->model->final_norm[i]);
        }
    }

    // Compute logits (with 4x unrolling)
    if (ctx->model->tok_embeddings && ctx->model->tok_embeddings->data) {
        float* final_h = ctx->scratch.x;
        int dim = ctx->model->dim;
        int vocab = ctx->model->vocab_size;
        for (int i = 0; i < vocab; i++) {
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
            int j = 0;
            for (; j + 3 < dim; j += 4) {
                sum0 += final_h[j+0] * f16_lookup(ctx->model->tok_embeddings->data[i * dim + j + 0]);
                sum1 += final_h[j+1] * f16_lookup(ctx->model->tok_embeddings->data[i * dim + j + 1]);
                sum2 += final_h[j+2] * f16_lookup(ctx->model->tok_embeddings->data[i * dim + j + 2]);
                sum3 += final_h[j+3] * f16_lookup(ctx->model->tok_embeddings->data[i * dim + j + 3]);
            }
            for (; j < dim; j++) {
                sum0 += final_h[j] * f16_lookup(ctx->model->tok_embeddings->data[i * dim + j]);
            }
            ctx->scratch.logits[i] = sum0 + sum1 + sum2 + sum3;
        }
    }

    // NOW sample from the newly computed logits
    int token = sm2_sample_token(ctx->scratch.logits, &ctx->params, &ctx->rng_state, ctx);

    // Track token for repetition penalty (add to next slot)
    if (ctx->scratch.recent_tokens && ctx->scratch.kv_cache_len < ctx->scratch.recent_max) {
        ctx->scratch.recent_tokens[ctx->scratch.kv_cache_len] = token;
    }

    // Update KV cache length to include this new token
    ctx->scratch.kv_cache_len = seq_pos + 1;
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

    fprintf(stderr, "First 20 logits:");
    for (int i = 0; i < 20 && i < vocab_size; i++) {
        fprintf(stderr, " [%d]=%.2f", i, logits[i]);
    }
    fprintf(stderr, "\n");

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
        logits[max_idx] = -1e9f;
    }
}

// ============================================================================
// STREAMING DECODE
// ============================================================================

int sm2_decode_stream(sm2_context* ctx, int max_new_tokens, sm2_stream_cb cb, void* user_data) {
    int n = 0;
    int token;
    (void)user_data; // unused

    while (n < max_new_tokens) {
        int ok = sm2_decode_next(ctx, &token);
        if (ok != 0 || token == 2) break; // EOS
        if (cb) cb(token, user_data);
        n++;
    }
    if (cb) cb(-1, user_data); // Signal end of stream

    return n;
}
