// sm2_context.c - Context creation and core inference functions

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "smollm2.h"
#include "sm2_utils.h"

// ============================================================================
// NEON-ACCELERATED MATMUL (ARM SIMD for 4x speedup)
// ============================================================================

#ifdef __ARM_NEON
#include <arm_neon.h>

// NEON matmul: y = x @ w where x is [1, k], w is [k, n] stored column-major
// Uses FMA (fused multiply-add) for maximum throughput
static inline void matmul_neon(float* out, const float* x, const float* w, int n, int k) {
    // Process4 output elements per iteration
    for (int j = 0; j < n; j += 4) {
        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);

        int l = 0;
        // Process 16 elements at a time for better pipelining
        for (; l + 15 < k; l += 16) {
            // Load x elements
            float32x4_t x0 = vdupq_n_f32(x[l+0]);
            float32x4_t x1 = vdupq_n_f32(x[l+1]);
            float32x4_t x2 = vdupq_n_f32(x[l+2]);
            float32x4_t x3 = vdupq_n_f32(x[l+3]);
            float32x4_t x4 = vdupq_n_f32(x[l+4]);
            float32x4_t x5 = vdupq_n_f32(x[l+5]);
            float32x4_t x6 = vdupq_n_f32(x[l+6]);
            float32x4_t x7 = vdupq_n_f32(x[l+7]);
            float32x4_t x8 = vdupq_n_f32(x[l+8]);
            float32x4_t x9 = vdupq_n_f32(x[l+9]);
            float32x4_t x10 = vdupq_n_f32(x[l+10]);
            float32x4_t x11 = vdupq_n_f32(x[l+11]);
            float32x4_t x12 = vdupq_n_f32(x[l+12]);
            float32x4_t x13 = vdupq_n_f32(x[l+13]);
            float32x4_t x14 = vdupq_n_f32(x[l+14]);
            float32x4_t x15 = vdupq_n_f32(x[l+15]);

            // Load w columns and FMA
            float32x4_t w0 = vld1q_f32(w + (l+0) * n + j);
            float32x4_t w1 = vld1q_f32(w + (l+1) * n + j);
            float32x4_t w2 = vld1q_f32(w + (l+2) * n + j);
            float32x4_t w3 = vld1q_f32(w + (l+3) * n + j);
            float32x4_t w4 = vld1q_f32(w + (l+4) * n + j);
            float32x4_t w5 = vld1q_f32(w + (l+5) * n + j);
            float32x4_t w6 = vld1q_f32(w + (l+6) * n + j);
            float32x4_t w7 = vld1q_f32(w + (l+7) * n + j);
            float32x4_t w8 = vld1q_f32(w + (l+8) * n + j);
            float32x4_t w9 = vld1q_f32(w + (l+9) * n + j);
            float32x4_t w10 = vld1q_f32(w + (l+10) * n + j);
            float32x4_t w11 = vld1q_f32(w + (l+11) * n + j);
            float32x4_t w12 = vld1q_f32(w + (l+12) * n + j);
            float32x4_t w13 = vld1q_f32(w + (l+13) * n + j);
            float32x4_t w14 = vld1q_f32(w + (l+14) * n + j);
            float32x4_t w15 = vld1q_f32(w + (l+15) * n + j);

            sum0 = vfmaq_f32(sum0, x0, w0);
            sum1 = vfmaq_f32(sum1, x1, w1);
            sum2 = vfmaq_f32(sum2, x2, w2);
            sum3 = vfmaq_f32(sum3, x3, w3);
            sum0 = vfmaq_f32(sum0, x4, w4);
            sum1 = vfmaq_f32(sum1, x5, w5);
            sum2 = vfmaq_f32(sum2, x6, w6);
            sum3 = vfmaq_f32(sum3, x7, w7);
            sum0 = vfmaq_f32(sum0, x8, w8);
            sum1 = vfmaq_f32(sum1, x9, w9);
            sum2 = vfmaq_f32(sum2, x10, w10);
            sum3 = vfmaq_f32(sum3, x11, w11);
            sum0 = vfmaq_f32(sum0, x12, w12);
            sum1 = vfmaq_f32(sum1, x13, w13);
            sum2 = vfmaq_f32(sum2, x14, w14);
            sum3 = vfmaq_f32(sum3, x15, w15);
        }

        // Handle remainder
        for (; l < k; l++) {
            float32x4_t x_val = vdupq_n_f32(x[l]);
            float32x4_t w_vec = vld1q_f32(w + l * n + j);
            sum0 = vfmaq_f32(sum0, x_val, w_vec);
        }

        // Store results
        vst1q_f32(out + j, sum0);
    }
}
#endif

// ============================================================================
// CONTEXT ALLOCATION (preallocated, no runtime allocation in hot path)
// ============================================================================

int sm2_create_context(sm2_model* model, sm2_context** out_ctx) {
    sm2_context* ctx = calloc(1, sizeof(sm2_context));
    if (!ctx) return -1;

    // Initialize F16 lookup table (256KB, done once)
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

static void embedding_lookup(float* out, int token, const sm2_model* model) {
    int dim = model->dim;
    float* embed_f32 = model->tok_embeddings_f32 + token * dim;
    int i = 0;
    // 4x unrolled
    for (; i + 3 < dim; i += 4) {
        out[i+0] = embed_f32[i+0];
        out[i+1] = embed_f32[i+1];
        out[i+2] = embed_f32[i+2];
        out[i+3] = embed_f32[i+3];
    }
    for (; i < dim; i++) {
        out[i] = embed_f32[i];
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
        // Use F32 directly - no F16 conversion needed
        xb_out[i] = (x_in[i] / rms) * model->input_layernorm_f32[ln_off + i];
    }

    // 2. Q projection: q = xb_out @ q_proj.T (with 8x unrolling)
    for (int i = 0; i < dim; i++) {
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;
        int j = 0;
        for (; j + 7 < dim; j += 8) {
            sum0 += xb_out[j+0] * model->q_proj_f32[q_off + i * dim + j + 0];
            sum1 += xb_out[j+1] * model->q_proj_f32[q_off + i * dim + j + 1];
            sum2 += xb_out[j+2] * model->q_proj_f32[q_off + i * dim + j + 2];
            sum3 += xb_out[j+3] * model->q_proj_f32[q_off + i * dim + j + 3];
            sum4 += xb_out[j+4] * model->q_proj_f32[q_off + i * dim + j + 4];
            sum5 += xb_out[j+5] * model->q_proj_f32[q_off + i * dim + j + 5];
            sum6 += xb_out[j+6] * model->q_proj_f32[q_off + i * dim + j + 6];
            sum7 += xb_out[j+7] * model->q_proj_f32[q_off + i * dim + j + 7];
        }
        for (; j < dim; j++) {
            sum0 += xb_out[j] * model->q_proj_f32[q_off + i * dim + j];
        }
        q[i] = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;
    }

    // 3. K projection: k = xb_out @ k_proj.T (with 4x unrolling)
    for (int i = 0; i < kv_dim; i++) {
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        int j = 0;
        for (; j + 3 < dim; j += 4) {
            sum0 += xb_out[j+0] * model->k_proj_f32[k_off + i * dim + j + 0];
            sum1 += xb_out[j+1] * model->k_proj_f32[k_off + i * dim + j + 1];
            sum2 += xb_out[j+2] * model->k_proj_f32[k_off + i * dim + j + 2];
            sum3 += xb_out[j+3] * model->k_proj_f32[k_off + i * dim + j + 3];
        }
        for (; j < dim; j++) {
            sum0 += xb_out[j] * model->k_proj_f32[k_off + i * dim + j];
        }
        k[i] = sum0 + sum1 + sum2 + sum3;
    }

    // 4. V projection: v = xb_out @ v_proj.T (with 4x unrolling)
    for (int i = 0; i < kv_dim; i++) {
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        int j = 0;
        for (; j + 3 < dim; j += 4) {
            sum0 += xb_out[j+0] * model->v_proj_f32[v_off + i * dim + j + 0];
            sum1 += xb_out[j+1] * model->v_proj_f32[v_off + i * dim + j + 1];
            sum2 += xb_out[j+2] * model->v_proj_f32[v_off + i * dim + j + 2];
            sum3 += xb_out[j+3] * model->v_proj_f32[v_off + i * dim + j + 3];
        }
        for (; j < dim; j++) {
            sum0 += xb_out[j] * model->v_proj_f32[v_off + i * dim + j];
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
            sum0 += attn_out[j+0] * model->o_proj_f32[o_off + i * dim + j + 0];
            sum1 += attn_out[j+1] * model->o_proj_f32[o_off + i * dim + j + 1];
            sum2 += attn_out[j+2] * model->o_proj_f32[o_off + i * dim + j + 2];
            sum3 += attn_out[j+3] * model->o_proj_f32[o_off + i * dim + j + 3];
        }
        for (; j < dim; j++) {
            sum0 += attn_out[j] * model->o_proj_f32[o_off + i * dim + j];
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
        // Use F32 directly - no F16 conversion needed
        xb_out[i] = (xb_out[i] / rms) * model->post_attention_layernorm_f32[post_off + i];
    }

    // 11. SwiGLU FFN using ffn_temp as workspace
    float* gate_out = ffn_temp;                  // first hidden_dim elements
    float* up_out = ffn_temp + hidden_dim;      // second hidden_dim elements

    // gate = xb_out @ gate_proj.T (with 4x unrolling)
    for (int i = 0; i < hidden_dim; i++) {
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        int j = 0;
        for (; j + 3 < dim; j += 4) {
            sum0 += xb_out[j+0] * model->gate_proj_f32[gate_off + i * dim + j + 0];
            sum1 += xb_out[j+1] * model->gate_proj_f32[gate_off + i * dim + j + 1];
            sum2 += xb_out[j+2] * model->gate_proj_f32[gate_off + i * dim + j + 2];
            sum3 += xb_out[j+3] * model->gate_proj_f32[gate_off + i * dim + j + 3];
        }
        for (; j < dim; j++) {
            sum0 += xb_out[j] * model->gate_proj_f32[gate_off + i * dim + j];
        }
        gate_out[i] = sum0 + sum1 + sum2 + sum3;
    }

    // up = xb_out @ up_proj.T (with 4x unrolling)
    for (int i = 0; i < hidden_dim; i++) {
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        int j = 0;
        for (; j + 3 < dim; j += 4) {
            sum0 += xb_out[j+0] * model->up_proj_f32[up_off + i * dim + j + 0];
            sum1 += xb_out[j+1] * model->up_proj_f32[up_off + i * dim + j + 1];
            sum2 += xb_out[j+2] * model->up_proj_f32[up_off + i * dim + j + 2];
            sum3 += xb_out[j+3] * model->up_proj_f32[up_off + i * dim + j + 3];
        }
        for (; j < dim; j++) {
            sum0 += xb_out[j] * model->up_proj_f32[up_off + i * dim + j];
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
            sum0 += gate_out[j+0] * model->down_proj_f32[down_off + i * hidden_dim + j + 0];
            sum1 += gate_out[j+1] * model->down_proj_f32[down_off + i * hidden_dim + j + 1];
            sum2 += gate_out[j+2] * model->down_proj_f32[down_off + i * hidden_dim + j + 2];
            sum3 += gate_out[j+3] * model->down_proj_f32[down_off + i * hidden_dim + j + 3];
        }
        for (; j < hidden_dim; j++) {
            sum0 += gate_out[j] * model->down_proj_f32[down_off + i * hidden_dim + j];
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
            embedding_lookup(ctx->scratch.x, tokens[t], model);
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
            final_h[i] = final_h[i] * scale * model->final_norm_f32[i];
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
                sum0 += final_h[j+0] * model->tok_embeddings_f32[i * dim + j + 0];
                sum1 += final_h[j+1] * model->tok_embeddings_f32[i * dim + j + 1];
                sum2 += final_h[j+2] * model->tok_embeddings_f32[i * dim + j + 2];
                sum3 += final_h[j+3] * model->tok_embeddings_f32[i * dim + j + 3];
            }
            for (; j < dim; j++) {
                sum0 += final_h[j] * model->tok_embeddings_f32[i * dim + j];
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
        embedding_lookup(ctx->scratch.x, ctx->last_token, ctx->model);
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
            final_h[i] = final_h[i] * scale * ctx->model->final_norm_f32[i];
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
                sum0 += final_h[j+0] * ctx->model->tok_embeddings_f32[i * dim + j + 0];
                sum1 += final_h[j+1] * ctx->model->tok_embeddings_f32[i * dim + j + 1];
                sum2 += final_h[j+2] * ctx->model->tok_embeddings_f32[i * dim + j + 2];
                sum3 += final_h[j+3] * ctx->model->tok_embeddings_f32[i * dim + j + 3];
            }
            for (; j < dim; j++) {
                sum0 += final_h[j] * ctx->model->tok_embeddings_f32[i * dim + j];
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
