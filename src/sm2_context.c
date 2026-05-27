// sm2_context.c - Context creation and core inference functions

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "smollm2.h"
#include "sm2_utils.h"

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
    for (int i = 0; i < dim; i++) {
        uint16_t h = embed->data[token * dim + i];
        out[i] = sm2_f16_to_float(h);
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

    // For each query head, attend over all cached K/V positions
    for (int qh = 0; qh < n_heads; qh++) {
        int kv_head = qh / group_size;

        // Compute attention scores for all cached positions
        float max_score = -1e9f;
        float scores[2048];

        for (int pos = 0; pos < kv_seq; pos++) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                float qv = q[qh * head_dim + d];
                int cache_idx = kv_head * ctx->params.max_context * head_dim + pos * head_dim + d;
                float kv = ctx->scratch.k_cache[layer][cache_idx];
                dot += qv * kv;
            }
            dot *= 1.0f / sqrtf((float)head_dim);
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
        xb_out[i] = (x_in[i] / rms) * sm2_f16_to_float(w);
    }

    // 2. Q projection: q = xb_out @ q_proj.T
    // q_proj is [dim, dim] stored row-major, so q_proj[i, j] = q_proj[i * dim + j]
    for (int i = 0; i < dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < dim; j++) {
            uint16_t w = model->q_proj[q_off + i * dim + j];
            sum += xb_out[j] * sm2_f16_to_float(w);
        }
        q[i] = sum;
    }

    // 3. K projection: k = xb_out @ k_proj.T
    // k_proj is [kv_dim, dim] stored row-major, so k_proj[i, j] = k_proj[i * dim + j]
    for (int i = 0; i < kv_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < dim; j++) {
            uint16_t w = model->k_proj[k_off + i * dim + j];
            sum += xb_out[j] * sm2_f16_to_float(w);
        }
        k[i] = sum;
    }

    // 4. V projection: v = xb_out @ v_proj.T
    // v_proj is [kv_dim, dim] stored row-major, so v_proj[i, j] = v_proj[i * dim + j]
    for (int i = 0; i < kv_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < dim; j++) {
            uint16_t w = model->v_proj[v_off + i * dim + j];
            sum += xb_out[j] * sm2_f16_to_float(w);
        }
        v[i] = sum;
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

    // 8. O projection: xb_out = attn_out @ o_proj.T
    // o_proj is [dim, dim] stored row-major, so o_proj[i, j] = o_proj[i * dim + j]
    for (int i = 0; i < dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < dim; j++) {
            uint16_t w = model->o_proj[o_off + i * dim + j];
            sum += attn_out[j] * sm2_f16_to_float(w);
        }
        xb_out[i] = sum;
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
        xb_out[i] = (xb_out[i] / rms) * sm2_f16_to_float(w);
    }

    // 11. SwiGLU FFN using ffn_temp as workspace
    float* gate_out = ffn_temp;                  // first hidden_dim elements
    float* up_out = ffn_temp + hidden_dim;      // second hidden_dim elements

    // gate = xb_out @ gate_proj.T
    // gate_proj is [hidden_dim, dim] stored row-major, so gate_proj[i, j] = gate_proj[i * dim + j]
    for (int i = 0; i < hidden_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < dim; j++) {
            uint16_t w = model->gate_proj[gate_off + i * dim + j];
            sum += xb_out[j] * sm2_f16_to_float(w);
        }
        gate_out[i] = sum;
    }

    // up = xb_out @ up_proj.T
    // up_proj is [hidden_dim, dim] stored row-major, so up_proj[i, j] = up_proj[i * dim + j]
    for (int i = 0; i < hidden_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < dim; j++) {
            uint16_t w = model->up_proj[up_off + i * dim + j];
            sum += xb_out[j] * sm2_f16_to_float(w);
        }
        up_out[i] = sum;
    }

    // SiLU: gate = gate * sigmoid(gate)
    for (int i = 0; i < hidden_dim; i++) {
        gate_out[i] *= 1.0f / (1.0f + expf(-gate_out[i]));
    }

    // Multiply: gate *= up
    for (int i = 0; i < hidden_dim; i++) gate_out[i] *= up_out[i];

    // down = gate_out @ down_proj.T, add residual to post-attention hidden state
    // down_proj is [dim, hidden_dim] stored row-major: down_proj[i, j] = down_proj[down_off + i * hidden_dim + j]
    for (int i = 0; i < dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < hidden_dim; j++) {
            uint16_t w = model->down_proj[down_off + i * hidden_dim + j];
            sum += gate_out[j] * sm2_f16_to_float(w);
        }
        // FFN output + residual (post-attention hidden state)
        xb_out[i] = x_post_attn[i] + sum;
    }
    // xb_out now holds the final layer output
    (void)hidden_dim; // unused
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
    }

    // Final RMSNorm
    if (model->final_norm) {
        float* final_h = ctx->scratch.x;
        float sum_sq = 0.0f;
        for (int i = 0; i < model->dim; i++) sum_sq += final_h[i] * final_h[i];
        float rms = sqrtf(sum_sq / (float)model->dim + 1e-5f);
        float scale = 1.0f / rms;
        for (int i = 0; i < model->dim; i++) {
            final_h[i] = final_h[i] * scale * sm2_f16_to_float(model->final_norm[i]);
        }
    }

    // Compute logits
    if (model->tok_embeddings && model->tok_embeddings->data) {
        float* final_h = ctx->scratch.x;
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
    for (int layer = 0; layer < ctx->model->n_layers; layer++) {
        layer_forward(ctx->scratch.xb, ctx->scratch.x, layer, ctx->model, ctx, seq_pos);
        // Swap input and output buffers
        float* tmp = ctx->scratch.x;
        ctx->scratch.x = ctx->scratch.xb;
        ctx->scratch.xb = tmp;
    }

    // Final RMSNorm
    if (ctx->model->final_norm) {
        float* final_h = ctx->scratch.x;
        float sum_sq = 0.0f;
        for (int i = 0; i < ctx->model->dim; i++) sum_sq += final_h[i] * final_h[i];
        float rms = sqrtf(sum_sq / (float)ctx->model->dim + 1e-5f);
        float scale = 1.0f / rms;
        for (int i = 0; i < ctx->model->dim; i++) {
            final_h[i] = final_h[i] * scale * sm2_f16_to_float(ctx->model->final_norm[i]);
        }
    }

    // Compute logits
    if (ctx->model->tok_embeddings && ctx->model->tok_embeddings->data) {
        float* final_h = ctx->scratch.x;
        for (int i = 0; i < ctx->model->vocab_size; i++) {
            float sum = 0.0f;
            for (int j = 0; j < ctx->model->dim; j++) {
                uint16_t h = ctx->model->tok_embeddings->data[i * ctx->model->dim + j];
                sum += final_h[j] * sm2_f16_to_float(h);
            }
            ctx->scratch.logits[i] = sum;
        }
    }

    // NOW sample from the newly computed logits
    int token = sm2_sample_token(ctx->scratch.logits, &ctx->params, &ctx->rng_state);

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
