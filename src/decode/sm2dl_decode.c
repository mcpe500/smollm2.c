// sm2dl_decode.c - smollm2dl decode layer core
// No-malloc hot path for decode_next

#include <stdio.h>
#include <math.h>
#include "smollm2.h"
#include "sm2_utils.h"

// ============================================================================
// DECODE CORE - NO ALLOCATION IN HOT PATH
//
// This is the critical path for fast token generation.
// Rules:
//   - NO malloc/free in decode_next
//   - NO JSON parsing in hot path
//   - NO string allocation
//   - All buffers preallocated in context
// ============================================================================

// Forward one token through the model
// This function MUST NOT allocate any memory
int sm2dl_forward_one_token(sm2_context* ctx) {
    sm2_model* model = ctx->model;
    const sm2_spec* spec = sm2_get_spec(model->variant);
    
    float* x = ctx->scratch.x;
    float* xb = ctx->scratch.xb;
    float* q = ctx->scratch.q;
    float* k = ctx->scratch.k;
    float* v = ctx->scratch.v;
    float* attn_out = ctx->scratch.attn_out;

    // ========================================================================
    // LAYER 0..n_layers-1
    // ========================================================================
    
    for (int layer = 0; layer < model->n_layers; layer++) {
        // ---- RMSNorm (input layernorm) ----
        // Using inline computation to avoid function call overhead
        float sum_sq = 0.0f;
        for (int i = 0; i < model->dim; i++) {
            sum_sq += x[i] * x[i];
        }
        float rms = sqrtf(sum_sq / (float)model->dim + spec->rms_eps);
        float scale = 1.0f / rms;

        uint16_t* ln_weight = model->input_layernorm;  // Now contiguous array [n_layers * dim]
        if (ln_weight) {
            uint16_t* layer_ln = ln_weight + layer * model->dim;
            for (int i = 0; i < model->dim; i++) {
                xb[i] = x[i] * scale * sm2_f16_to_float(layer_ln[i]);
            }
        } else {
            for (int i = 0; i < model->dim; i++) {
                xb[i] = x[i] * scale;
            }
        }
        
        // ---- Q, K, V projections ----
        // With actual weights
        int q_size = model->dim;
        int kv_size = model->n_kv_heads * model->head_dim;
        
        // Q projection: q = xb @ q_proj.T, q_proj is [dim, dim] row-major
        size_t q_off = (size_t)layer * model->dim * model->dim;
        for (int i = 0; i < q_size; i++) {
            float sum = 0.0f;
            for (int j = 0; j < model->dim; j++) {
                uint16_t w = model->q_proj[q_off + j * model->dim + i];  // [j][i]
                sum += xb[j] * sm2_f16_to_float(w);
            }
            q[i] = sum;
        }
        
        // K projection: k = xb @ k_proj.T, k_proj is [kv_dim, dim] row-major
        size_t k_off = (size_t)layer * kv_size * model->dim;
        for (int i = 0; i < kv_size; i++) {
            float sum = 0.0f;
            for (int j = 0; j < model->dim; j++) {
                uint16_t w = model->k_proj[k_off + j * kv_size + i];  // [j][i]
                sum += xb[j] * sm2_f16_to_float(w);
            }
            k[i] = sum;
        }
        
        // V projection: v = xb @ v_proj.T, v_proj is [kv_dim, dim] row-major
        size_t v_off = (size_t)layer * kv_size * model->dim;
        for (int i = 0; i < kv_size; i++) {
            float sum = 0.0f;
            for (int j = 0; j < model->dim; j++) {
                uint16_t w = model->v_proj[v_off + j * kv_size + i];  // [j][i]
                sum += xb[j] * sm2_f16_to_float(w);
            }
            v[i] = sum;
        }
        
        // ---- RoPE ----
        sm2_rope(q, k, model->head_dim, ctx->pos, 
                 model->n_heads, model->n_kv_heads, spec->rope_theta);
        
        // ---- Attention ----
        // Simplified GQA attention with cache-friendly access pattern
        int group_size = model->n_heads / model->n_kv_heads;
        const float attn_scale = 0.125f;  // 1/sqrt(64)
        
        for (int qh = 0; qh < model->n_heads; qh++) {
            int kv_head = qh / group_size;
            float* q_base = &q[qh * model->head_dim];
            float* v_base = &v[kv_head * model->head_dim];
            
            // Compute attention scores and find max
            float scores[256]; // max seq_len
            float max_score = -1e9f;
            int seq_len = ctx->kv.seq_len + 1;
            
            for (int pos = 0; pos < seq_len; pos++) {
                float* k_pos = &k[kv_head * model->head_dim + pos * model->head_dim];
                float dot = 0.0f;
                for (int d = 0; d < model->head_dim; d++) {
                    dot += q_base[d] * k_pos[d];
                }
                scores[pos] = dot * attn_scale;
                if (scores[pos] > max_score) max_score = scores[pos];
            }
            
            // Softmax
            float sum_exp = 0.0f;
            for (int pos = 0; pos < seq_len; pos++) {
                scores[pos] = expf(scores[pos] - max_score);
                sum_exp += scores[pos];
            }
            float inv_sum = 1.0f / sum_exp;
            
            // Weighted sum with scaled scores
            for (int d = 0; d < model->head_dim; d++) {
                float sum = 0.0f;
                for (int pos = 0; pos < seq_len; pos++) {
                    sum += scores[pos] * v_base[pos * model->head_dim + d];
                }
                attn_out[qh * model->head_dim + d] = sum * inv_sum;
            }
        }
        
        // ---- O projection + residual ----
        // o_proj is [dim, dim] row-major
        size_t o_off = (size_t)layer * model->dim * model->dim;
        for (int i = 0; i < model->dim; i++) {
            float sum = 0.0f;
            for (int j = 0; j < model->dim; j++) {
                uint16_t w = model->o_proj[o_off + j * model->dim + i];  // [j][i]
                sum += attn_out[j] * sm2_f16_to_float(w);
            }
            // Residual connection
            x[i] = xb[i] + sum;
        }
        
        // ---- Post-attention RMSNorm ----
        uint16_t* post_ln = model->post_attention_layernorm + layer * model->dim;
        float sum_sq2 = 0.0f;
        for (int i = 0; i < model->dim; i++) {
            sum_sq2 += x[i] * x[i];
        }
        float rms2 = sqrtf(sum_sq2 / (float)model->dim + spec->rms_eps);
        float scale2 = 1.0f / rms2;
        
        for (int i = 0; i < model->dim; i++) {
            xb[i] = x[i] * scale2 * sm2_f16_to_float(post_ln[i]);
        }
        
        // ---- SwiGLU FFN ----
        int hidden = model->hidden_dim;
        
        // gate_proj: [hidden_dim, dim] -> [hidden]
        size_t gate_off = (size_t)layer * hidden * model->dim;
        for (int i = 0; i < hidden; i++) {
            float sum = 0.0f;
            for (int j = 0; j < model->dim; j++) {
                uint16_t w = model->gate_proj[gate_off + j * hidden + i];  // [j][i]
                sum += xb[j] * sm2_f16_to_float(w);
            }
            q[i] = sum;  // reuse q buffer for gate
        }
        
        // up_proj: [hidden_dim, dim] -> [hidden]
        size_t up_off = (size_t)layer * hidden * model->dim;
        for (int i = 0; i < hidden; i++) {
            float sum = 0.0f;
            for (int j = 0; j < model->dim; j++) {
                uint16_t w = model->up_proj[up_off + j * hidden + i];  // [j][i]
                sum += xb[j] * sm2_f16_to_float(w);
            }
            k[i] = sum;  // reuse k buffer for up
        }
        
        // Apply silu to gate: silu(x) = x * sigmoid(x)
        // Inline: silu(x) = x / (1 + exp(-x))
        for (int i = 0; i < hidden; i++) {
            float x = q[i];
            float s = 1.0f / (1.0f + expf(-x));
            q[i] = x * s;
            // Multiply with up projection (reuse k buffer)
            q[i] = q[i] * k[i];
        }
        
        // down_proj: [dim, hidden_dim] -> [dim]
        size_t down_off = (size_t)layer * model->dim * hidden;
        for (int i = 0; i < model->dim; i++) {
            float sum = 0.0f;
            for (int j = 0; j < hidden; j++) {
                uint16_t w = model->down_proj[down_off + j * model->dim + i];  // [j][i]
                sum += q[j] * sm2_f16_to_float(w);
            }
            x[i] = xb[i] + sum;
        }
    }
    
    return 0;
}

// Decode next token - main entry point
// CRITICAL: No allocation allowed in this function
int sm2dl_decode_next(sm2_context* ctx, int* out_token) {
    // 1. Forward one token through the model
    int result = sm2dl_forward_one_token(ctx);
    if (result != 0) {
        *out_token = 0;
        return result;
    }

    // 2. Final RMSNorm
    if (ctx->model->final_norm) {
        float* vec = ctx->scratch.x;
        const sm2_spec* spec = sm2_get_spec(ctx->model->variant);
        float sum_sq = 0.0f;
        for (int i = 0; i < ctx->model->dim; i++) {
            sum_sq += vec[i] * vec[i];
        }
        float rms = sqrtf(sum_sq / (float)ctx->model->dim + spec->rms_eps);
        float scale = 1.0f / rms;
        uint16_t* fn = ctx->model->final_norm;  // [dim] of F16
        for (int i = 0; i < ctx->model->dim; i++) {
            vec[i] = vec[i] * scale * sm2_f16_to_float(fn[i]);
        }
    }
    
    // 3. Compute logits (embedding matrix multiply) - unrolled 4x
    if (ctx->model->tok_embeddings && ctx->model->tok_embeddings->data) {
        int vocab = ctx->model->vocab_size;
        int dim = ctx->model->dim;

        for (int i = 0; i < vocab; i++) {
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
            int j = 0;
            // Unroll 4x
            for (; j + 4 <= dim; j += 4) {
                uint16_t h0 = ctx->model->tok_embeddings->data[i * dim + j + 0];
                uint16_t h1 = ctx->model->tok_embeddings->data[i * dim + j + 1];
                uint16_t h2 = ctx->model->tok_embeddings->data[i * dim + j + 2];
                uint16_t h3 = ctx->model->tok_embeddings->data[i * dim + j + 3];
                sum0 += ctx->scratch.x[j + 0] * sm2_f16_to_float(h0);
                sum1 += ctx->scratch.x[j + 1] * sm2_f16_to_float(h1);
                sum2 += ctx->scratch.x[j + 2] * sm2_f16_to_float(h2);
                sum3 += ctx->scratch.x[j + 3] * sm2_f16_to_float(h3);
            }
            // Handle remainder
            for (; j < dim; j++) {
                uint16_t h = ctx->model->tok_embeddings->data[i * dim + j];
                sum0 += ctx->scratch.x[j] * sm2_f16_to_float(h);
            }
            ctx->scratch.logits[i] = sum0 + sum1 + sum2 + sum3;
        }
    }
    
    // 4. Sample token
    int token = sm2_sample_token(ctx->scratch.logits, &ctx->params, &ctx->rng_state, ctx);
    
    // 5. Update state (NO ALLOCATION)
    ctx->pos++;
    ctx->last_token = token;
    
    // 6. Get embedding for next token (for next decode step)
    float* x = ctx->scratch.x;
    if (ctx->model->tok_embeddings && ctx->model->tok_embeddings->data) {
        for (int i = 0; i < ctx->model->dim; i++) {
            uint16_t h = ctx->model->tok_embeddings->data[token * ctx->model->dim + i];
            x[i] = sm2_f16_to_float(h);
        }
    } else {
        for (int i = 0; i < ctx->model->dim; i++) {
            x[i] = (float)(token % 256) / 256.0f;
        }
    }
    
    *out_token = token;
    return 0;
}