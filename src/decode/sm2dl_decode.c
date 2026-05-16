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
        
        float* ln_weight = model->input_layernorm ? model->input_layernorm->data : NULL;
        if (ln_weight) {
            for (int i = 0; i < model->dim; i++) {
                xb[i] = x[i] * scale * ln_weight[i];
            }
        } else {
            for (int i = 0; i < model->dim; i++) {
                xb[i] = x[i] * scale;
            }
        }
        
        // ---- Q, K, V projections ----
        // Placeholder: identity matrix (real impl uses actual weights)
        int q_size = model->dim;
        int kv_size = model->n_kv_heads * model->head_dim;
        
        // Q projection
        for (int i = 0; i < q_size; i++) {
            float sum = 0.0f;
            for (int j = 0; j < model->dim; j++) {
                sum += xb[j] * (i < model->dim && j < model->dim ? 1.0f : 0.0f);
            }
            q[i] = sum;
        }
        
        // K projection
        for (int i = 0; i < kv_size; i++) {
            float sum = 0.0f;
            for (int j = 0; j < model->dim; j++) {
                sum += xb[j] * (i < kv_size && j < model->dim ? 1.0f : 0.0f);
            }
            k[i] = sum;
        }
        
        // V projection
        for (int i = 0; i < kv_size; i++) {
            float sum = 0.0f;
            for (int j = 0; j < model->dim; j++) {
                sum += xb[j] * (i < kv_size && j < model->dim ? 1.0f : 0.0f);
            }
            v[i] = sum;
        }
        
        // ---- RoPE ----
        sm2_rope(q, k, model->head_dim, ctx->pos, 
                 model->n_heads, model->n_kv_heads, spec->rope_theta);
        
        // ---- Attention ----
        // Simplified GQA attention
        int group_size = model->n_heads / model->n_kv_heads;
        
        for (int qh = 0; qh < model->n_heads; qh++) {
            int kv_head = qh / group_size;
            
            // Compute attention scores
            float scores[256]; // max seq_len
            float max_score = -1e9f;
            int seq_len = ctx->kv.seq_len + 1;
            
            for (int pos = 0; pos < seq_len; pos++) {
                float dot = 0.0f;
                for (int d = 0; d < model->head_dim; d++) {
                    dot += q[qh * model->head_dim + d] * 
                          k[kv_head * model->head_dim + pos * model->head_dim + d];
                }
                dot /= sqrtf((float)model->head_dim);
                scores[pos] = dot;
                if (dot > max_score) max_score = dot;
            }
            
            // Softmax
            float sum_exp = 0.0f;
            for (int pos = 0; pos < seq_len; pos++) {
                scores[pos] = expf(scores[pos] - max_score);
                sum_exp += scores[pos];
            }
            
            // Weighted sum
            for (int d = 0; d < model->head_dim; d++) {
                float sum = 0.0f;
                for (int pos = 0; pos < seq_len; pos++) {
                    sum += scores[pos] * v[kv_head * model->head_dim + pos * model->head_dim + d];
                }
                attn_out[qh * model->head_dim + d] = sum / sum_exp;
            }
        }
        
        // ---- O projection + residual ----
        for (int i = 0; i < model->dim; i++) {
            float sum = 0.0f;
            for (int j = 0; j < model->dim; j++) {
                sum += attn_out[j] * (i < model->dim && j < model->dim ? 1.0f : 0.0f);
            }
            // Residual connection
            x[i] = xb[i] + sum;  // x now holds residual
            // xb will hold next layer's input
        }
    }
    
    return 0;
}

// Decode next token - main entry point
// CRITICAL: No allocation allowed in this function
int sm2dl_decode_next(sm2_context* ctx, int* out_token) {
    // 1. Forward one token through the model
    sm2dl_forward_one_token(ctx);
    
    // 2. Final RMSNorm
    if (ctx->model->final_norm) {
        sm2_rmsnorm_inplace(ctx->scratch.x, ctx->model->final_norm->data, 
                           ctx->model->dim, 1e-5f);
    }
    
    // 3. Compute logits (embedding matrix multiply)
    if (ctx->model->tok_embeddings && ctx->model->tok_embeddings->data) {
        // logits = x @ embed (shared weights, tie_word_embeddings)
        int vocab = ctx->model->vocab_size;
        int dim = ctx->model->dim;
        
        for (int i = 0; i < vocab; i++) {
            float sum = 0.0f;
            for (int j = 0; j < dim; j++) {
                uint16_t h = ctx->model->tok_embeddings->data[i * dim + j];
                sum += ctx->scratch.x[j] * sm2_f16_to_float(h);
            }
            ctx->scratch.logits[i] = sum;
        }
    }
    
    // 4. Sample token
    int token = sm2_sample_token(ctx->scratch.logits, &ctx->params, &ctx->rng_state);
    
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