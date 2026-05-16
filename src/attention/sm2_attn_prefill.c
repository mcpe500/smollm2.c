// sm2_attn_prefill.c - Attention prefill kernels

#include <math.h>
#include "smollm2.h"

// Prefill attention: full attention over prompt tokens
// Computes attention scores for each head
void sm2_attn_prefill(
    float* out,              // [n_heads, seq_len, head_dim]
    const float* q,          // [n_heads, seq_len, head_dim]
    const float* k,          // [n_kv_heads, seq_len, head_dim]
    const float* v,          // [n_kv_heads, seq_len, head_dim]
    int seq_len,
    int n_heads,
    int n_kv_heads,
    int head_dim
) {
    int group_size = n_heads / n_kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    
    // For each position (could be parallelized)
    for (int pos = 0; pos < seq_len; pos++) {
        // For each query head
        for (int qh = 0; qh < n_heads; qh++) {
            int kv_head = qh / group_size;
            
            // Compute attention scores for this query at this position
            float scores[256]; // max seq_len
            float max_s = -1e9f;
            
            // Calculate q dot k for all previous positions
            for (int k_pos = 0; k_pos <= pos; k_pos++) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    dot += q[qh * seq_len * head_dim + pos * head_dim + d] *
                          k[kv_head * seq_len * head_dim + k_pos * head_dim + d];
                }
                scores[k_pos] = dot * scale;
                if (scores[k_pos] > max_s) max_s = scores[k_pos];
            }
            
            // Softmax
            float sum_exp = 0.0f;
            for (int k_pos = 0; k_pos <= pos; k_pos++) {
                scores[k_pos] = expf(scores[k_pos] - max_s);
                sum_exp += scores[k_pos];
            }
            
            // Weighted sum of values
            for (int d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (int k_pos = 0; k_pos <= pos; k_pos++) {
                    sum += scores[k_pos] * 
                          v[kv_head * seq_len * head_dim + k_pos * head_dim + d];
                }
                out[qh * seq_len * head_dim + pos * head_dim + d] = sum / sum_exp;
            }
        }
    }
}