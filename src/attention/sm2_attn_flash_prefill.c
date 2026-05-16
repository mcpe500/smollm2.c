// sm2_attn_flash_prefill.c - Flash attention for prefill

#include <math.h>
#include "smollm2.h"

// Flash attention for prefill
// Uses online softmax to avoid materializing full attention matrix
// Tile-based processing for cache efficiency
void sm2_flash_attn_prefill(
    float* out,
    const float* q,
    const float* k,
    const float* v,
    int seq_len,
    int n_heads,
    int n_kv_heads,
    int head_dim
) {
    int group_size = n_heads / n_kv_heads;
    int tile_size = 64; // Process in tiles for cache efficiency
    float scale = 1.0f / sqrtf((float)head_dim);
    
    // For each query position
    for (int q_pos = 0; q_pos < seq_len; q_pos++) {
        // Initialize accumulator
        float acc[64] = {0};
        float l = 0.0f;
        float m = -1e9f;
        
        // Process K,V in tiles
        for (int tile_start = 0; tile_start <= q_pos; tile_start += tile_size) {
            int tile_end = tile_start + tile_size;
            if (tile_end > q_pos + 1) tile_end = q_pos + 1;
            int tile_len = tile_end - tile_start;
            
            // Compute tile scores
            float tile_scores[64];
            float tile_max = -1e9f;
            
            for (int k_pos = tile_start; k_pos < tile_end; k_pos++) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    dot += q[q_pos * head_dim + d] * 
                          k[k_pos * head_dim + d];
                }
                tile_scores[k_pos - tile_start] = dot * scale;
                if (tile_scores[k_pos - tile_start] > tile_max) {
                    tile_max = tile_scores[k_pos - tile_start];
                }
            }
            
            // Online softmax update
            float alpha = expf(m - tile_max);
            float beta_sum = 0.0f;
            
            for (int i = 0; i < tile_len; i++) {
                tile_scores[i] = expf(tile_scores[i] - tile_max);
                beta_sum += tile_scores[i];
            }
            
            // Update accumulator
            for (int d = 0; d < head_dim; d++) {
                float acc_tile = 0.0f;
                for (int i = 0; i < tile_len; i++) {
                    acc_tile += tile_scores[i] * 
                               v[(tile_start + i) * head_dim + d];
                }
                acc[d] = acc[d] * alpha + acc_tile;
            }
            
            l = l * alpha + beta_sum;
            m = tile_max + logf(l);
        }
        
        // Normalize and store output
        for (int d = 0; d < head_dim; d++) {
            out[q_pos * head_dim + d] = acc[d] / l;
        }
    }
}