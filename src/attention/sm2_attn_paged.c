// sm2_attn_paged.c - Paged attention kernel

#include <math.h>
#include "smollm2.h"

// Paged attention for decode
// Reads KV cache from paged storage
void sm2_attn_paged(
    float* out,
    const float* q,
    sm2_kv_pool* pool,
    sm2_kv_table* table,
    int seq_len,
    int n_heads,
    int n_kv_heads,
    int head_dim
) {
    int group_size = n_heads / n_kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    
    for (int qh = 0; qh < n_heads; qh++) {
        int kv_head = qh / group_size;
        
        float acc[64] = {0};
        float l = 0.0f;
        float max_s = -1e9f;
        
        // Iterate over pages
        int n_pages = (seq_len + SM2_KV_PAGE_TOKENS - 1) / SM2_KV_PAGE_TOKENS;
        
        for (int p = 0; p < n_pages; p++) {
            uint32_t page_id = table->page_ids[p];
            sm2_kv_page* page = &pool->pages[page_id];
            
            int start = p * SM2_KV_PAGE_TOKENS;
            int len = SM2_KV_PAGE_TOKENS;
            if (start + len > seq_len) len = seq_len - start;
            
            // Read K,V from page
            float k_page[16][64];
            float v_page[16][64];
            
            float* k_base = (float*)page->k_data;
            float* v_base = (float*)page->v_data;
            
            for (int t = 0; t < len; t++) {
                int offset = (kv_head * SM2_KV_PAGE_TOKENS + t) * head_dim;
                for (int d = 0; d < head_dim; d++) {
                    k_page[t][d] = k_base[offset + t * head_dim + d];
                    v_page[t][d] = v_base[offset + t * head_dim + d];
                }
            }
            
            // Compute scores for this tile
            float tile_max = -1e9f;
            float scores[16];
            
            for (int t = 0; t < len; t++) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    dot += q[qh * head_dim + d] * k_page[t][d];
                }
                scores[t] = dot * scale;
                if (scores[t] > tile_max) tile_max = scores[t];
            }
            
            // Online softmax update
            float alpha = expf(max_s - tile_max);
            
            for (int t = 0; t < len; t++) {
                scores[t] = expf(scores[t] - tile_max);
            }
            
            for (int d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (int t = 0; t < len; t++) {
                    sum += scores[t] * v_page[t][d];
                }
                acc[d] = acc[d] * alpha + sum;
            }
            l = l * alpha;
            for (int t = 0; t < len; t++) l += scores[t];
            max_s = tile_max;
        }
        
        // Normalize
        for (int d = 0; d < head_dim; d++) {
            out[qh * head_dim + d] = acc[d] / l;
        }
    }
}