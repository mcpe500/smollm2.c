// sm2dl_flash_decode.c - Flash Decode implementation
// Online softmax attention with tile-based processing

#include <math.h>
#include "smollm2.h"

// ============================================================================
// FLASH DECODE - Online Softmax Attention
//
// Unlike standard attention which stores all scores in a buffer,
// Flash Decode uses online softmax: processes tokens in tiles without
// materializing full attention score matrix.
//
// Algorithm:
//   m = -inf, l = 0, acc = 0
//   for each tile:
//     s_tile = q @ K_tile^T
//     m_tile = max(s_tile)
//     alpha = exp(m - m_tile)
//     beta = exp(s_tile - m_tile)
//     acc = acc * alpha + beta @ V_tile
//     l = l * alpha + sum(beta)
//     m = m_tile + log(l)
//   out = acc / l
// ============================================================================

// Process one tile of KV for flash decode
static void flash_decode_tile(
    float* acc,        // accumulator [head_dim]
    float* l,          // running sum of exp [1]
    float* m,          // running max [1]
    const float* q,    // query vector [head_dim]
    const float* k_tile, // key tile [tile_size, head_dim]
    const float* v_tile, // value tile [tile_size, head_dim]
    int tile_size,
    int head_dim
) {
    // Compute scores: s = q @ k_tile^T
    float s[64]; // max tile size
    float m_tile = -1e9f;
    
    for (int i = 0; i < tile_size; i++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q[d] * k_tile[i * head_dim + d];
        }
        s[i] = dot;
        if (s[i] > m_tile) m_tile = s[i];
    }
    
    // Compute exp terms
    float alpha = expf(*m - m_tile);
    float beta_sum = 0.0f;
    
    for (int i = 0; i < tile_size; i++) {
        s[i] = expf(s[i] - m_tile);
        beta_sum += s[i];
    }
    
    // Update accumulator
    for (int d = 0; d < head_dim; d++) {
        float acc_tile = 0.0f;
        for (int i = 0; i < tile_size; i++) {
            acc_tile += s[i] * v_tile[i * head_dim + d];
        }
        acc[d] = acc[d] * alpha + acc_tile;
    }
    
    // Update running sum and max
    *l = *l * alpha + beta_sum;
    *m = m_tile + logf(*l);
}

// Flash decode for a single head
void sm2dl_flash_decode_head(
    float* out,
    const float* q,
    const float* k_cache,  // full KV cache [seq_len, head_dim]
    const float* v_cache,
    int seq_len,
    int head_dim
) {
    // Initialize accumulator
    float acc[64] = {0}; // max head_dim
    float l = 1.0f;
    float m = -1e9f;
    
    // Process in tiles of 16 (SM2_KV_PAGE_TOKENS)
    int tile_size = 16;
    int n_tiles = (seq_len + tile_size - 1) / tile_size;
    
    for (int t = 0; t < n_tiles; t++) {
        int start = t * tile_size;
        int len = tile_size;
        if (start + len > seq_len) len = seq_len - start;
        
        if (len > 0) {
            flash_decode_tile(acc, &l, &m, q,
                             k_cache + start * head_dim,
                             v_cache + start * head_dim,
                             len, head_dim);
        }
    }
    
    // Normalize
    for (int d = 0; d < head_dim; d++) {
        out[d] = acc[d] / l;
    }
}

// Flash decode with GQA support
void sm2dl_flash_decode_gqa(
    float* out,              // [n_heads, head_dim]
    const float* q,          // [n_heads, head_dim]
    const float* k_cache,   // [n_kv_heads, seq_len, head_dim]
    const float* v_cache,   // [n_kv_heads, seq_len, head_dim]
    int seq_len,
    int n_heads,
    int n_kv_heads,
    int head_dim
) {
    int group_size = n_heads / n_kv_heads;
    
    // Process each query head
    for (int qh = 0; qh < n_heads; qh++) {
        int kv_head = qh / group_size;
        
        sm2dl_flash_decode_head(
            out + qh * head_dim,
            q + qh * head_dim,
            k_cache + kv_head * seq_len * head_dim,
            v_cache + kv_head * seq_len * head_dim,
            seq_len, head_dim
        );
    }
}

// Flash decode with paged KV cache
void sm2dl_flash_decode_paged(
    float* out,
    const float* q,
    sm2_kv_pool* pool,
    sm2_kv_table* table,
    int layer,
    int n_heads,
    int n_kv_heads,
    int head_dim
) {
    int group_size = n_heads / n_kv_heads;
    
    for (int qh = 0; qh < n_heads; qh++) {
        int kv_head = qh / group_size;
        
        // Accumulator
        float acc[64] = {0};
        float l = 1.0f;
        float m = -1e9f;
        
        // Iterate over pages in the sequence
        int seq_len = table->seq_len;
        int n_pages = (seq_len + SM2_KV_PAGE_TOKENS - 1) / SM2_KV_PAGE_TOKENS;
        
        for (int p = 0; p < n_pages; p++) {
            uint32_t page_id = table->page_ids[p];
            sm2_kv_page* page = &pool->pages[page_id];
            
            int start = p * SM2_KV_PAGE_TOKENS;
            int len = SM2_KV_PAGE_TOKENS;
            if (start + len > seq_len) len = seq_len - start;
            
            if (len > 0) {
                // Get pointers to K and V data for this page
                float* k_page = (float*)page->k_data;
                float* v_page = (float*)page->v_data;
                
                flash_decode_tile(acc, &l, &m,
                                 q + qh * head_dim,
                                 k_page + layer * n_kv_heads * SM2_KV_PAGE_TOKENS * head_dim + kv_head * SM2_KV_PAGE_TOKENS * head_dim,
                                 v_page + layer * n_kv_heads * SM2_KV_PAGE_TOKENS * head_dim + kv_head * SM2_KV_PAGE_TOKENS * head_dim,
                                 len, head_dim);
            }
        }
        
        // Normalize and store
        for (int d = 0; d < head_dim; d++) {
            out[qh * head_dim + d] = acc[d] / l;
        }
    }
}