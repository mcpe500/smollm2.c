// sm2dl_paged_attention.c - Paged KV attention for decode layer

#include <math.h>
#include "smollm2.h"

// ============================================================================
// PAGED ATTENTION - KV cache with page-based management
//
// Pages are 16 tokens each. Each sequence has a page table mapping
// token positions to physical pages in the KV pool.
//
// Benefits:
//   - No external fragmentation (fixed page size)
//   - Sharing possible for prefix caching
//   - Efficient memory usage for variable length sequences
// ============================================================================

// Compute attention for a paged KV sequence
void sm2dl_paged_attn(
    float* out,              // [n_heads, head_dim] output
    const float* q,          // [n_heads, head_dim] query
    sm2_kv_pool* pool,       // KV pool
    sm2_kv_table* table,     // Sequence page table
    int layer,
    int n_heads,
    int n_kv_heads,
    int head_dim
) {
    int group_size = n_heads / n_kv_heads;
    int seq_len = table->seq_len;
    
    // For each query head
    for (int qh = 0; qh < n_heads; qh++) {
        int kv_head = qh / group_size;
        
        // Collect K,V from pages
        // Simple approach: read all pages into contiguous buffer
        int n_pages = (seq_len + SM2_KV_PAGE_TOKENS - 1) / SM2_KV_PAGE_TOKENS;
        
        // Allocate temp buffer for this head (first token only in decode)
        // For decode, seq_len is small (single token)
        // We could avoid this allocation by reading directly from pages
        
        // For now, do simple attention
        // Full paged attention would use flash_decode_paged
        float acc[64] = {0};
        float l = 0.0f;
        float max_s = -1e9f;
        
        for (int pos = 0; pos < seq_len; pos++) {
            // Find which page this position is in
            int page_idx = pos / SM2_KV_PAGE_TOKENS;
            int offset = pos % SM2_KV_PAGE_TOKENS;
            uint32_t page_id = table->page_ids[page_idx];
            sm2_kv_page* page = &pool->pages[page_id];
            
            // Get K,V for this position
            float k_vec[64];
            float v_vec[64];
            
            int k_offset = (layer * n_kv_heads + kv_head) * SM2_KV_PAGE_TOKENS * head_dim 
                         + offset * head_dim;
            int v_offset = k_offset; // Same layout for V
            
            // Copy K vector
            for (int d = 0; d < head_dim; d++) {
                float* k_base = (float*)(page->k_data);
                k_vec[d] = k_base[k_offset + d];
            }
            
            // Score
            float s = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                s += q[qh * head_dim + d] * k_vec[d];
            }
            s /= sqrtf((float)head_dim);
            
            // Online softmax
            if (s > max_s) max_s = s;
        }
        
        // Second pass: compute softmax and accumulate
        float sum_exp = 0.0f;
        for (int pos = 0; pos < seq_len; pos++) {
            int page_idx = pos / SM2_KV_PAGE_TOKENS;
            int offset = pos % SM2_KV_PAGE_TOKENS;
            uint32_t page_id = table->page_ids[page_idx];
            sm2_kv_page* page = &pool->pages[page_id];
            
            float* k_base = (float*)(page->k_data);
            float* v_base = (float*)(page->v_data);
            
            int k_offset = (layer * n_kv_heads + kv_head) * SM2_KV_PAGE_TOKENS * head_dim 
                         + offset * head_dim;
            int v_offset = k_offset;
            
            float s = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                s += q[qh * head_dim + d] * k_base[k_offset + d];
            }
            s /= sqrtf((float)head_dim);
            
            float exp_s = expf(s - max_s);
            sum_exp += exp_s;
            
            for (int d = 0; d < head_dim; d++) {
                acc[d] += exp_s * v_base[v_offset + d];
            }
        }
        
        // Normalize
        for (int d = 0; d < head_dim; d++) {
            out[qh * head_dim + d] = acc[d] / sum_exp;
        }
    }
}

// Update KV cache with new token (append to pages)
int sm2dl_kv_append(
    sm2_kv_pool* pool,
    sm2_kv_table* table,
    int layer,
    int kv_head,
    const float* k_new,
    const float* v_new
) {
    // Get or allocate page
    int last_page_idx = table->n_pages - 1;
    if (last_page_idx < 0 || table->n_pages == 0) {
        // Allocate first page
        if (pool->free_top <= 0) return -1; // No pages available
        uint32_t page_id = pool->free_stack[--pool->free_top];
        table->page_ids[0] = page_id;
        table->n_pages = 1;
        last_page_idx = 0;
        pool->pages[page_id].used = 0;
    }
    
    sm2_kv_page* page = &pool->pages[table->page_ids[last_page_idx]];
    
    // Check if page is full
    if (page->used >= pool->page_tokens) {
        // Allocate new page
        if (pool->free_top <= 0) return -1;
        uint32_t new_id = pool->free_stack[--pool->free_top];
        table->page_ids[table->n_pages++] = new_id;
        page = &pool->pages[new_id];
        page->used = 0;
    }
    
    // Write K,V to page
    int offset = page->used;
    int data_offset = (layer * pool->n_kv_heads + kv_head) * pool->page_tokens * pool->head_dim 
                     + offset * pool->head_dim;
    
    float* k_base = (float*)page->k_data;
    float* v_base = (float*)page->v_data;
    
    for (int d = 0; d < pool->head_dim; d++) {
        k_base[data_offset + d] = k_new[d];
        v_base[data_offset + d] = v_new[d];
    }
    
    page->used++;
    table->seq_len++;
    
    return 0;
}

// Free a sequence's KV pages
void sm2dl_kv_free_seq(sm2_kv_pool* pool, sm2_kv_table* table) {
    for (int i = 0; i < table->n_pages; i++) {
        uint32_t page_id = table->page_ids[i];
        pool->pages[page_id].used = 0;
        pool->free_stack[pool->free_top++] = page_id;
    }
    table->seq_len = 0;
    table->n_pages = 0;
}