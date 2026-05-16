// sm2_kv_pool.c - KV cache pool management

#include <stdlib.h>
#include <math.h>
#include "smollm2.h"
#include <string.h>

// Initialize KV pool
int sm2_kv_pool_init(sm2_kv_pool* pool, int n_layers, int n_kv_heads, int max_pages, sm2_kv_dtype dtype) {
    memset(pool, 0, sizeof(sm2_kv_pool));
    
    pool->n_layers = n_layers;
    pool->n_kv_heads = n_kv_heads;
    pool->head_dim = 64; // Fixed for SmolLM2
    pool->page_tokens = SM2_KV_PAGE_TOKENS;
    pool->dtype = dtype;
    pool->max_pages = max_pages;
    
    // Allocate page array
    pool->pages = calloc(max_pages, sizeof(sm2_kv_page));
    if (!pool->pages) return -1;
    
    // Allocate free stack
    pool->free_stack = calloc(max_pages, sizeof(uint32_t));
    if (!pool->free_stack) {
        free(pool->pages);
        return -1;
    }
    
    // Initialize free stack (stack grows from top)
    pool->free_top = max_pages;
    for (int i = 0; i < max_pages; i++) {
        pool->free_stack[i] = (uint32_t)i;
    }
    
    // Calculate bytes per page based on dtype
    size_t bytes_per_element = 2; // F16 default
    if (dtype == SM2_KV_Q8) bytes_per_element = 1;
    else if (dtype == SM2_KV_Q4) bytes_per_element = 0.5;
    else if (dtype == SM2_KV_TURBO2) bytes_per_element = 0.25;
    
    size_t kv_bytes_per_head = (size_t)n_kv_heads * SM2_KV_PAGE_TOKENS * 64 * bytes_per_element;
    size_t page_bytes = n_layers * kv_bytes_per_head * 2; // K + V
    
    // Allocate page data
    for (int i = 0; i < max_pages; i++) {
        pool->pages[i].id = i;
        pool->pages[i].k_data = calloc(page_bytes, 1);
        pool->pages[i].v_data = calloc(page_bytes, 1);
        pool->pages[i].k_scale = calloc(n_layers * n_kv_heads * sizeof(float), 1);
        pool->pages[i].v_scale = calloc(n_layers * n_kv_heads * sizeof(float), 1);
        
        if (!pool->pages[i].k_data || !pool->pages[i].v_data) {
            // Cleanup on failure
            for (int j = 0; j < i; j++) {
                free(pool->pages[j].k_data);
                free(pool->pages[j].v_data);
                free(pool->pages[j].k_scale);
                free(pool->pages[j].v_scale);
            }
            free(pool->pages);
            free(pool->free_stack);
            return -1;
        }
    }
    
    return 0;
}

// Free KV pool
void sm2_kv_pool_free(sm2_kv_pool* pool) {
    if (!pool) return;
    
    if (pool->pages) {
        for (int i = 0; i < pool->max_pages; i++) {
            if (pool->pages[i].k_data) free(pool->pages[i].k_data);
            if (pool->pages[i].v_data) free(pool->pages[i].v_data);
            if (pool->pages[i].k_scale) free(pool->pages[i].k_scale);
            if (pool->pages[i].v_scale) free(pool->pages[i].v_scale);
        }
        free(pool->pages);
    }
    
    if (pool->free_stack) free(pool->free_stack);
}

// Allocate a page from pool
uint32_t sm2_kv_alloc_page(sm2_kv_pool* pool) {
    if (pool->free_top <= 0) return 0; // No free pages
    return pool->free_stack[--pool->free_top];
}

// Free a page back to pool
void sm2_kv_free_page(sm2_kv_pool* pool, uint32_t page_id) {
    if (page_id >= pool->max_pages) return;
    pool->pages[page_id].used = 0;
    pool->free_stack[pool->free_top++] = page_id;
}

// Get page utilization stats
int sm2_kv_pool_stats(sm2_kv_pool* pool, int* out_free, int* out_used) {
    *out_free = pool->free_top;
    *out_used = pool->max_pages - pool->free_top;
    return 0;
}