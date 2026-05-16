// sm2_kv_page.c - KV page operations

#include <math.h>
#include "smollm2.h"

// Write a token to a KV page
int sm2_kv_write_token(sm2_kv_page* page, int offset, int kv_head, 
                       const float* k, const float* v) {
    if (offset >= SM2_KV_PAGE_TOKENS) return -1;
    
    size_t head_offset = (size_t)kv_head * SM2_KV_PAGE_TOKENS * 64;
    size_t token_offset = head_offset + offset * 64;
    
    float* k_base = (float*)page->k_data;
    float* v_base = (float*)page->v_data;
    
    for (int d = 0; d < 64; d++) {
        k_base[token_offset + d] = k[d];
        v_base[token_offset + d] = v[d];
    }
    
    return 0;
}

// Read a token from a KV page
int sm2_kv_read_token(sm2_kv_page* page, int offset, int kv_head,
                      float* k_out, float* v_out) {
    if (offset >= page->used) return -1;
    
    size_t head_offset = (size_t)kv_head * SM2_KV_PAGE_TOKENS * 64;
    size_t token_offset = head_offset + offset * 64;
    
    float* k_base = (float*)page->k_data;
    float* v_base = (float*)page->v_data;
    
    for (int d = 0; d < 64; d++) {
        k_out[d] = k_base[token_offset + d];
        v_out[d] = v_base[token_offset + d];
    }
    
    return 0;
}

// Write quantized token
int sm2_kv_write_token_q8(sm2_kv_page* page, int offset, int kv_head,
                          const float* k, const float* v, 
                          float k_scale, float v_scale) {
    if (offset >= SM2_KV_PAGE_TOKENS) return -1;
    
    size_t head_offset = (size_t)kv_head * SM2_KV_PAGE_TOKENS;
    size_t token_offset = head_offset + offset;
    
    int8_t* k_base = (int8_t*)page->k_data;
    int8_t* v_base = (int8_t*)page->v_data;
    
    for (int d = 0; d < 64; d++) {
        k_base[token_offset * 64 + d] = (int8_t)(k[d] / k_scale);
        v_base[token_offset * 64 + d] = (int8_t)(v[d] / v_scale);
    }
    
    // Store scales
    page->k_scale[kv_head] = k_scale;
    page->v_scale[kv_head] = v_scale;
    
    return 0;
}

// Clear a page
void sm2_kv_page_clear(sm2_kv_page* page) {
    page->used = 0;
    page->refcount = 0;
}