// sm2_rope.c - Rotary Position Embedding (RoPE)
//
// RoPE encodes position information by rotating query and key vectors.
// Each head dimension d is split into pairs, rotated by theta^(2i/d).
// 

#include <stdio.h>
#include <math.h>
#include "smollm2.h"

// ============================================================================
// Precompute RoPE frequencies for efficiency
// ============================================================================

void sm2_rope_build_freqs(float* freqs, int n_heads, int head_dim, int max_seq, float theta) {
    // Precompute rotation frequencies for each head dimension
    for (int i = 0; i < head_dim / 2; i++) {
        float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
        
        for (int pos = 0; pos < max_seq; pos++) {
            freqs[pos * (head_dim / 2) + i] = freq * (float)pos;
        }
    }
}

// Apply RoPE to a single Q or K vector
void sm2_rope_apply(float* vec, const float* freqs, int head_dim, int seq_pos) {
    int half = head_dim / 2;
    
    for (int i = 0; i < half; i++) {
        float freq = freqs[seq_pos * half + i];
        float cos_theta = cosf(freq);
        float sin_theta = sinf(freq);
        
        float x0 = vec[i];
        float x1 = vec[i + half];
        
        vec[i] = x0 * cos_theta - x1 * sin_theta;
        vec[i + half] = x0 * sin_theta + x1 * cos_theta;
    }
}

// Apply RoPE to query and key vectors (inline)
void sm2_rope(float* q, float* k, int head_dim, int pos, int n_heads, int n_kv_heads, float rope_theta) {
    // Build frequency table for this head
    float freqs[64]; // max head_dim = 64
    
    int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        freqs[i] = (float)pos / powf(rope_theta, (float)(2 * i) / (float)head_dim);
    }
    
    // Apply to Q (all heads)
    for (int h = 0; h < n_heads; h++) {
        float* q_head = q + h * head_dim;
        sm2_rope_apply(q_head, freqs, head_dim, 0);
    }
    
    // Apply to K (only KV heads)
    for (int h = 0; h < n_kv_heads; h++) {
        float* k_head = k + h * head_dim;
        sm2_rope_apply(k_head, freqs, head_dim, 0);
    }
}

// Apply RoPE to already-computed Q/K in the computation
void sm2_rope_full(float* q, float* k, int n_heads, int n_kv_heads, int head_dim, 
                   int seq_len, float rope_theta, float* output_q, float* output_k) {
    // Full RoPE across sequence length
    // For decode, we just apply for single position
    (void)seq_len;
    (void)output_q;
    (void)output_k;
    
    sm2_rope(q, k, head_dim, seq_len, n_heads, n_kv_heads, rope_theta);
}