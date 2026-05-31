// sm2_rope.c - Rotary Position Embedding (RoPE)
//
// RoPE encodes position information by rotating query and key vectors.
// Each head dimension d is split into pairs, rotated by theta^(2i/d).
//

#include <stdio.h>
#include <math.h>
#include "smollm2.h"

// ============================================================================
// Precomputed RoPE frequencies (computed once)
// ============================================================================

// Freq table: freq_base[i] = theta^(-2i/head_dim) for i in [0, head_dim/2)
static float rope_freq_table[64];  // head_dim max is 64
static int rope_freq_table_size = 0;
static float rope_theta_global = 0;

// Build frequency table once
void sm2_rope_build_freqs(float* freqs, int n_heads, int head_dim, int max_seq, float theta) {
    rope_theta_global = theta;
    rope_freq_table_size = head_dim / 2;

    // Precompute 1/theta^(2i/head_dim) for each i
    for (int i = 0; i < rope_freq_table_size; i++) {
        rope_freq_table[i] = powf(theta, -(float)(2 * i) / (float)head_dim);
    }

    // Also fill the output freqs array for compatibility
    for (int i = 0; i < head_dim / 2; i++) {
        float freq = rope_freq_table[i];
        for (int pos = 0; pos < max_seq; pos++) {
            freqs[pos * (head_dim / 2) + i] = freq * (float)pos;
        }
    }
}

// Apply RoPE to a single Q or K vector
void sm2_rope_apply(float* vec, const float* freqs, int head_dim, int seq_pos) {
    int half = head_dim / 2;

    // Precompute freq values for this position
    float freq_cache[32];
    for (int i = 0; i < half; i++) {
        freq_cache[i] = freqs[seq_pos * half + i];
    }

    // 2x unrolled with cached freq values
    int i = 0;
    for (; i + 1 < half; i += 2) {
        float cos0 = cosf(freq_cache[i]);
        float sin0 = sinf(freq_cache[i]);
        float cos1 = cosf(freq_cache[i+1]);
        float sin1 = sinf(freq_cache[i+1]);

        float x00 = vec[i];
        float x01 = vec[i + half];
        float x10 = vec[i+1];
        float x11 = vec[i+1 + half];

        vec[i] = x00 * cos0 - x01 * sin0;
        vec[i + half] = x00 * sin0 + x01 * cos0;
        vec[i+1] = x10 * cos1 - x11 * sin1;
        vec[i+1 + half] = x10 * sin1 + x11 * cos1;
    }
    for (; i < half; i++) {
        float cos_theta = cosf(freq_cache[i]);
        float sin_theta = sinf(freq_cache[i]);

        float x0 = vec[i];
        float x1 = vec[i + half];

        vec[i] = x0 * cos_theta - x1 * sin_theta;
        vec[i + half] = x0 * sin_theta + x1 * cos_theta;
    }
}

// Apply RoPE to query and key vectors (uses precomputed freq table)
void sm2_rope(float* q, float* k, int head_dim, int pos, int n_heads, int n_kv_heads, float rope_theta) {
    int half = head_dim / 2;

    // Rebuild freq table if theta changed (should only happen at init)
    if (rope_theta_global != rope_theta || rope_freq_table_size == 0) {
        rope_theta_global = rope_theta;
        rope_freq_table_size = half;
        for (int i = 0; i < half; i++) {
            rope_freq_table[i] = powf(rope_theta, -(float)(2 * i) / (float)head_dim);
        }
    }

    // Process Q heads with 2x unrolling for cos/sin computation
    for (int h = 0; h < n_heads; h++) {
        float* q_head = q + h * head_dim;
        int i = 0;
        for (; i + 1 < half; i += 2) {
            float freq0 = rope_freq_table[i] * (float)pos;
            float freq1 = rope_freq_table[i+1] * (float)pos;
            float cos0 = cosf(freq0);
            float sin0 = sinf(freq0);
            float cos1 = cosf(freq1);
            float sin1 = sinf(freq1);

            float x00 = q_head[i];
            float x01 = q_head[i + half];
            float x10 = q_head[i+1];
            float x11 = q_head[i+1 + half];

            q_head[i] = x00 * cos0 - x01 * sin0;
            q_head[i + half] = x00 * sin0 + x01 * cos0;
            q_head[i+1] = x10 * cos1 - x11 * sin1;
            q_head[i+1 + half] = x10 * sin1 + x11 * cos1;
        }
        for (; i < half; i++) {
            float freq = rope_freq_table[i] * (float)pos;
            float cos_theta = cosf(freq);
            float sin_theta = sinf(freq);

            float x0 = q_head[i];
            float x1 = q_head[i + half];

            q_head[i] = x0 * cos_theta - x1 * sin_theta;
            q_head[i + half] = x0 * sin_theta + x1 * cos_theta;
        }
    }

    // Process K heads
    for (int h = 0; h < n_kv_heads; h++) {
        float* k_head = k + h * head_dim;
        int i = 0;
        for (; i + 1 < half; i += 2) {
            float freq0 = rope_freq_table[i] * (float)pos;
            float freq1 = rope_freq_table[i+1] * (float)pos;
            float cos0 = cosf(freq0);
            float sin0 = sinf(freq0);
            float cos1 = cosf(freq1);
            float sin1 = sinf(freq1);

            float x00 = k_head[i];
            float x01 = k_head[i + half];
            float x10 = k_head[i+1];
            float x11 = k_head[i+1 + half];

            k_head[i] = x00 * cos0 - x01 * sin0;
            k_head[i + half] = x00 * sin0 + x01 * cos0;
            k_head[i+1] = x10 * cos1 - x11 * sin1;
            k_head[i+1 + half] = x10 * sin1 + x11 * cos1;
        }
        for (; i < half; i++) {
            float freq = rope_freq_table[i] * (float)pos;
            float cos_theta = cosf(freq);
            float sin_theta = sinf(freq);

            float x0 = k_head[i];
            float x1 = k_head[i + half];

            k_head[i] = x0 * cos_theta - x1 * sin_theta;
            k_head[i + half] = x0 * sin_theta + x1 * cos_theta;
        }
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