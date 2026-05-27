// sm2_matmul_fast.c - Fast matrix multiplication with loop unrolling
//
// Optimizations:
// - Inner loop unrolling (4x, 8x)
// - Reduced function call overhead
// - Better cache access patterns

#include <stdio.h>
#include <math.h>
#include "smollm2.h"
#include "sm2_utils.h"

// ============================================================================
// FAST F16 MATMUL WITH 8x UNROLLING
// out = a @ b, where a is [m, k], b is [k, n] float16
// ============================================================================

void sm2_matmul_f16_fast(float* out, const float* a, const uint16_t* wb,
                        int m, int n, int k) {
    // 8x inner loop unrolling for better ILP
    for (int i = 0; i < m; i++) {
        const float* a_row = a + i * k;

        for (int j = 0; j < n; j++) {
            // Main accumulator
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
            float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;

            // 8x unrolled inner loop
            int l = 0;
            for (; l + 7 < k; l += 8) {
                float a0 = a_row[l], a1 = a_row[l+1], a2 = a_row[l+2], a3 = a_row[l+3];
                float a4 = a_row[l+4], a5 = a_row[l+5], a6 = a_row[l+6], a7 = a_row[l+7];

                // Load 8 weights and convert
                uint16_t w0 = wb[l * n + j];
                uint16_t w1 = wb[(l+1) * n + j];
                uint16_t w2 = wb[(l+2) * n + j];
                uint16_t w3 = wb[(l+3) * n + j];
                uint16_t w4 = wb[(l+4) * n + j];
                uint16_t w5 = wb[(l+5) * n + j];
                uint16_t w6 = wb[(l+6) * n + j];
                uint16_t w7 = wb[(l+7) * n + j];

                // Convert F16 to float inline (avoid function call)
                float b0 = sm2_f16_to_float(w0);
                float b1 = sm2_f16_to_float(w1);
                float b2 = sm2_f16_to_float(w2);
                float b3 = sm2_f16_to_float(w3);
                float b4 = sm2_f16_to_float(w4);
                float b5 = sm2_f16_to_float(w5);
                float b6 = sm2_f16_to_float(w6);
                float b7 = sm2_f16_to_float(w7);

                sum0 += a0 * b0;
                sum1 += a1 * b1;
                sum2 += a2 * b2;
                sum3 += a3 * b3;
                sum4 += a4 * b4;
                sum5 += a5 * b5;
                sum6 += a6 * b6;
                sum7 += a7 * b7;
            }

            // Handle remainder (4x unrolled)
            for (; l + 3 < k; l += 4) {
                uint16_t w0 = wb[l * n + j];
                uint16_t w1 = wb[(l+1) * n + j];
                uint16_t w2 = wb[(l+2) * n + j];
                uint16_t w3 = wb[(l+3) * n + j];

                sum0 += a_row[l] * sm2_f16_to_float(w0);
                sum1 += a_row[l+1] * sm2_f16_to_float(w1);
                sum2 += a_row[l+2] * sm2_f16_to_float(w2);
                sum3 += a_row[l+3] * sm2_f16_to_float(w3);
            }

            // Handle remaining
            for (; l < k; l++) {
                sum0 += a_row[l] * sm2_f16_to_float(wb[l * n + j]);
            }

            out[i * n + j] = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;
        }
    }
}

// ============================================================================
// FAST RMSNORM WITH 4x UNROLLING
// ============================================================================

void sm2_rmsnorm_fast(float* out, const float* input, const uint16_t* weight,
                     int size, float eps) {
    // Compute sum of squares
    float sum_sq = 0.0f;
    int i = 0;

    // 4x unrolled summation
    for (; i + 3 < size; i += 4) {
        float v0 = input[i], v1 = input[i+1], v2 = input[i+2], v3 = input[i+3];
        sum_sq += v0*v0 + v1*v1 + v2*v2 + v3*v3;
    }
    for (; i < size; i++) {
        float v = input[i];
        sum_sq += v * v;
    }

    // Compute scale
    float rms = sqrtf(sum_sq / (float)size + eps);
    float scale = 1.0f / rms;

    // Apply normalization with weight, 4x unrolled
    i = 0;
    for (; i + 3 < size; i += 4) {
        float w0 = sm2_f16_to_float(weight[i]);
        float w1 = sm2_f16_to_float(weight[i+1]);
        float w2 = sm2_f16_to_float(weight[i+2]);
        float w3 = sm2_f16_to_float(weight[i+3]);

        out[i]   = input[i]   * scale * w0;
        out[i+1] = input[i+1] * scale * w1;
        out[i+2] = input[i+2] * scale * w2;
        out[i+3] = input[i+3] * scale * w3;
 }
    for (; i < size; i++) {
        out[i] = input[i] * scale * sm2_f16_to_float(weight[i]);
    }
}

// ============================================================================
// FAST ATTENTION SCORES (simplified, single head)
// ============================================================================

void sm2_attention_scores_fast(float* scores, const float* q, const float* k_cache,
                              int head_dim, int seq_len) {
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int pos = 0; pos < seq_len; pos++) {
        float dot = 0.0f;
        int d = 0;

        // 4x unrolled inner product
        for (; d + 3 < head_dim; d += 4) {
            float q0 = q[d], q1 = q[d+1], q2 = q[d+2], q3 = q[d+3];
            float k0 = k_cache[pos * head_dim + d];
            float k1 = k_cache[pos * head_dim + d + 1];
            float k2 = k_cache[pos * head_dim + d + 2];
            float k3 = k_cache[pos * head_dim + d + 3];
            dot += q0*k0 + q1*k1 + q2*k2 + q3*k3;
        }
        for (; d < head_dim; d++) {
            dot += q[d] * k_cache[pos * head_dim + d];
        }

        scores[pos] = dot * scale;
    }
}
