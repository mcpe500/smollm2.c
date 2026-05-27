// sm2_backend_neon.c - ARM NEON optimized matmul
//
// ARM NEON provides 128-bit registers (4x float32 or 8x int16)
// This gives up to 4x speedup over scalar code for matmul operations.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <arm_neon.h>  // ARM NEON intrinsics
#include "smollm2.h"
#include "sm2_utils.h"

// ============================================================================
// NEON-OPTIMIZED F16 TO FLOAT
// Converts 4x float16 values to 4x float32 using NEON
// ============================================================================

static void f16_to_float4(float* out, uint16_t* in) {
    // Each float16 is 16 bits, pack into uint32 for processing
    uint32x4_t bits = vmovl_n_u16(vld1_u16(in));

    uint32x4_t sign = vshrq_n_u32(bits, 15);
    uint32x4_t exp = vshrq_n_u32(bits, 10) & vdupq_n_u32(0x1F);
    uint32x4_t frac = bits & vdupq_n_u32(0x3FF);

    // Calculate float values
    // out = (-1)^sign * (1 + frac/1024) * 2^(exp-15)
    uint32x4_t biased_exp = exp - vdupq_n_u32(15);
    uint32x4_t mantissa = vaddq_u32(frac, vdupq_n_u32(512));

    // Use vgetq_lane_u32 to extract and compute
    for (int i = 0; i < 4; i++) {
        uint32_t s = vgetq_lane_u32(sign, i);
        uint32_t e = vgetq_lane_u32(biased_exp, i);
        uint32_t m = vgetq_lane_u32(mantissa, i);

        float result;
        if (e == 0) {
            result = (float)m / 1024.0f;
        } else {
            result = (1.0f + (float)m / 1024.0f) * powf(2.0f, (float)(e + 15));
        }
        out[i] = s ? -result : result;
    }
}

// Fallback scalar for small batches
static inline float f16_to_float_scalar(uint16_t h) {
    unsigned int bits = h;
    int sign = (bits >> 15) & 1;
    int exp = (bits >> 10) & 0x1F;
    int frac = bits & 0x3FF;
    float result;

    if (exp == 0) {
        result = (float)frac / 1024.0f;
    } else if (exp == 31) {
        result = (frac == 0) ? 1.0f / 0.0f : 0.0f / 0.0f;
    } else {
        result = (1.0f + (float)frac / 1024.0f) * powf(2.0f, (float)(exp - 15));
    }

    return sign ? -result : result;
}

// ============================================================================
// NEON-OPTIMIZED MATMUL: out = a @ b.T
// a: [m, k] float32, b: [n, k] float16 -> out: [m, n]
// Weight matrix b is transposed (column-major access pattern for cache efficiency)
// ============================================================================

void sm2_matmul_neon_f16(float* out, const float* a, const uint16_t* wb,
                        int m, int n, int k) {
    // Process 4 output columns at a time with NEON
    for (int i = 0; i < m; i++) {
        const float* a_row = a + i * k;

        for (int j = 0; j < n; j += 4) {
            int j_max = (j + 4 > n) ? n : j + 4;
            int remaining = j_max - j;

            float32x4_t sum_vec = vdupq_n_f32(0.0f);

            // Process in chunks of 4 for cache efficiency
            int l = 0;
            for (; l + 3 < k; l += 4) {
                // Load 4 elements from a
                float32x4_t a_vec = vld1q_f32(a_row + l);

                // Load 4 weight elements for each output column
                float32x4_t w0 = vdupq_n_f32(f16_to_float_scalar(wb[l * n + j + 0]));
                float32x4_t w1 = vdupq_n_f32(f16_to_float_scalar(wb[l * n + j + 1]));
                float32x4_t w2 = vdupq_n_f32(f16_to_float_scalar(wb[l * n + j + 2]));
                float32x4_t w3 = vdupq_n_f32(f16_to_float_scalar(wb[l * n + j + 3]));

                // Multiply accumulate and store result in separate accumulators
                // This is simplified - real version would interleave better
                sum_vec = vfmaq_f32(sum_vec, w0, a_vec);
            }

            // Handle remaining elements
            for (; l < k; l++) {
                float a_val = a_row[l];
                for (int jj = j; jj < j_max; jj++) {
                    out[i * n + jj] += a_val * f16_to_float_scalar(wb[l * n + jj]);
                }
            }
        }
    }
}

// ============================================================================
// NEON-OPTIMIZED RMSNORM
// Normalizes vector x with weight w: y = (x / rms(w)) * w
// ============================================================================

void sm2_rmsnorm_neon(float* out, const float* input, const uint16_t* weight,
                     int size, float eps) {
    // Compute sum of squares
    float32x4_t sum_sq_vec = vdupq_n_f32(0.0f);
    int i = 0;

    for (; i + 3 < size; i += 4) {
        float32x4_t v = vld1q_f32(input + i);
        sum_sq_vec = vfmaq_f32(sum_sq_vec, v, v);
    }
    for (; i < size; i++) {
        float32x4_t v = vdupq_n_f32(input[i]);
        sum_sq_vec = vfmaq_f32(sum_sq_vec, v, v);
    }

    // Horizontal sum
    float32x2_t sum_sq_low = vget_low_f32(sum_sq_vec);
    float32x2_t sum_sq_high = vget_high_f32(sum_sq_vec);
    float32x2_t sum_sq = vpadd_f32(sum_sq_low, sum_sq_high);
    sum_sq = vpadd_f32(sum_sq, sum_sq);
    float total_sum_sq = vget_lane_f32(sum_sq, 0);

    // Compute RMS
    float rms = sqrtf(total_sum_sq / (float)size + eps);
    float scale = 1.0f / rms;

    // Apply normalization and weight
    float32x4_t scale_vec = vdupq_n_f32(scale);

    for (i = 0; i + 3 < size; i += 4) {
        float32x4_t v = vld1q_f32(input + i);
        float32x4_t w = {
            f16_to_float_scalar(weight[i]),
            f16_to_float_scalar(weight[i + 1]),
            f16_to_float_scalar(weight[i + 2]),
            f16_to_float_scalar(weight[i + 3])
        };
        float32x4_t normalized = vmulq_f32(v, scale_vec);
        vst1q_f32(out + i, vmulq_f32(normalized, w));
    }
    for (; i < size; i++) {
        out[i] = (input[i] * scale) * f16_to_float_scalar(weight[i]);
    }
}

// ============================================================================
// NEON-OPTIMIZED ATTENTION SCORES
// Computes Q @ K^T / sqrt(d) for attention
// ============================================================================

void sm2_attention_scores_neon(float* scores, const float* q, const float* k_cache,
                              int n_heads, int n_kv_heads, int head_dim,
                              int seq_len) {
    int group_size = n_heads / n_kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int qh = 0; qh < n_heads; qh++) {
        int kv_head = qh / group_size;
        const float* q_head = q + qh * head_dim;
        const float* k_head = k_cache + kv_head * head_dim;

        float32x4_t sum_vec = vdupq_n_f32(0.0f);

        // Inner product in chunks
        for (int d = 0; d + 3 < head_dim; d += 4) {
            float32x4_t q_vec = vld1q_f32(q_head + d);
            float32x4_t k_vec = vld1q_f32(k_head + d);
            sum_vec = vfmaq_f32(sum_vec, q_vec, k_vec);
        }

        // Horizontal sum
        float32x2_t sum_low = vget_low_f32(sum_vec);
        float32x2_t sum_high = vget_high_f32(sum_vec);
        float32x2_t sum = vpadd_f32(sum_low, sum_high);
        sum = vpadd_f32(sum, sum);
        scores[qh] = scale * vget_lane_f32(sum, 0);
    }
}

// ============================================================================
// BACKEND REGISTRATION
// ============================================================================

sm2_backend sm2_backend_neon = {
    sm2_matmul_neon_f16,      // matmul_f16
    sm2_matmul_neon_f16,      // matmul_q4_k (use f16 path for now)
    sm2_matmul_neon_f16,      // matmul_q8_0 (use f16 path for now)
};
