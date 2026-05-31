// sm2_matmul_fast.c - Fast matrix multiplication with loop unrolling
//
// Optimizations:
// - Inner loop unrolling (4x)
// - Inline F16 conversion with lookup table
// - Non-Kahan summation (faster but sufficient precision)

#include <stdio.h>
#include <math.h>
#include "smollm2.h"

// F16 exponent lookup table (computed once)
static float f16_exp_table[32];
static int f16_exp_init = 0;

static void init_f16_exp(void) {
    if (f16_exp_init) return;
    for (int e = 0; e < 32; e++) {
        if (e == 0) {
            f16_exp_table[e] = 1.0f / 1024.0f;
        } else if (e == 31) {
            f16_exp_table[e] = 1.0f / 0.0f;
        } else {
            f16_exp_table[e] = ldexpf(1.0f, e - 15);
        }
    }
    f16_exp_init = 1;
}

// Fast F16 to float using lookup table
static inline float f16_to_f32(uint16_t h) {
    if (!f16_exp_init) init_f16_exp();
    unsigned int bits = h;
    int sign = (bits >> 15) & 1;
    int exp = (bits >> 10) & 0x1F;
    int frac = bits & 0x3FF;
    float result;

    if (exp == 0) {
        result = (float)frac * f16_exp_table[0];
    } else if (exp == 31) {
        result = (frac == 0) ? 1.0f / 0.0f : 0.0f / 0.0f;
    } else {
        result = (1.0f + (float)frac / 1024.0f) * f16_exp_table[exp];
    }

    return sign ? -result : result;
}

// Fast matmul: out = a @ b, where a is [m, k], b is [k, n] F16
// Uses 8x inner loop unrolling with aliasing hints
void sm2_matmul_f16_fast(float* restrict out, const float* restrict a, 
                         const uint16_t* restrict wb, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        const float* a_row = a + i * k;

        for (int j = 0; j < n; j++) {
            // 8x unrolled accumulation
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
            float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;
            int l = 0;

            for (; l + 7 < k; l += 8) {
                uint16_t w0 = wb[l * n + j];
                uint16_t w1 = wb[(l+1) * n + j];
                uint16_t w2 = wb[(l+2) * n + j];
                uint16_t w3 = wb[(l+3) * n + j];
                uint16_t w4 = wb[(l+4) * n + j];
                uint16_t w5 = wb[(l+5) * n + j];
                uint16_t w6 = wb[(l+6) * n + j];
                uint16_t w7 = wb[(l+7) * n + j];

                sum0 += a_row[l] * f16_to_f32(w0);
                sum1 += a_row[l+1] * f16_to_f32(w1);
                sum2 += a_row[l+2] * f16_to_f32(w2);
                sum3 += a_row[l+3] * f16_to_f32(w3);
                sum4 += a_row[l+4] * f16_to_f32(w4);
                sum5 += a_row[l+5] * f16_to_f32(w5);
                sum6 += a_row[l+6] * f16_to_f32(w6);
                sum7 += a_row[l+7] * f16_to_f32(w7);
            }

            // Handle remainder
            for (; l < k; l++) {
                sum0 += a_row[l] * f16_to_f32(wb[l * n + j]);
            }

            out[i * n + j] = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;
        }
    }
}

// Fast RMSNorm with 4x unrolling
void sm2_rmsnorm_fast(float* out, const float* input, const uint16_t* weight, int size, float eps) {
    // Compute sum of squares
    float sum_sq = 0.0f;
    int i = 0;

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

    // Apply normalization with weight
    i = 0;
    for (; i + 3 < size; i += 4) {
        float w0 = f16_to_f32(weight[i]);
        float w1 = f16_to_f32(weight[i+1]);
        float w2 = f16_to_f32(weight[i+2]);
        float w3 = f16_to_f32(weight[i+3]);

        out[i]   = input[i]   * scale * w0;
        out[i+1] = input[i+1] * scale * w1;
        out[i+2] = input[i+2] * scale * w2;
        out[i+3] = input[i+3] * scale * w3;
    }
    for (; i < size; i++) {
        out[i] = input[i] * scale * f16_to_f32(weight[i]);
    }
}
