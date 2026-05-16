// sm2_matmul_ref.c - Reference matmul implementations
//
// Portable C implementations of matrix multiplication.
// No SIMD, no arch-specific optimizations - pure reference.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "smollm2.h"

// ============================================================================
// FLOAT16 UTILITIES
// ============================================================================

// Convert IEEE-754 float16 to float
static float f16_to_float(uint16_t h) {
    unsigned int bits = h;
    int sign = (bits >> 15) & 1;
    int exp = (bits >> 10) & 0x1F;
    int frac = bits & 0x3FF;
    float result;

    if (exp == 0) {
        result = (float)frac / 1024.0f;
    } else if (exp == 31) {
        result = (frac == 0) ? 1.0f / 0.0f : 0.0f / 0.0f; // inf or nan
    } else {
        result = (1.0f + (float)frac / 1024.0f) * powf(2.0f, (float)(exp - 15));
    }

    return sign ? -result : result;
}

// ============================================================================
// F16 MATMUL (float16 reference)
// ============================================================================

void sm2_matmul_f16_ref(float* out, const float* a, const sm2_tensor_f16* wb, int m, int n, int k) {
    // out = a @ b, where a is [m, k], b is [k, n]
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                uint16_t b_h = wb->data[l * n + j];
                float b_f = f16_to_float(b_h);
                sum += a[i * k + l] * b_f;
            }
            out[i * n + j] = sum;
        }
    }
}

// Faster F16 matmul with inner loop unrolled 4x
void sm2_matmul_f16_ref_fast(float* out, const float* a, const sm2_tensor_f16* wb, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
            int l = 0;
            
            // Unroll by 4
            for (; l + 3 < k; l += 4) {
                uint16_t b0 = wb->data[l * n + j];
                uint16_t b1 = wb->data[(l+1) * n + j];
                uint16_t b2 = wb->data[(l+2) * n + j];
                uint16_t b3 = wb->data[(l+3) * n + j];
                
                sum0 += a[i * k + l] * f16_to_float(b0);
                sum1 += a[i * k + l + 1] * f16_to_float(b1);
                sum2 += a[i * k + l + 2] * f16_to_float(b2);
                sum3 += a[i * k + l + 3] * f16_to_float(b3);
            }
            
            // Remainder
            for (; l < k; l++) {
                uint16_t bh = wb->data[l * n + j];
                sum0 += a[i * k + l] * f16_to_float(bh);
            }
            
            out[i * n + j] = sum0 + sum1 + sum2 + sum3;
        }
    }
}

// ============================================================================
// Q4_K MATMUL (4-bit block quantization)
// Block size: 32 elements per block
// Each block: 16 4-bit values + 1 scale (float) + 1 zero (float)
// ============================================================================

// Dequantize a Q4_K block to float
static void dequant_q4_k_block(const uint8_t* qdata, const float* scale, const float* zero,
                                float* out, int n) {
    for (int i = 0; i < n; i++) {
        int byte_idx = i / 2;
        int shift = (i % 2) * 4;
        int qval = (qdata[byte_idx] >> shift) & 0x0F;
        out[i] = (float)qval * (*scale) + (*zero);
    }
}

void sm2_matmul_q4_k_ref(float* out, const float* a, const sm2_tensor_q4_k* wb, int m, int n, int k) {
    // For each row of output (m rows)
    for (int i = 0; i < m; i++) {
        // For each column of output (n cols)
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            
            // k dimension in blocks of 32
            int n_blocks = k / SM2_Q4_K_BLOCK_SIZE;
            
            for (int bl = 0; bl < n_blocks; bl++) {
                // Get scale and zero for this block
                float scale = wb->scales[bl * n + j];
                float zero = wb->zeros ? wb->zeros[bl * n + j] : 0.0f;
                
                // Get Q4 data offset
                // Q4 layout: [n_blocks][n][16 nibbles] - nibbles packed 2 per byte
                int block_offset = bl * n * 16 / 2; // nibbles to bytes
                int col_offset = j * 16 / 2;
                
                const uint8_t* qdata = wb->data + block_offset + col_offset;
                
                // Dequantize block and accumulate
                float vals[32];
                dequant_q4_k_block(qdata, &scale, &zero, vals, 32);
                
                // Multiply and accumulate
                for (int l = 0; l < 32; l++) {
                    sum += a[i * k + bl * 32 + l] * vals[l];
                }
            }
            
            out[i * n + j] = sum;
        }
    }
}

// ============================================================================
// Q8_0 MATMUL (8-bit quantization)
// Block size: 32 elements
// Each block: 32 8-bit values + 1 scale (float)
// ============================================================================

void sm2_matmul_q8_0_ref(float* out, const float* a, const sm2_tensor_q8_0* wb, int m, int n, int k) {
    int n_blocks = k / SM2_Q4_K_BLOCK_SIZE; // Same block size
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            
            for (int bl = 0; bl < n_blocks; bl++) {
                float scale = wb->scales[bl * n + j];
                uint8_t* qdata = wb->data + bl * n * 32 + j * 32;
                
                for (int l = 0; l < 32; l++) {
                    sum += a[i * k + bl * 32 + l] * ((float)qdata[l] * scale);
                }
            }
            
            out[i * n + j] = sum;
        }
    }
}

// ============================================================================
// BACKEND REGISTRATION
// Note: sm2_backend_ref is defined in src/backend/sm2_backend_ref.c
// to avoid duplicate symbol errors when linking.
// ============================================================================
extern sm2_backend sm2_backend_ref;