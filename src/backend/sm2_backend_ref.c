// sm2_backend_ref.c - Reference backend implementation

#include "smollm2.h"
#include "sm2_utils.h"
#include <string.h>

// Portable reference matmul - no SIMD, pure C
// This is the fallback when no optimized backend is available

// F16 matmul reference implementation
static void sm2_matmul_f16_impl(float* out, const float* a, const sm2_tensor_f16* b, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                uint16_t b_h = b->data[l * n + j];
                float b_f = sm2_f16_to_float(b_h);
                sum += a[i * k + l] * b_f;
            }
            out[i * n + j] = sum;
        }
    }
}

// Q4_K matmul reference
static void sm2_matmul_q4_k_impl(float* out, const float* a, const sm2_tensor_q4_k* b, int m, int n, int k) {
    int n_blocks = k / SM2_Q4_K_BLOCK_SIZE;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;

            for (int bl = 0; bl < n_blocks; bl++) {
                float scale = b->scales[bl * n + j];
                float zero = b->zeros ? b->zeros[bl * n + j] : 0.0f;

                int block_offset = bl * n * 16 / 2;
                int col_offset = j * 16 / 2;

                const uint8_t* qdata = b->data + block_offset + col_offset;

                for (int l = 0; l < SM2_Q4_K_BLOCK_SIZE; l++) {
                    int byte_idx = l / 2;
                    int shift = (l % 2) * 4;
                    int qval = (qdata[byte_idx] >> shift) & 0x0F;
                    // Sign extend
                    if (qval > 7) qval -= 16;

                    float dequant = (float)qval * scale + zero;
                    sum += a[i * k + bl * SM2_Q4_K_BLOCK_SIZE + l] * dequant;
                }
            }

            out[i * n + j] = sum;
        }
    }
}

// Q8_0 matmul reference
static void sm2_matmul_q8_0_impl(float* out, const float* a, const sm2_tensor_q8_0* b, int m, int n, int k) {
    int n_blocks = k / 32;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;

            for (int bl = 0; bl < n_blocks; bl++) {
                float scale = b->scales[bl * n + j];
                const uint8_t* qdata = b->data + bl * n * 32 + j * 32;

                for (int l = 0; l < 32; l++) {
                    sum += a[i * k + bl * 32 + l] * ((float)qdata[l] * scale);
                }
            }

            out[i * n + j] = sum;
        }
    }
}

// Wrapper functions for function pointer compatibility
static void sm2_matmul_f16_wrap(float* out, const float* a, const void* b, int m, int n, int k) {
    sm2_matmul_f16_impl(out, a, (const sm2_tensor_f16*)b, m, n, k);
}

static void sm2_matmul_q4_k_wrap(float* out, const float* a, const void* b, int m, int n, int k) {
    sm2_matmul_q4_k_impl(out, a, (const sm2_tensor_q4_k*)b, m, n, k);
}

static void sm2_matmul_q8_0_wrap(float* out, const float* a, const void* b, int m, int n, int k) {
    sm2_matmul_q8_0_impl(out, a, (const sm2_tensor_q8_0*)b, m, n, k);
}

// Register reference backend
sm2_backend sm2_backend_ref = {
    .matmul_f16 = sm2_matmul_f16_wrap,
    .matmul_q4_k = sm2_matmul_q4_k_wrap,
    .matmul_q8_0 = sm2_matmul_q8_0_wrap,
};