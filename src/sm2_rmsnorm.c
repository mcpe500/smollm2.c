// sm2_rmsnorm.c - Root Mean Square Layer Normalization

#include <stdio.h>
#include <math.h>
#include "smollm2.h"

// ============================================================================
// RMSNorm (Root Mean Square Normalization)
// 
// Unlike LayerNorm, RMSNorm only normalizes by RMS without centering.
// Formula: output = (input / RMS(input)) * weight
// RMS = sqrt(mean(x^2) + eps)
// ============================================================================

void sm2_rmsnorm(float* out, const float* input, const float* weight, int size, float eps) {
    // Calculate RMS = sqrt(sum(x^2) / n + eps)
    float sum_sq = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_sq += input[i] * input[i];
    }
    
    float rms = sqrtf(sum_sq / (float)size + eps);
    float scale = 1.0f / rms;
    
    // Multiply and scale by weight
    for (int i = 0; i < size; i++) {
        out[i] = input[i] * scale * weight[i];
    }
}

// In-place variant for pre-normalized residual
void sm2_rmsnorm_inplace(float* vec, const float* weight, int size, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_sq += vec[i] * vec[i];
    }
    
    float rms = sqrtf(sum_sq / (float)size + eps);
    float scale = 1.0f / rms;
    
    for (int i = 0; i < size; i++) {
        vec[i] = vec[i] * scale * weight[i];
    }
}

// SIMD version (requires <xmmintrin.h> for SSE or <arm_neon.h> for NEON)
// Currently using portable C - SIMD variants can be added per-backend
void sm2_rmsnorm_simd(float* out, const float* input, const float* weight, int size, float eps) {
    // Fallback to scalar
    sm2_rmsnorm(out, input, weight, size, eps);
}