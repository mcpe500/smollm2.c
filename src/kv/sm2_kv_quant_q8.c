// sm2_kv_quant_q8.c - KV Q8 quantization

#include <math.h>
#include "smollm2.h"

// Quantize KV to Q8
void sm2_kv_quantize_q8(float* out, const float* in, int n, float* scale) {
    float max_val = 0.0f;
    for (int i = 0; i < n; i++) {
        float abs = in[i] >= 0 ? in[i] : -in[i];
        if (abs > max_val) max_val = abs;
    }
    
    *scale = max_val / 127.0f;
    if (*scale < 1e-7f) *scale = 1e-7f;
    
    for (int i = 0; i < n; i++) {
        float v = in[i] / (*scale);
        int q = (int)(v >= 0 ? v + 0.5f : v - 0.5f);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        out[i] = (float)q;
    }
}

// Dequantize Q8 to F16
void sm2_kv_dequant_q8(float* out, const float* in, int n, float scale) {
    for (int i = 0; i < n; i++) {
        out[i] = in[i] * scale;
    }
}

// Quantize entire KV block (all heads, all tokens)
int sm2_kv_quantize_block_q8(
    float* k_out, float* k_scale_out,
    float* v_out, float* v_scale_out,
    const float* k_in, const float* v_in,
    int n_kv_heads, int n_tokens
) {
    for (int h = 0; h < n_kv_heads; h++) {
        float k_scale, v_scale;
        
        // Quantize K for this head
        sm2_kv_quantize_q8(
            k_out + h * n_tokens * 64,
            k_in + h * n_tokens * 64,
            n_tokens * 64,
            &k_scale
        );
        k_scale_out[h] = k_scale;
        
        // Quantize V for this head
        sm2_kv_quantize_q8(
            v_out + h * n_tokens * 64,
            v_in + h * n_tokens * 64,
            n_tokens * 64,
            &v_scale
        );
        v_scale_out[h] = v_scale;
    }
    
    return 0;
}