// sm2_kv_turbo2.c - KIVI-like TURBO2 quantization
// K per-channel, V per-token (~2 bits effective)

#include <math.h>
#include "smollm2.h"

// TURBO2 quantize: K per-channel, V per-token
void sm2_kv_turbo2_quantize(
    float* k_out, float* k_scale_out,
    float* v_out, float* v_scale_out,
    const float* k_in, const float* v_in,
    int n_kv_heads, int head_dim, int n_tokens
) {
    // K: per-channel quantization
    for (int h = 0; h < n_kv_heads; h++) {
        for (int d = 0; d < head_dim; d++) {
            // Find max in this channel across all tokens
            float max_val = 0.0f;
            for (int t = 0; t < n_tokens; t++) {
                float abs = k_in[h * n_tokens * head_dim + t * head_dim + d];
                if (abs < 0) abs = -abs;
                if (abs > max_val) max_val = abs;
            }
            
            float scale = max_val / 7.0f;
            if (scale < 1e-7f) scale = 1e-7f;
            
            // Quantize this channel
            for (int t = 0; t < n_tokens; t++) {
                float v = k_in[h * n_tokens * head_dim + t * head_dim + d];
                int q = (int)(v / scale + 0.5f);
                if (q > 7) q = 7;
                if (q < -7) q = -7;
                k_out[h * n_tokens * head_dim + t * head_dim + d] = (float)q;
            }
            k_scale_out[h * head_dim + d] = scale;
        }
    }
    
    // V: per-token quantization
    for (int t = 0; t < n_tokens; t++) {
        float max_val = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            float abs = v_in[t * head_dim + d];
            if (abs < 0) abs = -abs;
            if (abs > max_val) max_val = abs;
        }
        
        float scale = max_val / 7.0f;
        if (scale < 1e-7f) scale = 1e-7f;
        
        for (int d = 0; d < head_dim; d++) {
            float v = v_in[t * head_dim + d];
            int q = (int)(v / scale + 0.5f);
            if (q > 7) q = 7;
            if (q < -7) q = -7;
            v_out[t * head_dim + d] = (float)q;
        }
        v_scale_out[t] = scale;
    }
}

// TURBO2 dequantize
void sm2_kv_turbo2_dequant(
    float* out,
    const float* qdata,
    const float* scales,
    int n_kv_heads, int head_dim, int n_tokens,
    int is_k
) {
    if (is_k) {
        // K: per-channel scales
        for (int h = 0; h < n_kv_heads; h++) {
            for (int d = 0; d < head_dim; d++) {
                float scale = scales[h * head_dim + d];
                for (int t = 0; t < n_tokens; t++) {
                    out[h * n_tokens * head_dim + t * head_dim + d] = 
                        qdata[h * n_tokens * head_dim + t * head_dim + d] * scale;
                }
            }
        }
    } else {
        // V: per-token scales
        for (int t = 0; t < n_tokens; t++) {
            float scale = scales[t];
            for (int d = 0; d < head_dim; d++) {
                out[t * head_dim + d] = qdata[t * head_dim + d] * scale;
            }
        }
    }
}