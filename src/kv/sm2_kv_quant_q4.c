// sm2_kv_quant_q4.c - KV Q4 quantization

#include <math.h>
#include "smollm2.h"

// Quantize to Q4 (4-bit, packed 2 values per byte)
void sm2_kv_quantize_q4(float* out, const float* in, int n, float* scale) {
    float max_val = 0.0f;
    for (int i = 0; i < n; i++) {
        float abs = in[i] >= 0 ? in[i] : -in[i];
        if (abs > max_val) max_val = abs;
    }
    
    *scale = max_val / 7.0f;
    if (*scale < 1e-7f) *scale = 1e-7f;
    
    for (int i = 0; i < n; i++) {
        float v = in[i] / (*scale);
        int q = (int)(v >= 0 ? v + 0.5f : v - 0.5f);
        if (q > 7) q = 7;
        if (q < -7) q = -7;
        out[i] = (float)q;
    }
}

// Dequantize Q4 to F16
void sm2_kv_dequant_q4(float* out, const float* in, int n, float scale) {
    for (int i = 0; i < n; i++) {
        out[i] = in[i] * scale;
    }
}

// Pack Q4 nibbles into bytes
void sm2_kv_pack_q4(uint8_t* out, const float* in, int n) {
    for (int i = 0; i < n / 2; i++) {
        uint8_t lo = (uint8_t)((int)in[2*i] & 0x0F);
        uint8_t hi = (uint8_t)((int)in[2*i + 1] & 0x0F);
        out[i] = lo | (hi << 4);
    }
}

// Unpack Q4 nibbles from bytes
void sm2_kv_unpack_q4(float* out, const uint8_t* in, int n) {
    for (int i = 0; i < n; i++) {
        int byte_idx = i / 2;
        int shift = (i % 2) * 4;
        out[i] = (float)((in[byte_idx] >> shift) & 0x0F);
    }
}