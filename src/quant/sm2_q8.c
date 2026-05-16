// sm2_q8.c - Q8_0 weight quantization

#include "smollm2.h"
#include "sm2_utils.h"

// Quantize F16 -> Q8_0
int sm2_quantize_q8_0(void* out, const void* in, int n, float* scale) {
    float max_val = 0.0f;
    uint16_t* input = (uint16_t*)in;
    int8_t* output = (int8_t*)out;
    
    // Find max
    for (int i = 0; i < n; i++) {
        float v = sm2_f16_to_float(input[i]);
        float abs = v >= 0 ? v : -v;
        if (abs > max_val) max_val = abs;
    }
    
    *scale = max_val / 127.0f;
    if (*scale < 1e-7f) *scale = 1e-7f;
    
    // Quantize
    for (int i = 0; i < n; i++) {
        float v = sm2_f16_to_float(input[i]);
        int q = (int)(v / (*scale));
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        output[i] = (int8_t)q;
    }
    
    return 0;
}

// Dequantize Q8_0 -> F16
int sm2_dequant_q8_0(void* out, const void* in, int n, float scale) {
    uint16_t* output = (uint16_t*)out;
    int8_t* input = (int8_t*)in;
    
    for (int i = 0; i < n; i++) {
        float v = (float)input[i] * scale;
        output[i] = sm2_float_to_f16(v);
    }
    
    return 0;
}