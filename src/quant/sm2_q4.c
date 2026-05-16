// sm2_q4.c - Q4_0 weight quantization (4-bit, simple)

#include "smollm2.h"
#include "sm2_utils.h"

// Quantize F16 -> Q4_0 (4-bit, no block structure)
int sm2_quantize_q4_0(void* out, const void* in, int n, float* scale) {
    float max_val = 0.0f;
    uint16_t* input = (uint16_t*)in;
    uint8_t* output = (uint8_t*)out;
    
    // Find max
    for (int i = 0; i < n; i++) {
        float v = sm2_f16_to_float(input[i]);
        float abs = v >= 0 ? v : -v;
        if (abs > max_val) max_val = abs;
    }
    
    *scale = max_val / 7.0f;
    if (*scale < 1e-7f) *scale = 1e-7f;
    
    // Quantize and pack 2 values per byte
    for (int i = 0; i < n / 2; i++) {
        float v0 = sm2_f16_to_float(input[2*i]);
        float v1 = sm2_f16_to_float(input[2*i + 1]);
        
        int q0 = (int)(v0 / (*scale));
        int q1 = (int)(v1 / (*scale));
        
        if (q0 > 7) q0 = 7;
        if (q0 < -7) q0 = -7;
        if (q1 > 7) q1 = 7;
        if (q1 < -7) q1 = -7;
        
        output[i] = (uint8_t)((q0 & 0x0F) | ((q1 & 0x0F) << 4));
    }
    
    return 0;
}

// Dequantize Q4_0 -> F16
int sm2_dequant_q4_0(void* out, const void* in, int n, float scale) {
    uint16_t* output = (uint16_t*)out;
    uint8_t* input = (uint8_t*)in;
    
    for (int i = 0; i < n / 2; i++) {
        int q0 = (input[i] & 0x0F);
        int q1 = (input[i] >> 4) & 0x0F;
        
        // Sign extend
        if (q0 > 7) q0 -= 16;
        if (q1 > 7) q1 -= 16;
        
        output[2*i] = sm2_float_to_f16((float)q0 * scale);
        output[2*i + 1] = sm2_float_to_f16((float)q1 * scale);
    }
    
    return 0;
}