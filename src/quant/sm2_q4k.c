// sm2_q4k.c - Q4_K weight quantization (block-based 4-bit)
// Block size: 32 elements per block
// Each block: 16 4-bit values + 1 scale + 1 zero point

#include "smollm2.h"
#include "sm2_utils.h"
#include <stdlib.h>
#include <string.h>

#define Q4K_BLOCK_SIZE 32

// Quantize F16 -> Q4_K
int sm2_quantize_q4_k(void* out, const void* in, int rows, int cols, float** scales_out) {
    uint16_t* input = (uint16_t*)in;
    uint8_t* output = (uint8_t*)out;
    
    int n_blocks = (rows * cols + Q4K_BLOCK_SIZE - 1) / Q4K_BLOCK_SIZE;
    float* scales = calloc(n_blocks, sizeof(float));
    float* zeros = calloc(n_blocks, sizeof(float));
    
    if (!scales || !zeros) {
        free(scales);
        free(zeros);
        return -1;
    }
    
    // Process each block
    for (int blk = 0; blk < n_blocks; blk++) {
        int start = blk * Q4K_BLOCK_SIZE;
        int end = start + Q4K_BLOCK_SIZE;
        if (end > rows * cols) end = rows * cols;
        
        // Find max absolute value in block
        float max_val = 0.0f;
        for (int i = start; i < end; i++) {
            float v = sm2_f16_to_float(input[i]);
            float abs = v >= 0 ? v : -v;
            if (abs > max_val) max_val = abs;
        }
        
        float scale = max_val / 7.0f;
        if (scale < 1e-7f) scale = 1e-7f;
        scales[blk] = scale;
        zeros[blk] = 0.0f; // symmetric quantization
        
        // Quantize block
        for (int i = start; i < end; i++) {
            int idx = (i - start) / 2;
            int shift = (i - start) % 2 * 4;
            
            float v = sm2_f16_to_float(input[i]);
            int q = (int)(v / scale);
            if (q > 7) q = 7;
            if (q < -7) q = -7;
            
            if ((i - start) % 2 == 0) {
                output[blk * (Q4K_BLOCK_SIZE / 2) + idx] = (uint8_t)(q & 0x0F);
            } else {
                output[blk * (Q4K_BLOCK_SIZE / 2) + idx] |= (uint8_t)((q & 0x0F) << 4);
            }
        }
    }
    
    *scales_out = scales;
    return 0;
}

// Dequantize Q4_K -> F16
int sm2_dequant_q4_k(void* out, const void* in, int rows, int cols, const float* scales) {
    uint16_t* output = (uint16_t*)out;
    uint8_t* input = (uint8_t*)in;
    
    int n_blocks = (rows * cols + Q4K_BLOCK_SIZE - 1) / Q4K_BLOCK_SIZE;
    
    for (int blk = 0; blk < n_blocks; blk++) {
        float scale = scales[blk];
        int start = blk * Q4K_BLOCK_SIZE;
        int end = start + Q4K_BLOCK_SIZE;
        if (end > rows * cols) end = rows * cols;
        
        for (int i = start; i < end; i++) {
            int idx = (i - start) / 2;
            int shift = (i - start) % 2 * 4;
            
            int q = (input[blk * (Q4K_BLOCK_SIZE / 2) + idx] >> shift) & 0x0F;
            // Sign extend
            if (q > 7) q -= 16;
            
            float v = (float)q * scale;
            output[i] = sm2_float_to_f16(v);
        }
    }
    
    return 0;
}