// sm2_q5k.c - Q5_K weight quantization (5-bit block quantization)
// Q5_K uses 5 bits per value, packed with scale

#include "smollm2.h"
#include "sm2_utils.h"
#include <stdlib.h>
#include <string.h>

#define Q5K_BLOCK_SIZE 32

// Quantize F16 -> Q5_K
// Layout: [n_blocks][block_size * 5/8 bytes] (40 bits per block, 5 bytes + padding)
int sm2_quantize_q5_k(void* out, const void* in, int rows, int cols, float** scales_out) {
    uint16_t* input = (uint16_t*)in;
    uint8_t* output = (uint8_t*)out;
    
    int n_blocks = (rows * cols + Q5K_BLOCK_SIZE - 1) / Q5K_BLOCK_SIZE;
    float* scales = calloc(n_blocks, sizeof(float));
    
    if (!scales) return -1;
    
    // Process each block
    for (int blk = 0; blk < n_blocks; blk++) {
        int start = blk * Q5K_BLOCK_SIZE;
        int end = start + Q5K_BLOCK_SIZE;
        if (end > rows * cols) end = rows * cols;
        
        // Find max absolute value
        float max_val = 0.0f;
        for (int i = start; i < end; i++) {
            float v = sm2_f16_to_float(input[i]);
            float abs = v >= 0 ? v : -v;
            if (abs > max_val) max_val = abs;
        }
        
        float scale = max_val / 15.0f; // 5-bit: -15..15
        if (scale < 1e-7f) scale = 1e-7f;
        scales[blk] = scale;
        
        // Quantize: 5 bits packed into bytes
        // 32 values * 5 bits = 160 bits = 20 bytes
        for (int i = 0; i < end - start; i++) {
            float v = sm2_f16_to_float(input[start + i]);
            int q = (int)(v / scale);
            if (q > 15) q = 15;
            if (q < -15) q = -15;
            
            int byte_idx = (i * 5) / 8;
            int bit_shift = (i * 5) % 8;
            
            // Pack 5-bit value
            uint16_t val = (uint16_t)(q & 0x1F);
            output[blk * 20 + byte_idx] |= (uint8_t)(val << bit_shift);
            if (bit_shift > 3) {
                output[blk * 20 + byte_idx + 1] |= (uint8_t)(val >> (8 - bit_shift));
            }
        }
    }
    
    *scales_out = scales;
    return 0;
}

// Dequantize Q5_K -> F16
int sm2_dequant_q5_k(void* out, const void* in, int rows, int cols, const float* scales) {
    uint16_t* output = (uint16_t*)out;
    uint8_t* input = (uint8_t*)in;
    
    int n_blocks = (rows * cols + Q5K_BLOCK_SIZE - 1) / Q5K_BLOCK_SIZE;
    
    for (int blk = 0; blk < n_blocks; blk++) {
        float scale = scales[blk];
        
        for (int i = 0; i < Q5K_BLOCK_SIZE && (blk * Q5K_BLOCK_SIZE + i) < rows * cols; i++) {
            int byte_idx = (i * 5) / 8;
            int bit_shift = (i * 5) % 8;
            
            int q = (input[blk * 20 + byte_idx] >> bit_shift) & 0x1F;
            if (bit_shift > 3) {
                q |= (input[blk * 20 + byte_idx + 1] << (8 - bit_shift)) & 0x1F;
            }
            
            // Sign extend 5-bit
            if (q > 15) q -= 32;
            
            float v = (float)q * scale;
            output[blk * Q5K_BLOCK_SIZE + i] = sm2_float_to_f16(v);
        }
    }
    
    return 0;
}