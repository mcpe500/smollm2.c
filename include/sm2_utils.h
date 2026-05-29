// sm2_utils.h - Common utility functions for smollm2
#ifndef SM2_UTILS_H
#define SM2_UTILS_H

#include <stdint.h>
#include <string.h>
#include <math.h>

// Convert IEEE-754 float16 to float
// Optimized: uses shifts instead of powf
static inline float sm2_f16_to_float(uint16_t h) {
    unsigned int bits = h;
    int sign = (bits >> 15) & 1;
    int exp = (bits >> 10) & 0x1F;
    int frac = bits & 0x3FF;

    if (exp == 0) {
        // Denormalized
        return sign ? -(float)frac / 65536.0f : (float)frac / 65536.0f;
    }
    if (exp == 31) {
        // Inf or NaN
        return sign ? -1.0f / 0.0f : 1.0f / 0.0f;
    }

    // Fast path: compute 2^(exp-15) using bit manipulation
    // Use direct float construction
    unsigned int ieee = ((sign << 31) |
                         ((exp + 127 - 15) << 23) |
                         (frac << (23 - 10)));
    float result;
    memcpy(&result, &ieee, sizeof(float));
    return result;
}

// Convert float to IEEE-754 float16
static inline uint16_t sm2_float_to_f16(float f) {
    unsigned int bits;
    memcpy(&bits, &f, sizeof(float));
    int sign = (bits >> 16) & 0x8000;
    int exp = (bits >> 23) & 0xFF;
    int frac = bits & 0x7FFFFF;

    // Handle special cases
    if (exp == 0) return sign; // Zero
    if (exp == 0xFF) {
        if (frac == 0) return sign | 0x7C00; // Inf
        return sign | 0x7C00 | (frac != 0); // NaN
    }

    // Normalize
    int e = exp - 127;
    if (e > 15) return sign | 0x7C00; // Overflow -> inf
    if (e < -10) return sign; // Underflow -> zero

    int pre_exp = e + 15;
    int pre_frac = (frac >> 13) & 0x3FF; // 10 bits

    return (uint16_t)(sign | (pre_exp << 10) | pre_frac);
}

#endif // SM2_UTILS_H