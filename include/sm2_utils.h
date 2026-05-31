// sm2_utils.h - Common utility functions for smollm2
#ifndef SM2_UTILS_H
#define SM2_UTILS_H

#include <stdint.h>
#include <string.h>
#include <math.h>

// Lookup table for exponent bias offset: (exp + 127 - 15) << 23
// Maps F16 exp (0-31) to F32 biased exponent
static const uint32_t f16_exp_table[32] = {
    0x3C900000,  // exp=0:  (0+127-15)<<23 = 112<<23
    0x3D000000,  // exp=1:  (1+127-15)<<23 = 113<<23
    0x3D100000,  // exp=2:  (2+127-15)<<23 = 114<<23
    0x3D200000,  // exp=3:  (3+127-15)<<23 = 115<<23
    0x3D300000,  // exp=4:  (4+127-15)<<23 = 116<<23
    0x3D400000,  // exp=5:  (5+127-15)<<23 = 117<<23
    0x3D500000,  // exp=6:  (6+127-15)<<23 = 118<<23
    0x3D600000,  // exp=7:  (7+127-15)<<23 = 119<<23
    0x3D700000,  // exp=8:  (8+127-15)<<23 = 120<<23
    0x3D800000,  // exp=9:  (9+127-15)<<23 = 121<<23
    0x3D900000,  // exp=10: (10+127-15)<<23 = 122<<23
    0x3DA00000,  // exp=11: (11+127-15)<<23 = 123<<23
    0x3DB00000,  // exp=12: (12+127-15)<<23 = 124<<23
    0x3DC00000,  // exp=13: (13+127-15)<<23 = 125<<23
    0x3DD00000,  // exp=14: (14+127-15)<<23 = 126<<23
    0x3DE00000,  // exp=15: (15+127-15)<<23 = 127<<23
    0x3DF00000,  // exp=16: (16+127-15)<<23 = 128<<23
    0x3E000000,  // exp=17: (17+127-15)<<23 = 129<<23
    0x3E100000,  // exp=18: (18+127-15)<<23 = 130<<23
    0x3E200000,  // exp=19: (19+127-15)<<23 = 131<<23
    0x3E300000,  // exp=20: (20+127-15)<<23 = 132<<23
    0x3E400000,  // exp=21: (21+127-15)<<23 = 133<<23
    0x3E500000,  // exp=22: (22+127-15)<<23 = 134<<23
    0x3E600000,  // exp=23: (23+127-15)<<23 = 135<<23
    0x3E700000,  // exp=24: (24+127-15)<<23 = 136<<23
    0x3E800000,  // exp=25: (25+127-15)<<23 = 137<<23
    0x3E900000,  // exp=26: (26+127-15)<<23 = 138<<23
    0x3EA00000,  // exp=27: (27+127-15)<<23 = 139<<23
    0x3EB00000,  // exp=28: (28+127-15)<<23 = 140<<23
    0x3EC00000,  // exp=29: (29+127-15)<<23 = 141<<23
    0x3ED00000,  // exp=30: (30+127-15)<<23 = 142<<23
    0x3EE00000,  // exp=31: (31+127-15)<<23 = 143<<23
};


// Convert IEEE-754 float16 to float using lookup table
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

    // Fast path: use lookup table for exponent
    unsigned int ieee = ((sign << 31) | f16_exp_table[exp] | (frac << 13));
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