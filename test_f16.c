#include <stdio.h>
#include <stdint.h>
#include <math.h>

float f16_to_f32(uint16_t f16) {
    uint16_t sign = (f16 >> 15) & 0x1;
    uint16_t exp = (f16 >> 10) & 0x1F;
    uint16_t mant = f16 & 0x3FF;
    
    printf("  f16=0x%04x: sign=%d exp=%2d mant=%3d ", f16, sign, exp, mant);
    
    if (exp == 0) {
        if (mant == 0) { printf("-> %f (zero)\n", sign ? -0.0f : 0.0f); return sign ? -0.0f : 0.0f; }
        // denormal
        float val = powf(2.0f, -14.0f) * ((float)mant / 1024.0f);
        printf("-> %f (denorm)\n", sign ? -val : val);
        return sign ? -val : val;
    }
    if (exp == 31) {
        if (mant == 0) { printf("-> inf\n"); return sign ? -INFINITY : INFINITY; }
        printf("-> nan\n");
        return sign ? -NAN : NAN;
    }
    
    float val = (1.0f + mant / 1024.0f) * powf(2.0f, (float)exp - 15.0f);
    printf("-> %f\n", sign ? -val : val);
    return sign ? -val : val;
}

int main() {
    // Test with known values from HF
    // HF first value: BF16=0xbdd9 -> -0.105957
    
    // The conversion produces F16=0xaec8
    printf("Expected value for F16 0xaec8: -0.105957\n");
    float v = f16_to_f32(0xaec8);
    printf("Result: %f\n\n", v);
    
    // Test with various F16 values
    uint16_t test_vals[] = {
        0x3c00,  // 1.0
        0xbc00,  // -1.0
        0x0000,  // 0.0
        0xaec8,  // -0.105957
        0x2860,  // 0.034180
    };
    
    printf("Standard values:\n");
    for (int i = 0; i < 5; i++) {
        f16_to_f32(test_vals[i]);
    }
    
    return 0;
}
