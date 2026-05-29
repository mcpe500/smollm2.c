#include <stdio.h>
#include <stdint.h>
#include <math.h>

float f16_to_f32(uint16_t f16) {
    uint16_t sign = (f16 >> 15) & 0x1;
    uint16_t exp = (f16 >> 10) & 0x1F;
    uint16_t mant = f16 & 0x3FF;
    
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        // THIS IS WRONG for denormals - should compute 2^(-14) * mant/1024
        return sign ? -INFINITY : INFINITY;  // denorm handled specially
    }
    if (exp == 31) {
        if (mant == 0) return sign ? -INFINITY : INFINITY;
        return sign ? -NAN : NAN;
    }
    
    float value = (1.0f + mant / 1024.0f) * powf(2.0f, (float)exp - 15.0f);
    return sign ? -value : value;
}

int main() {
    // Check if there are denormal values in tok_embeddings
    // Read first 1000 values
    FILE* f = fopen("smollm2-135m.sm2", "rb");
    fseek(f, 1179115 + 8, SEEK_SET);
    
    int count = 0;
    int denorm_count = 0;
    int inf_count = 0;
    
    for (int i = 0; i < 1000; i++) {
        uint16_t val;
        fread(&val, 2, 1, f);
        
        uint16_t exp = (val >> 10) & 0x1F;
        if (exp == 0) {
            denorm_count++;
            uint16_t mant = val & 0x3FF;
            if (mant != 0) {
                printf("Denormal found at %d: 0x%04x, mant=%d\n", i, val, mant);
            }
        }
        float v = f16_to_f32(val);
        if (isinf(v)) inf_count++;
    }
    
    printf("\nTotal: %d denorm, %d inf\n", denorm_count, inf_count);
    fclose(f);
    return 0;
}
