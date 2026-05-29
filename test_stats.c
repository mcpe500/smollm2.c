#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "smollm2.h"

float f16_to_f32(uint16_t f16) {
    uint16_t sign = (f16 >> 15) & 0x1;
    uint16_t exp = (f16 >> 10) & 0x1F;
    uint16_t mant = f16 & 0x3FF;
    
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        return sign ? -INFINITY : INFINITY;  // denorm
    }
    if (exp == 31) {
        if (mant == 0) return sign ? -INFINITY : INFINITY;
        return sign ? -NAN : NAN;
    }
    
    float value = (1.0f + mant / 1024.0f) * powf(2.0f, (float)exp - 15.0f);
    return sign ? -value : value;
}

int main() {
    sm2_model* model;
    if (sm2_load_model("smollm2-135m.sm2", &model) != 0) {
        fprintf(stderr, "Failed to load\n");
        return 1;
    }
    
    uint16_t* data = model->tok_embeddings->data;
    int n = 100;
    
    float mn = f16_to_f32(data[0]);
    float mx = mn;
    double sum = 0.0;
    
    printf("First 10 converted values:\n");
    for (int i = 0; i < 10; i++) {
        float v = f16_to_f32(data[i]);
        printf("  [%d] f16=0x%04x -> %f\n", i, data[i], v);
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += fabsf(v);
    }
    
    printf("\nStats for first 100:\n");
    for (int i = 10; i < n; i++) {
        float v = f16_to_f32(data[i]);
        if (isinf(v)) { printf("  [%d] is inf!\n", i); }
        if (isnan(v)) { printf("  [%d] is nan!\n", i); }
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += fabsf(v);
    }
    
    printf("Min: %f, Max: %f, Sum of abs: %f\n", mn, mx, sum);
    
    sm2_free_model(model);
    return 0;
}
