#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "smollm2.h"

int main() {
    sm2_model* model;
    if (sm2_load_model("smollm2-135m.sm2", &model) != 0) {
        fprintf(stderr, "Failed to load\n");
        return 1;
    }
    
    printf("tok_embeddings first 10 values:\n");
    if (model->tok_embeddings) {
        uint16_t* data = model->tok_embeddings->data;
        for (int i = 0; i < 10; i++) {
            uint16_t v = data[i];
            uint16_t sign = (v >> 15) & 1;
            uint16_t exp = (v >> 10) & 31;
            uint16_t mant = v & 0x3FF;
            printf("  [%d] uint16=0x%04x sign=%d exp=%2d mant=%3d\n", i, v, sign, exp, mant);
        }
    }
    
    sm2_free_model(model);
    return 0;
}
