#include <stdio.h>
#include <stdlib.h>
#include "smollm2.h"

int main(int argc, char** argv) {
    sm2_model* model;
    sm2_load_model(argv[1], &model);
    
    sm2_context* ctx;
    sm2_create_context(model, &ctx);
    ctx->params.temperature = 0.7f;
    ctx->params.top_p = 90;
    ctx->params.max_context = 8192;
    ctx->params.repetition_penalty = 1.0f;
    
    int tokens[4096];
    int n = 0;
    tokens[n++] = 1; // im_start
    n += sm2_tokenizer_encode(model->tokenizer, "hello", tokens + n, 4096 - n);
    tokens[n++] = 2; // im_end
    tokens[n++] = 1; // im_start for assistant
    n += sm2_tokenizer_encode(model->tokenizer, "assistant\n", tokens + n, 4096 - n);
    
    sm2_prefill(ctx, tokens, n);
    
    // Print first 10 tokens individually
    printf("First 10 tokens:\n");
    for (int i = 0; i < 10; i++) {
        int token;
        sm2_decode_next(ctx, &token);
        char* dec = sm2_tokenizer_decode(model->tokenizer, &token, 1);
        
        // Print hex of decoded
        printf("  %d: ", token);
        if (dec) {
            for (char* p = dec; *p; p++) {
                printf("%02X ", (unsigned char)*p);
            }
            printf("'%s'", dec);
            free(dec);
        }
        printf("\n");
        if (token < 3) break;
    }
    
    sm2_free_context(ctx);
    sm2_free_model(model);
}
