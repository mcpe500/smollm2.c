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
    
    printf("Generated: ");
    int token;
    for (int i = 0; i < 30; i++) {
        sm2_decode_next(ctx, &token);
        if (token < 3) { printf("[EOS=%d]", token); break; }
        
        char* dec = sm2_tokenizer_decode(model->tokenizer, &token, 1);
        if (dec) {
            printf("%s", dec);
            free(dec);
        }
    }
    printf("\n");
    
    sm2_free_context(ctx);
    sm2_free_model(model);
}
