#include <stdio.h>
#include <stdlib.h>
#include "smollm2.h"

void print_token(const char* decoded) {
    if (!decoded) return;
    for (const char* p = decoded; *p; p++) {
        unsigned char c = (unsigned char)*p;
        if (c == 0xC4 && (unsigned char)p[1] == 0xA0) { putchar(' '); p++; }
        else if (c == 0xC4 && (unsigned char)p[1] == 0x8A) { putchar('\n'); p++; }
        else putchar(*p);
    }
}

int main(int argc, char** argv) {
    sm2_model* model;
    sm2_load_model(argv[1], &model);
    
    sm2_context* ctx;
    sm2_create_context(model, &ctx);
    ctx->params.temperature = 0.0f; // greedy
    ctx->params.top_p = 100;
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
    
    // Top logits BEFORE sampling
    float* logits = ctx->scratch.logits;
    printf("Top 5 logits after prefill:\n");
    for (int t = 0; t < 5; t++) {
        int max_idx = 0;
        float max_val = logits[0];
        for (int i = 1; i < 49152; i++) {
            if (logits[i] > max_val) { max_val = logits[i]; max_idx = i; }
        }
        char* dec = sm2_tokenizer_decode(model->tokenizer, &max_idx, 1);
        printf("  %d: %.2f '%s'\n", max_idx, max_val, dec ? dec : "?");
        if (dec) free(dec);
        logits[max_idx] = -1e9f;
    }
    
    printf("\nGreedy output:\n");
    int token;
    for (int i = 0; i < 20; i++) {
        sm2_decode_next(ctx, &token);
        if (token < 3) { printf("[EOS]\n"); break; }
        char* dec = sm2_tokenizer_decode(model->tokenizer, &token, 1);
        if (dec) { print_token(dec); free(dec); }
    }
    printf("\n");
    
    sm2_free_context(ctx);
    sm2_free_model(model);
}
