// Trace each generation step with sampling parameters
#include <stdio.h>
#include "smollm2.h"

int main() {
    sm2_model* model;
    if (sm2_load_model("./smollm2-135m.sm2", &model) != 0) return 1;

    sm2_context* ctx;
    if (sm2_create_context(model, &ctx) != 0) return 1;

    // Test with temp=0 (greedy)
    ctx->params.temperature = 0.0f;
    ctx->params.top_p = 100;
    ctx->params.top_k = 0;
    ctx->params.repetition_penalty = 1.0f;

    // Encode "hello"
    int tokens[] = {88, 85, 92, 92, 95};
    sm2_prefill(ctx, tokens, 5);

    printf("=== Greedy generation (temp=0) ===\n");
    for (int gen = 0; gen < 10; gen++) {
        int token;
        sm2_decode_next(ctx, &token);

        char* decoded = sm2_tokenizer_decode(model->tokenizer, &token, 1);
        printf("Gen %d: token=%d '%s'\n", gen, token, decoded ? decoded : "?");
        if (decoded) free(decoded);
    }

    // Reset and test with temp=0.7
    sm2_free_context(ctx);
    sm2_free_model(model);

    if (sm2_load_model("./smollm2-135m.sm2", &model) != 0) return 1;
    if (sm2_create_context(model, &ctx) != 0) return 1;

    ctx->params.temperature = 0.7f;
    ctx->params.top_p = 100;
    ctx->params.top_k = 0;
    ctx->params.repetition_penalty = 1.0f;

    sm2_prefill(ctx, tokens, 5);

    printf("\n=== Temperature 0.7 generation ===\n");
    for (int gen = 0; gen < 10; gen++) {
        int token;
        sm2_decode_next(ctx, &token);

        char* decoded = sm2_tokenizer_decode(model->tokenizer, &token, 1);
        printf("Gen %d: token=%d '%s'\n", gen, token, decoded ? decoded : "?");
        if (decoded) free(decoded);
    }

    sm2_free_context(ctx);
    sm2_free_model(model);
    return 0;
}