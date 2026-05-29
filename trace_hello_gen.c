// Check hidden state after each generation step with "hello"
#include <stdio.h>
#include "smollm2.h"

int main() {
    sm2_model* model;
    if (sm2_load_model("./smollm2-135m.sm2", &model) != 0) return 1;

    sm2_context* ctx;
    if (sm2_create_context(model, &ctx) != 0) return 1;

    ctx->params.temperature = 0.0f;  // Greedy

    // Encode "hello"
    char* prompt = "hello";
    int tokens[256];
    int n_tokens = sm2_tokenizer_encode(model->tokenizer, prompt, tokens, 256);

    printf("Prompt '%s' -> %d tokens: ", prompt, n_tokens);
    for (int i = 0; i < n_tokens; i++) printf("%d ", tokens[i]);
    printf("\n");

    // Prefill
    sm2_prefill(ctx, tokens, n_tokens);

    printf("\n=== Generation trace ===\n");

    // Generate 15 tokens
    for (int gen = 0; gen < 15; gen++) {
        int token;
        sm2_decode_next(ctx, &token);

        char* decoded = sm2_tokenizer_decode(model->tokenizer, &token, 1);
        printf("Gen %d: token=%d, '%s', hidden[0]=%.4f\n",
               gen, token,
               decoded ? decoded : "(null)",
               ctx->scratch.x[0]);
        if (decoded) free(decoded);

        // Print top 3 tokens this step
        float* logits = ctx->scratch.logits;
        float logit_copy[49152];
        for (int i = 0; i < model->vocab_size; i++) logit_copy[i] = logits[i];

        printf("  Top 3: ");
        for (int t = 0; t < 3; t++) {
            int max_idx = 0;
            float max_val = logit_copy[0];
            for (int i = 1; i < model->vocab_size; i++) {
                if (logit_copy[i] > max_val) {
                    max_val = logit_copy[i];
                    max_idx = i;
                }
            }
            char* dec = sm2_tokenizer_decode(model->tokenizer, &max_idx, 1);
            printf("%d('%.10s',%.2f) ", max_idx, dec ? dec : "?", max_val);
            if (dec) free(dec);
            logit_copy[max_idx] = -1e9f;
        }
        printf("\n");
    }

    sm2_free_context(ctx);
    sm2_free_model(model);
    return 0;
}