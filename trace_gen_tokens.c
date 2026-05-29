// Trace all generated tokens during generation
#include <stdio.h>
#include "smollm2.h"

int main() {
    sm2_model* model;
    if (sm2_load_model("./smollm2-135m.sm2", &model) != 0) return 1;

    sm2_context* ctx;
    if (sm2_create_context(model, &ctx) != 0) return 1;

    ctx->params.temperature = 0.0f;

    // Encode prompt
    char* prompt = "Once upon a time";
    int tokens[256];
    int n_tokens = sm2_tokenizer_encode(model->tokenizer, prompt, tokens, 256);
    printf("Prompt '%s' -> %d tokens: ", prompt, n_tokens);
    for (int i = 0; i < n_tokens; i++) printf("%d ", tokens[i]);
    printf("\n\n");

    // Prefill
    sm2_prefill(ctx, tokens, n_tokens);

    // Generate 20 tokens and trace
    printf("=== Generation trace ===\n");
    char output[1024] = {0};
    int out_pos = 0;

    for (int gen = 0; gen < 20; gen++) {
        int token;
        sm2_decode_next(ctx, &token);

        char* decoded = sm2_tokenizer_decode(model->tokenizer, &token, 1);
        printf("Gen %d: token=%d, decoded='%s'\n", gen, token, decoded ? decoded : "(null)");

        if (decoded) {
            int len = strlen(decoded);
            if (out_pos + len < 1023) {
                strcpy(output + out_pos, decoded);
                out_pos += len;
            }
            free(decoded);
        }

        if (token == 2) break;  // EOS
    }

    printf("\nFull output: '%s'\n", output);

    sm2_free_context(ctx);
    sm2_free_model(model);
    return 0;
}