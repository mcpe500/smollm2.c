// Detailed trace of generation with tokenizer output
#include <stdio.h>
#include "smollm2.h"

int main() {
    sm2_model* model;
    if (sm2_load_model("./smollm2-135m.sm2", &model) != 0) return 1;

    sm2_context* ctx;
    if (sm2_create_context(model, &ctx) != 0) return 1;

    ctx->params.temperature = 0.0f;

    // Encode prompt
    char* prompt = "hello";
    int tokens[256];
    int n_tokens = sm2_tokenizer_encode(model->tokenizer, prompt, tokens, 256);
    printf("Prompt '%s' -> %d tokens: ", prompt, n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        char* dec = sm2_tokenizer_decode(model->tokenizer, &tokens[i], 1);
        printf("%d('%s') ", tokens[i], dec ? dec : "?");
        if (dec) free(dec);
    }
    printf("\n\n");

    // Prefill
    sm2_prefill(ctx, tokens, n_tokens);

    // Generate
    printf("=== Generation ===\n");
    char buffer[4096] = {0};
    int pos = 0;

    for (int gen = 0; gen < 10; gen++) {
        int token;
        sm2_decode_next(ctx, &token);

        char* decoded = sm2_tokenizer_decode(model->tokenizer, &token, 1);
        printf("Gen %d: token=%d, raw='%s', len=%zu\n",
               gen, token,
               decoded ? decoded : "(null)",
               decoded ? strlen(decoded) : 0);

        if (decoded) {
            // Manually apply print_token logic
            const char* p = decoded;
            while (*p) {
                unsigned char c = (unsigned char)*p;
                if (c == 0xC4 && (unsigned char)p[1] == 0xA0) {
                    buffer[pos++] = ' ';
                    p += 2;
                } else if (c == 0xC4 && (unsigned char)p[1] == 0x8A) {
                    buffer[pos++] = '\n';
                    p += 2;
                } else {
                    buffer[pos++] = *p;
                    p++;
                }
            }
            free(decoded);
        }

        printf("  Buffer now: '%s'\n", buffer);
    }

    printf("\nFinal output: '%s'\n", buffer);

    sm2_free_context(ctx);
    sm2_free_model(model);
    return 0;
}