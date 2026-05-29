// chat_cli.c - Interactive CLI chat mode for SmolLM2
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "smollm2.h"
#include "chat_history.h"

static double time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void print_token(const char* decoded) {
    if (!decoded) return;
    for (const char* p = decoded; *p; p++) {
        unsigned char c = (unsigned char)*p;
        if (c == 0xC4 && (unsigned char)p[1] == 0xA0) {
            putchar(' ');
            p++;
        } else if (c == 0xC4 && (unsigned char)p[1] == 0x8A) {
            putchar('\n');
            p++;
        } else {
            putchar(*p);
        }
    }
}

int run_chat_cli(sm2_model* model, const cli_args* args, const sm2_generate_params* gen_params) {
    (void)args;  // args not used, sampling params come from gen_params
    sm2_tokenizer* tok = model->tokenizer;
    chat_history hist;
    chat_history_init(&hist);

    // Set default system prompt
    const char* system_prompt = "You are a helpful AI assistant named SmolLM, trained by Hugging Face.";
    chat_history_set_system(&hist, system_prompt);

    printf("================================\n");
    printf("  SmolLM2 Chat (CLI Mode)\n");
    printf("  Type 'quit' or 'exit' to stop\n");
    printf("  Type 'clear' to clear history\n");
    printf("================================\n\n");

    char input[1024];
    char response[8192];

    while (1) {
        printf("\n");
        printf("\033[1;36mYou:\033[0m ");
        fflush(stdout);

        if (!fgets(input, sizeof(input), stdin)) {
            printf("\nGoodbye!\n");
            break;
        }

        // Remove trailing newline
        input[strcspn(input, "\n")] = '\0';

        // Handle commands
        if (strcmp(input, "quit") == 0 || strcmp(input, "exit") == 0) {
            printf("\nGoodbye!\n");
            break;
        }

        if (strcmp(input, "clear") == 0) {
            chat_history_clear(&hist);
            printf("History cleared.\n");
            continue;
        }

        if (strlen(input) == 0) continue;

        // Add user message to history
        chat_history_add_user(&hist, input);

        // Generate response
        printf("\n");
        printf("\033[1;32mSmolLM:\033[0m ");
        fflush(stdout);

        double t0 = time_ms();
        int last_update = 0;
        int gen_tokens = 0;

        // Generate response using args for sampling
        int resp_len = 0;
        char response[8192] = {0};
        {
            char* prompt = chat_history_build_prompt(&hist, NULL);
            if (prompt) {
                int tokens[4096];
                int n_tokens = sm2_tokenizer_encode(tok, prompt, tokens, 4096);
                free(prompt);

                if (n_tokens > 0) {
                    sm2_context* ctx;
                    if (sm2_create_context(model, &ctx) == 0) {
                        ctx->params.temperature = gen_params->temperature;
                        ctx->params.top_p = gen_params->top_p;
                        ctx->params.top_k = gen_params->top_k;
                        ctx->params.max_context = 8192;
                        ctx->params.max_output = gen_params->max_output;
                        ctx->params.repetition_penalty = gen_params->repetition_penalty;
                        ctx->params.penalty_window = 32;

                        if (sm2_prefill(ctx, tokens, n_tokens) == 0) {
                            // Show loading indicator
                            printf("\033[90m[Generating...]\033[0m");
                            fflush(stdout);

                            while (resp_len < (int)sizeof(response) - 1) {
                                int token;
                                if (sm2_decode_next(ctx, &token) != 0) break;
                                if (token < 3) break;

                                char* decoded = sm2_tokenizer_decode(tok, &token, 1);
                                if (decoded) {
                                    print_token(decoded);
                                    for (const char* p = decoded; *p && resp_len < (int)sizeof(response) - 1; p++) {
                                        unsigned char c = (unsigned char)*p;
                                        if (c == 0xC4 && (unsigned char)p[1] == 0xA0) {
                                            response[resp_len++] = ' ';
                                            p++;
                                        } else if (c == 0xC4 && (unsigned char)p[1] == 0x8A) {
                                            response[resp_len++] = '\n';
                                            p++;
                                        } else {
                                            response[resp_len++] = *p;
                                        }
                                    }
                                    free(decoded);
                                }
                                gen_tokens++;

                                // Update progress every 300ms
                                double dt_gen = (time_ms() - t0) / 1000.0;
                                int now = (int)(time_ms() / 300);
                                if (now != last_update && gen_tokens > 0) {
                                    printf("\r\033[1;32mSmolLM:\033[0m %s\033[90m[%.1f tok/s]\033[0m",
                                           response + (resp_len > 50 ? resp_len - 50 : 0),
                                           gen_tokens / dt_gen);
                                    if (resp_len > 50) printf("...");
                                    fflush(stdout);
                                    last_update = now;
                                }
                            }
                        }
                        sm2_free_context(ctx);
                    }
                }
            }
        }

        double dt = time_ms() - t0;
        response[resp_len >= 0 ? resp_len : 0] = '\0';

        if (resp_len > 0) {
            printf("\n\033[90m(%.1f ms, %d chars, %.1f tok/s)\033[0m", dt, resp_len, gen_tokens / (dt / 1000.0));
            chat_history_add_assistant(&hist, response);
        } else {
            printf("\n\033[91mGeneration failed\033[0m");
        }

        // Trim history if too long
        chat_history_trim(&hist, 1800);
    }

    return 0;
}
