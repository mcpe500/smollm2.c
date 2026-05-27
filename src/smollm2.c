// smollm2.c - Decode-first SmolLM2 inference engine
// Main entry point and CLI

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "smollm2.h"

// ============================================================================
// PRINT TOKEN: Converts tokenizer output to terminal-safe ASCII
// Replaces UTF-8 BPE tokens with ASCII equivalents for display
// Ġ (U+0120 = 0xC4 0xA0) -> space (word start indicator)
// Ċ (U+010A = 0xC4 0x8A) -> newline
// ============================================================================

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

// ============================================================================
// CLI Argument Parsing
// ============================================================================

typedef struct {
    const char* model_path;
    const char* prompt;
    int ctx_size;
    int max_output;
    int n_threads;
    float temperature;
    int top_p;
    int top_k;
    float repetition_penalty;
    int interactive;
} cli_args;

static void print_help(const char* prog) {
    printf("smollm2.c - SmolLM2 inference engine\n\n");
    printf("Usage: %s -m <model.sm2> -p <prompt>\n", prog);
    printf("  -m, --model <path>    Model file (.sm2)\n");
    printf("  -p, --prompt <text>   Input prompt\n");
    printf("  -n, --max-output <n> Max tokens (default: 50)\n");
    printf("  -t, --temp <x>       Temperature (0.0=greedy, 0.7=recommended, 1.0=creative)\n");
    printf("  -q, --top-p <n>      Top-p nucleus sampling (0-100, default: 90)\n");
    printf("  -k, --top-k <n>      Top-k sampling (0=disabled)\n");
    printf("  -r, --rep-penalty <x> Repetition penalty (1.0=off, 1.3=recommended)\n");
    printf("  -h, --help           Show help\n");
}

static cli_args parse_args(int argc, char** argv) {
    cli_args args = {
        .model_path = NULL,
        .prompt = "Hello",
        .ctx_size = 2048,
        .max_output = 50,
        .n_threads = 4,
        .temperature = 0.7f,    // Default to 0.7 for diverse output (like Ollama)
        .top_p = 90,            // Nucleus sampling (0 = disabled)
        .top_k = 0,             // No top-k by default
        .repetition_penalty = 1.0f,
        .interactive = 0,
    };

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            args.model_path = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) {
            args.prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--max-output") == 0) {
            args.max_output = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--temp") == 0) {
            args.temperature = atof(argv[++i]);
        } else if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--top-p") == 0) {
            args.top_p = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-k") == 0 || strcmp(argv[i], "--top-k") == 0) {
            args.top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--rep-penalty") == 0) {
			args.repetition_penalty = atof(argv[++i]);
		} else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_help(argv[0]);
            exit(0);
        }
    }

    return args;
}

// ============================================================================
// TIME MEASUREMENT
// ============================================================================

static double time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// ============================================================================
// MAIN INFERENCE LOOP
// ============================================================================

static int run_inference(sm2_model* model, const cli_args* args) {
    double start_time = time_ms();

    // Create context
    sm2_context* ctx;
    if (sm2_create_context(model, &ctx) != 0) {
        fprintf(stderr, "Failed to create context\n");
        return -1;
    }

    ctx->params.temperature = args->temperature;
    ctx->params.top_p = args->top_p;
    ctx->params.top_k = args->top_k;
    ctx->params.max_context = args->ctx_size;
    ctx->params.max_output = args->max_output;
    ctx->params.repetition_penalty = args->repetition_penalty;

    // BPE tokenization for input using the loaded tokenizer
    int tokens[4096];
    int n_tokens = 0;

    if (model->tokenizer) {
        // Use proper BPE tokenizer (like HF does)
        n_tokens = sm2_tokenizer_encode(model->tokenizer, args->prompt, tokens, 4096);
    }

    // Fallback to byte tokenization if tokenizer not available
    if (n_tokens == 0) {
        for (int i = 0; args->prompt[i] && n_tokens < 4096; i++) {
            unsigned char byte_val = (unsigned char)args->prompt[i];
            if (model->tokenizer && model->tokenizer->byte_to_token) {
                tokens[n_tokens++] = model->tokenizer->byte_to_token[byte_val];
            } else {
                tokens[n_tokens++] = byte_val;
            }
        }
    }

    printf("Input: \"%s\" => %d tokens\n", args->prompt, n_tokens);

    // Prefill
    double t0 = time_ms();
    if (sm2_prefill(ctx, tokens, n_tokens) != 0) {
        fprintf(stderr, "Prefill failed\n");
        sm2_free_context(ctx);
        return -1;
    }
    printf("Prefill: %.1f ms\n", time_ms() - t0);

    // Generate tokens
    printf("Output: ");
    fflush(stdout);

    int gen_tokens = 0;
    int token;

    while (gen_tokens < args->max_output) {
        sm2_decode_next(ctx, &token);

        if (token < 3) break; // EOS

        if (model->tokenizer && model->tokenizer->tokens) {
            // Use proper BPE decode
            char* decoded = sm2_tokenizer_decode(model->tokenizer, &token, 1);
            if (decoded) {
                print_token(decoded);
                free(decoded);
            } else if (token >= 32 && token < 127) {
                putchar(token);
            }
        } else if (token >= 32 && token < 127) {
            putchar(token);
        } else {
            printf("[%d]", token);
        }
        fflush(stdout);
        gen_tokens++;
    }

    double dt = time_ms() - start_time;
    printf("\n\nGenerated %d tokens in %.1f ms (%.1f ms/token)\n",
           gen_tokens, dt, gen_tokens > 0 ? dt / gen_tokens : 0);

    sm2_free_context(ctx);
    return 0;
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    cli_args args = parse_args(argc, argv);

    if (!args.model_path) {
        fprintf(stderr, "Error: --model required\n");
        print_help(argv[0]);
        return 1;
    }

    sm2_model* model;
    if (sm2_load_model(args.model_path, &model) != 0) {
        fprintf(stderr, "Failed to load model: %s\n", args.model_path);
        return 1;
    }

    int ok = run_inference(model, &args);
    sm2_free_model(model);
    return ok;
}
