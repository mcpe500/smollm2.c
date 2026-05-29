// smollm2.c - Decode-first SmolLM2 inference engine
// Main entry point and CLI

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "smollm2.h"
#include "chat_history.h"

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

static void print_help(const char* prog) {
    printf("smollm2.c - SmolLM2 inference engine\n\n");
    printf("Usage: %s -m <model.sm2> [options]\n\n", prog);
    printf("Mode options:\n");
    printf("  --mode cli         Interactive CLI chat (default)\n");
    printf("  --mode tui         Text-based UI with chat history\n");
    printf("  --mode web         HTTP server with web UI\n\n");
    printf("Model options:\n");
    printf("  -m, --model <path>    Model file (.sm2) [required]\n\n");
    printf("Chat options:\n");
    printf("  -p, --prompt <text>   Initial prompt (default: \"Hello\")\n");
    printf("  --system-prompt <text> Custom system prompt\n\n");
    printf("Generation options:\n");
    printf("  -n, --max-output <n> Max tokens (default: 50)\n");
    printf("  -t, --temp <x>       Temperature (0.0=greedy, 0.7=recommended, 1.0=creative)\n");
    printf("  -q, --top-p <n>      Top-p nucleus sampling (0-100, default: 90)\n");
    printf("  -k, --top-k <n>      Top-k sampling (0=disabled)\n");
    printf("  -r, --rep-penalty <x> Repetition penalty (1.0=off, 1.2=default, 1.5=strong)\n\n");
    printf("Web mode options:\n");
    printf("  -P, --port <n>      HTTP server port (default: 7331)\n");
    printf("  --host <addr>       Bind address (default: 127.0.0.1)\n\n");
    printf("Examples:\n");
    printf("  %s -m model.sm2 --mode cli\n", prog);
    printf("  %s -m model.sm2 --mode tui\n", prog);
    printf("  %s -m model.sm2 --mode web --port 8080\n", prog);
}

static cli_args parse_args(int argc, char** argv) {
    cli_args args = {
        .model_path = NULL,
        .prompt = "Hello",
        .ctx_size = 2048,
        .max_output = 50,
        .n_threads = 4,
        .temperature = 0.7f,
        .top_p = 90,
        .top_k = 0,
        .repetition_penalty = 1.3f,
        .mode = MODE_CLI,
        .web_port = 7331,
        .web_host = "127.0.0.1",
        .system_prompt = NULL,
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
		} else if (strcmp(argv[i], "--mode") == 0) {
            i++;
            if (strcmp(argv[i], "cli") == 0) args.mode = MODE_CLI;
            else if (strcmp(argv[i], "tui") == 0) args.mode = MODE_TUI;
            else if (strcmp(argv[i], "web") == 0) args.mode = MODE_WEB;
            else fprintf(stderr, "Unknown mode: %s\n", argv[i]);
        } else if (strcmp(argv[i], "-P") == 0 || strcmp(argv[i], "--port") == 0) {
            args.web_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--host") == 0) {
            args.web_host = argv[++i];
        } else if (strcmp(argv[i], "--system-prompt") == 0) {
            args.system_prompt = argv[++i];
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
// CHAT MODE HANDLERS
// ============================================================================

int run_chat_cli(sm2_model* model, const cli_args* args, const sm2_generate_params* gen_params);
int run_chat_tui(sm2_model* model, const cli_args* args);
int run_chat_web(sm2_model* model, const cli_args* args);

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
    ctx->params.max_context = 8192;  // Use model max_seq, not CLI override
    ctx->params.max_output = args->max_output;
    ctx->params.repetition_penalty = args->repetition_penalty;
    ctx->params.penalty_window = 32;  // Track last 32 tokens for repetition

    // BPE tokenization for input using the loaded tokenizer
    int tokens[4096];
    int n_tokens = 0;

    // Add full chat template to match HF/Ollama behavior:
    // <|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
    // <|im_start|>user\n{prompt}<|im_end|>
    // <|im_start|>assistant\n
    tokens[n_tokens++] = 1;  // <|im_start|>
    if (model->tokenizer) {
        // System prompt
        n_tokens += sm2_tokenizer_encode(model->tokenizer, "system", tokens + n_tokens, 4096 - n_tokens);
        tokens[n_tokens++] = 198;  // Ċ (newline)
        n_tokens += sm2_tokenizer_encode(model->tokenizer, "You are a helpful AI assistant named SmolLM, trained by Hugging Face", tokens + n_tokens, 4096 - n_tokens);
    }
    tokens[n_tokens++] = 2;  // <|im_end|>
    tokens[n_tokens++] = 198;  // Ċ (newline)
    tokens[n_tokens++] = 1;  // <|im_start|> for user

    if (model->tokenizer) {
        // User message: "user" + Ċ + "{prompt}"
        n_tokens += sm2_tokenizer_encode(model->tokenizer, "user", tokens + n_tokens, 4096 - n_tokens);
        tokens[n_tokens++] = 198;  // Ċ (newline)
        n_tokens += sm2_tokenizer_encode(model->tokenizer, args->prompt, tokens + n_tokens, 4096 - n_tokens);
    }
    tokens[n_tokens++] = 2;  // <|im_end|>
    tokens[n_tokens++] = 198;  // Ċ (newline)
    tokens[n_tokens++] = 1;  // <|im_start|> for assistant
    if (model->tokenizer) {
        n_tokens += sm2_tokenizer_encode(model->tokenizer, "assistant", tokens + n_tokens, 4096 - n_tokens);
        tokens[n_tokens++] = 198;  // Ċ (newline)
    }

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
        int ok = sm2_decode_next(ctx, &token);
        if (ok != 0) {
            fprintf(stderr, "ERROR: decode_next returned %d\n", ok);
            break;
        }

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

    int ok;
    sm2_generate_params gen_params = {
        .temperature = args.temperature,
        .top_p = args.top_p,
        .top_k = args.top_k,
        .max_output = args.max_output,
        .repetition_penalty = args.repetition_penalty,
    };

    if (args.mode == MODE_TUI) {
        ok = run_chat_tui(model, &args);
    } else if (args.mode == MODE_WEB) {
        ok = run_chat_web(model, &args);
    } else {
        // MODE_CLI or default: interactive CLI chat mode
        ok = run_chat_cli(model, &args, &gen_params);
    }

    sm2_free_model(model);
    return ok;
}
