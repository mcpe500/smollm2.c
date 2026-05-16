// smollm2.c - Decode-first SmolLM2 inference engine
// Main entry point and CLI

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "smollm2.h"

// ============================================================================
// CLI Argument Parsing
// ============================================================================

typedef struct {
    const char* model_path;
    const char* prompt;
    const char* chat_prompt;  // JSON chat format
    int ctx_size;
    int max_output;
    int n_threads;
    float temperature;
    int top_p;
    int top_k;
    int low_memory;
    int interactive;
} cli_args;

static void print_help(const char* prog) {
    printf("smollm2.c - Decode-first SmolLM2 inference engine\n\n");
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  -m, --model <path>      Model file (.sm2)\n");
    printf("  -p, --prompt <text>     Input prompt\n");
    printf("  -c, --ctx <n>          Context size (default: 2048)\n");
    printf("  -n, --max-output <n>   Max output tokens (default: 256)\n");
    printf("  -t, --threads <n>      Threads (default: 4)\n");
    printf("  -T, --temp <f>         Temperature (default: 0.8)\n");
    printf("  -top-p <n>             Top-p sampling (default: 0.9)\n");
    printf("  -top-k <n>             Top-k sampling (default: 40)\n");
    printf("  -i, --interactive      Interactive mode\n");
    printf("  --low-mem              Low memory mode (512MB VPS)\n");
    printf("  -h, --help             Show this help\n");
    printf("\n");
    printf("Example:\n");
    printf("  %s --model smollm2-135m-q4.sm2 --prompt 'Hello world'\n", prog);
}

static cli_args parse_args(int argc, char** argv) {
    cli_args args = {
        .model_path = NULL,
        .prompt = "Hello",
        .ctx_size = 2048,
        .max_output = 256,
        .n_threads = 4,
        .temperature = 0.8f,
        .top_p = 90,
        .top_k = 40,
        .low_memory = 0,
        .interactive = 0,
    };
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            args.model_path = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) {
            args.prompt = argv[++i];
        } else if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--ctx") == 0) {
            args.ctx_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--max-output") == 0) {
            args.max_output = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0) {
            args.n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-T") == 0 || strcmp(argv[i], "--temp") == 0) {
            args.temperature = atof(argv[++i]);
        } else if (strcmp(argv[i], "-top-p") == 0) {
            args.top_p = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-top-k") == 0) {
            args.top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--interactive") == 0) {
            args.interactive = 1;
        } else if (strcmp(argv[i], "--low-mem") == 0) {
            args.low_memory = 1;
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
    
    // Set generation params
    ctx->params.temperature = args->temperature;
    ctx->params.top_p = args->top_p;
    ctx->params.top_k = args->top_k;
    ctx->params.max_context = args->ctx_size;
    ctx->params.max_output = args->max_output;
    
    // Tokenize prompt
    int tokens[4096];
    int n_tokens = 0;
    
    // For now, use simple tokenization (bytes)
    // TODO: integrate proper tokenizer
    const char* prompt = args->prompt;
    for (int i = 0; prompt[i] && i < 4096; i++) {
        tokens[i] = (unsigned char)prompt[i];
        n_tokens++;
    }
    
    printf("Processing %d tokens...\n", n_tokens);
    
    // Prefill
    double t0 = time_ms();
    if (sm2_prefill(ctx, tokens, n_tokens) != 0) {
        fprintf(stderr, "Prefill failed\n");
        sm2_free_context(ctx);
        return -1;
    }
    printf("Prefill done in %.1f ms\n", time_ms() - t0);
    
    // Generate tokens
    printf("\nOutput: ");
    fflush(stdout);
    
    int gen_tokens = 0;
    int token;
    double total_decode_time = 0;
    
    while (gen_tokens < args->max_output) {
        double t1 = time_ms();
        int ok = sm2_decode_next(ctx, &token);
        double dt = time_ms() - t1;
        total_decode_time += dt;
        
        if (ok != 0 || token == 2) { // EOS
            break;
        }
        
        // Print token (as byte for now)
        putchar(token >= 32 && token < 127 ? token : '?');
        fflush(stdout);
        gen_tokens++;
    }
    
    double total_time = time_ms() - start_time;
    printf("\n\nGenerated %d tokens in %.1f ms (%.1f ms/token)\n", 
           gen_tokens, total_decode_time, 
           gen_tokens > 0 ? total_decode_time / gen_tokens : 0);
    printf("Total time (incl. prefill): %.1f ms\n", total_time);
    
    sm2_free_context(ctx);
    return 0;
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    cli_args args = parse_args(argc, argv);
    
    if (args.model_path == NULL) {
        fprintf(stderr, "Error: --model required\n");
        print_help(argv[0]);
        return 1;
    }
    
    printf("smollm2.c - SmolLM2 inference engine\n");
    printf("Model: %s\n", args.model_path);
    printf("Prompt: %s\n", args.prompt);
    printf("Context: %d, Max output: %d, Threads: %d\n",
           args.ctx_size, args.max_output, args.n_threads);
    
    // Load model
    double t0 = time_ms();
    sm2_model* model;
    int ok = sm2_load_model(args.model_path, &model);
    if (ok != 0) {
        fprintf(stderr, "Failed to load model: %s\n", args.model_path);
        return 1;
    }
    printf("Model loaded in %.1f ms\n", time_ms() - t0);
    
    // Run inference
    ok = run_inference(model, &args);
    
    sm2_free_model(model);
    
    return ok;
}