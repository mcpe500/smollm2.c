// benchmark.c - SmolLM2 benchmark tool
// Measures tokens/second and generates performance report

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "smollm2.h"

static double time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

typedef struct {
    const char* model_path;
    int n_warmup;
    int n_tokens;
    float temperature;
} bench_args;

static bench_args parse_args(int argc, char** argv) {
    bench_args args = {
        .model_path = NULL,
        .n_warmup = 5,
        .n_tokens = 50,
        .temperature = 0.7f,
    };

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0) args.model_path = argv[++i];
        else if (strcmp(argv[i], "-n") == 0) args.n_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "-w") == 0) args.n_warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0) args.temperature = atof(argv[++i]);
    }
    return args;
}

static double run_generation(sm2_model* model, int n_tokens, float temp) {
    int tokens[64];
    int n = sm2_tokenizer_encode(model->tokenizer, "A", tokens, 64);
    if (n <= 0) return -1;

    sm2_context* ctx;
    if (sm2_create_context(model, &ctx) != 0) return -1;

    ctx->params.temperature = temp;
    ctx->params.top_p = 90;
    ctx->params.max_output = n_tokens;
    ctx->params.repetition_penalty = 1.0f;

    if (sm2_prefill(ctx, tokens, n) != 0) {
        sm2_free_context(ctx);
        return -1;
    }

    double t0 = time_ms();
    int gen = 0;

    while (gen < n_tokens) {
        int tok;
        if (sm2_decode_next(ctx, &tok) != 0) break;
        if (tok < 3) break;
        gen++;
    }

    double dt = time_ms() - t0;
    sm2_free_context(ctx);

    return dt;
}

int main(int argc, char** argv) {
    bench_args args = parse_args(argc, argv);

    if (!args.model_path) {
        fprintf(stderr, "Usage: %s -m <model.sm2> [-n tokens] [-w warmup] [-t temp]\n", argv[0]);
        return 1;
    }

    printf("============================================\n");
    printf("  SmolLM2 Benchmark\n");
    printf("============================================\n\n");

    // Load model
    printf("Loading model: %s\n", args.model_path);
    double load_start = time_ms();

    sm2_model* model;
    if (sm2_load_model(args.model_path, &model) != 0) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    printf("Model loaded in %.1f ms\n\n", time_ms() - load_start);

    // Warmup
    printf("Warming up (%d tokens)...\n", args.n_warmup);
    for (int i = 0; i < args.n_warmup; i++) {
        run_generation(model, 10, args.temperature);
    }
    printf("Warmup complete\n\n");

    // Benchmark runs
    printf("Running benchmark (%d tokens per run)...\n", args.n_tokens);
    printf("--------------------------------------------\n");

    double total_dt = 0;
    int runs = 0;

    for (int run = 0; run < 3; run++) {
        double dt = run_generation(model, args.n_tokens, args.temperature);
        if (dt > 0) {
            double tok_s = args.n_tokens * 1000.0 / dt;
            printf("Run %d: %d tokens in %.1f ms (%.1f tok/s)\n",
                   run + 1, args.n_tokens, dt, tok_s);
            total_dt += dt;
            runs++;
        }
    }

    printf("--------------------------------------------\n");

    if (runs > 0) {
        double avg_tok_s = args.n_tokens * 1000.0 * runs / total_dt;
        printf("\nAverage: %.1f tok/s\n", avg_tok_s);
    }

    sm2_free_model(model);
    printf("\nBenchmark complete!\n");

    return 0;
}