// bench.c - Simple benchmark (standalone)
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

int main(int argc, char** argv) {
    const char* model_path = "smollm2-135m.sm2";
    int n_tokens = 30;
    int n_warmup = 2;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0) model_path = argv[++i];
        else if (strcmp(argv[i], "-n") == 0) n_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "-w") == 0) n_warmup = atoi(argv[++i]);
    }

    fprintf(stderr, "Loading model: %s\n", model_path);
    sm2_model* model;
    if (sm2_load_model(model_path, &model) != 0) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // Warmup
    fprintf(stderr, "Warming up...\n");
    for (int i = 0; i < n_warmup; i++) {
        sm2_context* ctx;
        if (sm2_create_context(model, &ctx) != 0) continue;
        int tokens[4] = {1, 1234, 5678, 2};
        if (sm2_prefill(ctx, tokens, 4) == 0) {
            for (int j = 0; j < 5; j++) {
                int tok;
                if (sm2_decode_next(ctx, &tok) != 0) break;
                if (tok < 3) break;
            }
        }
        sm2_free_context(ctx);
    }

    // Benchmark
    fprintf(stderr, "Benchmarking %d tokens...\n", n_tokens);
    double total_ms = 0;
    int total_gen = 0;
    int runs = 3;

    for (int r = 0; r < runs; r++) {
        sm2_context* ctx;
        if (sm2_create_context(model, &ctx) != 0) continue;
        int tokens[4] = {1, 1234, 5678, 2};
        if (sm2_prefill(ctx, tokens, 4) != 0) {
            sm2_free_context(ctx);
            continue;
        }

        double t0 = time_ms();
        int gen = 0;

        for (int j = 0; j < n_tokens; j++) {
            int tok;
            int ok = sm2_decode_next(ctx, &tok);
            if (ok != 0 || tok < 3) break;
            gen++;
        }

        double dt = time_ms() - t0;
        total_ms += dt;
        total_gen += gen;

        fprintf(stderr, "Run %d: %d tokens in %.1f ms (%.1f ms/token)\n",
                r + 1, gen, dt, gen > 0 ? dt / gen : 0);

        sm2_free_context(ctx);
    }

    double avg_ms_per_token = total_ms / total_gen;
    double avg_tok_s = 1000.0 / avg_ms_per_token;
    fprintf(stderr, "Average: %.1f ms/token = %.1f tok/s\n", avg_ms_per_token, avg_tok_s);

    sm2_free_model(model);
    return 0;
}