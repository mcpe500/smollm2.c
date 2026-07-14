// studio.c — subcommand dispatcher for studio phase 1
#include "studio.h"
#include "data.h"
#include "gguf_write.h"
#include "backward.h"
#include "tokenizer.h"
#include "gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int usage() {
    fprintf(stderr,
        "studio subcommands:\n"
        "  studio data-build    --in <file> --out <packed.bin> [--fmt auto|raw|instruct|sharegpt] [--model <gguf>]\n"
        "  studio data-inspect --packed <packed.bin>\n"
        "  studio gguf-rewrite --in <base.gguf> --out <copy.gguf>\n"
        "  studio grad-check   [--m N --n N --k N] [--eps 1e-3]\n");
    return 1;
}

static int cmd_data_build(int argc, char** argv) {
    const char* in = NULL, *out = NULL, *fmt_s = "auto", *model = NULL;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in = argv[++i];
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out = argv[++i];
        else if (strcmp(argv[i], "--fmt") == 0 && i + 1 < argc) fmt_s = argv[++i];
        else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model = argv[++i];
    }
    if (!in || !out) { fprintf(stderr, "data-build: --in and --out required\n"); return 1; }

    data_fmt fmt = FMT_AUTO;
    if (strcmp(fmt_s, "raw") == 0) fmt = FMT_RAW;
    else if (strcmp(fmt_s, "instruct") == 0) fmt = FMT_INSTRUCT;
    else if (strcmp(fmt_s, "sharegpt") == 0) fmt = FMT_SHAREGPT;
    else if (strcmp(fmt_s, "auto") == 0) fmt = FMT_AUTO;
    else { fprintf(stderr, "unknown --fmt: %s\n", fmt_s); return 1; }

    tokenizer* tok = NULL;
    if (model) {
        gguf_ctx g;
        if (gguf_load(model, &g) < 0) {
            fprintf(stderr, "data-build: cannot load model %s\n", model);
            return 1;
        }
        if (tokenizer_load(&tok, &g) < 0) {
            fprintf(stderr, "data-build: tokenizer load failed\n");
            gguf_free(&g);
            return 1;
        }
        int rc = data_build(in, out, fmt, tok);
        tokenizer_free(tok);
        gguf_free(&g);
        return rc < 0 ? 1 : 0;
    }

    /* No model: still tokenize raw text as whitespace split (fallback) */
    fprintf(stderr, "data-build: --model required for phase 1 tokenization\n");
    return 1;
}

static int cmd_data_inspect(int argc, char** argv) {
    const char* packed = NULL;
    for (int i = 0; i < argc; i++)
        if (strcmp(argv[i], "--packed") == 0 && i + 1 < argc) packed = argv[++i];
    if (!packed) { fprintf(stderr, "--packed required\n"); return 1; }
    return data_inspect(packed) < 0 ? 1 : 0;
}

static int cmd_gguf_rewrite(int argc, char** argv) {
    const char* in = NULL, *out = NULL;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) in = argv[++i];
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out = argv[++i];
    }
    if (!in || !out) { fprintf(stderr, "--in and --out required\n"); return 1; }
    return gguf_copy(in, out) < 0 ? 1 : 0;
}

static int cmd_grad_check(int argc, char** argv) {
    int m = 4, n = 3, k = 5;
    float eps = 1e-3f;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--m") == 0 && i + 1 < argc) m = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--eps") == 0 && i + 1 < argc) eps = (float)atof(argv[++i]);
    }
    float err = backward_matmul_grad_check(m, n, k, eps);
    printf("grad-check: m=%d n=%d k=%d eps=%.2e max_abs_error=%.3e\n",
           m, n, k, eps, err);
    return err < 1e-3f ? 0 : 1;
}

int studio_dispatch(int argc, char** argv) {
    if (argc < 1) return usage();
    if (strcmp(argv[0], "data-build") == 0)    return cmd_data_build(argc - 1, argv + 1);
    if (strcmp(argv[0], "data-inspect") == 0)  return cmd_data_inspect(argc - 1, argv + 1);
    if (strcmp(argv[0], "gguf-rewrite") == 0) return cmd_gguf_rewrite(argc - 1, argv + 1);
    if (strcmp(argv[0], "grad-check") == 0)   return cmd_grad_check(argc - 1, argv + 1);
    return usage();
}