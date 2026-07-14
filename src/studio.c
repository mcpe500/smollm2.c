// studio.c — subcommand dispatcher for studio phase 1+2
#include "studio.h"
#include "data.h"
#include "gguf_write.h"
#include "backward.h"
#include "tokenizer.h"
#include "gguf.h"
#include "forward.h"
#include "hw_probe.h"
#include "train.h"
#include "attn_registry.h"
#include "sampling.h"
#include "web.h"
#include "resolve_model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int usage() {
    fprintf(stderr,
        "studio subcommands:\n"
        "  studio data-build    --in <file> --out <packed.bin> [--fmt auto|raw|instruct|sharegpt] [--model <gguf>]\n"
        "  studio data-inspect --packed <packed.bin>\n"
        "  studio gguf-rewrite --in <base.gguf> --out <copy.gguf>\n"
        "  studio grad-check   [--m N --n N --k N] [--eps 1e-3]\n"
        "  studio hw\n"
        "  studio train        --data <packed.bin> --mode lora|qlora|fullft [--rank N] [--epochs N]\n"
        "                      [--lr F] [--seq N] [--batch N] [--max-steps N] [--out-dir DIR] [--model <gguf>]\n"
        "                      [--simulate-mem-kb N]\n"
        "  studio merge        --base <gguf> --adapter <lora.bin> --out <merged.gguf>\n"
        "  studio attn-list\n"
        "  studio attn-config  --config <layers.json> [--layers N]\n"
        "  studio web          [--port 8082] [--model <gguf>]\n");
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

static int cmd_hw(int argc, char** argv) {
    (void)argc; (void)argv;
    hw_caps c;
    hw_probe(&c);
    hw_print(&c);
    return 0;
}

static int cmd_train(int argc, char** argv) {
    const char* data = NULL, *mode_s = "lora", *model = NULL, *out_dir = "adapters";
    train_params p;
    memset(&p, 0, sizeof(p));
    p.mode = TRAIN_LORA;
    p.lora_rank = 8;
    p.lora_alpha = 16;
    p.seq_max = 128;
    p.batch = 1;
    p.lr = 1e-4f;
    p.epochs = 1;
    p.checkpoint_every = 5;
    p.max_steps = 0;
    p.seed = 42;
    p.simulate_mem_kb = 0;

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) data = argv[++i];
        else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) mode_s = argv[++i];
        else if (strcmp(argv[i], "--rank") == 0 && i + 1 < argc) p.lora_rank = atoi(argv[++i]);
        else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) p.epochs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) p.lr = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc) p.seq_max = atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) p.batch = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-steps") == 0 && i + 1 < argc) p.max_steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--out-dir") == 0 && i + 1 < argc) out_dir = argv[++i];
        else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model = argv[++i];
        else if (strcmp(argv[i], "--simulate-mem-kb") == 0 && i + 1 < argc)
            p.simulate_mem_kb = atol(argv[++i]);
    }
    if (!data || !model) {
        fprintf(stderr, "train: --data and --model required\n");
        return 1;
    }
    if (strcmp(mode_s, "lora") == 0) p.mode = TRAIN_LORA;
    else if (strcmp(mode_s, "qlora") == 0) p.mode = TRAIN_QLORA;
    else if (strcmp(mode_s, "fullft") == 0) p.mode = TRAIN_FULLFT;
    else { fprintf(stderr, "train: unknown mode %s\n", mode_s); return 1; }

    hw_caps caps;
    hw_probe(&caps);
    if (p.simulate_mem_kb > 0) caps.mem_avail_kb = p.simulate_mem_kb;

    if (p.mode == TRAIN_FULLFT) {
        long need = 2.5L * 1024 * 1024;  /* 2.5 GB */
        if (caps.mem_avail_kb < need) {
            fprintf(stderr,
                "train: fullft refused — insufficient mem "
                "(avail=%ld MB, need>=2560 MB)\n",
                caps.mem_avail_kb / 1024);
            return 1;
        }
    }
    if (p.mode == TRAIN_LORA || p.mode == TRAIN_QLORA) {
        long need = 800 * 1024;  /* 800 MB */
        if (caps.mem_avail_kb < need) {
            fprintf(stderr,
                "train: lora refused — insufficient mem "
                "(avail=%ld MB, need>=800 MB)\n",
                caps.mem_avail_kb / 1024);
            return 1;
        }
    }

    gguf_ctx g;
    if (gguf_load(model, &g) < 0) {
        fprintf(stderr, "train: cannot load model %s\n", model);
        return 1;
    }
    forward_ctx* fwd = NULL;
    if (forward_load(&fwd, &g, p.seq_max + 8) < 0) {
        fprintf(stderr, "train: forward_load failed\n");
        gguf_free(&g);
        return 1;
    }
    train_state* ts = train_create(fwd, &p);
    if (!ts) {
        fprintf(stderr, "train: train_create failed\n");
        forward_free(fwd); gguf_free(&g);
        return 1;
    }
    int rc = train_run(ts, data, &p, out_dir);
    train_free(ts);
    forward_free(fwd);
    gguf_free(&g);
    return rc < 0 ? 1 : 0;
}

static int cmd_merge(int argc, char** argv) {
    const char* base = NULL, *adapter = NULL, *out = NULL;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--base") == 0 && i + 1 < argc) base = argv[++i];
        else if (strcmp(argv[i], "--adapter") == 0 && i + 1 < argc) adapter = argv[++i];
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out = argv[++i];
    }
    if (!base || !adapter || !out) {
        fprintf(stderr, "merge: --base --adapter --out required\n");
        return 1;
    }
    return train_merge(base, adapter, out) < 0 ? 1 : 0;
}

static int cmd_web(int argc, char** argv) {
    int port = 8082;
    const char* model = NULL;
    char* auto_model = NULL;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) port = atoi(argv[++i]);
        else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model = argv[++i];
    }
    if (!model) {
        auto_model = resolve_ollama_model_path();
        if (!auto_model) {
            fprintf(stderr,
                "web: --model <gguf> required "
                "(or install Ollama smollm2:135m)\n");
            return 1;
        }
        model = auto_model;
    }
    sample_params sp = {0.3f, 0.0f, 5, 1.1f, 0};
    int rc = web_run(model, port, &sp);
    free(auto_model);
    return rc < 0 ? 1 : 0;
}

static int cmd_attn_list(int argc, char** argv) {
    (void)argc; (void)argv;
    printf("registered variants: dense, swa, dilated(stub), bigbird(stub), glocal(stub), mla(stub)\n");
    int n = attn_n_layers();
    if (n <= 0) {
        /* Dump default */
        attn_spec s;
        attn_get_spec(0, &s);  /* returns default when empty */
        printf("default: type=%s window=%d dilation=%d n_global=%d latent_dim=%d\n",
               attn_type_name(s.type), s.window, s.dilation, s.n_global, s.latent_dim);
        return 0;
    }
    for (int i = 0; i < n; i++) {
        attn_spec s;
        attn_get_spec(i, &s);
        printf("L%02d type=%s window=%d dilation=%d n_global=%d latent_dim=%d\n",
               i, attn_type_name(s.type), s.window, s.dilation, s.n_global, s.latent_dim);
    }
    return 0;
}

static int cmd_attn_config(int argc, char** argv) {
    const char* cfg = NULL;
    int layers = 30;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) cfg = argv[++i];
        else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) layers = atoi(argv[++i]);
    }
    if (!cfg) { fprintf(stderr, "attn-config: --config required\n"); return 1; }
    if (attn_load_config(cfg, layers) < 0) {
        fprintf(stderr, "attn-config: load failed\n");
        return 1;
    }
    return cmd_attn_list(0, NULL);
}

int studio_dispatch(int argc, char** argv) {
    if (argc < 1) return usage();
    if (strcmp(argv[0], "data-build") == 0)    return cmd_data_build(argc - 1, argv + 1);
    if (strcmp(argv[0], "data-inspect") == 0)  return cmd_data_inspect(argc - 1, argv + 1);
    if (strcmp(argv[0], "gguf-rewrite") == 0) return cmd_gguf_rewrite(argc - 1, argv + 1);
    if (strcmp(argv[0], "grad-check") == 0)   return cmd_grad_check(argc - 1, argv + 1);
    if (strcmp(argv[0], "hw") == 0)           return cmd_hw(argc - 1, argv + 1);
    if (strcmp(argv[0], "train") == 0)        return cmd_train(argc - 1, argv + 1);
    if (strcmp(argv[0], "merge") == 0)        return cmd_merge(argc - 1, argv + 1);
    if (strcmp(argv[0], "attn-list") == 0)    return cmd_attn_list(argc - 1, argv + 1);
    if (strcmp(argv[0], "attn-config") == 0)  return cmd_attn_config(argc - 1, argv + 1);
    if (strcmp(argv[0], "web") == 0)          return cmd_web(argc - 1, argv + 1);
    return usage();
}