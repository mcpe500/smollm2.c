// main.c — smollm2 CLI

#include "gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// ----------------------------------------------------------------------------
// Default model resolution: read Ollama manifest, find model layer digest,
// return heap-allocated blob path (caller frees). Returns NULL on failure.
// ----------------------------------------------------------------------------
static char* resolve_ollama_model_path(void) {
    const char* home = getenv("HOME");
    if (!home) return NULL;

    char manifest_path[1024];
    snprintf(manifest_path, sizeof(manifest_path),
        "%s/.ollama/models/manifests/registry.ollama.ai/library/smollm2/135m", home);

    FILE* f = fopen(manifest_path, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz > 65536) { fclose(f); return NULL; }

    char* buf = malloc(sz + 1);
    fread(buf, 1, sz, f);
    buf[sz] = '\0';
    fclose(f);

    // Find the model layer: search for the mediaType, then the digest right AFTER it
    // (Ollama layer objects order: mediaType, digest, size, from).
    const char* needle = "\"application/vnd.ollama.image.model\"";
    char* p = strstr(buf, needle);
    if (!p) { free(buf); return NULL; }
    p += strlen(needle);

    char* q = strstr(p, "\"digest\":");
    if (!q) { free(buf); return NULL; }

    char* val_start = q + strlen("\"digest\":");
    while (*val_start == ' ' || *val_start == '\t') val_start++;
    if (*val_start != '"') { free(buf); return NULL; }
    val_start++;  // skip opening quote
    char* val_end = strchr(val_start, '"');
    if (!val_end) { free(buf); return NULL; }

    size_t digest_len = val_end - val_start;  // "sha256:HEX..."
    // Ollama stores blobs as "sha256-HEX..." (':' replaced by '-').
    char* blob_rel = malloc(digest_len + 1);
    memcpy(blob_rel, val_start, digest_len);
    blob_rel[digest_len] = '\0';
    char* colon = strchr(blob_rel, ':');
    if (colon) *colon = '-';

    free(buf);

    char* full = malloc(2048);
    snprintf(full, 2048, "%s/.ollama/models/blobs/%s", home, blob_rel);
    free(blob_rel);
    return full;
}

// ----------------------------------------------------------------------------
// --inspect
// ----------------------------------------------------------------------------
static int do_inspect(const char* path) {
    gguf_ctx ctx;
    if (gguf_load(path, &ctx) < 0) {
        fprintf(stderr, "failed to load %s\n", path);
        return 1;
    }

    printf("GGUF v%u, n_tensors=%llu, n_kv=%llu, size=%.1f MB\n\n",
           ctx.version,
           (unsigned long long)ctx.n_tensors,
           (unsigned long long)ctx.n_kv,
           ctx.size / 1048576.0);

    const char* arch = gguf_kv_str(&ctx, "general.architecture");
    printf("architecture: %s\n", arch ? arch : "(unknown)");

    if (arch) {
        char key[128];
        snprintf(key, sizeof(key), "%s.embedding_length", arch);
        printf("  embedding_length : %lld\n", (long long)gguf_kv_i64(&ctx, key, -1));
        snprintf(key, sizeof(key), "%s.block_count", arch);
        printf("  block_count      : %lld\n", (long long)gguf_kv_i64(&ctx, key, -1));
        snprintf(key, sizeof(key), "%s.attention.head_count", arch);
        printf("  head_count       : %lld\n", (long long)gguf_kv_i64(&ctx, key, -1));
        snprintf(key, sizeof(key), "%s.attention.head_count_kv", arch);
        printf("  head_count_kv    : %lld\n", (long long)gguf_kv_i64(&ctx, key, -1));
        snprintf(key, sizeof(key), "%s.feed_forward_length", arch);
        printf("  ffn_hidden       : %lld\n", (long long)gguf_kv_i64(&ctx, key, -1));
        snprintf(key, sizeof(key), "%s.attention.layer_norm_rms_epsilon", arch);
        printf("  rms_eps          : %g\n",
               (double)gguf_kv_f32(&ctx, key, 0.0f));
    }

    printf("  tie_word_embeddings : %lld\n",
           (long long)gguf_kv_i64(&ctx, "llama.tie_word_embeddings", -1));
    printf("  general.name        : %s\n",
           gguf_kv_str(&ctx, "general.name"));

    const char* tok_model = gguf_kv_str(&ctx, "tokenizer.ggml.model");
    printf("tokenizer model: %s\n", tok_model ? tok_model : "(none)");
    uint64_t vocab_n = 0;
    gguf_vtype et;
    const void* arr = gguf_kv_arr(&ctx, "tokenizer.ggml.tokens", &et, &vocab_n);
    (void)arr;
    printf("vocab size: %llu (elem_type=%d)\n",
           (unsigned long long)vocab_n, (int)et);

    printf("\nfirst 5 tensors:\n");
    for (uint64_t i = 0; i < ctx.n_tensors && i < 5; i++) {
        const gguf_tensor_info* t = &ctx.tensors[i];
        printf("  %-32s dtype=%2u dims=[", t->name, (unsigned)t->dtype);
        for (uint32_t d = 0; d < t->n_dims; d++) {
            printf("%llu%s", (unsigned long long)t->dims[d],
                   d + 1 < t->n_dims ? "," : "");
        }
        printf("]\n");
    }

    const char* needed[] = {
        "token_embd.weight", "output.weight",
        "blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
        "blk.0.attn_norm.weight", "blk.0.ffn_norm.weight",
        "output_norm.weight",
    };
    printf("\nkey tensor presence:\n");
    for (size_t i = 0; i < sizeof(needed)/sizeof(needed[0]); i++) {
        const gguf_tensor_info* t = gguf_tensor_get(&ctx, needed[i]);
        printf("  %-32s %s\n", needed[i], t ? "FOUND" : "MISSING");
    }

    gguf_free(&ctx);
    return 0;
}

// ----------------------------------------------------------------------------
// Usage
// ----------------------------------------------------------------------------
static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [--inspect] [-m <gguf-path>] [-p <prompt>] [-n <tokens>]\n"
        "\n"
        "  --inspect        Print GGUF metadata + tensor list, then exit.\n"
        "  -m <path>        GGUF model file. Default: auto-resolve from Ollama manifest.\n"
        "  -p <prompt>      Prompt (TODO: implemented in step 7).\n"
        "  -n <tokens>      Max tokens to generate (TODO).\n",
        prog);
}

int main(int argc, char** argv) {
    const char* model_path = NULL;
    int inspect = 0;
    const char* prompt = NULL;
    int n_tokens = 50;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--inspect") == 0) inspect = 1;
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]); return 0;
        } else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!model_path) {
        model_path = resolve_ollama_model_path();
        if (!model_path) {
            fprintf(stderr,
                "could not auto-resolve Ollama smollm2:135m; "
                "pass -m <gguf-path>\n");
            return 1;
        }
    }

    if (inspect) {
        int rc = do_inspect(model_path);
        if (!argv[1] || strcmp(argv[1], "--inspect") != 0) {
            // model_path was heap-allocated by resolver
        }
        return rc;
    }

    // TODO: Step 7 implements -p / -n.
    (void)prompt; (void)n_tokens;
    fprintf(stderr, "inference not yet implemented; use --inspect\n");
    return 1;
}
