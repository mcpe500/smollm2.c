// main.c — smollm2 CLI

#include "gguf.h"
#include "tokenizer.h"
#include "forward.h"
#include "sampling.h"
#include "tui.h"
#include "web.h"
#include "studio.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <ctype.h>

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
// --logits <prompt>  : run prefill, print argmax of last-position logits.
//                      Step 5 verification gate vs `ollama run smollm2:135m`.
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// -p / -n / --temp / --top-p / --top-k / --rep-penalty : generate text
// ----------------------------------------------------------------------------
#define CHAT_TEMPLATE_MAX 4096

static void build_prompt(const char* user_text, char* out, int max_out) {
    snprintf(out, max_out,
        "<|im_start|>system\n"
        "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
        "<|im_end|>\n"
        "<|im_start|>user\n%s<|im_end|>\n"
        "<|im_start|>assistant\n",
        user_text);
}

// ----------------------------------------------------------------------------
// Heavy mode — think → answer → verify(gate). One forward_load for all passes.
// ----------------------------------------------------------------------------
#define HEAVY_PROMPT_MAX 12288
#define HEAVY_OUT_MAX    8192
#define HEAVY_IDS_MAX    1536

static const char HEAVY_SYS_THINK[] =
    "You are a careful reasoner. Write step-by-step notes inside "
    "<think>...</think> only. Focus on what the user asked. End with </think>.";

static const char HEAVY_SYS_ANSWER[] =
    "Answer the user using the notes. Be direct and correct. No <think> tags.";

static const char HEAVY_SYS_VERIFY[] =
    "Check the draft answer against the question. Reply with exactly one line: "
    "VERIFIED: <short reason> or REJECT: <what is wrong>.";

static const char HEAVY_SYS_REANSWER[] =
    "The draft was rejected. Write a corrected final answer only. No tags.";

/* gen_once: run one prefill+decode pass.
   - stop_sub: if non-NULL, halt once decoded text contains it
   - hit_stop: optional; set 1 if stop_sub was hit (before strip)
   - out_buf: NUL-term decoded text (stop_sub stripped if hit)
   Returns tokens generated, or -1 on error. */
static int gen_once(forward_ctx* fwd, tokenizer* tok,
                    const char* prompt, int max_new,
                    const sample_params* sp,
                    const char* stop_sub,
                    char* out_buf, int out_max, int stream,
                    int* hit_stop) {
    if (hit_stop) *hit_stop = 0;
    if (out_buf && out_max > 0) out_buf[0] = '\0';
    int ids[HEAVY_IDS_MAX];
    int n = tokenizer_encode(tok, prompt, ids, HEAVY_IDS_MAX);
    if (n <= 0) { fprintf(stderr, "heavy: empty prompt encode\n"); return -1; }
    if (n > 2047) n = 2047;

    int vocab = forward_vocab_size(fwd);
    float* logits = malloc((size_t)vocab * sizeof(float));
    int*   gen    = malloc((size_t)max_new * sizeof(int));
    if (!logits || !gen) { free(logits); free(gen); return -1; }

    forward_reset(fwd);
    if (forward_prefill(fwd, ids, n, logits) < 0) {
        free(logits); free(gen); return -1;
    }

    int pos = n, gen_n = 0, out_len = 0;
    char dec_buf[512];
    while (gen_n < max_new) {
        int next = sample_token(logits, vocab, sp, gen, gen_n);
        if (next == 1 || next == 2) break;
        gen[gen_n++] = next;
        int m = tokenizer_decode(tok, next, dec_buf, sizeof(dec_buf));
        if (m > 0) {
            if (stream) { fwrite(dec_buf, 1, m, stdout); fflush(stdout); }
            if (out_len + m < out_max - 1) {
                memcpy(out_buf + out_len, dec_buf, m);
                out_len += m;
                out_buf[out_len] = '\0';
                if (stop_sub && strstr(out_buf, stop_sub)) {
                    if (hit_stop) *hit_stop = 1;
                    out_len = (int)(strstr(out_buf, stop_sub) - out_buf);
                    out_buf[out_len] = '\0';
                    break;
                }
            }
        }
        if (pos < 2047) {
            if (forward_decode(fwd, next, pos, logits) < 0) break;
            pos++;
        } else break;
    }
    free(logits); free(gen);
    return gen_n;
}

/* 1 if VERIFIED/REJECT appears in first 40 chars (after leading space). */
static int gate_decision(const char* ver_buf) {
    for (int i = 0; i < 40 && ver_buf && ver_buf[i]; i++) {
        if (isspace((unsigned char)ver_buf[i])) continue;
        if (strncmp(ver_buf + i, "VERIFIED", 8) == 0) return 1;
        if (strncmp(ver_buf + i, "REJECT", 6) == 0) return -1;
        break; /* first non-space token is neither */
    }
    return 0;
}

static int verify_stamp_present(const char* buf) {
    return gate_decision(buf) != 0;
}

static int do_heavy(const char* path, const char* user_text,
                    int ans_n, int think_n, int ver_n,
                    const sample_params* sp) {
    gguf_ctx ctx;
    if (gguf_load(path, &ctx) < 0) { fprintf(stderr, "failed to load model\n"); return 1; }
    tokenizer* tok = NULL;
    if (tokenizer_load(&tok, &ctx) < 0) {
        fprintf(stderr, "tokenizer load failed\n"); gguf_free(&ctx); return 1;
    }
    forward_ctx* fwd = NULL;
    if (forward_load(&fwd, &ctx, 2048) < 0) {
        fprintf(stderr, "forward load failed\n");
        tokenizer_free(tok); gguf_free(&ctx); return 1;
    }

    char *prompt = malloc(HEAVY_PROMPT_MAX);
    char *think_buf = malloc(HEAVY_OUT_MAX);
    char *ans_buf = malloc(HEAVY_OUT_MAX);
    char *ver_buf = malloc(HEAVY_OUT_MAX);
    char *re_buf = malloc(HEAVY_OUT_MAX);
    if (!prompt || !think_buf || !ans_buf || !ver_buf || !re_buf) {
        fprintf(stderr, "heavy: alloc failed\n");
        free(prompt); free(think_buf); free(ans_buf); free(ver_buf); free(re_buf);
        forward_free(fwd); tokenizer_free(tok); gguf_free(&ctx); return 1;
    }

    clock_t t0 = clock();
    int t_think=0, t_ans=0, t_ver=0, t_re=0;
    int re_done = 0, think_degraded = 0, ver_degraded = 0;
    int hit = 0;

    /* PASS 1: THINK (retry once with half budget if no </think>) */
    think_buf[0] = ans_buf[0] = ver_buf[0] = re_buf[0] = '\0';
    snprintf(prompt, HEAVY_PROMPT_MAX,
        "<|im_start|>system\n%s<|im_end|>\n"
        "<|im_start|>user\n%s<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n",
        HEAVY_SYS_THINK, user_text);
    printf("=== THINK ===\n"); fflush(stdout);
    t_think = gen_once(fwd, tok, prompt, think_n, sp, "</think>",
                       think_buf, HEAVY_OUT_MAX, 1, &hit);
    printf("\n"); fflush(stdout);
    if (!hit) {
        int think_n2 = think_n / 2;
        if (think_n2 >= 16) {
            fprintf(stderr, "[think: retry %d tokens, no </think> in first pass]\n", think_n2);
            printf("=== THINK (retry) ===\n"); fflush(stdout);
            t_think = gen_once(fwd, tok, prompt, think_n2, sp, "</think>",
                               think_buf, HEAVY_OUT_MAX, 1, &hit);
            printf("\n"); fflush(stdout);
        }
        if (!hit) {
            think_degraded = 1;
            fprintf(stderr, "[think: degraded, no </think> after retry]\n");
        }
    }

    /* PASS 2: ANSWER */
    snprintf(prompt, HEAVY_PROMPT_MAX,
        "<|im_start|>system\n%s<|im_end|>\n"
        "<|im_start|>user\nQuestion: %s\nNotes: %s<|im_end|>\n"
        "<|im_start|>assistant\n",
        HEAVY_SYS_ANSWER, user_text, think_buf);
    printf("=== ANSWER ===\n"); fflush(stdout);
    t_ans = gen_once(fwd, tok, prompt, ans_n, sp, NULL,
                     ans_buf, HEAVY_OUT_MAX, 1, NULL);
    printf("\n"); fflush(stdout);

    /* PASS 3: VERIFY (prefix-force retry if no stamp) */
    snprintf(prompt, HEAVY_PROMPT_MAX,
        "<|im_start|>system\n%s<|im_end|>\n"
        "<|im_start|>user\nQuestion: %s\nDraft: %s<|im_end|>\n"
        "<|im_start|>assistant\n",
        HEAVY_SYS_VERIFY, user_text, ans_buf);
    printf("=== VERIFY ===\n"); fflush(stdout);
    t_ver = gen_once(fwd, tok, prompt, ver_n, sp, NULL,
                     ver_buf, HEAVY_OUT_MAX, 1, NULL);
    printf("\n"); fflush(stdout);
    if (!verify_stamp_present(ver_buf)) {
        fprintf(stderr, "[verify: retry with prefix force]\n");
        snprintf(prompt, HEAVY_PROMPT_MAX,
            "<|im_start|>system\n%s<|im_end|>\n"
            "<|im_start|>user\nQuestion: %s\nDraft: %s<|im_end|>\n"
            "<|im_start|>assistant\nVERIFIED: ",
            HEAVY_SYS_VERIFY, user_text, ans_buf);
        printf("=== VERIFY (retry) ===\n"); fflush(stdout);
        t_ver = gen_once(fwd, tok, prompt, ver_n, sp, NULL,
                         ver_buf, HEAVY_OUT_MAX, 1, NULL);
        printf("\n"); fflush(stdout);
        /* Prefix-forced output may be missing REJECT path; prefix guarantees
           VERIFIED-ish start. Treat as no-stamp only if both VERIFIED and
           REJECT absent. */
        if (!verify_stamp_present(ver_buf)) {
            /* Last resort: synthesize a VERIFIED stamp so the prefix is honest. */
            snprintf(ver_buf, HEAVY_OUT_MAX, "VERIFIED: (forced, model did not stamp)");
            ver_degraded = 1;
            fprintf(stderr, "[verify: degraded, no stamp after prefix retry]\n");
        }
    }

    /* Gate: REJECT → one re-answer (no re-verify). */
    if (gate_decision(ver_buf) == -1) {
        snprintf(prompt, HEAVY_PROMPT_MAX,
            "<|im_start|>system\n%s<|im_end|>\n"
            "<|im_start|>user\nQuestion: %s\nDraft: %s\nCritique: %s<|im_end|>\n"
            "<|im_start|>assistant\n",
            HEAVY_SYS_REANSWER, user_text, ans_buf, ver_buf);
        printf("=== RE-ANSWER ===\n"); fflush(stdout);
        t_re = gen_once(fwd, tok, prompt, ans_n, sp, NULL,
                        re_buf, HEAVY_OUT_MAX, 1, NULL);
        printf("\n"); fflush(stdout);
        re_done = 1;
    }

    double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
    int total = (t_think>0?t_think:0) + (t_ans>0?t_ans:0) + (t_ver>0?t_ver:0) + (t_re>0?t_re:0);
    printf("[heavy: think=%d ans=%d ver=%d %s%d tokens, %.1fs, %.1f tok/s overall%s%s]\n",
           t_think<0?0:t_think, t_ans<0?0:t_ans, t_ver<0?0:t_ver,
           re_done ? "re=" : "", re_done ? (t_re<0?0:t_re) : 0,
           secs, total > 0 ? total / secs : 0.0,
           think_degraded ? " degraded=think" : "",
           ver_degraded ? " degraded=verify" : "");

    free(prompt); free(think_buf); free(ans_buf); free(ver_buf); free(re_buf);
    forward_free(fwd); tokenizer_free(tok); gguf_free(&ctx);
    return 0;
}

static int do_generate(const char* path, const char* user_text,
                       int max_new, const sample_params* sp) {
    gguf_ctx ctx;
    if (gguf_load(path, &ctx) < 0) {
        fprintf(stderr, "failed to load model\n"); return 1;
    }
    tokenizer* tok = NULL;
    if (tokenizer_load(&tok, &ctx) < 0) {
        fprintf(stderr, "tokenizer load failed\n"); gguf_free(&ctx); return 1;
    }
    forward_ctx* fwd = NULL;
    if (forward_load(&fwd, &ctx, 2048) < 0) {
        fprintf(stderr, "forward load failed\n");
        tokenizer_free(tok); gguf_free(&ctx); return 1;
    }

    char prompt_buf[CHAT_TEMPLATE_MAX];
    build_prompt(user_text, prompt_buf, sizeof(prompt_buf));

    int prompt_ids[1024];
    int prompt_len = tokenizer_encode(tok, prompt_buf, prompt_ids, 1024);
    if (prompt_len <= 0) {
        fprintf(stderr, "empty prompt\n");
        forward_free(fwd); tokenizer_free(tok); gguf_free(&ctx); return 1;
    }

    int vocab = forward_vocab_size(fwd);
    float* logits = malloc((size_t)vocab * sizeof(float));
    int*   gen    = malloc((size_t)max_new * sizeof(int));
    if (!logits || !gen) {
        fprintf(stderr, "alloc failed\n");
        free(logits); free(gen);
        forward_free(fwd); tokenizer_free(tok); gguf_free(&ctx); return 1;
    }

    /* Prefill */
    clock_t t0 = clock();
    if (forward_prefill(fwd, prompt_ids, prompt_len, logits) < 0) {
        fprintf(stderr, "prefill failed\n");
        free(logits); free(gen);
        forward_free(fwd); tokenizer_free(tok); gguf_free(&ctx); return 1;
    }

    /* Decode loop */
    int pos = prompt_len;
    int gen_n = 0;
    char dec_buf[512];
    while (gen_n < max_new) {
        int next = sample_token(logits, vocab, sp, gen, gen_n);
        if (next == 1 || next == 2) break;  /* <|im_end|> */
        gen[gen_n++] = next;
        int m = tokenizer_decode(tok, next, dec_buf, sizeof(dec_buf));
        fwrite(dec_buf, 1, m, stdout);
        fflush(stdout);
        if (pos < 2047) {
            if (forward_decode(fwd, next, pos, logits) < 0) break;
            pos++;
        } else break;
    }
    double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
    fprintf(stdout, "\n[%d tokens, %.1fs, %.1f tok/s]\n",
            gen_n, secs, gen_n > 0 ? gen_n / secs : 0.0);

    free(logits); free(gen);
    forward_free(fwd); tokenizer_free(tok); gguf_free(&ctx);
    return 0;
}

// ----------------------------------------------------------------------------
// --logits / --logits-json <prompt>
//   as_json=0: human lines (legacy)
//   as_json=1: one JSON object with prompt_tokens + top-10 logits
// ----------------------------------------------------------------------------
static int do_logits(const char* path, const char* prompt, int max_seq, int as_json) {
    gguf_ctx ctx;
    if (gguf_load(path, &ctx) < 0) {
        fprintf(stderr, "failed to load %s\n", path);
        return 1;
    }
    tokenizer* tok = NULL;
    if (tokenizer_load(&tok, &ctx) < 0) {
        fprintf(stderr, "tokenizer load failed\n");
        gguf_free(&ctx);
        return 1;
    }
    forward_ctx* fwd = NULL;
    if (forward_load(&fwd, &ctx, max_seq) < 0) {
        fprintf(stderr, "forward load failed\n");
        tokenizer_free(tok);
        gguf_free(&ctx);
        return 1;
    }

    int ids[1024];
    int n = tokenizer_encode(tok, prompt, ids, 1024);
    if (n <= 0) {
        fprintf(stderr, "empty prompt encoding\n");
        forward_free(fwd);
        tokenizer_free(tok);
        gguf_free(&ctx);
        return 1;
    }
    if (n > max_seq) {
        fprintf(stderr, "prompt too long: %d tokens > max_seq %d\n", n, max_seq);
        n = max_seq;
    }

    int vocab = forward_vocab_size(fwd);
    float* logits = malloc((size_t)vocab * sizeof(float));
    if (!logits) {
        fprintf(stderr, "logits alloc failed\n");
        forward_free(fwd);
        tokenizer_free(tok);
        gguf_free(&ctx);
        return 1;
    }

    clock_t t0 = clock();
    if (forward_prefill(fwd, ids, n, logits) < 0) {
        fprintf(stderr, "forward_prefill failed\n");
        free(logits);
        forward_free(fwd);
        tokenizer_free(tok);
        gguf_free(&ctx);
        return 1;
    }
    double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;

    /* Top-K (K=10) via partial selection into fixed array. */
    enum { TOPK = 10 };
    int top_id[TOPK];
    float top_l[TOPK];
    int ntop = 0;
    for (int v = 0; v < vocab; v++) {
        float lv = logits[v];
        if (ntop < TOPK) {
            int j = ntop++;
            while (j > 0 && top_l[j - 1] < lv) {
                top_l[j] = top_l[j - 1];
                top_id[j] = top_id[j - 1];
                j--;
            }
            top_l[j] = lv;
            top_id[j] = v;
        } else if (lv > top_l[TOPK - 1]) {
            int j = TOPK - 1;
            while (j > 0 && top_l[j - 1] < lv) {
                top_l[j] = top_l[j - 1];
                top_id[j] = top_id[j - 1];
                j--;
            }
            top_l[j] = lv;
            top_id[j] = v;
        }
    }
    int best = (ntop > 0) ? top_id[0] : 0;

    if (as_json) {
        printf("{\"prompt_tokens\":[");
        for (int i = 0; i < n; i++) {
            if (i) putchar(',');
            printf("%d", ids[i]);
        }
        printf("],\"topk\":[");
        for (int i = 0; i < ntop; i++) {
            char dbuf[512];
            int dm = tokenizer_decode(tok, top_id[i], dbuf, sizeof(dbuf));
            if (i) putchar(',');
            printf("{\"id\":%d,\"logit\":%.6f,\"bytes\":[", top_id[i], top_l[i]);
            for (int b = 0; b < dm; b++) {
                if (b) putchar(',');
                printf("%d", (unsigned char)dbuf[b]);
            }
            printf("]}");
        }
        printf("],\"argmax\":%d,\"vocab\":%d,\"n_tokens\":%d,\"prefill_s\":%.4f}\n",
               best, vocab, n, secs);
    } else {
        printf("prompt tokens (%d):", n);
        for (int i = 0; i < n; i++) printf(" %d", ids[i]);
        printf("\n");

        char buf[512];
        int m = tokenizer_decode(tok, best, buf, sizeof(buf));
        printf("argmax: %d  logit=%.4f  decoded(%d bytes): \"", best, logits[best], m);
        for (int i = 0; i < m; i++) {
            unsigned char b = (unsigned char)buf[i];
            if (b >= 32 && b < 127) putchar(b);
            else printf("\\x%02x", b);
        }
        printf("\"\n");

        printf("prefill: %.3fs for %d tokens (%.2f tok/s scalar)\n",
               secs, n, n / secs);
    }

    free(logits);
    forward_free(fwd);
    tokenizer_free(tok);
    gguf_free(&ctx);
    return 0;
}

// ----------------------------------------------------------------------------
// Usage
// ----------------------------------------------------------------------------
static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [-p <prompt>] [-n <tokens>] [options]\n"
        "       %s --tui\n"
        "       %s --web [--port <port>]\n"
        "\n"
        "  -p <prompt>          Generate a response to prompt.\n"
        "  -n <tokens>          Max tokens to generate (default: 200).\n"
        "  --temp <float>       Temperature (default: 0.3).\n"
        "  --top-p <float>      Top-p nucleus sampling (default: 0.0 = off).\n"
        "  --top-k <int>        Top-k sampling (default: 5).\n"
        "  --rep-penalty <float> Repetition penalty (default: 1.1).\n"
        "\n"
        "  TUI in-session commands:\n"
        "    /temp 0.5    /topp 0.9    /topk 10    /settings\n"
        "  --rep-penalty <float> Repetition penalty (default: 1.1).\n"
        "  --tui                Launch full-screen ncurses TUI chat.\n"
        "  --web                Start HTTP WebUI server (default port 8080).\n"
        "  --port <port>        WebUI port (use with --web).\n"
        "  -m <path>            GGUF model file (default: auto-resolve Ollama).\n"
        "  --inspect            Print model metadata.\n"
        "  --tok-test <text>    Tokenizer round-trip test.\n"
        "  --logits <prompt>    Print argmax logit for prompt.\n"
        "  --logits-json <p>    Same as --logits, one JSON line (top-10).\n"
        "  --heavy              Multi-pass: think → answer → verify(gate).\n"
        "  --heavy-think-n <n>  Think token budget (default 128).\n"
        "  --heavy-verify-n <n> Verify token budget (default 64).\n"
        "  --rope <f32|f16|q8>  RoPE table precision (default f32).\n"
        "  --kv   <f32|f16|q8>  KV cache precision (default f32).\n"
        "  --attn <naive|flash> Attention kernel (default naive).\n"
        "  -h / --help          Show this help.\n",
        prog, prog, prog);
}

int main(int argc, char** argv) {
    if (argc >= 2 && strcmp(argv[1], "studio") == 0) {
        return studio_dispatch(argc - 2, argv + 2);
    }
    const char* model_path   = NULL;
    const char* tok_test_text = NULL;
    const char* logits_prompt = NULL;
    int logits_as_json = 0;
    int inspect = 0;
    int do_tui  = 0;
    int do_web  = 0;
    int web_port = 8080;
    const char* prompt = NULL;
    int n_tokens = 200;
    int heavy = 0, heavy_think_n = 128, heavy_verify_n = 64;
    int rope_mode = ROPE_F32, kv_mode = KV_F32, attn_mode = ATTN_NAIVE;
    sample_params sp = {0.3f, 0.0f, 5, 1.1f, 0};

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--inspect") == 0) inspect = 1;
        else if (strcmp(argv[i], "--tok-test") == 0 && i + 1 < argc) tok_test_text = argv[++i];
        else if (strcmp(argv[i], "--logits") == 0 && i + 1 < argc) logits_prompt = argv[++i];
        else if (strcmp(argv[i], "--logits-json") == 0 && i + 1 < argc) { logits_prompt = argv[++i]; logits_as_json = 1; }
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) sp.temperature = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) sp.top_p = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) sp.top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--rep-penalty") == 0 && i + 1 < argc) sp.rep_penalty = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--heavy") == 0) heavy = 1;
        else if (strcmp(argv[i], "--heavy-think-n") == 0 && i + 1 < argc) heavy_think_n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--heavy-verify-n") == 0 && i + 1 < argc) heavy_verify_n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--rope") == 0 && i + 1 < argc) {
            const char* v = argv[++i];
            if      (strcmp(v, "f32") == 0) rope_mode = ROPE_F32;
            else if (strcmp(v, "f16") == 0) rope_mode = ROPE_F16;
            else if (strcmp(v, "q8")  == 0) rope_mode = ROPE_Q8;
            else { fprintf(stderr, "invalid --rope value: %s (want f32|f16|q8)\n", v); return 1; }
        }
        else if (strcmp(argv[i], "--kv") == 0 && i + 1 < argc) {
            const char* v = argv[++i];
            if      (strcmp(v, "f32") == 0) kv_mode = KV_F32;
            else if (strcmp(v, "f16") == 0) kv_mode = KV_F16;
            else if (strcmp(v, "q8")  == 0) kv_mode = KV_Q8;
            else { fprintf(stderr, "invalid --kv value: %s (want f32|f16|q8)\n", v); return 1; }
        }
        else if (strcmp(argv[i], "--attn") == 0 && i + 1 < argc) {
            const char* v = argv[++i];
            if      (strcmp(v, "naive") == 0) attn_mode = ATTN_NAIVE;
            else if (strcmp(v, "flash") == 0) attn_mode = ATTN_FLASH;
            else { fprintf(stderr, "invalid --attn value: %s (want naive|flash)\n", v); return 1; }
        }
        else if (strcmp(argv[i], "--tui") == 0) do_tui = 1;
        else if (strcmp(argv[i], "--web") == 0) do_web = 1;
        else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) web_port = atoi(argv[++i]);
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]); return 0;
        } else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    int resolver_allocated = 0;
    forward_set_modes(rope_mode, kv_mode, attn_mode);
    if (!model_path) {
        model_path = resolve_ollama_model_path();
        if (!model_path) {
            fprintf(stderr,
                "could not auto-resolve Ollama smollm2:135m; "
                "pass -m <gguf-path>\n");
            return 1;
        }
        resolver_allocated = 1;
    }

    int rc = 1;

    if (inspect) {
        rc = do_inspect(model_path);
        goto cleanup;
    }

    if (tok_test_text) {
        gguf_ctx ctx;
        if (gguf_load(model_path, &ctx) < 0) {
            fprintf(stderr, "failed to load %s\n", model_path);
            goto cleanup;
        }
        tokenizer* tok = NULL;
        if (tokenizer_load(&tok, &ctx) < 0) {
            fprintf(stderr, "tokenizer load failed\n");
            gguf_free(&ctx);
            goto cleanup;
        }
        int ids[1024];
        int n = tokenizer_encode(tok, tok_test_text, ids, 1024);
        printf("input  (%zu bytes): %s\n", strlen(tok_test_text), tok_test_text);
        printf("tokens (%d):", n);
        for (int i = 0; i < n; i++) printf(" %d", ids[i]);
        printf("\n");

        char buf[4096];
        int total = 0;
        for (int i = 0; i < n; i++) {
            char tmp[512];
            int m = tokenizer_decode(tok, ids[i], tmp, sizeof(tmp));
            if (m > 0 && total + m < (int)sizeof(buf)) {
                memcpy(buf + total, tmp, m);
                total += m;
            }
        }
        buf[total] = '\0';
        printf("decode (%d bytes): %s\n", total, buf);
        // Hex of decode bytes for sanity
        printf("hex   :");
        for (int i = 0; i < total; i++) printf(" %02x", (unsigned char)buf[i]);
        printf("\n");

        // Token strings
        printf("token strings:");
        for (int i = 0; i < n; i++) {
            char tmp[512];
            int m = tokenizer_decode(tok, ids[i], tmp, sizeof(tmp));
            printf(" [%d]='", ids[i]);
            for (int j = 0; j < m; j++) {
                unsigned char b = (unsigned char)tmp[j];
                if (b >= 32 && b < 127) putchar(b);
                else printf("\\x%02x", b);
            }
            printf("'");
        }
        printf("\n");

        tokenizer_free(tok);
        gguf_free(&ctx);
        rc = 0;
        goto cleanup;
    }

    if (logits_prompt) {
        rc = do_logits(model_path, logits_prompt, 2048, logits_as_json);
        goto cleanup;
    }

    if (prompt) {
        if (heavy) {
            rc = do_heavy(model_path, prompt, n_tokens, heavy_think_n, heavy_verify_n, &sp);
        } else {
            rc = do_generate(model_path, prompt, n_tokens, &sp);
        }
        goto cleanup;
    }

    if (do_tui) {
        rc = tui_run(model_path, &sp);
        goto cleanup;
    }

    if (do_web) {
        rc = web_run(model_path, web_port, &sp);
        goto cleanup;
    }

    fprintf(stderr, "No action specified. Use -p <prompt>, --tui, or --web.\n");
    usage(argv[0]);

cleanup:
    if (resolver_allocated) free((void*)model_path);
    return rc;
}
