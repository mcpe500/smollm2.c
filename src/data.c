// data.c — dataset adapter (auto-detect, template, packed token IDs)
#include "data.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/stat.h>

#define MAX_LINE  (256 * 1024)
#define MAX_TOKENS 2048

const char* data_fmt_name(data_fmt f) {
    switch (f) {
    case FMT_RAW:      return "RAW";
    case FMT_INSTRUCT: return "INSTRUCT";
    case FMT_SHAREGPT: return "SHAREGPT";
    case FMT_AUTO:     return "AUTO";
    }
    return "?";
}

/* ---- JSON sniffing: bare-bones key check on first non-blank line ---- */
static const char* find_key(const char* s, const char* key) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    return strstr(s, pat);
}

data_fmt data_detect(const char* in_path) {
    FILE* f = fopen(in_path, "r");
    if (!f) return FMT_RAW;  /* fallback */
    char buf[4096];
    char first[4096] = {0};
    while (fgets(buf, sizeof(buf), f)) {
        char* p = buf;
        while (*p && isspace((unsigned char)*p)) p++;
        if (*p == '\0') continue;
        if (*p == '#') continue;
        strncpy(first, p, sizeof(first) - 1);
        break;
    }
    fclose(f);

    /* Trim trailing newline */
    size_t n = strlen(first);
    while (n > 0 && (first[n-1] == '\n' || first[n-1] == '\r')) first[--n] = 0;

    if (first[0] != '{') return FMT_RAW;
    if (find_key(first, "messages"))                       return FMT_SHAREGPT;
    if (find_key(first, "prompt") && find_key(first, "completion"))
                                                          return FMT_INSTRUCT;
    if (find_key(first, "text"))                           return FMT_RAW;
    return FMT_RAW;
}

/* ---- Template builders (ChatML style used in main.c) ---- */
static void append_chatml(char* out, size_t cap, const char* role,
                          const char* content) {
    size_t cur = strlen(out);
    if (cur >= cap - 32) return;
    snprintf(out + cur, cap - cur, "<|im_start|>%s\n%s<|im_end|>\n", role, content);
}

static int tokenize_text(const tokenizer* tok, const char* text,
                         int* out, int max_out) {
    return tokenizer_encode(tok, text, out, max_out);
}

/* ---- Sample builders ---- */
typedef int (*sample_builder)(const tokenizer* tok, FILE* in, FILE* out_idx,
                              FILE* out_bin, long* total_tokens);

static int build_raw(const tokenizer* tok, FILE* in, FILE* out_idx,
                     FILE* out_bin, long* total) {
    char line[MAX_LINE];
    int n_samples = 0;
    while (fgets(line, sizeof(line), in)) {
        size_t L = strlen(line);
        while (L > 0 && (line[L-1] == '\n' || line[L-1] == '\r')) line[--L] = 0;
        if (L == 0) continue;
        int ids[MAX_TOKENS];
        int n = tokenize_text(tok, line, ids, MAX_TOKENS);
        if (n <= 0) continue;
        long off = ftell(out_bin);
        if (fwrite(ids, sizeof(int), n, out_bin) != (size_t)n) return -1;
        sample_idx si = { .n_tokens = n, .offset = off };
        if (fwrite(&si, sizeof(si), 1, out_idx) != 1) return -1;
        *total += n;
        n_samples++;
    }
    return n_samples;
}

/* Minimal JSON value extractor (avoid full parser dep). Only handles
   string values after a given key. */
static int extract_str(const char* json, const char* key, char* out, size_t cap) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char* p = strstr(json, pat);
    if (!p) return 0;
    p += strlen(pat);
    while (*p && (isspace((unsigned char)*p) || *p == ':')) p++;
    if (*p != '"') return 0;
    p++;
    size_t i = 0;
    while (*p && *p != '"' && i < cap - 1) {
        if (*p == '\\' && p[1]) { p++; }
        out[i++] = *p++;
    }
    out[i] = 0;
    return 1;
}

/* Iterate messages array — call cb for each {"role":..., "content":...}.
   cb builds ChatML string. We only support flat arrays (no nested). */
typedef void (*msg_cb)(const char* role, const char* content, char* out, size_t cap);

static void extract_messages(const char* line, msg_cb cb, char* out, size_t cap) {
    const char* p = strstr(line, "\"messages\"");
    if (!p) return;
    p = strchr(p, '[');
    if (!p) return;
    while (*p && *p != ']') {
        const char* obj = strchr(p, '{');
        if (!obj) break;
        const char* end = strchr(obj, '}');
        if (!end) break;
        char buf[MAX_LINE];
        size_t L = (size_t)(end - obj + 1);
        if (L >= sizeof(buf)) L = sizeof(buf) - 1;
        memcpy(buf, obj, L);
        buf[L] = 0;
        char role[64] = {0}, content[MAX_LINE] = {0};
        extract_str(buf, "role", role, sizeof(role));
        extract_str(buf, "content", content, sizeof(content));
        if (role[0] && content[0]) cb(role, content, out, cap);
        p = end + 1;
    }
}

static void cb_chatml(const char* role, const char* content,
                     char* out, size_t cap) {
    append_chatml(out, cap, role, content);
}

static int build_instruct(const tokenizer* tok, FILE* in, FILE* out_idx,
                          FILE* out_bin, long* total) {
    char line[MAX_LINE];
    int n_samples = 0;
    while (fgets(line, sizeof(line), in)) {
        char prompt[MAX_LINE] = {0}, completion[MAX_LINE] = {0};
        if (!extract_str(line, "prompt", prompt, sizeof(prompt))) continue;
        if (!extract_str(line, "completion", completion, sizeof(completion))) continue;
        char tmpl[MAX_LINE * 2] = {0};
        append_chatml(tmpl, sizeof(tmpl), "user", prompt);
        append_chatml(tmpl, sizeof(tmpl), "assistant", completion);
        int ids[MAX_TOKENS];
        int n = tokenize_text(tok, tmpl, ids, MAX_TOKENS);
        if (n <= 0) continue;
        long off = ftell(out_bin);
        if (fwrite(ids, sizeof(int), n, out_bin) != (size_t)n) return -1;
        sample_idx si = { .n_tokens = n, .offset = off };
        if (fwrite(&si, sizeof(si), 1, out_idx) != 1) return -1;
        *total += n;
        n_samples++;
    }
    return n_samples;
}

static int build_sharegpt(const tokenizer* tok, FILE* in, FILE* out_idx,
                          FILE* out_bin, long* total) {
    char line[MAX_LINE];
    int n_samples = 0;
    while (fgets(line, sizeof(line), in)) {
        char tmpl[MAX_LINE * 4] = {0};
        extract_messages(line, cb_chatml, tmpl, sizeof(tmpl));
        if (tmpl[0] == 0) continue;
        int ids[MAX_TOKENS];
        int n = tokenize_text(tok, tmpl, ids, MAX_TOKENS);
        if (n <= 0) continue;
        long off = ftell(out_bin);
        if (fwrite(ids, sizeof(int), n, out_bin) != (size_t)n) return -1;
        sample_idx si = { .n_tokens = n, .offset = off };
        if (fwrite(&si, sizeof(si), 1, out_idx) != 1) return -1;
        *total += n;
        n_samples++;
    }
    return n_samples;
}

int data_build(const char* in_path, const char* out_path,
               data_fmt fmt, const tokenizer* tok) {
    if (!tok) {
        fprintf(stderr, "data_build: tokenizer required\n");
        return -1;
    }
    if (fmt == FMT_AUTO) fmt = data_detect(in_path);

    FILE* in = fopen(in_path, "r");
    if (!in) { perror("data_build fopen in"); return -1; }

    char idx_path[1024];
    snprintf(idx_path, sizeof(idx_path), "%s.idx", out_path);

    FILE* out_idx = fopen(idx_path, "wb");
    if (!out_idx) { fclose(in); perror("fopen idx"); return -1; }

    FILE* out_bin = fopen(out_path, "wb");
    if (!out_bin) { fclose(in); fclose(out_idx); perror("fopen out"); return -1; }

    long total = 0;
    int n_samples = 0;
    sample_builder fn = NULL;
    switch (fmt) {
    case FMT_RAW:      fn = build_raw;      break;
    case FMT_INSTRUCT: fn = build_instruct; break;
    case FMT_SHAREGPT: fn = build_sharegpt; break;
    default:           fn = build_raw;      break;
    }
    n_samples = fn(tok, in, out_idx, out_bin, &total);

    fclose(in);
    fclose(out_idx);
    fclose(out_bin);

    if (n_samples <= 0) {
        fprintf(stderr, "data_build: 0 samples produced\n");
        return -1;
    }

    /* Append header to idx: n_samples, total_tokens, format, magic */
    FILE* f = fopen(idx_path, "ab");
    if (!f) return -1;
    int32_t hdr[3] = { n_samples, (int)total, (int)fmt };
    fwrite(hdr, sizeof(int32_t), 3, f);
    const char magic[8] = "STUDIO\0";
    fwrite(magic, 1, 8, f);
    fclose(f);

    printf("data_build: fmt=%s n_samples=%d total_tokens=%ld out=%s idx=%s\n",
           data_fmt_name(fmt), n_samples, total, out_path, idx_path);
    return 0;
}

int data_inspect(const char* packed_path) {
    char idx_path[1024];
    snprintf(idx_path, sizeof(idx_path), "%s.idx", packed_path);
    FILE* f = fopen(idx_path, "rb");
    if (!f) { fprintf(stderr, "data_inspect: cannot open %s\n", idx_path); return -1; }
    fseek(f, -20, SEEK_END);
    int32_t hdr[3] = {0};
    char magic[8] = {0};
    fread(hdr, sizeof(int32_t), 3, f);
    fread(magic, 1, 8, f);
    fclose(f);
    if (memcmp(magic, "STUDIO", 7) != 0) {
        fprintf(stderr, "data_inspect: bad magic\n");
        return -1;
    }
    printf("data_inspect: n_samples=%d total_tokens=%d fmt=%s\n",
           hdr[0], hdr[1], data_fmt_name((data_fmt)hdr[2]));
    return 0;
}

void data_free(dataset* d) {
    if (!d) return;
    free(d->index);
    free(d->packed_path);
    free(d);
}