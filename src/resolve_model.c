// resolve_model.c — Ollama manifest → blob path
#include "resolve_model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* resolve_ollama_model_path(void) {
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

    char* buf = malloc((size_t)sz + 1);
    fread(buf, 1, (size_t)sz, f);
    buf[sz] = '\0';
    fclose(f);

    const char* needle = "\"application/vnd.ollama.image.model\"";
    char* p = strstr(buf, needle);
    if (!p) { free(buf); return NULL; }
    p += strlen(needle);

    char* q = strstr(p, "\"digest\":");
    if (!q) { free(buf); return NULL; }

    char* val_start = q + strlen("\"digest\":");
    while (*val_start == ' ' || *val_start == '\t') val_start++;
    if (*val_start != '"') { free(buf); return NULL; }
    val_start++;
    char* val_end = strchr(val_start, '"');
    if (!val_end) { free(buf); return NULL; }

    size_t digest_len = (size_t)(val_end - val_start);
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
