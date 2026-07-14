// gguf_write.c — phase 1: byte copy. Phase 2 will add full writer.
#include "gguf_write.h"

#include <stdio.h>
#include <stdlib.h>

int gguf_copy(const char* src_path, const char* dst_path) {
    FILE* in = fopen(src_path, "rb");
    if (!in) { perror("gguf_copy fopen src"); return -1; }
    FILE* out = fopen(dst_path, "wb");
    if (!out) { fclose(in); perror("gguf_copy fopen dst"); return -1; }

    char buf[64 * 1024];
    size_t n;
    long total = 0;
    while ((n = fread(buf, 1, sizeof(buf), in)) > 0) {
        if (fwrite(buf, 1, n, out) != n) {
            fclose(in); fclose(out);
            fprintf(stderr, "gguf_copy: write failed\n");
            return -1;
        }
        total += (long)n;
    }
    fclose(in);
    if (fclose(out) != 0) {
        fprintf(stderr, "gguf_copy: close failed\n");
        return -1;
    }
    printf("gguf_copy: %ld bytes written to %s\n", total, dst_path);
    return 0;
}