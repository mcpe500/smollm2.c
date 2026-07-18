// gguf_write.c — byte copy + tensor patch for LoRA merge
#include "gguf_write.h"
#include "gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    (void)total;
    return 0;
}

/* Resolve absolute file offset of a tensor's data in src GGUF. */
static long resolve_tensor_offset(const char* src_path, const char* name,
                                  size_t* out_size) {
    gguf_ctx g;
    if (gguf_load(src_path, &g) < 0) return -1;
    long result = -1;
    const gguf_tensor_info* t = gguf_tensor_get(&g, name);
    if (!t) {
        fprintf(stderr, "gguf_patch: tensor '%s' not found\n", name);
        goto done;
    }
    const uint8_t* base = (const uint8_t*)g.map;
    const uint8_t* tdata = (const uint8_t*)g.tensor_data;
    result = (long)(tdata - base) + (long)t->offset;
    if (out_size) {
        size_t elems = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) elems *= (size_t)t->dims[d];
        *out_size = elems * gguf_dtype_size(t->dtype);
    }
done:
    gguf_free(&g);
    return result;
}

static int patch_one(FILE* out, long file_off,
                     const void* data, size_t n) {
    if (fseek(out, file_off, SEEK_SET) != 0) {
        fprintf(stderr, "gguf_patch: seek to %ld failed\n", file_off);
        return -1;
    }
    if (fwrite(data, 1, n, out) != n) {
        fprintf(stderr, "gguf_patch: write failed at %ld\n", file_off);
        return -1;
    }
    return 0;
}

int gguf_patch_tensors(const char* src_path, const char* dst_path,
                       const char* const* names,
                       const void* const* data,
                       const size_t* sizes,
                       int count) {
    if (!src_path || !dst_path || count <= 0) return -1;
    if (gguf_copy(src_path, dst_path) < 0) return -1;

    FILE* out = fopen(dst_path, "r+b");
    if (!out) { perror("gguf_patch fopen dst"); return -1; }

    for (int i = 0; i < count; i++) {
        size_t expect_size = 0;
        long off = resolve_tensor_offset(src_path, names[i], &expect_size);
        if (off < 0) {
            fprintf(stderr, "gguf_patch: cannot resolve '%s'\n", names[i]);
            fclose(out);
            return -1;
        }
        if (sizes[i] != expect_size) {
            fprintf(stderr,
                "gguf_patch: size mismatch for '%s' (got %zu, expected %zu)\n",
                names[i], sizes[i], expect_size);
            fclose(out);
            return -1;
        }
        if (patch_one(out, off, data[i], sizes[i]) < 0) {
            fclose(out);
            return -1;
        }
    }
    if (fflush(out) != 0 || fclose(out) != 0) {
        fprintf(stderr, "gguf_patch: finalize failed\n");
        return -1;
    }
    return 0;
}

int gguf_patch_tensor(const char* src_path, const char* dst_path,
                      const char* tensor_name,
                      const void* new_data, size_t n_bytes) {
    return gguf_patch_tensors(src_path, dst_path,
                              &tensor_name, &new_data, &n_bytes, 1);
}
