// gguf.c — Minimal GGUF v3 reader

#define _GNU_SOURCE
#include "gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

// ============================================================================
// Cursor: read primitive LE values from mmap buffer
// ============================================================================

typedef struct {
    const uint8_t* p;
    const uint8_t* end;
} cursor;

static int cur_need(cursor* c, uint64_t n) {
    if ((uint64_t)(c->end - c->p) < n) return -1;
    return 0;
}

static uint8_t cur_u8(cursor* c) {
    uint8_t v = *c->p++;
    return v;
}

static uint16_t cur_u16(cursor* c) {
    uint16_t v;
    memcpy(&v, c->p, 2);
    c->p += 2;
    return v;
}

static uint32_t cur_u32(cursor* c) {
    uint32_t v;
    memcpy(&v, c->p, 4);
    c->p += 4;
    return v;
}

static uint64_t cur_u64(cursor* c) {
    uint64_t v;
    memcpy(&v, c->p, 8);
    c->p += 8;
    return v;
}

static float cur_f32(cursor* c) {
    float v;
    memcpy(&v, c->p, 4);
    c->p += 4;
    return v;
}

static double cur_f64(cursor* c) {
    double v;
    memcpy(&v, c->p, 8);
    c->p += 8;
    return v;
}

// Read string (uint64 length + bytes). Heap-allocates a null-terminated copy.
static int cur_str(cursor* c, char** out) {
    if (cur_need(c, 8) < 0) return -1;
    uint64_t n = cur_u64(c);
    if (cur_need(c, n) < 0) return -1;
    char* s = malloc(n + 1);
    if (!s) return -1;
    memcpy(s, c->p, (size_t)n);
    s[n] = '\0';
    c->p += n;
    *out = s;
    return 0;
}

// ============================================================================
// KV parsing
// ============================================================================

static int parse_value(cursor* c, gguf_vtype t, gguf_kv* kv) {
    switch (t) {
        case GGUF_V_UINT8:   kv->v_u64 = cur_u8(c);              return 0;
        case GGUF_V_INT8:    kv->v_i64 = (int8_t)cur_u8(c);      return 0;
        case GGUF_V_UINT16:  kv->v_u64 = cur_u16(c);             return 0;
        case GGUF_V_INT16:   kv->v_i64 = (int16_t)cur_u16(c);    return 0;
        case GGUF_V_UINT32:  kv->v_u64 = cur_u32(c);             return 0;
        case GGUF_V_INT32:   kv->v_i64 = (int32_t)cur_u32(c);    return 0;
        case GGUF_V_FLOAT32: kv->v_f32 = cur_f32(c);             return 0;
        case GGUF_V_BOOL:    kv->v_bool = cur_u8(c) ? 1 : 0;     return 0;
        case GGUF_V_STRING:  return cur_str(c, &kv->v_str.data),
                             (kv->v_str.n = strlen(kv->v_str.data)), 0;
        case GGUF_V_UINT64:  kv->v_u64 = cur_u64(c);             return 0;
        case GGUF_V_INT64:   kv->v_i64 = (int64_t)cur_u64(c);    return 0;
        case GGUF_V_FLOAT64: kv->v_f32 = (float)cur_f64(c);      return 0;
        case GGUF_V_ARRAY: {
            if (cur_need(c, 4) < 0) return -1;
            gguf_vtype et = (gguf_vtype)cur_u32(c);
            if (cur_need(c, 8) < 0) return -1;
            uint64_t n = cur_u64(c);
            kv->v_arr.elem_type = et;
            kv->v_arr.n = n;
            // We treat strings/arrays-of-strings specially (heap-alloc array of char*).
            if (et == GGUF_V_STRING) {
                char** arr = calloc(n, sizeof(char*));
                if (!arr) return -1;
                for (uint64_t i = 0; i < n; i++) {
                    if (cur_str(c, &arr[i]) < 0) {
                        for (uint64_t j = 0; j < i; j++) free(arr[j]);
                        free(arr);
                        return -1;
                    }
                }
                kv->v_arr.data = arr;
                return 0;
            }
            // POD arrays: read into a heap buffer sized by element size.
            size_t esz = (et == GGUF_V_UINT8 || et == GGUF_V_INT8)   ? 1 :
                         (et == GGUF_V_UINT16 || et == GGUF_V_INT16) ? 2 :
                         (et == GGUF_V_FLOAT32 || et == GGUF_V_UINT32 ||
                          et == GGUF_V_INT32 || et == GGUF_V_BOOL)   ? 4 :
                         (et == GGUF_V_UINT64 || et == GGUF_V_INT64 ||
                          et == GGUF_V_FLOAT64)                      ? 8 : 0;
            if (esz == 0) return -1;
            uint64_t bytes = n * esz;
            if (cur_need(c, bytes) < 0) return -1;
            void* buf = malloc(bytes);
            if (!buf) return -1;
            memcpy(buf, c->p, (size_t)bytes);
            c->p += bytes;
            kv->v_arr.data = buf;
            return 0;
        }
    }
    return -1;
}

// ============================================================================
// Public API
// ============================================================================

int gguf_load(const char* path, gguf_ctx* ctx) {
    memset(ctx, 0, sizeof(*ctx));

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return -1; }

    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat"); close(fd); return -1; }
    uint64_t size = (uint64_t)st.st_size;

    void* map = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); close(fd); return -1; }

    ctx->fd = fd;
    ctx->map = map;
    ctx->size = size;
    ctx->alignment = GGUF_DEFAULT_ALIGNMENT;

    cursor c = { (const uint8_t*)map, (const uint8_t*)map + size };

    uint32_t magic = cur_u32(&c);
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "gguf: bad magic 0x%08x (expected 0x%08x)\n", magic, GGUF_MAGIC);
        gguf_free(ctx);
        return -1;
    }
    ctx->version = cur_u32(&c);
    if (ctx->version != GGUF_VERSION) {
        fprintf(stderr, "gguf: unsupported version %u (expected %u)\n",
                ctx->version, GGUF_VERSION);
        gguf_free(ctx);
        return -1;
    }
    ctx->n_tensors = cur_u64(&c);
    ctx->n_kv      = cur_u64(&c);

    // KV table
    ctx->kv = calloc(ctx->n_kv, sizeof(gguf_kv));
    if (!ctx->kv) { gguf_free(ctx); return -1; }
    for (uint64_t i = 0; i < ctx->n_kv; i++) {
        gguf_kv* kv = &ctx->kv[i];
        char* key = NULL;
        if (cur_str(&c, &key) < 0) { gguf_free(ctx); return -1; }
        strncpy(kv->key, key, sizeof(kv->key) - 1);
        free(key);
        uint32_t t = cur_u32(&c);
        kv->type = (gguf_vtype)t;
        if (parse_value(&c, kv->type, kv) < 0) {
            fprintf(stderr, "gguf: parse error at kv %llu key='%s'\n",
                    (unsigned long long)i, kv->key);
            gguf_free(ctx);
            return -1;
        }
    }

    // Tensor info table
    ctx->tensors = calloc(ctx->n_tensors, sizeof(gguf_tensor_info));
    if (!ctx->tensors) { gguf_free(ctx); return -1; }
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        gguf_tensor_info* t = &ctx->tensors[i];
        char* name = NULL;
        if (cur_str(&c, &name) < 0) { gguf_free(ctx); return -1; }
        strncpy(t->name, name, sizeof(t->name) - 1);
        free(name);
        t->n_dims = cur_u32(&c);
        if (t->n_dims > 8) { gguf_free(ctx); return -1; }
        for (uint32_t d = 0; d < t->n_dims; d++) {
            t->dims[d] = cur_u64(&c);
        }
        t->dtype  = (gguf_dtype)cur_u32(&c);
        t->offset = cur_u64(&c);
    }

    // Tensor data section starts at the next aligned offset after the
    // current cursor position.
    uint64_t here = (uint64_t)(c.p - (const uint8_t*)map);
    uint64_t aligned = (here + ctx->alignment - 1) & ~((uint64_t)ctx->alignment - 1);
    ctx->tensor_data = (const uint8_t*)map + aligned;

    return 0;
}

void gguf_free(gguf_ctx* ctx) {
    if (!ctx) return;
    if (ctx->kv) {
        for (uint64_t i = 0; i < ctx->n_kv; i++) {
            gguf_kv* kv = &ctx->kv[i];
            if (kv->type == GGUF_V_STRING) {
                free(kv->v_str.data);
            } else if (kv->type == GGUF_V_ARRAY) {
                if (kv->v_arr.elem_type == GGUF_V_STRING && kv->v_arr.data) {
                    char** arr = (char**)kv->v_arr.data;
                    for (uint64_t j = 0; j < kv->v_arr.n; j++) free(arr[j]);
                }
                free(kv->v_arr.data);
            }
        }
        free(ctx->kv);
    }
    free(ctx->tensors);
    if (ctx->map && ctx->map != MAP_FAILED) munmap(ctx->map, ctx->size);
    if (ctx->fd >= 0) close(ctx->fd);
    memset(ctx, 0, sizeof(*ctx));
    ctx->fd = -1;
}

const gguf_kv* gguf_kv_get(const gguf_ctx* ctx, const char* key) {
    for (uint64_t i = 0; i < ctx->n_kv; i++) {
        if (strcmp(ctx->kv[i].key, key) == 0) return &ctx->kv[i];
    }
    return NULL;
}

int64_t gguf_kv_i64(const gguf_ctx* ctx, const char* key, int64_t def) {
    const gguf_kv* kv = gguf_kv_get(ctx, key);
    if (!kv) return def;
    switch (kv->type) {
        case GGUF_V_UINT8:   return (int64_t)kv->v_u64;
        case GGUF_V_INT8:    return kv->v_i64;
        case GGUF_V_UINT16:  return (int64_t)kv->v_u64;
        case GGUF_V_INT16:   return kv->v_i64;
        case GGUF_V_UINT32:  return (int64_t)kv->v_u64;
        case GGUF_V_INT32:   return kv->v_i64;
        case GGUF_V_BOOL:    return kv->v_bool;
        case GGUF_V_UINT64:  return (int64_t)kv->v_u64;
        case GGUF_V_INT64:   return kv->v_i64;
        default:             return def;
    }
}

float gguf_kv_f32(const gguf_ctx* ctx, const char* key, float def) {
    const gguf_kv* kv = gguf_kv_get(ctx, key);
    if (!kv) return def;
    switch (kv->type) {
        case GGUF_V_FLOAT32: return kv->v_f32;
        case GGUF_V_FLOAT64: return kv->v_f32;
        default:             return (float)gguf_kv_i64(ctx, key, (int64_t)def);
    }
}

const char* gguf_kv_str(const gguf_ctx* ctx, const char* key) {
    const gguf_kv* kv = gguf_kv_get(ctx, key);
    if (!kv || kv->type != GGUF_V_STRING) return NULL;
    return kv->v_str.data;
}

const void* gguf_kv_arr(const gguf_ctx* ctx, const char* key,
                        gguf_vtype* elem_type, uint64_t* n) {
    const gguf_kv* kv = gguf_kv_get(ctx, key);
    if (!kv || kv->type != GGUF_V_ARRAY) return NULL;
    if (elem_type) *elem_type = kv->v_arr.elem_type;
    if (n) *n = kv->v_arr.n;
    return kv->v_arr.data;
}

const gguf_tensor_info* gguf_tensor_get(const gguf_ctx* ctx, const char* name) {
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        if (strcmp(ctx->tensors[i].name, name) == 0) return &ctx->tensors[i];
    }
    return NULL;
}

const void* gguf_tensor_data(const gguf_ctx* ctx, const gguf_tensor_info* t) {
    return (const uint8_t*)ctx->tensor_data + t->offset;
}

size_t gguf_dtype_size(gguf_dtype dt) {
    switch (dt) {
        case GGUF_DT_F32: return 4;
        case GGUF_DT_F16: return 2;
        case GGUF_DT_Q8_0: return 1 + 16; // 1 scale (f16) + 16 int8
        case GGUF_DT_Q8_1: return 2 + 16; // 2 (d, dsum) + 16 int8
        case GGUF_DT_Q4_0: return 1 + 16; // 1 scale + 16 packed (4-bit × 32)
        case GGUF_DT_Q4_1: return 2 + 16;
        case GGUF_DT_Q5_0: return 1 + 20;
        case GGUF_DT_Q5_1: return 2 + 20;
        default: return 0; // K-quants variable, handle separately if needed
    }
}
