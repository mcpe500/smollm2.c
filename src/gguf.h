// gguf.h — Minimal GGUF v3 reader (mmap, no dequant)

#ifndef GGUF_H
#define GGUF_H

#include <stdint.h>
#include <stddef.h>

// GGUF magic = 'GGUF' little-endian
#define GGUF_MAGIC 0x46554747u
#define GGUF_VERSION 3u
#define GGUF_DEFAULT_ALIGNMENT 32u

// Metadata value types (gguf_metadata_value_type)
typedef enum {
    GGUF_V_UINT8   = 0,
    GGUF_V_INT8    = 1,
    GGUF_V_UINT16  = 2,
    GGUF_V_INT16   = 3,
    GGUF_V_UINT32  = 4,
    GGUF_V_INT32   = 5,
    GGUF_V_FLOAT32 = 6,
    GGUF_V_BOOL    = 7,
    GGUF_V_STRING  = 8,
    GGUF_V_ARRAY   = 9,
    GGUF_V_UINT64  = 10,
    GGUF_V_INT64   = 11,
    GGUF_V_FLOAT64 = 12,
} gguf_vtype;

// Tensor dtypes (ggml_type)
typedef enum {
    GGUF_DT_F32  = 0,
    GGUF_DT_F16  = 1,
    GGUF_DT_Q4_0 = 2,
    GGUF_DT_Q4_1 = 3,
    GGUF_DT_Q5_0 = 6,
    GGUF_DT_Q5_1 = 7,
    GGUF_DT_Q8_0 = 8,
    GGUF_DT_Q8_1 = 9,
    GGUF_DT_Q2_K = 10,
    GGUF_DT_Q3_K = 11,
    GGUF_DT_Q4_K = 12,
    GGUF_DT_Q5_K = 13,
    GGUF_DT_Q6_K = 14,
    GGUF_DT_Q8_K = 15,
} gguf_dtype;

typedef struct {
    char     key[128];
    gguf_vtype type;
    // For scalar types use v_i64/v_u64/v_f32/v_bool.
    // For string use v_str (heap-allocated, freed in gguf_free).
    // For array use v_arr (heap-allocated element buffer, freed in gguf_free).
    int64_t   v_i64;
    uint64_t  v_u64;
    float     v_f32;
    int       v_bool;
    struct { char* data; uint64_t n; } v_str;
    struct { gguf_vtype elem_type; uint64_t n; void* data; } v_arr;
} gguf_kv;

typedef struct {
    char        name[128];
    uint32_t    n_dims;
    uint64_t    dims[8];   // dims[0] = fastest changing (innermost)
    gguf_dtype  dtype;
    uint64_t    offset;    // from start of tensor data section
} gguf_tensor_info;

typedef struct {
    int                 fd;
    void*               map;
    uint64_t            size;
    uint32_t            version;
    uint64_t            n_tensors;
    uint64_t            n_kv;
    gguf_kv*            kv;
    gguf_tensor_info*   tensors;
    const void*         tensor_data;   // pointer into mmap
    uint32_t            alignment;
} gguf_ctx;

// Load GGUF file (mmap). Returns 0 on success, -1 on error.
int  gguf_load(const char* path, gguf_ctx* ctx);

// Release mmap and allocations.
void gguf_free(gguf_ctx* ctx);

// KV accessors. Return NULL / default if not found.
const gguf_kv* gguf_kv_get (const gguf_ctx* ctx, const char* key);
int64_t        gguf_kv_i64 (const gguf_ctx* ctx, const char* key, int64_t def);
float          gguf_kv_f32 (const gguf_ctx* ctx, const char* key, float def);
const char*    gguf_kv_str (const gguf_ctx* ctx, const char* key);
// Array: returns pointer to raw element buffer + writes element type + count.
const void*    gguf_kv_arr (const gguf_ctx* ctx, const char* key,
                            gguf_vtype* elem_type, uint64_t* n);

// Tensor accessors.
const gguf_tensor_info* gguf_tensor_get (const gguf_ctx* ctx, const char* name);
const void*             gguf_tensor_data(const gguf_ctx* ctx,
                                         const gguf_tensor_info* t);

// Utility: bytes per element for a dtype.
size_t gguf_dtype_size(gguf_dtype dt);

#endif // GGUF_H
