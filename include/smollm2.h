#ifndef SMOLLMW_H
#define SMOLLMW_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

// ============================================================================
// SMOLLM2.C - Decode-First SmolLM2 Inference Engine
// Portable C99, no dependencies, optimized for VPS 512 MB
// ============================================================================

#define SM2_MAGIC "SM2C001"
#define SM2_VERSION 1
#define SM2_KV_PAGE_TOKENS 16

// ============================================================================
// VARIANT SPECS
// ============================================================================

typedef enum {
    SM2_135M = 135,
    SM2_360M = 360,
    SM2_1700M = 1700
} sm2_variant;

typedef enum {
    SM2_F16,
    SM2_Q8_0,
    SM2_Q4_0,
    SM2_Q4_K
} sm2_quant_type;

typedef enum {
    SM2_KV_F16,
    SM2_KV_Q8,
    SM2_KV_Q4,
    SM2_KV_TURBO2,
    SM2_KV_MIXED
} sm2_kv_dtype;

// ============================================================================
// MODEL SPEC CONSTANTS (per variant)
// ============================================================================

typedef struct {
    sm2_variant id;
    int n_layers;
    int dim;           // embedding dimension
    int hidden_dim;    // FFN intermediate size
    int n_heads;       // query heads
    int n_kv_heads;    // key/value heads (GQA)
    int head_dim;      // 64 for all variants
    int vocab_size;    // 49152 for SmolLM2
    int max_seq_len;   // 8192
    float rms_eps;
    float rope_theta;
    uint32_t bos_token_id;
    uint32_t eos_token_id;
    uint32_t pad_token_id;
} sm2_spec;

// Global spec table - indexed by variant
static const sm2_spec sm2_specs[] = {
    // 135M: 30 layers, dim 576, hidden 1536, GQA 9/3
    { SM2_135M,  30,  576, 1536,  9,  3, 64, 49152, 8192, 1e-5f, 100000.0f, 1, 2, 0 },
    // 360M: 32 layers, dim 960, hidden 2560, GQA 15/5
    { SM2_360M,  32,  960, 2560, 15,  5, 64, 49152, 8192, 1e-5f, 100000.0f, 1, 2, 0 },
    // 1.7B: 24 layers, dim 2048, hidden 8192, MHA 32/32
    { SM2_1700M, 24, 2048, 8192, 32, 32, 64, 49152, 8192, 1e-5f, 130000.0f, 1, 2, 0 },
};

// Maximum dimension across all variants (for stack allocation)
#define DIM_MAX 2048

static inline const sm2_spec* sm2_get_spec(sm2_variant v) {
    for (int i = 0; i < 3; i++) {
        if (sm2_specs[i].id == v) return &sm2_specs[i];
    }
    return &sm2_specs[0]; // default to 135M
}

// ============================================================================
// .SM2 FILE FORMAT HEADER
// ============================================================================

#define SM2_FILE_HEADER_SIZE 256

typedef struct {
    char magic[8];           // "SM2C001"
    uint32_t version;
    uint32_t variant_id;
    uint32_t quant_type;
    uint32_t flags;
    
    uint32_t vocab_size;
    uint32_t n_layers;
    uint32_t dim;
    uint32_t hidden_dim;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t max_seq_len;
    
    float rms_eps;
    float rope_theta;
    
    uint32_t bos_token_id;
    uint32_t eos_token_id;
    uint32_t pad_token_id;
    
    uint64_t tokenizer_offset;
    uint64_t tokenizer_size;
    uint64_t tensor_index_offset;
    uint64_t tensor_index_size;
    uint64_t weights_offset;
    uint64_t weights_size;
    uint64_t checksum;
} sm2_file_header;

// Tensor metadata for index
typedef struct {
    uint32_t name_len;
    char name[64];           // tensor name (up to 64 chars)
    uint32_t n_dims;
    uint32_t dims[4];
    uint32_t dtype;
    uint64_t offset;
    uint64_t size;
} sm2_tensor_meta;

// ============================================================================
// WEIGHT STORAGE (quantized)
// ============================================================================

// F16 weights - direct float16 values
typedef struct {
    int rows;
    int cols;
    uint16_t* data;          // float16 values
} sm2_tensor_f16;

// Q4_K weights - block quantized 4-bit
// Block size: 32 elements, scale + zero per block
#define SM2_Q4_K_BLOCK_SIZE 32

typedef struct {
    int rows;
    int cols;
    uint8_t* data;           // 4-bit quantized (2 values per byte)
    float* scales;           // per-block scale
    float* zeros;            // per-block zero point
    int n_blocks;            // rows * cols / 32
} sm2_tensor_q4_k;

// Q8_0 weights - 8-bit quantization
typedef struct {
    int rows;
    int cols;
    uint8_t* data;           // 8-bit values
    float* scales;           // per-block scale
    int n_blocks;
} sm2_tensor_q8_0;

// Generic tensor handle
typedef struct {
    char name[64];
    int n_dims;
    uint32_t dims[4];
    uint32_t dtype;          // 0=F16, 1=Q4_K, 2=Q8_0
    void* data;
    size_t size;
} sm2_tensor;

// ============================================================================
// MODEL WEIGHTS STRUCTURE
// ============================================================================

// Forward declaration - tokenizer is defined later
typedef struct sm2_tokenizer sm2_tokenizer;

typedef struct {
    sm2_variant variant;
    sm2_quant_type quant_type;
    
    // Embedding + output (shared for tie_word_embeddings)
    sm2_tensor_f16* tok_embeddings;    // [vocab, dim]
    
    // Layer weights (allocated as arrays per layer)
    // 1D weights: [n_layers * dim] of F16 (stored as uint16_t)
    uint16_t* input_layernorm;        // [n_layers * dim]
    uint16_t* post_attention_layernorm; // [n_layers * dim]
    uint16_t* final_norm;             // [dim] (not per-layer)

    // 2D weights: [n_layers * rows * cols] of F16
    uint16_t* q_proj;                 // [n_layers * dim * dim]
    uint16_t* k_proj;                 // [n_layers * kv_dim * dim]
    uint16_t* v_proj;                 // [n_layers * kv_dim * dim]
    uint16_t* o_proj;                 // [n_layers * dim * kv_dim]
    uint16_t* gate_proj;              // [n_layers * hidden_dim * dim]
    uint16_t* up_proj;                // [n_layers * hidden_dim * dim]
    uint16_t* down_proj;              // [n_layers * hidden_dim * dim]

    sm2_tokenizer* tokenizer;         // loaded from .sm2 file
    
    int n_layers;
    int dim;
    int hidden_dim;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int vocab_size;
} sm2_model;

// ============================================================================
// KV CACHE STRUCTURES
// ============================================================================

// KV page - stores 16 tokens of K and V for one layer/head
typedef struct {
    uint32_t id;
    uint16_t used;           // tokens in use (0-16)
    uint16_t refcount;       // reference count
    uint32_t next_free;      // free list link
    
    // Data pointers (can be F16, Q8, Q4, or TURBO2)
    uint8_t* k_data;
    uint8_t* v_data;
    uint8_t* k_scale;
    uint8_t* v_scale;
} sm2_kv_page;

// KV pool - global pool of pages
typedef struct {
    int n_layers;
    int n_kv_heads;
    int head_dim;
    int page_tokens;         // 16
    sm2_kv_dtype dtype;
    
    int max_pages;
    sm2_kv_page* pages;
    uint32_t* free_stack;
    int free_top;
} sm2_kv_pool;

// Per-sequence page table
typedef struct {
    uint64_t seq_id;
    int seq_len;
    int n_pages;
    uint32_t page_ids[256];  // max ~4K tokens
} sm2_kv_table;

// ============================================================================
// CONTEXT WITH PREALLOCATED BUFFERS (no runtime allocation)
// ============================================================================

// Preallocated scratch buffers for inference
// These are allocated once at init, never during decode
typedef struct {
    float* x;           // input embedding [dim]
    float* xb;          // residual buffer [dim]
    float* xb2;         // FFN temp buffer [max(dim, hidden_dim)]
    float* q;           // query [dim]
    float* k;           // key [n_kv_heads * head_dim]
    float* v;           // value [n_kv_heads * head_dim]
    float* attn_out;    // attention output [dim]
    float* logits;      // output logits [vocab_size]
    // KV cache for full context
    float** k_cache;    // per-layer K cache [n_layers][n_kv_heads][max_seq][head_dim]
    float** v_cache;    // per-layer V cache [n_layers][n_kv_heads][max_seq][head_dim]
    int kv_cache_len;   // current number of positions in KV cache
    // Repetition penalty: tracking recent tokens
    int* recent_tokens; // ring buffer of recent token IDs
    int recent_head;    // index into ring buffer
    int recent_max;     // max size of ring buffer
} sm2_scratch;

// Generation parameters
typedef struct {
    float temperature;
    int top_p;
    int top_k;
    int max_context;
    int max_output;
    float repetition_penalty;    // penalty for repeated tokens (1.0 = disabled)
    int penalty_window;          // how many recent tokens to check for repetition
} sm2_generate_params;

// Main inference context
typedef struct {
    sm2_model* model;
    sm2_kv_pool* kv_pool;
    sm2_kv_table kv;
    sm2_scratch scratch;
    
    int pos;                    // current position in sequence
    int last_token;             // last generated token
    uint64_t rng_state;
    
    sm2_generate_params params;
} sm2_context;

// ============================================================================
// TOKENIZER
// ============================================================================

typedef struct sm2_tokenizer {
    int vocab_size;
    char** tokens;
    float* scores;
    int* token_to_id;

    int num_merges;
    char** merges;              // pair of tokens to merge

    uint8_t* vocab_data;        // raw vocab json
    size_t vocab_size_bytes;

    // Byte-to-token mapping: byte_to_token[256] maps raw byte values to vocab IDs
    // For example, byte 84 ('T') -> vocab token 68
    int byte_to_token[256];
} sm2_tokenizer;

// ============================================================================
// SERVER REQUEST STATES
// ============================================================================

typedef enum {
    SM2_REQ_WAITING,
    SM2_REQ_PREFILL,
    SM2_REQ_DECODE,
    SM2_REQ_VERIFY,
    SM2_REQ_DONE,
    SM2_REQ_CANCELLED
} sm2_req_state;

// ============================================================================
// CLI ARGUMENTS (shared across modes)
// ============================================================================

typedef enum {
    MODE_CLI,
    MODE_TUI,
    MODE_WEB,
} run_mode;

typedef struct {
    const char* model_path;
    const char* prompt;
    int ctx_size;
    int max_output;
    int n_threads;
    float temperature;
    int top_p;
    int top_k;
    float repetition_penalty;
    run_mode mode;
    int web_port;
    const char* web_host;
    const char* system_prompt;
} cli_args;

// ============================================================================
// DFLASH CONFIG (Phase 8b)
// ============================================================================

typedef struct {
    sm2_model* draft_model;
    sm2_model* target_model;
    int num_draft_tokens;       // 16 default
    int block_size;             // 16
    float accept_threshold;
} sm2_dflash_config;

// ============================================================================
// CORE API FUNCTIONS
// ============================================================================

// Model loading from .sm2 file
int sm2_load_model(const char* path, sm2_model** out_model);
int sm2_free_model(sm2_model* model);

// Memory-mapped model loading (zero-copy)
int sm2_mmap_model(const char* path, sm2_model** out_model);
int sm2_unmmap_model(sm2_model* model);

// Context creation with preallocated buffers
int sm2_create_context(sm2_model* model, sm2_context** out_ctx);
int sm2_free_context(sm2_context* ctx);

// KV pool management
int sm2_kv_pool_init(sm2_kv_pool* pool, int n_layers, int n_kv_heads, int max_pages, sm2_kv_dtype dtype);
void sm2_kv_pool_free(sm2_kv_pool* pool);
uint32_t sm2_kv_alloc_page(sm2_kv_pool* pool);
void sm2_kv_free_page(sm2_kv_pool* pool, uint32_t page_id);

// Prefill - process prompt tokens
int sm2_prefill(sm2_context* ctx, const int* tokens, int n_tokens);

// Add after sm2_prefill declaration in include/smollm2.h
// Debug: print top logits
void sm2_debug_print_logits(sm2_context* ctx, int top_n);

// Decode - generate next token (no malloc in hot path)
int sm2_decode_next(sm2_context* ctx, int* out_token);

// Fast decode path with no allocation
int sm2dl_decode_next(sm2_context* ctx, int* out_token);

// Streaming decode with callback
typedef void (*sm2_stream_cb)(int token, void* user_data);
int sm2_decode_stream(sm2_context* ctx, int max_new_tokens, sm2_stream_cb cb, void* user_data);

// Sampling utilities
int sm2_sample_token(const float* logits, const sm2_generate_params* params, uint64_t* rng_state,
                     sm2_context* ctx);
float sm2_sample_temperature(float x, float temp, uint64_t* rng_state);

// RoPE (Rotary Position Embedding)
void sm2_rope(float* q, float* k, int head_dim, int pos, int n_heads, int n_kv_heads, float rope_theta);

// RMSNorm
void sm2_rmsnorm(float* out, const float* input, const float* weight, int size, float eps);
void sm2_rmsnorm_inplace(float* vec, const float* weight, int size, float eps);

// ============================================================================
// MATMUL BACKEND
// ============================================================================

typedef void (*sm2_matmul_fn)(float* out, const float* a, const void* b, int m, int n, int k);

typedef struct {
    sm2_matmul_fn matmul_f16;
    sm2_matmul_fn matmul_q4_k;
    sm2_matmul_fn matmul_q8_0;
} sm2_backend;

// Reference implementation
void sm2_matmul_f16_ref(float* out, const float* a, const sm2_tensor_f16* b, int m, int n, int k);
void sm2_matmul_q4_k_ref(float* out, const float* a, const sm2_tensor_q4_k* b, int m, int n, int k);
void sm2_matmul_q8_0_ref(float* out, const float* a, const sm2_tensor_q8_0* b, int m, int n, int k);

extern sm2_backend sm2_backend_ref;

// ============================================================================
// TOKENIZER API
// ============================================================================

int sm2_tokenizer_init(const char* vocab_path, const char* merges_path, sm2_tokenizer** out_tok);
void sm2_tokenizer_free(sm2_tokenizer* tok);
int sm2_tokenizer_encode(sm2_tokenizer* tok, const char* text, int* ids, int max_len);
char* sm2_tokenizer_decode(sm2_tokenizer* tok, const int* ids, int n_ids);
int sm2_load_tokenizer_from_sm2(sm2_tokenizer* tok, FILE* f, uint64_t offset, uint64_t size);

// Convert byte value to token ID using tokenizer's byte mapping
int sm2_tokenizer_byte_to_token(sm2_tokenizer* tok, unsigned char byte_val);

// ============================================================================
// SERVER API (Phase 6)
// ============================================================================

typedef struct {
    const char* model_path;
    const char* host;
    int port;
    int n_threads;
    int max_ctx;
    int max_output;
    sm2_kv_dtype kv_dtype;
    int low_memory;
} sm2_server_config;

int sm2_server_run(const sm2_server_config* config);

#endif // SMOLLMW_H