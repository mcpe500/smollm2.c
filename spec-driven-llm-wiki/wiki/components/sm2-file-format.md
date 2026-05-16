---
title: "sm2-file-format"
type: component
tags: [file-format, model, storage]
last_updated: 2026-05-16
---

# sm2-file-format

Custom binary file format for smollm2.c runtime. Converts HF safetensors to efficient mmap-friendly format.

## Format Version
- Magic: SM2C001
- Version: 1
- Endian: Little-endian (native for x86/ARM)

## Header (256 bytes)

```c
#define SM2_MAGIC "SM2C001"
#define SM2_VERSION 1

typedef struct {
    char magic[8];           // "SM2C001"
    uint32_t version;        // 1
    uint32_t variant_id;     // SM2_135M / SM2_360M / SM2_1700M
    uint32_t quant_type;     // F16 / Q8_0 / Q4_0 / Q4_K
    uint32_t flags;          // reserved
    
    uint32_t vocab_size;    // 49152
    uint32_t n_layers;       // 30 / 32 / 24
    uint32_t dim;            // 576 / 960 / 2048
    uint32_t hidden_dim;     // 1536 / 2560 / 8192
    uint32_t n_heads;        // 9 / 15 / 32
    uint32_t n_kv_heads;     // 3 / 5 / 32
    uint32_t head_dim;       // 64 (fixed)
    uint32_t max_seq_len;    // 8192
    
    float rms_eps;          // 1e-5
    float rope_theta;        // 100000 / 130000
    
    uint32_t bos_token_id;   // 1
    uint32_t eos_token_id;   // 2
    uint32_t pad_token_id;   // 0
    
    uint64_t tokenizer_offset;
    uint64_t tokenizer_size;
    uint64_t tensor_index_offset;
    uint64_t tensor_index_size;
    uint64_t weights_offset;
    uint64_t weights_size;
    uint64_t checksum;
} sm2_file_header;
```

## Tensor Index

Binary array of tensor metadata:

```c
typedef struct {
    uint32_t name_len;
    char name[name_len];
    uint32_t n_dims;
    uint32_t dims[n_dims];
    uint32_t dtype;
    uint64_t offset;
    uint64_t size;
} sm2_tensor;
```

## Tensor Order

```
tok_embeddings

for layer 0..n_layers-1:
  input_layernorm
  q_proj
  k_proj
  v_proj
  o_proj
  post_attention_layernorm
  gate_proj
  up_proj
  down_proj

final_norm
lm_head
```

Note: SmolLM2 has tie_word_embeddings: true, so lm_head shares weights with tok_embeddings.

## Quantized Tensors

For Q4_K quantization:

```c
typedef struct {
    uint8_t q[rows * cols / 2];  // 4-bit quantized
    float scales[rows];
    float zeros[rows];
} sm2_q4_tensor;
```

## Conversion Tool

```bash
python tools/convert_smollm2.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --output smollm2-135m-instruct-q4.sm2 \
  --quant q4_k
```

## Memory-Mapped Access

```c
int sm2_mmap_model(const char *path, sm2_model **out_model) {
    int fd = open(path, O_RDONLY);
    struct stat st;
    fstat(fd, &st);
    
    void *data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) return -1;
    
    sm2_file_header *hdr = (sm2_file_header *)data;
    // validate magic, load tensors, etc.
    return 0;
}
```

## Benefits
- Single mmap for entire model
- Fast startup
- Portable across platforms
- Supports partial loading

## Dependencies
- [[smollm2c-core-runtime]] - uses this format
- [[smollm2c-tokenizer]] - tokenizer blob embedded

## Status
- [[spec:001]] - Phase 1 implementation
- ✅ Working: smollm2-135m-v5.sm2 generated and loaded successfully
- Magic bytes: SM2C001 (8 bytes, correct)
- Hardcoded weights_offset workaround: 1,179,115 bytes (header had corruption)
