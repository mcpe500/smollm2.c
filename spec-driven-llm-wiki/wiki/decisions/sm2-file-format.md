# ADR-002: .sm2 File Format

**Status:** Accepted  
**Date:** 2026-05-16  
**Spec:** [[spec:001]]

## Context

C runtime cannot load HF safetensors directly. Need efficient binary format.

## Decision

Custom `.sm2` format with magic `SM2C001`, mmap-friendly layout.

## Format Structure

```c
// Header (256 bytes)
#define SM2_MAGIC "SM2C001"
typedef struct {
    char magic[8];           // "SM2C001"
    uint32_t version;        // 1
    uint32_t variant_id;     // SM2_135M/360M/1700M
    uint32_t quant_type;     // F16/Q8/Q4/Q4K
    uint32_t flags;
    uint32_t vocab_size, n_layers, dim, hidden_dim;
    uint32_t n_heads, n_kv_heads, head_dim;
    uint32_t max_seq_len;
    float rms_eps, rope_theta;
    uint32_t bos_token_id, eos_token_id, pad_token_id;
    uint64_t tokenizer_offset, tokenizer_size;
    uint64_t tensor_index_offset, tensor_index_size;
    uint64_t weights_offset, weights_size;
    uint64_t checksum;
} sm2_file_header;
```

## Tensor Order
```
tok_embeddings
for layer 0..n_layers-1:
  input_layernorm
  q_proj, k_proj, v_proj, o_proj
  post_attention_layernorm
  gate_proj, up_proj, down_proj
final_norm
lm_head (ref to tok_embeddings if tied)
tokenizer_blob
chat_template_blob
```

## Conversion
```
HF safetensors + tokenizer
        ↓
python tools/convert_smollm2.py
        ↓
smollm2-135m-instruct-q4.sm2
```

## Consequences

**Positive:**
- Single mmap for entire model
- Fast startup
- Portable across platforms

**Negative:**
- Separate conversion step
- Custom tooling needed