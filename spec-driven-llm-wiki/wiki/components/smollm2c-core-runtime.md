---
title: "smollm2c-core-runtime"
type: component
tags: [core, runtime, inference]
last_updated: 2026-04-24
---

# smollm2c-core-runtime

Core C runtime for SmolLM2 inference engine.

## Components
- [[smollm2dl-decode-layer]] - decode optimization layer
- [[smollm2d-server-daemon]] - HTTP server
- [[sm2-backend-ref]] - portable C matmul backend

## Architecture
```
smollm2.c
  model loader -> .sm2 file format
  tokenizer -> HF vocab/merges
  forward kernels -> RMSNorm, RoPE, Attention, MLP
  quantized matmul -> Q4_K, Q8_0 support
  KV cache -> paged + quantized
  sampling -> temperature, top-p
```

## Key Structs
```c
typedef struct {
    sm2_variant id;
    int n_layers, dim, hidden_dim;
    int n_heads, n_kv_heads, head_dim;
    int vocab_size, max_seq_len;
    float rms_eps, rope_theta;
} sm2_spec;
```

## Variants
| Variant | Params | Layers | Dim | Heads (Q/KV) | Context |
|---------|--------|--------|-----|--------------|---------|
| 135M | 135M | 30 | 576 | 9/3 (GQA) | 8192 |
| 360M | 360M | 32 | 960 | 15/5 (GQA) | 8192 |
| 1.7B | 1.7B | 24 | 2048 | 32/32 | 8192 |

## Memory Budget (135M, Q4_K, ctx 1024)
- Weights: ~70 MB (Q4_K)
- KV cache F16: ~23 MB
- KV cache Q8: ~12 MB
- Scratch: ~2 MB
- **Total: ~85 MB** (fits in 512 MB VPS)

## Status
- [[spec:001]] - master blueprint
- Phase 1-3 implementation target
