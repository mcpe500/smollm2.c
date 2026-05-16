---
title: "Project Overview"
type: synthesis
tags: [overview]
last_updated: 2026-05-17
---

# Project Overview

smollm2.c is a decode-first inference engine for SmolLM2 models (135M, 360M, 1.7B) written in portable C.

## Vision

High-performance, memory-efficient LLM inference that runs on VPS 512 MB. SmolLM2 family deserves a purpose-built runtime optimized for decode, not a generic forward-pass implementation.

## Architecture

```
smollm2.c
  smollm2c-core-runtime    - Core C runtime
  smollm2dl-decode-layer   - Decode optimization (smollm2dl)
  smollm2d-server-daemon  - HTTP server with OpenAI API
  sm2-kv-cache            - Paged KV cache
  sm2-file-format         - Custom .sm2 binary format
  sm2-backend-ref         - Portable matmul backend
  sm2-dflash-module       - DFlash block diffusion (Phase 8b)
```

## Key Features

| Feature | Phase | Description |
|---------|-------|-------------|
| .sm2 Format | 1 | Custom binary format with mmap |
| F16 Forward | 1 | Reference forward pass |
| Q4_K Quantization | 2 | 4-bit weight quantization |
| No-malloc Decode | 3 | Preallocated buffers |
| Flash Decode | 3 | Online softmax attention |
| Paged KV | 4 | Block-based KV cache |
| KV Turbo Quant | 5 | KIVI-like 2-bit KV cache |
| Server API | 6 | OpenAI-compatible endpoints |
| Speculative Decoding | 8a | 135M -> 360M/1.7B draft |
| DFlash | 8b | Block diffusion (research) |

## Memory Targets

| Config | Memory | Use Case |
|--------|--------|----------|
| 135M Q4_K + KV Q8 | ~85 MB | VPS 512 MB |
| 360M Q4_K + KV Q8 | ~200 MB | Central worker |
| 1.7B Q4_K + KV Q8 | ~900 MB | Premium server |

## Project Status

### ✅ Phase 1-3 Implementation COMPLETE

**Working Implementation:**
- `smollm2-cli` binary compiles and runs
- Model generates valid tokens (tested with "Hello", "Hi", "Hello world")
- Generate ~5 tokens per run before EOS
- No segfault on any input length

**Key Files:**
- `src/sm2_context.c` - Core layer_forward, RMSNorm, attention, FFN matmul
- `src/smollm2.c` - CLI main loop
- `src/sm2_sampling.c` - Token sampling
- `smollm2-135m-v5.sm2` - Working model file (270 MB)

**Bugs Fixed (Critical):**
1. **down_proj matmul indexing** - Was `j*hidden_dim+i`, fixed to `i*hidden_dim+j` (transposed)
2. **Post-attention RMSNorm** - Was simple `xb[i] * weight[i]`, fixed to proper RMSNorm formula
3. **sm2_decode_next sequence** - Now: embed → forward layers → compute logits → sample (correct order)
4. **Final RMSNorm** - Was using LayerNorm (mean subtraction), fixed to pure RMSNorm

### ⚠️ Known Issues

1. **Tokenizer uses byte fallback** - Full tokenizer.json integration not complete
2. **EOS detection stops at 5 tokens** - Token 0 (`<|endoftext|>`) selected too early
3. **Generation stops prematurely** - Should generate more tokens per request

### 📋 Remaining Work

1. Fix tokenizer integration (load tokenizer.json properly)
2. Fix EOS detection to allow longer generation
3. Verify output quality against HuggingFace reference
4. Test edge cases (single char, special characters)

## Tech Stack

- C99 portable (no dependencies)
- Optional: AVX2, AVX512, NEON, WASM SIMD
- Optional: CUDA, Metal GPU acceleration
- Python tools for model conversion

## Dependencies

- [[smollm2c-core-runtime]]
- [[smollm2dl-decode-layer]]
- [[smollm2d-server-daemon]]
- [[sm2-kv-cache]]
- [[sm2-file-format]]
- [[sm2-backend-ref]]

## Related Specs

- [[spec:001]] - smollm2c-master-blueprint

## References

- SmolLM2 Models: https://huggingface.co/collections/HuggingFaceTB/smollm2
- DFlash: https://github.com/z-lab/dflash
- KIVI KV Cache: https://github.com/jy-yuan/KIVI
- FlashAttention: https://github.com/flashinfer-ai/flashinfer