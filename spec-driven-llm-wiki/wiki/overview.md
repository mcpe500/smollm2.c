---
title: "Project Overview"
type: synthesis
tags: [overview]
last_updated: 2026-05-16
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

- [[spec:001]] - Master blueprint (Phase 1-8b)
- 6 components defined
- 6 architecture decisions (ADRs)
- 4 patterns documented
- Implementation: Not started

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

## Next Steps

1. Implement .sm2 file loader (Phase 1)
2. Implement tokenizer (Phase 1)
3. Implement F16 forward pass (Phase 1)
4. Verify logits vs HuggingFace reference

## References

- SmolLM2 Models: https://huggingface.co/collections/HuggingFaceTB/smollm2
- DFlash: https://github.com/z-lab/dflash
- KIVI KV Cache: https://github.com/jy-yuan/KIVI
- FlashAttention: https://github.com/flashinfer-ai/flashinfer
