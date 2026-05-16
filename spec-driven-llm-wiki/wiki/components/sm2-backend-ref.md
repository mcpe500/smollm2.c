---
title: "sm2-backend-ref"
type: component
tags: [backend, matmul, quantization, reference]
last_updated: 2026-04-24
---

# sm2-backend-ref

Portable C reference matmul backend for SmolLM2. Provides pure-C implementations of quantized matrix multiplication kernels.

## Dependencies
- [[smollm2c-core-runtime]] - calls into matmul

## Supported Quantization Formats
| Format | Bits/Weight | Description |
|--------|-------------|-------------|
| F32 | 32 | Reference float32 |
| F16 | 16 | Half-precision |
| Q8_0 | 8 | Symmetric 8-bit quant |
| Q4_K | 4.5 | K-quant 4-bit (block) |

## Kernel Implementations
```c
// Core matmul dispatch
void sm2_matmul(sm2_tensor* out, sm2_tensor* a, sm2_tensor* b, sm2_backend* backend);

// Quantized kernels  
void sm2_matmul_q4_k(float* out, sm2_q4_k* a, sm2_q4_k* b, int m, int n, int k);
void sm2_matmul_q8_0(float* out, sm2_q8_0* a, sm2_q8_0* b, int m, int n, int k);
```

## Performance Notes
- Reference impl: portable C, no arch-specific intrinsics
- Q4_K uses block scaling (blocksize 32) for accuracy
- Q8_0 uses per-block symmetric scaling
- Fallback: F32 matmul always available

## Extension Points
- CUDA backend: implement `sm2_backend` vtable
- Metal backend: implement `sm2_backend` vtable
- ARM NEON: optimize Q4_K dot product

## Status
- [[spec:001]] - backend interface spec'd
- Phase 1-2 implementation
