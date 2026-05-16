# ADR-006: Weight Quantization Strategy

**Status:** Accepted  
**Date:** 2026-05-16  
**Spec:** [[spec:001]]

## Context

We need to decide on weight quantization formats for smollm2.c. The goal is to support VPS 512 MB (needs aggressive quantization) while maintaining quality for server deployments.

## Decision

Support these quantization formats in order of priority:

### Priority 1: F16 (Baseline)
- Import only, not default
- Reference for quality comparison
- Full precision for GPU serving

### Priority 2: Q4_K (Default for VPS)
- 4-bit with K-quantization (per-channel scales)
- ~70MB for 135M (vs 135MB F16)
- Standard in llama.cpp ecosystem
- Quality: ~95% of F16

### Priority 3: Q8_0 (High Quality Option)
- 8-bit with per-tensor quantization
- ~85MB for 135M
- Quality: ~99% of F16
- Good balance for central servers

### Priority 4: Q5_K (Optional)
- 5-bit K-quantization
- ~78MB for 135M
- Quality: ~97% of F16
- Between Q4 and Q8

## Format Specifications

### Q4_K Structure

```c
typedef struct {
    uint8_t q[rows * cols / 2];   // 4-bit packed (2 values per byte)
    float scales[rows];           // per-row scales
    float zeros[rows];            // per-row zero points
} sm2_q4k_tensor;
```

### Q8_0 Structure

```c
typedef struct {
    int8_t q[rows * cols];        // 8-bit values
    float scale;                  // per-tensor scale
} sm2_q8_tensor;
```

### Q5_K Structure

```c
typedef struct {
    uint8_t q[rows * cols * 5 / 8];  // 5-bit packed
    float scales[rows];
    float zeros[rows];
} sm2_q5k_tensor;
```

## Memory Budgets

| Model | F16 | Q8_0 | Q5_K | Q4_K |
|-------|-----|------|------|------|
| 135M | 135MB | 85MB | 78MB | 70MB |
| 360M | 360MB | 225MB | 207MB | 185MB |
| 1.7B | 1.7GB | 1.1GB | 1.0GB | 0.9GB |

## Recommended Defaults

### VPS 512 MB (135M only)
```
--quant q4_k
--ctx 1024
--threads 1
```

### Central Server (360M)
```
--quant q4_k
--kv-dtype q8
--ctx 2048
```

### Premium Server (1.7B GPU)
```
--quant f16
--kv-dtype q8
--ctx 4096
--gpu
```

## Matmul API

```c
void sm2_matmul_q(
    float *out,
    const float *x,
    const sm2_qtensor *w,
    int rows,
    int cols,
    sm2_backend backend
);
```

## Backend Support

| Backend | Q4_K | Q8_0 | Q5_K | F16 |
|---------|------|------|------|-----|
| Portable C | Yes | Yes | Yes | Yes |
| AVX2 | Yes | Yes | Yes | Yes |
| AVX512 | Yes | Yes | Yes | Yes |
| NEON | Yes | Yes | Yes | Yes |
| WASM SIMD | Yes | Yes | No | Yes |
| CUDA | Yes | Yes | Yes | Yes |

## Consequences

**Positive:**
- VPS 512 MB viable with Q4_K
- Server memory reduced 50%+
- Multiple quality/performance tradeoffs

**Negative:**
- Quantization error accumulates
- Quality degradation at aggressive settings
- More complex matmul kernels

## See Also
- [[smollm2c-core-runtime]]
- [[sm2-backend-ref]]
- [[low-memory-mode]]