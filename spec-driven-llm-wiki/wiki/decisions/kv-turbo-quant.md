# ADR-004: KV Turbo Quantization

**Status:** Accepted  
**Date:** 2026-05-16  
**Spec:** [[spec:001]]

## Context

KV cache becomes memory bottleneck at high concurrency/long context. KIVI paper shows 2-bit quantization with asymmetric K/V reduces memory 2.6x and improves throughput 2.35x-3.47x.

## Decision

Implement layered KV quantization:

| Mode | K Quant | V Quant | Memory | Quality |
|------|---------|---------|--------|---------|
| F16 | none | none | 100% | 100% |
| Q8 | per-token | per-token | 50% | ~99% |
| Q4 | per-token | per-token | 25% | ~95% |
| TURBO2 | per-channel | per-token | 12.5% | ~90% |
| MIXED | mixed | mixed | ~30% | ~97% |

## TURBO2 Layout (KIVI-like)
```
K cache: quantize per-channel group (group=32, 2 groups/head)
V cache: quantize per-token
Recent N tokens: F16 or Q8 (residual window)
```

## Config Options
```bash
--kv-dtype mixed \
--kv-recent 64 \      # residual window size
--kv-recent-dtype q8 \
--kv-old turbo2
```

## Memory Calculation (135M)
```
Base: 30 layers × 3 KV heads × 64 dim × 2 bytes × ctx
     = 23,040 bytes/token ≈ 22.5 KB/token

F16 ctx 1024:  23 MB
Q8  ctx 1024:  12 MB
TURBO2 ctx 1024: ~6 MB
```

## Consequences

**Positive:**
- 2-4x memory reduction
- Higher throughput
- VPS 512 MB viable

**Negative:**
- Quality degradation at TURBO2
- Complex dequantization
- Must tune residual window

## References
- KIVI: arXiv:2402.02750
- jy-yuan/KIVI GitHub