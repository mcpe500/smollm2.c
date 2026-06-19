# Autoresearch Ideas Backlog

## FINAL STATE — session complete
All major optimizations implemented. Current steady state: 100-105 tok/s.
vs baseline 27.8 = +271%. vs Ollama 13.7 = 7.5x faster.

## What was implemented
1. INT8 vdotq_s32 weight quantization (x3 from 27.8 to ~50 tok/s)
2. NEON activation quantize (quantize_row_to_q8_tensor)
3. NEON rmsnorm
4. NEON attention QK dot + value weighted sum
5. Precomputed RoPE cos/sin table
6. fast_expf (Schraudolph bit trick) for SiLU and softmax
7. INT8 prefill path (eliminates F16 cache pollution)
8. Free F16 weights after INT8 quantization (saves 202MB)
9. Per-tensor activation quantization (faster than per-block-64)
10. Per-row weight scale (tighter vdotq loop, no per-block hadd)
11. 2x unroll matmul inner loop (128 bytes/iter)
12. Dead code cleanup

## Definitively NOT worth trying
- INT4 weight quantization: argmax fails (too coarse for 135M model)
- Threading: overhead > benefit on mobile (all forms tried)
- -O3 -ffast-math: no gain with manual NEON intrinsics
- 2-row matmul tiling: register pressure
- F16 KV cache: KV is tiny, no bandwidth benefit
- Software prefetch: HW prefetcher already optimal for sequential access

## Theoretical remaining headroom
- We're at ~10.5 GB/s effective DRAM bandwidth
- Single-core LPDDR5 limit: ~10-12 GB/s
- Essentially at the hardware limit

## Session continuation (post-compaction)
- **NEON silu_mul**: argmax shifts 504->57 due to vfmaq FP rounding difference accumulating over 30 layers. DEAD END.
- **256-byte vdotq unroll**: within noise, no measurable gain on throttled device. DEAD END.
- **Device thermal state**: consistently throttled at 50-64% max CPU freq during extended sessions. Steady-state ~85-87 tok/s throttled, 100-103 tok/s cool.
- **Correctness gate**: argmax=504 ('The') for 4-token '<|im_start|>assistant\n' prefix. Must be preserved.
- **Conclusion**: All optimization paths exhausted. Session complete at 100-103 tok/s peak (3.6-3.7x baseline, 7.4x Ollama).
