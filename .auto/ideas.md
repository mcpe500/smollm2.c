# Autoresearch Ideas Backlog

## FINAL STATE — session fully complete
All optimization paths exhausted. Current steady state: 87-95 tok/s (throttled), 100-103 tok/s (cool).
vs baseline 27.8 = +3.2-3.7x. vs Ollama 13.7 = 6.4-7.5x faster.

## What was implemented (chronological)
1. INT8 vdotq_s32 weight quantization (+79% from F16 baseline)
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
13. **2-batch FFN matmul in prefill** (+4-6 tok/s): reads gate/up/down W 3x instead of 6x for 6-token prefill

## Definitively NOT worth trying
- INT4 weight quantization: argmax fails (too coarse for 135M model)
- Threading: overhead > benefit on mobile (mutex/cond pool, spinlock, pthread_create all tried)
- -O3 -ffast-math / PGO / LTO: no gain with manual NEON intrinsics
- 2-row or 4-row matmul tiling: register pressure exceeds savings
- F16 KV cache: KV is tiny, no bandwidth benefit
- Software prefetch: HW prefetcher already optimal for sequential access
- NEON silu_mul: argmax shifts (fp rounding in matmul ctx); element-wise vsilu_mul tested = within noise
- 4-token batch FFN: 16 acc regs causes spills, slower than 2-batch
- Full 2-batch (QKV+O+FFN): QKV matrices (330KB) are L2-resident, no DRAM savings vs FFN (880KB+)
- 256-byte vdotq unroll: within noise (already at HW limit)
- 128-byte inner loop in 2-batch: register pressure (use 64-byte inner loop for 2-batch)

## Architecture facts
- matmul_q8_dot_perrow: 128-byte unroll (optimal single-row)
- matmul_q8_2batch: 64-byte inner loop (optimal 2-batch, 12 regs = 4 W + 8 acc)
- Sweet spot: 2-batch for large matrices (gate/up/down at 880KB each), single-row for small (QKV at 110-330KB)

## Hardware limit
- Effective bandwidth: ~10.5 GB/s (measured)
- LPDDR5 single-core limit: ~10-12 GB/s
- Weight bytes per decode step: ~78MB (INT8)
- Theoretical max: 78MB / (1/10.5GB/s) = 135 tok/s (never achievable: decode is sequential)
- Current: 87-95 tok/s throttled = ~90% of theoretical
