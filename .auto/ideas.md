# Autoresearch Ideas Backlog

## High priority
- **INT4 quantization for large matmuls**: gate/up/logit matrices are 39%+21%=60% of weight bytes.
  INT4 for those only, keep INT8 for q/k/v/o (small matrices where INT4 is slower).
  Micro-bench: logit 1.34x, gate+up 1.13x faster. Net ~8-10% overall improvement.
  Risk: argmax=504 may change with INT4 quantization noise.
  Implementation: nibble-packed INT4 + matmul_q4_dot using vshrq/vandq unpack + vdotq_s32.
  Status: NOT YET TRIED on actual model weights.

## Medium priority
- **Free F32 token embeddings after use**: w_token_embd is 113MB F32. Used only for embedding lookup.
  Could convert to F16 to save 56MB. Embedding lookup precision loss is negligible (no accumulation).
  Status: NOT TRIED.

- **Skip F16 weight allocation entirely**: currently we alloc F16, load from GGUF, quantize INT8, free F16.
  If GGUF API supported row-by-row reading, could quantize directly without 202MB intermediate.
  Complex: requires GGUF parser changes. Status: NOT TRIED.

## Low priority / explored
- -O3 -ffast-math: tried, slower
- RoPE precompute: tried, within noise
- 2-row tiling: tried, register pressure
- unroll-32: tried, register pressure  
- NEON rmsnorm: KEPT (small gain)
- NEON quantize: KEPT (small gain)
- NEON attention: KEPT (small gain)
- Software prefetch: tried, slower
- F16 KV cache: tried, KV fits in L2 so no BW benefit
- Logit projection F16: KEPT (+1 tok/s real, now superseded by INT8)
- INT8 vdotq_s32 per-block-64: KEPT (+79% from 27.8->52 tok/s)
- INT8 prefill path: KEPT (+30% from 56->75 tok/s, eliminates F16 cache pollution)
- Free F16 weights: KEPT (202MB savings, further cache improvement)
- Threading (pthread pool, spinlock): tried, all slower (sync overhead > BW gain)
- 2-row matmul tiling: slower (register pressure)
- vmlaq_n block accumulation: no gain (compiler already does similar)
- Block size 128 vs 64: not measurably faster
