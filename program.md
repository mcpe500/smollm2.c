# SmolLM2 Autoresearch - Agent Instructions

You are optimizing SmolLM2 C inference for speed.
**Current baseline: ~1-2 tok/s**
**Target: 100 tok/s**

## Experiment Rules

1. **One change at a time** - Don't mix multiple optimizations
2. **Measure before/after** - Run benchmark to verify speedup
3. **Only keep improvements** - Revert if tok/s decreases
4. **Focus on hot path** - `sm2_decode_next()` in src/sm2_context.c

## Benchmark Command

```bash
cd benchmarks && make && ./benchmark -m ../smollm2-135m.sm2 -n 50
```

## Priority Order (Experiment Sequence)

### 🔴 Phase 1: Quick Wins (Expected: 10x speedup)

1. **Remove DEBUG prints** from `src/decode/sm2dl_decode.c`
   - Lines 221, 224, 229, 233, 249, 253, 257, 262, 265, 269
   - Each fprintf adds ~10ms overhead per token
   - Expected: 4x speedup

2. **Enable NEON backend** in `Makefile`
   - Change `SRC_BACKEND = $(SRC)backend/sm2_backend_ref.c`
   - To `SRC_BACKEND = $(SRC)backend/sm2_backend_neon.c`
   - If NEON fails, keep ref backend
   - Expected: 2-4x speedup on ARM

3. **Remove Kahan summation** in `src/sm2_context.c`
   - Simplify matmul loops (lines 225-235)
   - Use simple accumulation: `sum += a * b`
   - Expected: 1.5x speedup

### 🟡 Phase 2: F16 Optimization (Expected: 3-5x speedup)

4. **Implement fast F16->F32** in `src/smollm2.c`
   - Replace `sm2_f16_to_float()` with table lookup
   - Create 65536-entry float table at startup
   - Expected: 2x speedup

5. **Optimize NEON F16 conversion** in `src/backend/sm2_backend_neon.c`
   - Use proper NEON intrinsics for batch conversion
   - Expected: additional 2x

### 🟡 Phase 3: RoPE Optimization (Expected: 2-3x speedup)

6. **Precompute RoPE frequencies** in `src/sm2_rope.c`
   - Build `freq_table[head_dim/2]` once at model load
   - Replace `powf(theta, x)` with `freq_table[i] * pos`
   - Expected: 3x speedup

7. **Cache RoPE cos/sin** in context
   - Store precomputed cos/sin for each position
   - Avoid recalculating same positions
   - Expected: 1.5x speedup

### 🟢 Phase 4: Advanced Optimizations (Expected: 2-4x speedup)

8. **Integrate flash attention** from `src/decode/sm2dl_flash_decode.c`
   - Use online softmax in decode path
   - Expected: 2x speedup

9. **Enable KV quantization** in `src/sm2_context.c`
   - Use Q8 KV cache instead of F16
   - Halve memory bandwidth
   - Expected: 1.5x speedup

10. **SIMD matmul unrolling** in `src/sm2_context.c`
    - Unroll inner loops 4x
    - Use temp accumulator to hide loads
    - Expected: 2x speedup

### 🚀 Phase 5: Expert (Target: 100 tok/s)

11. **Batch decode** - Process multiple sequences
12. **Speculative decoding** - Draft-verifier pattern
13. **Quantized weights** - Q4/Q8 matmul in hot path

## Success Criteria

| Phase | Target tok/s | Notes |
|-------|-------------|-------|
| Baseline | ~1-2 | Current |
| Phase 1 | >10 | Quick wins |
| Phase 2 | >30 | F16 optimization |
| Phase 3 | >60 | RoPE optimization |
| Phase 4 | >80 | Advanced |
| Phase 5 | >100 | Expert |

## Files to Modify

```
Critical files (hot path):
- src/sm2_context.c       # Main decode loop
- src/decode/sm2dl_decode.c # Decode layer
- src/smollm2.c            # F16 conversion
- src/sm2_rope.c           # RoPE calculations

Build files:
- Makefile                 # Enable NEON backend

Backend files:
- src/backend/sm2_backend_neon.c # NEON SIMD
- src/backend/sm2_backend_ref.c    # Reference impl
```

## Experiment Loop

```
WHILE current_tok_s < 100:
    1. Pick next optimization from Priority Order
    2. Make the change
    3. Run: make clean && make && ./benchmark -m smollm2-135m.sm2 -n 50
    4. IF tok_s increased:
         - Keep change
         - Commit with message: "perf: <optimization_name> +X tok/s"
    5. ELSE:
         - Revert change
         - Continue to next optimization
    6. Report new tok/s

GIVE UP after 10 consecutive failures
```

## Tips

- Run benchmark 3 times and take average
- Ignore first run (cold cache)
- Watch for variance >20% between runs
- If NEON backend fails to compile, revert and skip
- Some optimizations may need to be combined (F16 + NEON)

## Exit Condition

STOP when tok/s >= 100 OR all optimizations exhausted.

Report final tok/s and list of optimizations that worked.