# Evo-Research: SmolLM2 Inference Optimization

## Primary Metric
- **tok/s** (tokens per second) — higher is better
- Target: 100 tok/s (from baseline ~9 tok/s)

## Experiment Rules

1. **No benchmark overfitting** — Changes must improve real inference, not gaming the benchmark
2. **One change at a time** — Each candidate modifies one optimization
3. **Verify correctness** — Generated text must remain coherent and readable
4. **Keep improvements only** — Discard if tok/s decreases
5. **Seed diverse families** — Start with 6+ different optimization approaches

## Candidate Families (Genome Structure)

Each candidate has:
- `family`: Optimization category
- `operator`: Specific transformation
- `hypothesis`: Why this should help
- `genome`: What to change and where

### Family Categories

1. **DEBUG_REMOVAL** — Strip debug prints from hot path
2. **NEON_SIMD** — Enable ARM NEON vectorization in matmul
3. **ROPE_OPT** — Precompute RoPE frequencies, cache cos/sin
4. **F16_FAST** — Fast half-precision conversion with lookup tables
5. **MATMUL_UNROLL** — Loop unrolling and accumulator optimization
6. **KV_CACHE** — Enable KV cache for attention reuse
7. **QUANTIZATION** — Weight quantization (Q4/Q8) for faster matmul
8. **MEMORY_LAYOUT** — Optimize data locality (AoS vs SoA)
9. **FLASH_ATTN** — Integrate flash attention in decode path
10. **BATCH_DECODE** — Process multiple tokens simultaneously

## ASI Schema (per experiment)
```json
{
  "candidate_id": "c001",
  "family": "DEBUG_REMOVAL",
  "operator": "strip_fprintf_decode",
  "hypothesis": "Debug prints add ~10ms overhead per token",
  "genome": {"file": "src/decode/sm2dl_decode.c", "change": "remove fprintf calls"},
  "baseline_tok_s": 9.3,
  "result_tok_s": 12.1,
  "improvement_pct": 30.1,
  "correctness": "PASS"
}
```

## Hot Path Files (Priority Order)
1. `src/sm2_context.c` — Main decode loop
2. `src/decode/sm2dl_decode.c` — Decode layer with DEBUG prints
3. `src/sm2_rope.c` — RoPE calculation (called per layer per token)
4. `src/smollm2.c` — F16 conversion in main
5. `src/sm2_matmul_ref.c` — Reference matmul implementation

## Experiment Loop

```
FOR each candidate in population:
    1. Apply genome changes to codebase
    2. Rebuild (make clean && make)
    3. Run benchmark: ./bench_run -n 50 -w 3
    4. If tok_s > baseline:
        - Keep change
        - Update population with new tok_s
        - Commit: "evo: {family} {operator} +{pct}% tok/s"
    5. Else:
        - Revert change
        - Mark candidate as failed

SELECT next generation:
    - Keep top performers by family
    - Mutate best candidates
    - Introduce random new candidates
    - Log ASI for each
```

## Success Criteria

| Phase | Target | Families |
|-------|--------|----------|
| Phase 1 | >20 tok/s | DEBUG_REMOVAL, NEON_SIMD |
| Phase 2 | >40 tok/s | ROPE_OPT, F16_FAST |
| Phase 3 | >70 tok/s | MATMUL_UNROLL, KV_CACHE |
| Phase 4 | >100 tok/s | QUANTIZATION, FLASH_ATTN |

## Anti-Cheating Rules

- Do NOT modify the benchmark tool itself
- Do NOT hardcode token values that bypass model computation
- Do NOT skip actual inference steps
- Verify output text is coherent (not garbage tokens)
- Changes must work on actual model weights

## Commit Format

```
evo: {family} {operator} +{pct}% ({tok_s} tok/s)
```

Example:
```
evo: DEBUG_REMOVAL strip_fprintf_decode +31% (12.2 tok/s)
```