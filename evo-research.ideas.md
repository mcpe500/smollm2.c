# Evo-Research Ideas

## Deferred Optimizations (Not Pursued Yet)

### High Priority
- **DEBUG_REMOVAL**: Strip fprintf from decode layer (candidates c001-c003)
- **NEON_SIMD**: Enable ARM NEON in matmul loops
- **ROPE_OPT**: Precompute RoPE frequencies at model load

### Medium Priority
- **F16_FAST**: Fast half-precision conversion with lookup table
- **MATMUL_UNROLL**: Inner loop unrolling 4x
- **KV_CACHE**: Implement KV cache for attention reuse

### Low Priority
- **QUANTIZATION**: Q4/Q8 weight quantization
- **FLASH_ATTN**: Flash attention in decode path
- **BATCH_DECODE**: Process multiple sequences

## Dead Ends
- None yet

## Observations
- Baseline: ~8-9 tok/s on ARM64 Android
- Run-to-run variance is high (7-15 tok/s)
- Need 3+ runs per candidate for confidence