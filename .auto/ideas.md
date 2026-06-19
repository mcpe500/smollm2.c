# Autoresearch Ideas Backlog

## High priority
- **Multi-threaded matmul (pthreads)**: single-threaded decode uses 1 of 8 cores. Divide output rows across N threads for each matmul. Memory BW scales with threads up to DRAM limit. This is the highest-leverage remaining optimization. Implementation: add thread pool, split matmul output rows across threads. Risk: pthread overhead per call may hurt for small matmuls (kvdim=192).

- **INT8 per-block quantization**: quantize F16 weights to INT8 at load time (per-32 elements, like GGUF Q8_0). Halves weight bytes (202MB->101MB). Use vdotq_s32 (asimddp available). Risk: quantization error may change argmax=504 correctness gate.

## Medium priority  
- **INT4 quantization**: quarter the weight bytes (202MB->51MB). 4x potential speedup if purely BW-limited. Much more complex, higher quality loss.
- **Merge Makefile CC=clang default**: currently requires `make CC=clang`, simplify to use clang by default

## Low priority / explored
- -O3 -ffast-math: tried, slower
- RoPE precompute: tried, within noise
- 2-row tiling: tried, register pressure
- unroll-32: tried, register pressure  
- NEON rmsnorm: tried, within noise
- Software prefetch: tried, slower
- F16 KV cache: tried, KV fits in L2 so no BW benefit
- NEON attention: tried, within noise
- Logit projection F16: KEPT (+1 tok/s real)
