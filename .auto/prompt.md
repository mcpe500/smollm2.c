# Autoresearch: smollm2.c — maximize decode tok/s + close quality gap vs Ollama

## Objective
Optimize the C inference engine (smollm2.c) for:
1. **Primary**: decode tok/s (higher is better)
2. **Secondary**: output quality vs Ollama reference (greedy temp=0, 30 tokens, prompt "Hello, how are you?")

Ollama reference (temp=0): "I'm doing great! How can I help you today?" — 13 tokens, ~21 tok/s on this device.
Our baseline: ~24 tok/s decode (temp=0.3 top-k=5), output coherent but no system prompt / assistant persona.

## Metrics
- **Primary**: `tok_s` (tok/s, higher is better) — decode throughput on 50-token generation
- **Secondary**: `prefill_toks` — prefill speed on 6-token prompt

## How to Run
`./.auto/measure.sh` — outputs `METRIC name=number` lines.

## Files in Scope
- `src/forward.c` — transformer forward pass, matmul, RoPE, attention, FFN (main perf target)
- `src/sampling.c` — token sampling (top-k, top-p, temperature)
- `src/main.c` — CLI entry point
- `src/tokenizer.c` — tokenizer
- `Makefile` — build config

## Off Limits
- `src/gguf.c` / `src/gguf.h` — GGUF parser (don't touch)
- `src/tui.c` / `src/web.c` — UI code (don't touch unless needed)
- Do NOT cheat benchmarks (hardcode outputs, skip computation, etc.)
- Do NOT overfit to benchmark prompts

## Constraints
- Must build with `make CC=clang` with no errors/warnings
- `./smollm2 --inspect` must succeed and show correct model metadata
- `./smollm2 --logits $'<|im_start|>assistant\n'` argmax must remain 76 (correctness gate)
- No new external dependencies
- NEON intrinsics are fine (already used), stay ARM64-compatible

## Architecture Facts (don't re-derive)
- SmolLM2-135M: dim=576, n_layers=30, n_heads=9, n_kv_heads=3, head_dim=64, ffn_hidden=1536, vocab=49152
- GQA: q_head h uses kv_head h/3 (h * nkv / nh = h/3)
- rope_theta=100000 (from GGUF, NOT 10000)
- Tied embeddings: logits = x_norm @ w_token_embd^T (no output.weight tensor)
- Weights stored F16 in GGUF, loaded as F16 in memory, converted on-the-fly in NEON matmul
- w_token_embd stored F32 (converted at load), used for both embedding lookup and logit projection
- KV cache: [n_layers × max_seq × kv_dim] F32
- NEON matmul already implemented with 4-accumulator unroll-16 in `matmul()` (F16 weights)
- EOS suppression: tokens 1 and 2 suppressed in sample_token() to prevent early stop

## Current Bottlenecks (profiling insight)
- `matmul()` is the hot loop: called 30×(q+k+v+o+gate+up+down) = 210 times per decode step
- Shapes: q/k/v/o are [576×576] or [192×576], gate/up are [1536×576], down is [576×1536]
- RoPE: cosf/sinf per token per head — could be precomputed
- Attention softmax: expf in inner loop
- `matmul_f32` for logit projection: vocab=49152 rows × 576 cols — expensive!

## What's Been Tried
- NEON F16 matmul with 4 accumulators, unroll-16: ~24 tok/s (was ~5 tok/s scalar)
- top-p rewritten with qsort
- EOS suppression for natural generation
- F32 token_embd (needed for tied logit projection accuracy)
- **KEPT**: F16 logit projection (w_token_embd_f16): +1 tok/s → baseline now 28.8 tok/s
- -O3 -ffast-math: no gain (within noise)
- Precompute RoPE table: no gain (RoPE not bottleneck)
- 2-row matmul tiling: slower (register pressure)
- unroll-32 with vld1q_f16: slower (register pressure)
- NEON attention QK dot + value accum: noisy, no clear gain
- matmul_f32 4-acc upgrade: within noise
- NEON rmsnorm: within noise
- Software prefetch (distance=2): slower (HW prefetcher already handles sequential)
- F16 KV cache: slower (KV cache fits in L2, F32→F16 overhead outweighs bandwidth saving)

## Benchmark Context
- **Our engine**: 28.8 tok/s baseline (F16 logit proj commit)
- **Ollama smollm2:135m**: 13.7 tok/s avg (same device, CPU inference)
- **We are already 2.1x faster than Ollama**
- Noise floor: ~2 tok/s. Need >30 tok/s to be a clear win over baseline.

## Ideas to Try
- Precompute RoPE sin/cos table at load time
- Batch attention score computation with NEON dot products
- Lazy logit projection: only compute top-k logits instead of full vocab
- Loop unrolling for attention accumulation
- Prefetch hints for weight access
- matmul_f32 (logit projection) with NEON — already F32, might benefit from unroll
- Try -O3 -ffast-math in CFLAGS
- Try multi-accumulator matmul (8 accumulators instead of 4)
- Quantize KV cache to F16 to reduce memory bandwidth
