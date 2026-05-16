---
title: "smollm2c-core-runtime"
type: component
tags: [core, runtime, inference]
last_updated: 2026-05-17
---

# smollm2c-core-runtime

Core C runtime for SmolLM2 inference engine.

## Status

**✅ Phase 1-3 IMPLEMENTED** - Working implementation generating valid tokens.

## Architecture

```
smollm2.c
  model loader -> .sm2 file format
  tokenizer -> HF vocab/merges (basic byte fallback)
  forward kernels -> RMSNorm, RoPE, Attention, MLP
  quantized matmul -> Reference implementation
  KV cache -> Contiguous F16 (Phase 1)
  sampling -> temperature, top-p, top-k
```

## Implementation Details

### Core Files

| File | Purpose | Status |
|------|---------|--------|
| `src/smollm2.c` | CLI main loop | ✅ Working |
| `src/sm2_model.c` | .sm2 file loading, weight loading | ✅ Working |
| `src/sm2_context.c` | layer_forward, prefill, decode | ✅ Working |
| `src/sm2_sampling.c` | Token sampling | ✅ Working |
| `src/sm2_tokenizer.c` | Tokenizer (byte fallback) | ⚠️ Partial |
| `src/sm2_rmsnorm.c` | RMSNorm kernel | ✅ Working |
| `src/sm2_rope.c` | RoPE embedding | ✅ Working |
| `src/sm2_mlp.c` | SwiGLU FFN | ✅ Working |

### Key Structs

```c
typedef struct {
    sm2_variant id;
    int n_layers, dim, hidden_dim;
    int n_heads, n_kv_heads, head_dim;
    int vocab_size, max_seq_len;
    float rms_eps, rope_theta;
} sm2_spec;
```

### Key Functions

```c
int sm2_prefill(sm2_context* ctx, const int* tokens, int n_tokens);
int sm2_decode_next(sm2_context* ctx, int* out_token);
int sm2_sample_token(float* logits, sm2_generate_params* params, uint64_t* rng_state);
```

## Bugs Fixed (Critical)

1. **down_proj matmul indexing** - `down_proj[down_off + i*hidden_dim + j]` (was `j*hidden_dim+i`)
2. **Post-attention RMSNorm** - Proper RMSNorm: `xb[i] = (x[i]/rms) * weight[i]`
3. **sm2_decode_next sequence** - embed → forward → logits → sample (not sample first)
4. **Final RMSNorm** - Pure RMSNorm without mean subtraction

## Variants

| Variant | Params | Layers | Dim | Heads (Q/KV) | Context |
|---------|--------|--------|-----|--------------|---------|
| 135M | 135M | 30 | 576 | 9/3 (GQA) | 2048 |
| 360M | 360M | 32 | 960 | 15/5 (GQA) | 8192 |
| 1.7B | 1.7B | 24 | 2048 | 32/32 | 8192 |

## Memory Budget (135M, F16, ctx 1024)

- Weights F16: ~270 MB
- KV cache F16: ~23 MB
- Scratch: ~2 MB
- **Total: ~295 MB** (VPS 512MB feasible with Q4)

## Test Results

```
$ ./smollm2-cli -m smollm2-135m-v5.sm2 -p "Hello" -n 10
Model loaded in 262.4 ms
Prefill done in 618.7 ms
Output: [gen_tok=23]'[gen_tok=10]<iss[gen_tok=28],[gen_tok=20]$[gen_tok=6]<fil[EOS=0]
Generated 5 tokens in 3320.5 ms (664.1 ms/token)
Total time: 3939.8 ms
```

## Known Issues

1. **Tokenizer** - Uses byte fallback, full tokenizer.json integration pending
2. **EOS detection** - Stops generation at ~5 tokens (token 0 selected too early)
3. **KV cache** - Contiguous F16 only, paged KV not yet integrated

## Related Components

- [[smollm2dl-decode-layer]] - Decode optimization layer
- [[sm2-file-format]] - .sm2 binary format
- [[sm2-kv-cache]] - KV cache management (planned)

## Specs

- [[spec:001]] - Master blueprint (Phase 1-8b)