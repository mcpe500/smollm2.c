---
title: "smollm2dl-decode-layer"
type: component
tags: [decode, optimization, layer, dflash]
last_updated: 2026-05-16
---

# smollm2dl-decode-layer

Decode optimization layer for smollm2.c. The "dl" stands for "decode layer" - this is the critical path for fast token generation.

## Parent
- [[smollm2c-core-runtime]]

## Subcomponents
- [[sm2dl-decode-core]] - core decode loop
- [[sm2dl-flash-decode]] - flash decode implementation
- [[sm2dl-paged-attention]] - paged KV attention
- [[sm2dl-kv-quant]] - KV quantization
- [[sm2dl-batch-decode]] - batch decode support
- [[sm2dl-speculative]] - speculative decode (Phase 8a)
- [[sm2-dflash-module]] - DFlash block diffusion (Phase 8b)

## Design Philosophy

**Decode-first, not forward-pass-first.**

Unlike generic forward pass implementations, smollm2dl optimizes specifically for:
- Single token generation per step
- No allocation in hot path
- Online softmax without score buffer
- Paged KV read
- Streaming token output

## Hard Rules

```
malloc/free FORBIDDEN inside decode_next
JSON parsing FORBIDDEN inside decode_next
token string allocation FORBIDDEN inside decode_next
full attention vector allocation FORBIDDEN
```

## Core Loop

```c
int sm2dl_decode_next(sm2_context *ctx, sm2_generate_params *p) {
    // 1. Single token forward (no allocation)
    sm2_forward_one_token(ctx);
    
    // 2. Sample from logits
    int token = sm2_sample(ctx->logits, p);
    
    // 3. Stream token immediately
    ctx->last_token = token;
    return token;
}
```

## Flash Decode Algorithm

For each q_head (with kv_head = q_head / group_size):

```c
float m = -INFINITY, l = 0.0f;
float acc[64] = {0};

for each kv_page:
    for each token in page:
        score = dot(q, K[kv_head][token]) * scale;
        new_m = max(m, score);
        alpha = expf(m - new_m);
        beta = expf(score - new_m);
        
        // Online softmax accumulation
        for d in 0..63:
            acc[d] = acc[d] * alpha + V[token][d] * beta;
        
        l = l * alpha + beta;
        m = new_m;

out = acc / l; // Normalize
```

## Speculative Decoding (Phase 8a)

Standard autoregressive draft model:

```c
typedef struct {
    sm2_model *draft_model;
    sm2_model *target_model;
    int draft_tokens;       // 2, 4, 8
    float accept_threshold;
} sm2_spec_decode_config;
```

Flow: 135M draft -> 360M/1.7B verifier

## DFlash Integration (Phase 8b)

DFlash (z-lab/dflash) block diffusion for parallel drafting:

```c
typedef struct {
    sm2_model *draft_model;     // DFlash diffusion draft
    sm2_model *target_model;    // Target SmolLM2
    int num_draft_tokens;       // 16 default
    int block_size;             // 16
} sm2_dflash_config;
```

DFlash generates ~16 tokens in parallel via diffusion, then target verifies in one pass.

**Reference:** https://github.com/z-lab/dflash

## Files

```
src/decode/
  sm2dl_decode.c           - core decode loop
  sm2dl_flash_decode.c     - flash decode implementation
  sm2dl_paged_attention.c  - paged KV attention
  sm2dl_kv_quant.c         - KV quantization
  sm2dl_batch_decode.c     - batch decode support
  sm2dl_speculative.c      - speculative decode
  sm2dl_dflash.c           - DFlash block diffusion

src/dflash/
  sm2_dflash.c             - block diffusion forward
  sm2_dflash_model.c       - draft model loader
  sm2_dflash_verify.c      - verification logic
```

## Performance Targets

| Mode | Target | Configuration |
|------|--------|----------------|
| Prefill | >100 tok/s | 135M Q4_K |
| Decode | >30 tok/s | 135M Q4_K |
| Speculative | >60 tok/s | 135M -> 360M |
| DFlash | >180 tok/s | DFlash-135M -> 360M |

## Status
- [[spec:001]] - Phase 3 (core), Phase 8a/8b (speculative)
- Critical path for <50ms/token on 135M
- DFlash Phase 8b is research (requires trained draft model)

## Dependencies
- [[sm2-kv-cache]] - KV cache management
- [[sm2-backend-ref]] - matmul kernels
- [[dflash-integration]] - ADR for DFlash