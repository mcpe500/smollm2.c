# Pattern: Speculative Decoding

**For:** [[spec:001]], [[smollm2dl-decode-layer]]  
**Type:** Decoding Pattern

## Overview

Speculative decoding uses a smaller/faster draft model to predict multiple tokens, which are then verified by the larger target model in parallel. This achieves lossless speedup.

## Architecture

```
Target Model: SmolLM2-360M or 1.7B
Draft Model: SmolLM2-135M

Decode Loop:
  1. Draft generates K tokens autoregressively
  2. Target verifies all K tokens in one prefill-like pass
  3. Accept matching prefix
  4. On rejection, sample from target distribution
  5. Continue
```

## Standard Speculative (Phase 8a)

```c
int sm2_spec_decode(sm2_context *target_ctx,
                    sm2_context *draft_ctx,
                    sm2_generate_params *params,
                    int *out_token) {
    // Step 1: Draft generates K tokens
    int draft_tokens[K_MAX];
    for (int i = 0; i < K_MAX; i++) {
        sm2_forward_one_token(draft_ctx);
        draft_tokens[i] = sm2_sample(draft_ctx->logits, params);
        sm2_kv_append(draft_ctx->kv_pool, &draft_ctx->kv, 
                     draft_tokens[i]);
    }
    
    // Step 2: Target verifies in one pass
    float target_logits[K_MAX];
    sm2_verify_batch(target_ctx, draft_tokens, K_MAX, target_logits);
    
    // Step 3: Accept/reject
    int accept_len = 0;
    for (int i = 0; i < K_MAX; i++) {
        int draft_tok = draft_tokens[i];
        float draft_p = softmax(draft_logits[i])[draft_tok];
        float target_p = softmax(target_logits[i])[draft_tok];
        
        if (target_p >= draft_p) {
            accept_len = i + 1;
        } else if (rand() < target_p / draft_p) {
            accept_len = i + 1;
        } else {
            break;
        }
    }
    
    // Step 4: Return accepted token
    if (accept_len > 0) {
        *out_token = draft_tokens[accept_len - 1];
    } else {
        *out_token = sm2_sample(target_logits[0], params);
    }
    
    return accept_len;
}
```

## DFlash-Style Block Diffusion (Phase 8b)

DFlash generates blocks of tokens in parallel using diffusion:

```c
int sm2_dflash_block(sm2_context *ctx,
                     sm2_dflash_config *cfg,
                     int *out_tokens) {
    // Block diffusion forward (parallel token generation)
    sm2_dflash_draft_block(ctx, cfg, out_tokens, cfg->block_size);
    
    // Verify entire block in one pass
    int n_accepted;
    sm2_dflash_verify(ctx, cfg, out_tokens, cfg->block_size, &n_accepted);
    
    return n_accepted;
}
```

## Acceptance Rate Targets

| Draft -> Target | Expected Acceptance | Speedup |
|----------------|---------------------|---------|
| 135M -> 360M | 70-80% | 2-3x |
| 135M -> 1.7B | 60-70% | 1.5-2x |
| DFlash-135M -> 360M | 85-95% | 4-6x |

## Memory Considerations

Standard speculative:
- Target model loaded
- Draft model loaded
- 2x KV cache (or share if compatible)

DFlash:
- Target model loaded
- DFlash draft model loaded (diffusion weights)
- Draft model is separate architecture (not standard AR)

## Limitations

1. Draft and target must share tokenizer
2. KV cache incompatibility between architectures
3. DFlash requires trained draft model
4. VPS 512 MB: NOT suitable (memory overhead)

## See Also
- [[smollm2dl-decode-layer]]
- [[dflash-integration]]
- [[spec:001]] Phase 8a/8b
