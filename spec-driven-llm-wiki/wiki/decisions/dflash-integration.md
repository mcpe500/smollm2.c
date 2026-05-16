# ADR-005: DFlash Integration

**Status:** Proposed  
**Date:** 2026-05-16  
**Spec:** [[spec:001]]

## Context

DFlash (z-lab/dflash) is a block diffusion model for speculative decoding. We need to decide how to integrate DFlash into smollm2.c.

**References:**
- GitHub: https://github.com/z-lab/dflash (4.6k stars)
- Paper: arXiv:2602.06036
- Claims: 6x+ lossless acceleration, 2.5x better than EAGLE-3

## Decision

Implement DFlash in Phase 8b (Research), with two-stage approach:

### Stage 1: Standard Speculative Decoding (Phase 8a - Production)
- 135M autoregressive draft -> 360M/1.7B verifier
- No additional model training required
- Target: 2-4x speedup

### Stage 2: DFlash Block Diffusion (Phase 8b - Research)
- Lightweight diffusion draft model for SmolLM2
- Requires training DFlash-SmolLM2 draft model
- NOT for VPS 512 MB mode
- Target: 6x+ speedup

## DFlash Architecture

DFlash uses block diffusion to predict multiple tokens in parallel:

```
Input: [tokens 0..pos]
         |
         v
+------------------+
| DFlash Draft     |  -> Generate K tokens (parallel, not AR)
| (diffusion model)|
+------------------+
         |
         v
Draft tokens: [t1, t2, t3, ... tK]
         |
         v
+------------------+
| Target Verifier  |  -> Verify all K tokens in one pass
| (SmolLM2)        |
+------------------+
         |
         v
Accepted prefix: [t1, t2, t3] (matching tokens)
Rejected: [t4] -> sample from target
```

## API Design

```c
typedef struct {
    sm2_model *draft_model;
    sm2_model *target_model;
    int num_draft_tokens;       // 16 default
    int block_size;             // 16
    float accept_threshold;
} sm2_dflash_config;

int sm2_dflash_init(sm2_dflash_config *cfg, 
                    const char *draft_path, 
                    const char *target_path);

int sm2_dflash_generate(sm2_context *ctx, 
                        sm2_dflash_config *cfg,
                        int *out_token);
```

## Configuration

For server mode:

```c
sm2_dflash_config cfg = {
    .draft_model = dflash_model,
    .target_model = target_model,
    .num_draft_tokens = 16,
    .block_size = 16,
    .accept_threshold = 0.5f
};
```

## Consequences

**Positive:**
- 6x+ speedup achievable (according to paper)
- Parallel token generation (no sequential bottleneck)
- Compatible with existing verification loop

**Negative:**
- Requires trained DFlash draft model (research work)
- Draft model adds memory overhead (~135M extra for 135M target)
- Not suitable for 512 MB VPS mode

## Implementation Notes

1. DFlash draft model must be trained separately (not available from HF)
2. Block diffusion forward pass is non-trivial in C
3. GPU backend recommended for DFlash performance
4. Standard speculative decode is fallback if DFlash model unavailable

## See Also
- [[smollm2dl-decode-layer]]
- [[spec:001]] Phase 8b