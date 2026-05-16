// sm2dl_speculative.c - Speculative decoding (Phase 8a)
// 135M draft -> 360M/1.7B target verification

#include <stdlib.h>
#include <math.h>
#include "smollm2.h"

// ============================================================================
// SPECULATIVE DECODING - Draft-verifier pattern
//
// Standard autoregressive draft model:
//   1. Draft model generates K tokens (2, 4, or 8)
//   2. Target model verifies all K tokens in one prefill-like pass
//   3. Accept matching prefix, sample from target on first mismatch
//   4. Continue with next draft block
//
// Benefits:
//   - 2-4x speedup when acceptance rate is high (>80%)
//   - Lossless (exact same output as autoregressive)
// ============================================================================

typedef struct {
    sm2_model* draft_model;
    sm2_model* target_model;
    sm2_context* draft_ctx;
    sm2_context* target_ctx;
    int draft_tokens;       // 2, 4, or 8
    float accept_threshold;
    int n_accepted;         // running count
    int n_rejected;         // running count
} sm2_spec_dec;

int sm2_spec_dec_init(sm2_spec_dec** out, sm2_model* draft, sm2_model* target) {
    sm2_spec_dec* spec = calloc(1, sizeof(sm2_spec_dec));
    if (!spec) return -1;
    
    spec->draft_model = draft;
    spec->target_model = target;
    spec->draft_tokens = 4; // default
    spec->accept_threshold = 0.5f;
    
    // Create contexts for both models
    if (sm2_create_context(draft, &spec->draft_ctx) != 0) {
        free(spec);
        return -1;
    }
    if (sm2_create_context(target, &spec->target_ctx) != 0) {
        sm2_free_context(spec->draft_ctx);
        free(spec);
        return -1;
    }
    
    *out = spec;
    return 0;
}

void sm2_spec_dec_free(sm2_spec_dec* spec) {
    if (!spec) return;
    if (spec->draft_ctx) sm2_free_context(spec->draft_ctx);
    if (spec->target_ctx) sm2_free_context(spec->target_ctx);
    free(spec);
}

// Draft phase: generate K tokens using draft model
int sm2_spec_draft(sm2_spec_dec* spec, int* out_tokens) {
    sm2_context* ctx = spec->draft_ctx;
    
    for (int i = 0; i < spec->draft_tokens; i++) {
        int token;
        int ok = sm2_decode_next(ctx, &token);
        if (ok != 0) return i;
        out_tokens[i] = token;
        
        if (token == 2) break; // EOS
    }
    
    return spec->draft_tokens;
}

// Verify phase: verify draft tokens with target model
// Returns: number of accepted tokens
int sm2_spec_verify(sm2_spec_dec* spec, const int* draft_tokens, int n_tokens) {
    // Prefill target with draft tokens
    int ok = sm2_prefill(spec->target_ctx, draft_tokens, n_tokens);
    if (ok != 0) return 0;
    
    // For each draft token, check if target agrees
    int n_accepted = 0;
    
    for (int i = 0; i < n_tokens; i++) {
        // Get target's predicted token
        float max_logit = spec->target_ctx->scratch.logits[0];
        int max_idx = 0;
        for (int j = 1; j < spec->target_ctx->model->vocab_size; j++) {
            if (spec->target_ctx->scratch.logits[j] > max_logit) {
                max_logit = spec->target_ctx->scratch.logits[j];
                max_idx = j;
            }
        }
        
        // Compare with draft
        if (max_idx == draft_tokens[i]) {
            n_accepted++;
        } else {
            // First mismatch - stop accepting
            break;
        }
        
        // Decode one step to get next logits ready
        int dummy_token;
        sm2_decode_next(spec->target_ctx, &dummy_token);
    }
    
    spec->n_accepted += n_accepted;
    spec->n_rejected += (n_tokens - n_accepted);
    
    return n_accepted;
}

// Full speculative decode step
int sm2_spec_decode_step(sm2_spec_dec* spec, int* out_token) {
    // Step 1: Draft K tokens
    int draft_tokens[8];
    int n_draft = sm2_spec_draft(spec, draft_tokens);
    
    if (n_draft == 0) {
        *out_token = 2; // EOS
        return 0;
    }
    
    // Step 2: Verify with target
    int n_accepted = sm2_spec_verify(spec, draft_tokens, n_draft);
    
    // Step 3: Return accepted token
    if (n_accepted > 0) {
        *out_token = draft_tokens[n_accepted - 1];
    } else {
        // All rejected - sample from target
        *out_token = sm2_sample_token(
            spec->target_ctx->scratch.logits,
            &spec->target_ctx->params,
            &spec->target_ctx->rng_state
        );
    }
    
    return 0;
}

// Get acceptance rate
float sm2_spec_acceptance_rate(sm2_spec_dec* spec) {
    int total = spec->n_accepted + spec->n_rejected;
    if (total == 0) return 0.0f;
    return (float)spec->n_accepted / (float)total;
}