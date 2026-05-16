// sm2_dflash_verify.c - DFlash verification logic

#include <math.h>
#include "smollm2.h"

// Verify a block of draft tokens against target model
// Returns number of accepted tokens from the prefix
int sm2_dflash_verify_block(
    sm2_dflash* df,
    const int* draft_tokens,
    int n_tokens,
    int* out_accepted
) {
    if (!df || !df->target_ctx || !draft_tokens) return -1;
    
    // Prefill target with draft block
    int ok = sm2_prefill(df->target_ctx, draft_tokens, n_tokens);
    if (ok != 0) return -1;
    
    // Check each position
    int n_accepted = 0;
    
    for (int i = 0; i < n_tokens; i++) {
        // Get target's prediction
        float max_logit = df->target_ctx->scratch.logits[0];
        int pred_token = 0;
        
        for (int j = 1; j < df->target_ctx->model->vocab_size; j++) {
            if (df->target_ctx->scratch.logits[j] > max_logit) {
                max_logit = df->target_ctx->scratch.logits[j];
                pred_token = j;
            }
        }
        
        if (pred_token == draft_tokens[i]) {
            n_accepted++;
        } else {
            break; // First mismatch
        }
        
        // Advance to next position
        if (i < n_tokens - 1) {
            int dummy;
            sm2_decode_next(df->target_ctx, &dummy);
        }
    }
    
    if (out_accepted) *out_accepted = n_accepted;
    return 0;
}