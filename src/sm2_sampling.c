// sm2_sampling.c - Token sampling utilities

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "smollm2.h"

// ============================================================================
// ENTROPY CALCULATION (simplified, fast version)
// Returns normalized entropy in [0, 1], where 0 = model is confident/fixed
// ============================================================================

static float compute_logits_entropy_fast(const float* logits, int vocab_size) {
    // Compute entropy using strategy: scan all logits once for max, then sample
    if (!logits || vocab_size <= 0) return 1.0f;

    int n_top = vocab_size > 64 ? 64 : vocab_size;
    if (n_top <= 0) return 1.0f;

    // Find max logit
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    // Find top 64 using partial selection (avoids sorting all 49152 elements)
    float exp_vals[64];
    int indices[64];
    int n_found = 0;

    for (int i = 0; i < vocab_size; i++) {
        float exp_val = expf(logits[i] - max_logit);

        if (n_found < n_top) {
            indices[n_found] = i;
            exp_vals[n_found] = exp_val;
            n_found++;

            // Keep sorted
            for (int j = n_found - 1; j > 0 && exp_vals[j] > exp_vals[j-1]; j--) {
                float tmp = exp_vals[j];
                exp_vals[j] = exp_vals[j-1];
                exp_vals[j-1] = tmp;
                int ti = indices[j];
                indices[j] = indices[j-1];
                indices[j-1] = ti;
            }
        } else if (exp_val > exp_vals[n_top - 1]) {
            // Replace smallest and bubble up
            exp_vals[n_top - 1] = exp_val;
            indices[n_top - 1] = i;

            for (int j = n_top - 1; j > 0 && exp_vals[j] > exp_vals[j-1]; j--) {
                float tmp = exp_vals[j];
                exp_vals[j] = exp_vals[j-1];
                exp_vals[j-1] = tmp;
                int ti = indices[j];
                indices[j] = indices[j-1];
                indices[j-1] = ti;
            }
        }
    }

    // Compute entropy on top 64
    float prob_sum = 0.0f;
    for (int i = 0; i < n_found; i++) prob_sum += exp_vals[i];
    if (prob_sum <= 0.0f) return 1.0f;

    float entropy = 0.0f;
    for (int i = 0; i < n_found; i++) {
        float p = exp_vals[i] / prob_sum;
        if (p > 1e-10f) {
            entropy -= p * logf(p);
        }
    }

    // Normalize: max entropy with 64 options ≈ 4.16
    float max_entropy = 4.16f;
    return entropy / max_entropy;
}

// ============================================================================
// N-GRAM REPETITION DETECTION
// Detects repeating sequences like "II II" (immediate consecutive repetition)
// ============================================================================

static int detect_ngram_repetition(const int* recent_tokens, int n_recent) {
    if (!recent_tokens || n_recent < 6) return 0;
    if (n_recent > 256) n_recent = 256;  // Safety cap

    int rep_count = 0;

    // Detect immediate repetitions: ABABAB pattern
    // For trailing "II II" pattern: tokens[n-1] == tokens[n-2]
    // Count consecutive pairs that are identical
    int max_check = n_recent > 16 ? 16 : n_recent;

    for (int i = 0; i < max_check - 1; i++) {
        int idx1 = n_recent - 1 - i;
        int idx2 = n_recent - 2 - i;

        if (idx1 < 0 || idx2 < 0) continue;

        int t1 = recent_tokens[idx1];
        int t2 = recent_tokens[idx2];

        if (t1 == t2) {
            // Consecutive tokens are the same - strong repetition signal
            rep_count++;
        }
    }

    return rep_count;
}

// ============================================================================
// REPETITION PENALTY
// Applied to logits before sampling to reduce token repetition
// ============================================================================

void sm2_apply_repetition_penalty(float* logits, int vocab_size,
                                 const int* recent_tokens, int n_recent,
                                 float penalty) {
    if (penalty <= 1.0f || n_recent <= 0) return;

    // Divide logits for tokens that appeared in recent_window by penalty
    // This reduces their probability of being selected again
    for (int i = 0; i < n_recent; i++) {
        int token_id = recent_tokens[i];
        if (token_id >= 0 && token_id < vocab_size) {
            if (logits[token_id] > 0) {
                logits[token_id] /= penalty;
            } else {
                logits[token_id] *= penalty;
            }
        }
    }
}

// ============================================================================
// RANDOM NUMBER GENERATION (xorshift64)
// ============================================================================

static uint64_t xorshift64(uint64_t state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

static float random_01(uint64_t* rng_state) {
    *rng_state = xorshift64(*rng_state);
    // Convert 64-bit to float [0, 1)
    return (float)(*rng_state >> 11) / (float)(1ULL << 53);
}

// ============================================================================
// TEMPERATURE SAMPLING
// ============================================================================

float sm2_sample_temperature(float x, float temp, uint64_t* rng_state) {
    if (temp <= 0.0f) {
        // Greedy: return max
        return x;
    }

    // Apply temperature
    x /= temp;

    // Softmax numerically stable
    // Find max
    float max_x = x;
    // Not needed for single value

    // Convert to probability via softmax
    float exp_x = expf(x - max_x);
    float r = random_01(rng_state) * exp_x;

    return r;
}

// ============================================================================
// TOP-P (NUCLEUS) SAMPLING
// ============================================================================

int sm2_sample_top_p(float* logits, int vocab_size, float top_p, float temp, uint64_t* rng_state) {
    // Apply temperature first
    if (temp > 0.0f) {
        for (int i = 0; i < vocab_size; i++) {
            logits[i] /= temp;
        }
    }

    // Find max logit for numerical stability
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    // Compute exp probabilities
    float sum_exp = 0.0f;
    float exp_logits[512];  // Only need top candidates
    int indices[512];
    int n_candidates = 0;

    // First pass: compute exp and find top candidates (pre-filter to reduce work)
    for (int i = 0; i < vocab_size; i++) {
        float e = expf(logits[i] - max_logit);
        sum_exp += e;

        // Keep top 512 candidates for sorting
        if (n_candidates < 512 || e > exp_logits[n_candidates - 1]) {
            // Insert into sorted position
            int j = n_candidates;
            while (j > 0 && e > exp_logits[j - 1]) {
                if (j < 512) {
                    exp_logits[j] = exp_logits[j - 1];
                    indices[j] = indices[j - 1];
                }
                j--;
            }
            if (j < 512) {
                exp_logits[j] = e;
                indices[j] = i;
            }
            if (n_candidates < 512) n_candidates++;
        }
    }

    // Top-p: find smallest set with cumulative prob >= top_p
    float cum = 0.0f;
    int top_count = n_candidates;
    for (int i = 0; i < n_candidates; i++) {
        cum += exp_logits[i] / sum_exp;
        if (cum >= top_p) {
            top_count = i + 1;
            break;
        }
    }

    // Normalize top-p candidates
    float top_sum = 0.0f;
    for (int i = 0; i < top_count; i++) {
        top_sum += exp_logits[i];
    }

    // Sample from top-p
    float r = random_01(rng_state) * top_sum;
    cum = 0.0f;
    for (int i = 0; i < top_count; i++) {
        cum += exp_logits[i];
        if (r <= cum) {
            return indices[i];
        }
    }

    // Fallback to most likely
    return indices[0];
}

// ============================================================================
// TOP-K SAMPLING
// ============================================================================

int sm2_sample_top_k(float* logits, int vocab_size, int top_k, float temp, uint64_t* rng_state) {
    // Apply temperature (any temp > 0 uses temperature scaling)
    if (temp > 0.0f) {
        for (int i = 0; i < vocab_size; i++) {
            logits[i] /= temp;
        }
    }

    // Find max logit for numerical stability
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    // Limit top_k to reasonable range
    if (top_k <= 0 || top_k > 256) top_k = 256;
    if (top_k > vocab_size) top_k = vocab_size;

    // Pre-filter: find top-k tokens using partial selection
    int indices[256];
    float exp_vals[256];
    int n_selected = 0;

    for (int i = 0; i < vocab_size; i++) {
        float e = expf(logits[i] - max_logit);

        if (n_selected < top_k) {
            // Fill up the array
            indices[n_selected] = i;
            exp_vals[n_selected] = e;
            n_selected++;

            // Keep sorted (simple insertion sort)
            for (int j = n_selected - 1; j > 0; j--) {
                if (exp_vals[j] > exp_vals[j - 1]) {
                    int ti = indices[j];
                    indices[j] = indices[j - 1];
                    indices[j - 1] = ti;
                    float tv = exp_vals[j];
                    exp_vals[j] = exp_vals[j - 1];
                    exp_vals[j - 1] = tv;
                } else {
                    break;
                }
            }
        } else {
            // Check if this should replace the smallest in top-k
            if (e > exp_vals[top_k - 1]) {
                // Replace last element
                exp_vals[top_k - 1] = e;
                indices[top_k - 1] = i;

                // Bubble up
                for (int j = top_k - 1; j > 0; j--) {
                    if (exp_vals[j] > exp_vals[j - 1]) {
                        int ti = indices[j];
                        indices[j] = indices[j - 1];
                        indices[j - 1] = ti;
                        float tv = exp_vals[j];
                        exp_vals[j] = exp_vals[j - 1];
                        exp_vals[j - 1] = tv;
                    } else {
                        break;
                    }
                }
            }
        }
    }

    // Normalize and sample
    float sum_exp = 0.0f;
    for (int i = 0; i < top_k; i++) {
        sum_exp += exp_vals[i];
    }

    float r = random_01(rng_state) * sum_exp;
    float cum = 0.0f;
    for (int i = 0; i < top_k; i++) {
        cum += exp_vals[i];
        if (r <= cum) {
            return indices[i];
        }
    }

    return indices[0];  // Fallback
}

// ============================================================================
// MAIN SAMPLE FUNCTION
// ============================================================================

int sm2_sample_token(const float* logits, const sm2_generate_params* params,
                     uint64_t* rng_state, sm2_context* ctx) {
    // Make a copy of logits so we can apply repetition penalty without modifying originals
    float logits_copy[49152];
    int vocab_size = 49152;
    memcpy(logits_copy, logits, vocab_size * sizeof(float));

    // === AUTOMATIC REPETITION HANDLING ===
    // If the model is producing repetitive output, the code will handle it.
    // Find and block tokens that appear multiple times in recent output.

    float effective_penalty = params->repetition_penalty;

    // Apply after initial generation (skip first 2 tokens)
    const int* rt = ctx->scratch.recent_tokens;
    int n_total = ctx->scratch.kv_cache_len;
    int n_gen = n_total - 2;  // First 2 tokens are prompt

    if (n_gen >= 6 && rt) {
        // RANGE: Only penalize generated tokens, not prompt
        // gen_start = prompt tokens (n_total - n_gen = 2 for default: BOS + actual prompt)
        int gen_start = 2;  // Default: BOS(1) + prompt tokens
        if (n_total > n_gen) gen_start = n_total - n_gen;

        // LOOK_BACK window on generated tokens only
        int look_back = (n_gen < 8) ? n_gen : 8;

        for (int i = 0; i < look_back; i++) {
            int ti = rt[n_total - 1 - i];
            if (ti < 0 || ti >= vocab_size) continue;

            // Count occurrences in look_back window (generated tokens only)
            int count = 0;
            for (int j = 0; j < look_back; j++) {
                int idx = n_total - 1 - j;
                // Only count within the generated portion
                if (idx >= gen_start && rt[idx] == ti) count++;
            }

            // CHANGED: count > 2 (3+ occurrences) instead of count > 1
            // This is less aggressive - token needs to appear 3+ times in last 8
            if (count > 2) {
                logits_copy[ti] = -1e9f;
            } else if (count > 1) {
                // Scale down instead of hard block - reduces but doesn't eliminate
                logits_copy[ti] *= 0.5f;
            }
        }
    }

    // Apply repetition penalty if enabled (only to generated tokens, not prompt)
    if (effective_penalty > 1.0f && ctx) {
        int gen_start = 2;  // BOS + prompt tokens
        int n_total = ctx->scratch.kv_cache_len;
        int n_gen = n_total - gen_start;
        if (n_gen > 0 && ctx->scratch.recent_tokens) {
            sm2_apply_repetition_penalty(logits_copy, vocab_size,
                                         ctx->scratch.recent_tokens + gen_start,
                                         n_gen,
                                         effective_penalty);
        }
    }

    // === MINIMUM RESPONSE LENGTH ===
    // Block EOS tokens in the first few generated tokens to ensure minimum response length
    // This prevents the model from immediately ending when the chat template confuses it
    //
    // ctx->pos starts at 0, incremented in each decode step
    // Block EOS in first 2 decode steps (pos = 0 or 1)
    if (ctx->pos < 2) {
        // Block ALL special tokens (0, 1, 2) in first 2 generated tokens
        // This ensures we get actual content text
        logits_copy[0] = -1e9f;  // <|endoftext|>
        logits_copy[1] = -1e9f;  // <|im_start|>
        logits_copy[2] = -1e9f;  // <|im_end|>
    }

    // Combine top-k and top-p
    // top_p is stored as integer 0-100, convert to 0.0-1.0 for comparison
    float top_p_val = params->top_p / 100.0f;
    if (params->top_k > 0 && top_p_val < 1.0f) {
        return sm2_sample_top_k(logits_copy, vocab_size, params->top_k, params->temperature, rng_state);
    } else if (top_p_val < 1.0f) {
        return sm2_sample_top_p(logits_copy, vocab_size, top_p_val, params->temperature, rng_state);
    } else if (params->top_k > 0) {
        return sm2_sample_top_k(logits_copy, vocab_size, params->top_k, params->temperature, rng_state);
    } else {
        // Greedy
        int max_idx = 0;
        float max_val = logits_copy[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits_copy[i] > max_val) {
                max_val = logits_copy[i];
                max_idx = i;
            }
        }
        return max_idx;
    }
}
