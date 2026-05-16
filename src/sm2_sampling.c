// sm2_sampling.c - Token sampling utilities

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "smollm2.h"

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
    
    // Compute exp and sort by probability
    typedef struct { int id; float p; } prob_t;
    prob_t* probs = (prob_t*)malloc(vocab_size * sizeof(prob_t));
    
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        probs[i].id = i;
        probs[i].p = expf(logits[i] - max_logit);
        sum_exp += probs[i].p;
    }
    
    // Normalize
    for (int i = 0; i < vocab_size; i++) {
        probs[i].p /= sum_exp;
    }
    
    // Sort by probability descending (simple bubble sort for small vocab)
    for (int i = 0; i < vocab_size - 1; i++) {
        for (int j = i + 1; j < vocab_size; j++) {
            if (probs[j].p > probs[i].p) {
                prob_t tmp = probs[i];
                probs[i] = probs[j];
                probs[j] = tmp;
            }
        }
    }
    
    // Top-p: find smallest set with cumulative prob >= top_p
    float cum = 0.0f;
    int top_count = vocab_size;
    for (int i = 0; i < vocab_size; i++) {
        cum += probs[i].p;
        if (cum >= top_p) {
            top_count = i + 1;
            break;
        }
    }
    
    // Sample from top-p
    float r = random_01(rng_state) * cum;
    cum = 0.0f;
    int sampled = 0;
    for (int i = 0; i < top_count; i++) {
        cum += probs[i].p;
        if (r <= cum) {
            sampled = probs[i].id;
            break;
        }
    }
    
    free(probs);
    return sampled;
}

// ============================================================================
// TOP-K SAMPLING
// ============================================================================

int sm2_sample_top_k(float* logits, int vocab_size, int top_k, float temp, uint64_t* rng_state) {
    // Apply temperature
    if (temp > 0.0f && temp != 1.0f) {
        for (int i = 0; i < vocab_size; i++) {
            logits[i] /= temp;
        }
    }
    
    // Find max logit
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    // Top-k mask
    float* probs = (float*)malloc(vocab_size * sizeof(float));
    float sum_exp = 0.0f;
    
    for (int i = 0; i < vocab_size; i++) {
        if (top_k > 0 && i >= top_k) {
            probs[i] = 0.0f;
        } else {
            probs[i] = expf(logits[i] - max_logit);
            sum_exp += probs[i];
        }
    }
    
    // Normalize
    for (int i = 0; i < vocab_size; i++) {
        probs[i] /= sum_exp;
    }
    
    // Sample
    float r = random_01(rng_state);
    float cum = 0.0f;
    int sampled = 0;
    for (int i = 0; i < vocab_size; i++) {
        cum += probs[i];
        if (r <= cum) {
            sampled = i;
            break;
        }
    }
    
    free(probs);
    return sampled;
}

// ============================================================================
// MAIN SAMPLE FUNCTION
// ============================================================================

int sm2_sample_token(const float* logits, const sm2_generate_params* params, uint64_t* rng_state) {
    // Combine top-k and top-p
    if (params->top_k > 0 && params->top_p < 1.0f) {
        // Use top-k then top-p
        return sm2_sample_top_k((float*)logits, 49152, params->top_k, params->temperature, rng_state);
    } else if (params->top_p < 1.0f) {
        return sm2_sample_top_p((float*)logits, 49152, params->top_p / 100.0f, params->temperature, rng_state);
    } else if (params->top_k > 0) {
        return sm2_sample_top_k((float*)logits, 49152, params->top_k, params->temperature, rng_state);
    } else {
        // Greedy
        int max_idx = 0;
        float max_val = logits[0];
        for (int i = 1; i < 49152; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }
        return max_idx;
    }
}