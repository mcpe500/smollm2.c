// sampling.c — token sampling

#include "sampling.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <stdint.h>
#include <time.h>

static unsigned g_rng = 0;

static float* g_topp_probs = NULL;
static int topp_cmp(const void* a, const void* b) {
    float pa = g_topp_probs[*(const int*)a];
    float pb = g_topp_probs[*(const int*)b];
    return (pa < pb) - (pa > pb);
}

static float rng_f32(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return (float)(g_rng & 0x7fffffffu) / (float)0x7fffffffu;
}

int sample_token(float* logits, int vocab, const sample_params* p,
                 const int* history, int hist_len) {
    if (!logits || vocab <= 0 || !p) return 0;

    /* Apply rep_penalty for both greedy and sampled paths. */
    if (p->rep_penalty > 1.0f && history && hist_len > 0) {
        for (int i = 0; i < hist_len; i++) {
            int t = history[i];
            if (t >= 0 && t < vocab) {
                if (logits[t] > 0.0f)
                    logits[t] /= p->rep_penalty;
                else
                    logits[t] *= p->rep_penalty;
            }
        }
    }

    /* Greedy shortcut */
    if (p->temperature <= 0.0f) {
        int best = 0;
        for (int v = 1; v < vocab; v++)
            if (logits[v] > logits[best]) best = v;
        return best;
    }


    /* Temperature */
    float inv_t = 1.0f / p->temperature;
    float max_l = -FLT_MAX;
    for (int v = 0; v < vocab; v++) if (logits[v] > max_l) max_l = logits[v];

    /* Softmax into logits (reuse buffer) */
    float sum = 0.0f;
    for (int v = 0; v < vocab; v++) {
        logits[v] = expf((logits[v] - max_l) * inv_t);
        sum += logits[v];
    }
    float inv_sum = 1.0f / sum;
    for (int v = 0; v < vocab; v++) logits[v] *= inv_sum;

    /* Top-k: zero out all but top_k entries. */
    if (p->top_k > 0 && p->top_k < vocab) {
        float kth = 0.0f;
        float seen[64]; int ns = 0;
        int ks = (p->top_k < 64) ? p->top_k : 64;
        for (int v = 0; v < vocab; v++) {
            if (ns < ks) {
                seen[ns++] = logits[v];
            } else {
                float mn = seen[0]; int mi = 0;
                for (int j = 1; j < ks; j++) if (seen[j] < mn) { mn = seen[j]; mi = j; }
                if (logits[v] > mn) seen[mi] = logits[v];
            }
        }
        float mn = seen[0];
        for (int j = 1; j < ns; j++) if (seen[j] < mn) mn = seen[j];
        kth = mn;
        sum = 0.0f;
        for (int v = 0; v < vocab; v++) {
            if (logits[v] < kth) logits[v] = 0.0f;
            sum += logits[v];
        }
        if (sum > 0.0f) { inv_sum = 1.0f/sum; for (int v=0;v<vocab;v++) logits[v]*=inv_sum; }
    }

    /* Top-p (nucleus sampling): qsort indices by prob desc, keep cumsum >= top_p */
    if (p->top_p > 0.0f && p->top_p < 1.0f) {
        static int idx[49152];
        static float probs_copy[49152];
        int n = vocab < 49152 ? vocab : 49152;
        for (int v = 0; v < n; v++) idx[v] = v;
        memcpy(probs_copy, logits, (size_t)n * sizeof(float));
        g_topp_probs = probs_copy;
        qsort(idx, (size_t)n, sizeof(int), topp_cmp);
        float cum = 0.0f;
        int cutoff = n;
        for (int i = 0; i < n; i++) {
            cum += probs_copy[idx[i]];
            if (cum >= p->top_p) { cutoff = i + 1; break; }
        }
        for (int i = cutoff; i < n; i++) logits[idx[i]] = 0.0f;
        sum = 0.0f;
        for (int v = 0; v < n; v++) sum += logits[v];
        if (sum > 0.0f) { inv_sum = 1.0f/sum; for (int v = 0; v < n; v++) logits[v] *= inv_sum; }
    }

    /* Seed RNG if not seeded — mix time + pointer for varied outputs per run */
    if (g_rng == 0) {
        if (p->seed != 0) {
            g_rng = p->seed;
        } else {
            unsigned t = (unsigned)time(NULL) ^ (unsigned)(uintptr_t)logits;
            t ^= t << 13; t ^= t >> 17; t ^= t << 5;
            g_rng = t ? t : 12345u;
        }
    }

    /* Multinomial sample */
    float r = rng_f32();
    float cum = 0.0f;
    for (int v = 0; v < vocab; v++) {
        cum += logits[v];
        if (r <= cum) return v;
    }
    return vocab - 1;
}
