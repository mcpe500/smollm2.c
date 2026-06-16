// sampling.c — token sampling

#include "sampling.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

static unsigned g_rng = 0;

static float rng_f32(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return (float)(g_rng & 0x7fffffffu) / (float)0x7fffffffu;
}

int sample_token(float* logits, int vocab, const sample_params* p,
                 const int* history, int hist_len) {
    if (!logits || vocab <= 0 || !p) return 0;

    /* Greedy shortcut */
    if (p->temperature <= 0.0f) {
        int best = 0;
        for (int v = 1; v < vocab; v++)
            if (logits[v] > logits[best]) best = v;
        return best;
    }

    /* Repetition penalty */
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

    /* Top-k: zero out all but top_k entries */
    if (p->top_k > 0 && p->top_k < vocab) {
        /* find k-th largest via partial selection */
        float kth = 0.0f;
        /* simple O(k*n) for small k */
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

    /* Top-p: zero out tail below cumulative threshold */
    if (p->top_p > 0.0f && p->top_p < 1.0f) {
        /* sort descending by prob then truncate - use scratch sort on pairs */
        /* for vocab=49152 a full sort is expensive; use a simpler threshold scan */
        /* find the prob threshold such that cumsum >= top_p */
        /* approximation: iterate sorted descending using max-heap would be ideal,
           but for simplicity we do two passes (find threshold, then zero out) */
        float cum = 0.0f;
        float threshold = 0.0f;
        /* first pass: collect top probs until cumsum >= top_p */
        float tmp;
        /* simple approach: repeatedly find max unselected */
        float* sorted = (float*)malloc((size_t)vocab * sizeof(float));
        if (sorted) {
            memcpy(sorted, logits, (size_t)vocab * sizeof(float));
            /* partial sort: just scan and accumulate until top_p hit */
            for (int pass = 0; pass < vocab; pass++) {
                float mx = -FLT_MAX; int mi = 0;
                for (int v = 0; v < vocab; v++) if (sorted[v] > mx) { mx = sorted[v]; mi = v; }
                if (mx <= 0.0f) break;
                cum += mx;
                sorted[mi] = -FLT_MAX;
                threshold = mx;
                if (cum >= p->top_p) break;
            }
            free(sorted);
            sum = 0.0f;
            for (int v = 0; v < vocab; v++) {
                if (logits[v] < threshold) logits[v] = 0.0f;
                sum += logits[v];
            }
            if (sum > 0.0f) { inv_sum=1.0f/sum; for(int v=0;v<vocab;v++) logits[v]*=inv_sum; }
        }
        (void)tmp;
    }

    /* Seed RNG if not seeded */
    if (g_rng == 0) g_rng = (p->seed != 0) ? p->seed : 12345u;

    /* Multinomial sample */
    float r = rng_f32();
    float cum = 0.0f;
    for (int v = 0; v < vocab; v++) {
        cum += logits[v];
        if (r <= cum) return v;
    }
    return vocab - 1;
}
