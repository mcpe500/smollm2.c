// backward.c — Phase B: per-op analytical backward + finite-diff grad check.
//
// Each op has:
//   - <op>_forward: forward pass used by both analytical and numerical paths
//   - <op>_backward: analytical gradient via chain rule
//   - grad_check_<op>: centered finite-diff comparison vs analytical
//
// Loss L = sum(Y * grad_Y). Finite-diff perturbs each input ±eps and measures
// (L+ - L-) / (2*eps).
#include "backward.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Deterministic RNG so failures are reproducible. */
static void rand_seed(int s) { srand((unsigned)s); }
static float randf() { return ((float)rand() / 2147483647.0f) - 0.5f; }

/* ----------------------------------------------------------------- helpers */
static float* f32_alloc(size_t n) { return (float*)malloc(n * sizeof(float)); }
static void   f32_fill_rand(float* p, size_t n, float scale) {
    for (size_t i = 0; i < n; i++) p[i] = randf() * scale;
}
static float max_abs_diff(const float* a, const float* b, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

/* ======================================================= MATMUL ============ */
/* Y[m,n] = X[m,k] @ W[k,n] */
static void matmul_fwd(const float* X, const float* W, float* Y,
                       int m, int n, int k) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int p = 0; p < k; p++) s += X[i * k + p] * W[p * n + j];
            Y[i * n + j] = s;
        }
}
/* gradX[m,k] = gradY[m,n] @ W[k,n]^T ; gradW[k,n] = X[m,k]^T @ gradY[m,n] */
static void matmul_backward(const float* X, const float* W, const float* gY,
                            float* gX, float* gW, int m, int n, int k) {
    if (gX) {
        for (int i = 0; i < m; i++)
            for (int p = 0; p < k; p++) {
                float s = 0;
                for (int j = 0; j < n; j++) s += gY[i * n + j] * W[p * n + j];
                gX[i * k + p] = s;
            }
    }
    if (gW) {
        for (int p = 0; p < k; p++)
            for (int j = 0; j < n; j++) {
                float s = 0;
                for (int i = 0; i < m; i++) s += X[i * k + p] * gY[i * n + j];
                gW[p * n + j] = s;
            }
    }
}

float grad_check_matmul(int m, int n, int k, float eps) {
    rand_seed(42);
    float *X = f32_alloc((size_t)m*k), *W = f32_alloc((size_t)k*n);
    float *Y = f32_alloc((size_t)m*n), *gY = f32_alloc((size_t)m*n);
    float *gX_a = f32_alloc((size_t)m*k), *gW_a = f32_alloc((size_t)k*n);
    float *gX_n = f32_alloc((size_t)m*k), *gW_n = f32_alloc((size_t)k*n);
    if (!X||!W||!Y||!gY||!gX_a||!gW_a||!gX_n||!gW_n) goto fail;
    f32_fill_rand(X, (size_t)m*k, 0.5f);
    f32_fill_rand(W, (size_t)k*n, 0.5f);
    f32_fill_rand(gY, (size_t)m*n, 0.5f);
    matmul_fwd(X, W, Y, m, n, k);
    matmul_backward(X, W, gY, gX_a, gW_a, m, n, k);

    /* numerical on X */
    for (int i = 0; i < m*k; i++) {
        float o = X[i];
        X[i] = o + eps; matmul_fwd(X, W, Y, m, n, k);
        float Lp = 0; for (int j = 0; j < m*n; j++) Lp += Y[j] * gY[j];
        X[i] = o - eps; matmul_fwd(X, W, Y, m, n, k);
        float Lm = 0; for (int j = 0; j < m*n; j++) Lm += Y[j] * gY[j];
        X[i] = o;
        gX_n[i] = (Lp - Lm) / (2.0f * eps);
    }
    /* numerical on W */
    for (int i = 0; i < k*n; i++) {
        float o = W[i];
        W[i] = o + eps; matmul_fwd(X, W, Y, m, n, k);
        float Lp = 0; for (int j = 0; j < m*n; j++) Lp += Y[j] * gY[j];
        W[i] = o - eps; matmul_fwd(X, W, Y, m, n, k);
        float Lm = 0; for (int j = 0; j < m*n; j++) Lm += Y[j] * gY[j];
        W[i] = o;
        gW_n[i] = (Lp - Lm) / (2.0f * eps);
    }
    float err = max_abs_diff(gX_a, gX_n, (size_t)m*k);
    float errW = max_abs_diff(gW_a, gW_n, (size_t)k*n);
    if (errW > err) err = errW;
    free(X);free(W);free(Y);free(gY);free(gX_a);free(gW_a);free(gX_n);free(gW_n);
    return err;
fail:
    free(X);free(W);free(Y);free(gY);free(gX_a);free(gW_a);free(gX_n);free(gW_n);
    return -1.0f;
}

/* ======================================================= RMSNORM =========== */
/* y[i] = x[i] * w[i] / rms, rms = sqrt(mean(x^2) + eps_rms) */
static float rmsnorm_fwd_one(const float* x, const float* w, float* y,
                             int n, float eps_rms) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float rms = sqrtf(ss / (float)n + eps_rms);
    float inv = 1.0f / rms;
    for (int i = 0; i < n; i++) y[i] = x[i] * w[i] * inv;
    return rms;
}
/* See backward.h comment for derivation. */
static void rmsnorm_backward(const float* x, const float* w, const float* gY,
                             float* gX, float* gW, int n, float eps_rms) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float rms = sqrtf(ss / (float)n + eps_rms);
    float inv = 1.0f / rms;
    /* mean_term = (1/n) * sum_j gY[j] * w[j] * x[j] */
    float mt = 0;
    for (int i = 0; i < n; i++) mt += gY[i] * w[i] * x[i];
    mt /= (float)n;
    if (gX) {
        for (int i = 0; i < n; i++)
            gX[i] = inv * w[i] * gY[i] - x[i] * inv * inv * inv * mt;
    }
    if (gW) {
        for (int i = 0; i < n; i++) gW[i] = gY[i] * x[i] * inv;
    }
    (void)eps_rms;
}

float grad_check_rmsnorm(int n, float eps) {
    rand_seed(42);
    float *x = f32_alloc(n), *w = f32_alloc(n), *y = f32_alloc(n);
    float *gY = f32_alloc(n), *gX_a = f32_alloc(n), *gW_a = f32_alloc(n);
    float *gX_n = f32_alloc(n), *gW_n = f32_alloc(n);
    if (!x||!w||!y||!gY||!gX_a||!gW_a||!gX_n||!gW_n) goto fail;
    f32_fill_rand(x, n, 0.5f);
    f32_fill_rand(w, n, 0.5f);
    f32_fill_rand(gY, n, 0.5f);
    /* add 1 to w so they're not all near 0 */
    for (int i = 0; i < n; i++) w[i] += 1.0f;
    const float eps_rms = 1e-5f;
    rmsnorm_fwd_one(x, w, y, n, eps_rms);
    rmsnorm_backward(x, w, gY, gX_a, gW_a, n, eps_rms);
    /* numerical on x */
    for (int i = 0; i < n; i++) {
        float o = x[i];
        x[i] = o + eps; rmsnorm_fwd_one(x, w, y, n, eps_rms);
        float Lp = 0; for (int j = 0; j < n; j++) Lp += y[j] * gY[j];
        x[i] = o - eps; rmsnorm_fwd_one(x, w, y, n, eps_rms);
        float Lm = 0; for (int j = 0; j < n; j++) Lm += y[j] * gY[j];
        x[i] = o;
        gX_n[i] = (Lp - Lm) / (2.0f * eps);
    }
    /* numerical on w */
    for (int i = 0; i < n; i++) {
        float o = w[i];
        w[i] = o + eps; rmsnorm_fwd_one(x, w, y, n, eps_rms);
        float Lp = 0; for (int j = 0; j < n; j++) Lp += y[j] * gY[j];
        w[i] = o - eps; rmsnorm_fwd_one(x, w, y, n, eps_rms);
        float Lm = 0; for (int j = 0; j < n; j++) Lm += y[j] * gY[j];
        w[i] = o;
        gW_n[i] = (Lp - Lm) / (2.0f * eps);
    }
    float err = max_abs_diff(gX_a, gX_n, n);
    float errW = max_abs_diff(gW_a, gW_n, n);
    if (errW > err) err = errW;
    free(x);free(w);free(y);free(gY);free(gX_a);free(gW_a);free(gX_n);free(gW_n);
    return err;
fail:
    free(x);free(w);free(y);free(gY);free(gX_a);free(gW_a);free(gX_n);free(gW_n);
    return -1.0f;
}

/* ======================================================= ROPE (Llama) ====== */
/* For position p, pair (2i, 2i+1): cos/sin from theta_p_i = p * base^(-2i/d)
 * y[2i]   = x[2i]*cos - x[2i+1]*sin
 * y[2i+1] = x[2i]*sin + x[2i+1]*cos
 * Backward: rotate grad by -theta. */
static void rope_fwd(const float* x, float* y, int T, int hd, float base) {
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < hd / 2; i++) {
            float theta = (float)t * powf(base, -(float)(2*i) / (float)hd);
            float c = cosf(theta), s = sinf(theta);
            float a = x[t * hd + 2*i];
            float b = x[t * hd + 2*i + 1];
            y[t * hd + 2*i]     = a * c - b * s;
            y[t * hd + 2*i + 1] = a * s + b * c;
        }
    }
}
static void rope_backward(const float* gY, float* gX, int T, int hd, float base) {
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < hd / 2; i++) {
            float theta = (float)t * powf(base, -(float)(2*i) / (float)hd);
            float c = cosf(theta), s = sinf(theta);
            float ga = gY[t * hd + 2*i];
            float gb = gY[t * hd + 2*i + 1];
            /* inverse rotation */
            gX[t * hd + 2*i]     = ga * c + gb * s;
            gX[t * hd + 2*i + 1] = -ga * s + gb * c;
        }
    }
}

float grad_check_rope(int n_heads, int head_dim, float eps) {
    int T = 4;
    int N = T * head_dim;
    rand_seed(42);
    float *x = f32_alloc((size_t)n_heads * N), *y = f32_alloc((size_t)n_heads * N);
    float *gY = f32_alloc((size_t)n_heads * N), *gX_a = f32_alloc((size_t)n_heads * N);
    float *gX_n = f32_alloc((size_t)n_heads * N);
    if (!x||!y||!gY||!gX_a||!gX_n) goto fail;
    f32_fill_rand(x, (size_t)n_heads * N, 0.5f);
    f32_fill_rand(gY, (size_t)n_heads * N, 0.5f);
    const float base = 10000.0f;
    for (int h = 0; h < n_heads; h++) {
        rope_fwd(x + h*N, y + h*N, T, head_dim, base);
        rope_backward(gY + h*N, gX_a + h*N, T, head_dim, base);
    }
    /* numerical per element (per head, per position) */
    for (int h = 0; h < n_heads; h++) {
        for (int i = 0; i < N; i++) {
            float o = x[h*N + i];
            x[h*N + i] = o + eps; rope_fwd(x + h*N, y + h*N, T, head_dim, base);
            float Lp = 0; for (int j = 0; j < N; j++) Lp += y[h*N + j] * gY[h*N + j];
            x[h*N + i] = o - eps; rope_fwd(x + h*N, y + h*N, T, head_dim, base);
            float Lm = 0; for (int j = 0; j < N; j++) Lm += y[h*N + j] * gY[h*N + j];
            x[h*N + i] = o;
            gX_n[h*N + i] = (Lp - Lm) / (2.0f * eps);
        }
    }
    float err = max_abs_diff(gX_a, gX_n, (size_t)n_heads * N);
    free(x);free(y);free(gY);free(gX_a);free(gX_n);
    return err;
fail:
    free(x);free(y);free(gY);free(gX_a);free(gX_n);
    return -1.0f;
}

/* ======================================================= ATTENTION (1 head) */
/* scores[t,s] = (Q[t] . K[s]) * inv_sqrt_hd
 * probs[t,s] = softmax(scores[t,:])[s]
 * attn_out[t] = sum_s probs[t,s] * V[s] */
static void attn_fwd(const float* Q, const float* K, const float* V,
                     float* out, float* scores, float* probs,
                     int T, int hd) {
    float inv = 1.0f / sqrtf((float)hd);
    for (int t = 0; t < T; t++) {
        float mx = -1e30f;
        for (int s = 0; s < T; s++) {
            float d = 0;
            for (int h = 0; h < hd; h++) d += Q[t*hd + h] * K[s*hd + h];
            d *= inv;
            scores[t*T + s] = d;
            if (d > mx) mx = d;
        }
        float sum = 0;
        for (int s = 0; s < T; s++) {
            float e = expf(scores[t*T + s] - mx);
            probs[t*T + s] = e;
            sum += e;
        }
        float invsum = 1.0f / sum;
        for (int s = 0; s < T; s++) probs[t*T + s] *= invsum;
        for (int h = 0; h < hd; h++) {
            float a = 0;
            for (int s = 0; s < T; s++) a += probs[t*T + s] * V[s*hd + h];
            out[t*hd + h] = a;
        }
    }
}
static void attn_backward(const float* Q, const float* K, const float* V,
                          const float* probs, const float* gOut,
                          float* gQ, float* gK, float* gV,
                          int T, int hd) {
    float inv = 1.0f / sqrtf((float)hd);
    /* gV[s,h] += probs[t,s] * gOut[t,h] */
    if (gV) {
        for (int s = 0; s < T; s++)
            for (int h = 0; h < hd; h++) {
                float a = 0;
                for (int t = 0; t < T; t++) a += probs[t*T + s] * gOut[t*hd + h];
                gV[s*hd + h] = a;
            }
    }
    /* gprobs[t,s] = sum_h gOut[t,h] * V[s,h]
     * gscore[t,s] = probs[t,s] * (gprobs[t,s] - sum_s' probs[t,s'] * gprobs[t,s']) */
    float* gscore = (float*)malloc((size_t)T*T*sizeof(float));
    if (!gscore) return;
    for (int t = 0; t < T; t++) {
        float gp[256];
        float dot = 0;
        for (int s = 0; s < T; s++) {
            float a = 0;
            for (int h = 0; h < hd; h++) a += gOut[t*hd + h] * V[s*hd + h];
            gp[s] = a;
            dot += probs[t*T + s] * a;
        }
        for (int s = 0; s < T; s++)
            gscore[t*T + s] = probs[t*T + s] * (gp[s] - dot);
    }
    if (gQ) {
        for (int t = 0; t < T; t++)
            for (int h = 0; h < hd; h++) {
                float a = 0;
                for (int s = 0; s < T; s++) a += gscore[t*T + s] * K[s*hd + h];
                gQ[t*hd + h] = a * inv;
            }
    }
    if (gK) {
        for (int s = 0; s < T; s++)
            for (int h = 0; h < hd; h++) {
                float a = 0;
                for (int t = 0; t < T; t++) a += gscore[t*T + s] * Q[t*hd + h];
                gK[s*hd + h] = a * inv;
            }
    }
    free(gscore);
}

float grad_check_attention(int T, int hd, float eps) {
    rand_seed(42);
    int N = T * hd;
    int NS = T * T;
    float *Q=f32_alloc(N), *K=f32_alloc(N), *V=f32_alloc(N);
    float *out=f32_alloc(N), *scores=f32_alloc(NS), *probs=f32_alloc(NS);
    float *gOut=f32_alloc(N), *gQ_a=f32_alloc(N), *gK_a=f32_alloc(N), *gV_a=f32_alloc(N);
    float *gQ_n=f32_alloc(N), *gK_n=f32_alloc(N), *gV_n=f32_alloc(N);
    if (!Q||!K||!V||!out||!scores||!probs||!gOut||!gQ_a||!gK_a||!gV_a||!gQ_n||!gK_n||!gV_n)
        goto fail;
    f32_fill_rand(Q, N, 0.5f);
    f32_fill_rand(K, N, 0.5f);
    f32_fill_rand(V, N, 0.5f);
    f32_fill_rand(gOut, N, 0.5f);
    attn_fwd(Q, K, V, out, scores, probs, T, hd);
    attn_backward(Q, K, V, probs, gOut, gQ_a, gK_a, gV_a, T, hd);

    /* numerical on Q */
    for (int i = 0; i < N; i++) {
        float o = Q[i];
        Q[i] = o + eps; attn_fwd(Q, K, V, out, scores, probs, T, hd);
        float Lp = 0; for (int j = 0; j < N; j++) Lp += out[j] * gOut[j];
        Q[i] = o - eps; attn_fwd(Q, K, V, out, scores, probs, T, hd);
        float Lm = 0; for (int j = 0; j < N; j++) Lm += out[j] * gOut[j];
        Q[i] = o;
        gQ_n[i] = (Lp - Lm) / (2.0f * eps);
    }
    for (int i = 0; i < N; i++) {
        float o = K[i];
        K[i] = o + eps; attn_fwd(Q, K, V, out, scores, probs, T, hd);
        float Lp = 0; for (int j = 0; j < N; j++) Lp += out[j] * gOut[j];
        K[i] = o - eps; attn_fwd(Q, K, V, out, scores, probs, T, hd);
        float Lm = 0; for (int j = 0; j < N; j++) Lm += out[j] * gOut[j];
        K[i] = o;
        gK_n[i] = (Lp - Lm) / (2.0f * eps);
    }
    for (int i = 0; i < N; i++) {
        float o = V[i];
        V[i] = o + eps; attn_fwd(Q, K, V, out, scores, probs, T, hd);
        float Lp = 0; for (int j = 0; j < N; j++) Lp += out[j] * gOut[j];
        V[i] = o - eps; attn_fwd(Q, K, V, out, scores, probs, T, hd);
        float Lm = 0; for (int j = 0; j < N; j++) Lm += out[j] * gOut[j];
        V[i] = o;
        gV_n[i] = (Lp - Lm) / (2.0f * eps);
    }
    float err = max_abs_diff(gQ_a, gQ_n, N);
    float errK = max_abs_diff(gK_a, gK_n, N);
    float errV = max_abs_diff(gV_a, gV_n, N);
    if (errK > err) err = errK;
    if (errV > err) err = errV;
    free(Q);free(K);free(V);free(out);free(scores);free(probs);
    free(gOut);free(gQ_a);free(gK_a);free(gV_a);free(gQ_n);free(gK_n);free(gV_n);
    return err;
fail:
    free(Q);free(K);free(V);free(out);free(scores);free(probs);
    free(gOut);free(gQ_a);free(gK_a);free(gV_a);free(gQ_n);free(gK_n);free(gV_n);
    return -1.0f;
}

/* ======================================================= SILU GLU =========== */
/* y = silu(gate) * up, silu(x) = x * sigmoid(x) */
static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float siluf(float x) { return x * sigmoidf(x); }
static inline float silu_deriv(float x) {
    float s = sigmoidf(x);
    return s * (1.0f + x * (1.0f - s));
}
static void silu_glu_fwd(const float* gate, const float* up, float* y, int n) {
    for (int i = 0; i < n; i++) y[i] = siluf(gate[i]) * up[i];
}
static void silu_glu_backward(const float* gate, const float* up, const float* gY,
                              float* gGate, float* gUp, int n) {
    if (gGate) for (int i = 0; i < n; i++) gGate[i] = gY[i] * up[i] * silu_deriv(gate[i]);
    if (gUp)   for (int i = 0; i < n; i++) gUp[i]   = gY[i] * siluf(gate[i]);
}

float grad_check_silu_glu(int n, float eps) {
    rand_seed(42);
    float *g=f32_alloc(n), *u=f32_alloc(n), *y=f32_alloc(n);
    float *gY=f32_alloc(n), *gG_a=f32_alloc(n), *gU_a=f32_alloc(n);
    float *gG_n=f32_alloc(n), *gU_n=f32_alloc(n);
    if (!g||!u||!y||!gY||!gG_a||!gU_a||!gG_n||!gU_n) goto fail;
    f32_fill_rand(g, n, 0.5f);
    f32_fill_rand(u, n, 0.5f);
    f32_fill_rand(gY, n, 0.5f);
    silu_glu_fwd(g, u, y, n);
    silu_glu_backward(g, u, gY, gG_a, gU_a, n);
    for (int i = 0; i < n; i++) {
        float o = g[i];
        g[i] = o + eps; silu_glu_fwd(g, u, y, n);
        float Lp = 0; for (int j = 0; j < n; j++) Lp += y[j] * gY[j];
        g[i] = o - eps; silu_glu_fwd(g, u, y, n);
        float Lm = 0; for (int j = 0; j < n; j++) Lm += y[j] * gY[j];
        g[i] = o;
        gG_n[i] = (Lp - Lm) / (2.0f * eps);
    }
    for (int i = 0; i < n; i++) {
        float o = u[i];
        u[i] = o + eps; silu_glu_fwd(g, u, y, n);
        float Lp = 0; for (int j = 0; j < n; j++) Lp += y[j] * gY[j];
        u[i] = o - eps; silu_glu_fwd(g, u, y, n);
        float Lm = 0; for (int j = 0; j < n; j++) Lm += y[j] * gY[j];
        u[i] = o;
        gU_n[i] = (Lp - Lm) / (2.0f * eps);
    }
    float err = max_abs_diff(gG_a, gG_n, n);
    float errU = max_abs_diff(gU_a, gU_n, n);
    if (errU > err) err = errU;
    free(g);free(u);free(y);free(gY);free(gG_a);free(gU_a);free(gG_n);free(gU_n);
    return err;
fail:
    free(g);free(u);free(y);free(gY);free(gG_a);free(gU_a);free(gG_n);free(gU_n);
    return -1.0f;
}

/* ======================================================= SOFTMAX+CE ======== */
/* L = -log(softmax(logits)[target]) */
static float softmax_ce_fwd(const float* logits, int vocab, int target,
                            float* probs) {
    float mx = logits[0];
    for (int i = 1; i < vocab; i++) if (logits[i] > mx) mx = logits[i];
    float sum = 0;
    for (int i = 0; i < vocab; i++) {
        probs[i] = expf(logits[i] - mx);
        sum += probs[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < vocab; i++) probs[i] *= inv;
    return -logf(probs[target] + 1e-12f);
}
/* Double-precision variant for finite-diff reference (avoids 1e-3 float
 * cancellation when (Lp - Lm) is divided by 2*eps). */
static double softmax_ce_fwd_d(const float* logits, int vocab, int target) {
    double mx = logits[0];
    for (int i = 1; i < vocab; i++) if (logits[i] > mx) mx = logits[i];
    double sum = 0;
    for (int i = 0; i < vocab; i++) sum += exp((double)logits[i] - mx);
    double pt = exp((double)logits[target] - mx) / sum;
    return -log(pt + 1e-20);
}
/* gLogits[i] = probs[i] - onehot[i] */
static void softmax_ce_backward(const float* probs, int vocab, int target,
                                float* gLogits) {
    for (int i = 0; i < vocab; i++) gLogits[i] = probs[i];
    gLogits[target] -= 1.0f;
}

float grad_check_softmax_ce(int vocab, float eps) {
    rand_seed(42);
    float *logits=f32_alloc(vocab), *probs=f32_alloc(vocab), *gL_a=f32_alloc(vocab);
    float *gL_n=f32_alloc(vocab);
    if (!logits||!probs||!gL_a||!gL_n) goto fail;
    f32_fill_rand(logits, vocab, 0.5f);
    int target = vocab / 2;
    float L = softmax_ce_fwd(logits, vocab, target, probs);
    (void)L;
    softmax_ce_backward(probs, vocab, target, gL_a);
    for (int i = 0; i < vocab; i++) {
        float o = logits[i];
        logits[i] = o + eps; double Lp = softmax_ce_fwd_d(logits, vocab, target);
        logits[i] = o - eps; double Lm = softmax_ce_fwd_d(logits, vocab, target);
        logits[i] = o;
        gL_n[i] = (float)((Lp - Lm) / (2.0 * eps));
    }
    float err = max_abs_diff(gL_a, gL_n, vocab);
    free(logits);free(probs);free(gL_a);free(gL_n);
    return err;
fail:
    free(logits);free(probs);free(gL_a);free(gL_n);
    return -1.0f;
}

/* ======================================================= DISPATCH ========== */
grad_op grad_op_from_name(const char* name) {
    if (!name) return GRAD_MATMUL;
    if (strcmp(name, "matmul") == 0)     return GRAD_MATMUL;
    if (strcmp(name, "rmsnorm") == 0)    return GRAD_RMSNORM;
    if (strcmp(name, "rope") == 0)       return GRAD_ROPE;
    if (strcmp(name, "attention") == 0)  return GRAD_ATTENTION;
    if (strcmp(name, "silu_glu") == 0)   return GRAD_SILU_GLU;
    if (strcmp(name, "softmax_ce") == 0) return GRAD_SOFTMAX_CE;
    return GRAD_MATMUL;
}
const char* grad_op_name(grad_op op) {
    switch (op) {
        case GRAD_MATMUL:    return "matmul";
        case GRAD_RMSNORM:   return "rmsnorm";
        case GRAD_ROPE:      return "rope";
        case GRAD_ATTENTION: return "attention";
        case GRAD_SILU_GLU:  return "silu_glu";
        case GRAD_SOFTMAX_CE:return "softmax_ce";
        default:             return "unknown";
    }
}

float grad_check_dispatch(grad_op op, float eps) {
    switch (op) {
        case GRAD_MATMUL:     return grad_check_matmul(4, 3, 5, eps);
        case GRAD_RMSNORM:    return grad_check_rmsnorm(16, eps);
        case GRAD_ROPE:       return grad_check_rope(2, 8, eps);
        case GRAD_ATTENTION:  return grad_check_attention(4, 8, eps);
        case GRAD_SILU_GLU:   return grad_check_silu_glu(16, eps);
        case GRAD_SOFTMAX_CE: return grad_check_softmax_ce(32, eps);
        default:              return -1.0f;
    }
}

/* Phase 1 compat shim — delegates to grad_check_matmul. */
float backward_matmul_grad_check(int m, int n, int k, float eps) {
    return grad_check_matmul(m, n, k, eps);
}
