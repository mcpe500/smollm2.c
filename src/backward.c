// backward.c — phase 1 minimal: matmul numerical gradient check
#include "backward.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Analytical: dL/dX = grad_Y @ W  (where Y = X @ W^T, W: n×k) */
static void matmul_backward_input(const float* gradY, const float* W,
                                  float* gradX, int m, int n, int k) {
    memset(gradX, 0, m * k * sizeof(float));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++) {
            float s = 0;
            for (int p = 0; p < n; p++) s += gradY[i * n + p] * W[p * k + j];
            gradX[i * k + j] = s;
        }
}

static void matmul_fwd(const float* X, const float* W, float* Y,
                       int m, int n, int k) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int p = 0; p < k; p++) s += X[i * k + p] * W[j * k + p];
            Y[i * n + j] = s;
        }
}

float backward_matmul_grad_check(int m, int n, int k, float eps) {
    float* X = malloc((size_t)m * k * sizeof(float));
    float* W = malloc((size_t)n * k * sizeof(float));
    float* Y = malloc((size_t)m * n * sizeof(float));
    float* gY = malloc((size_t)m * n * sizeof(float));
    float* gX_anal = malloc((size_t)m * k * sizeof(float));
    if (!X || !W || !Y || !gY || !gX_anal) {
        free(X); free(W); free(Y); free(gY); free(gX_anal);
        return -1.0f;
    }
    /* Deterministic small seed */
    srand(42);
    for (int i = 0; i < m * k; i++) X[i] = ((float)rand() / 2147483647.0f) - 0.5f;
    for (int i = 0; i < n * k; i++) W[i] = ((float)rand() / 2147483647.0f) - 0.5f;

    matmul_fwd(X, W, Y, m, n, k);

    /* grad_Y = ones * 0.1 (any non-trivial) */
    for (int i = 0; i < m * n; i++) gY[i] = 0.1f;

    /* Analytical grad */
    matmul_backward_input(gY, W, gX_anal, m, n, k);

    /* Numerical grad via finite difference */
    float max_err = 0.0f;
    for (int i = 0; i < m * k; i++) {
        float orig = X[i];
        X[i] = orig + eps;
        float* Yp = malloc(m * n * sizeof(float));
        matmul_fwd(X, W, Yp, m, n, k);
        float Lp = 0; for (int j = 0; j < m * n; j++) Lp += Yp[j] * gY[j];
        X[i] = orig - eps;
        matmul_fwd(X, W, Yp, m, n, k);
        float Lm = 0; for (int j = 0; j < m * n; j++) Lm += Yp[j] * gY[j];
        X[i] = orig;
        float gnum = (Lp - Lm) / (2.0f * eps);
        float diff = fabsf(gX_anal[i] - gnum);
        if (diff > max_err) max_err = diff;
        free(Yp);
    }

    free(X); free(W); free(Y); free(gY); free(gX_anal);
    return max_err;
}