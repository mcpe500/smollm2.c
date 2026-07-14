// backward.h — autograd minimal (phase 1: grad check only)
#ifndef BACKWARD_H
#define BACKWARD_H

/* Numerical gradient check untuk Y = X @ W^T (X: m×k, W: n×k, Y: m×n).
   Compare analytical dL/dX = grad_Y @ W against finite-difference.
   Returns max abs error. eps = finite-diff step. */
float backward_matmul_grad_check(int m, int n, int k, float eps);

#endif // BACKWARD_H