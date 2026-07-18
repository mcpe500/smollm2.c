// backward.h — Phase B: per-op analytical backward + finite-diff grad check.
#ifndef BACKWARD_H
#define BACKWARD_H

typedef enum {
    GRAD_MATMUL = 0,
    GRAD_RMSNORM,
    GRAD_ROPE,
    GRAD_ATTENTION,
    GRAD_SILU_GLU,
    GRAD_SOFTMAX_CE,
    GRAD_OP_COUNT,
} grad_op;

grad_op     grad_op_from_name(const char* name);
const char* grad_op_name(grad_op op);

/* Each grad_check_<op> runs forward with random inputs, computes analytical
 * gradient, verifies against centered finite-diff. Returns max abs error,
 * or negative on error. eps = finite-diff step (1e-4 recommended). */
float grad_check_matmul(int m, int n, int k, float eps);
float grad_check_rmsnorm(int n, float eps);
float grad_check_rope(int n_heads, int head_dim, float eps);
float grad_check_attention(int n_tokens, int head_dim, float eps);
float grad_check_silu_glu(int n, float eps);
float grad_check_softmax_ce(int vocab, float eps);

/* Dispatch by enum. */
float grad_check_dispatch(grad_op op, float eps);

/* Backward-compat: original phase 1 matmul numerical check. */
float backward_matmul_grad_check(int m, int n, int k, float eps);

#endif // BACKWARD_H
