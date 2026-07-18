# 013_backprop_analytical.md

## Prompt

Implementasi analytical backpropagation (chain rule) untuk 6 op fundamental
transformer. Bukan numerical finite-diff yang 1000× lebih lambat. TDD:
max abs error < 1e-3 vs centered finite-diff per op.

## Goal

```bash
./smollm2 studio grad-check --op matmul       # max_abs_error < 1e-3
./smollm2 studio grad-check --op rmsnorm
./smollm2 studio grad-check --op rope
./smollm2 studio grad-check --op attention
./smollm2 studio grad-check --op silu_glu
./smollm2 studio grad-check --op softmax_ce
```

Setiap op punya: forward, analytical backward, finite-diff grad check
(centered, eps=1e-4). Threshold max abs error < 1e-3.

## Why

Backprop untuk LoRA/QLoRA/FullFT butuh gradient yang akurat terhadap setiap
op di transformer. Numerical finite-diff (forward-only) O(2N) per parameter
— untuk 135M param, tidak feasible. Analytical backward O(1) per parameter,
tapi harus diverified benar via finite-diff per op.

6 op ini compose menjadi full transformer backward:
```
softmax_ce → matmul(lm_head) → rmsnorm(final) → [layer L]
[layer L] = matmul(o_proj) → attention → rope → matmul(q/k/v) → rmsnorm(attn)
           + matmul(down) → silu_glu → matmul(gate/up) → rmsnorm(ffn)
           + residual additions
```

## Codebase Context

- `src/backward.{c,h}` — REPLACE stub numerical matmul check dengan 6 op analytical + dispatch
- `src/studio.c` — `cmd_grad_check` rewrite: parse `--op`, dispatch via `grad_check_dispatch`
- `eval/grad_check_test.py` — TDD: 6 ops, max_abs_error regex, op name echo check
- `Makefile` — backward.c tetap di SRC (sudah ada)

## Logical Change

### Op enum dan dispatch

```c
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
float       grad_check_dispatch(grad_op op, float eps);
```

### Per-op implementation pattern

Setiap op punya 3 fungsi:
1. `<op>_forward(...)` — forward pass
2. `<op>_backward(...)` — analytical gradient via chain rule
3. `grad_check_<op>(...)` — centered finite-diff vs analytical

Finite-diff formula (centered):
```
g_x[i] ≈ (L(x[i]+eps) - L(x[i]-eps)) / (2*eps)
```
Loss `L = sum(Y * gY)` (dot product dengan upstream grad).

### Matmul

Forward: `Y[m,n] = X[m,k] @ W[k,n]`
Backward:
- `gX[m,k] = gY[m,n] @ W[k,n]^T`
- `gW[k,n] = X[m,k]^T @ gY[m,n]`

### RMSNorm

Forward: `y[i] = x[i] * w[i] / rms`, `rms = sqrt(mean(x^2) + eps_rms)`
Backward (chain rule):
- `gX[j] = gY[j]*w[j]/rms - x[j]/(n*rms^3) * sum_i(gY[i]*w[i]*x[i])`
- `gW[i] = gY[i] * x[i] / rms`

Bug fix: formula awal punya faktor `w[i]` ekstra pada second term. Koreksi:
`gX[i] = inv*w[i]*gY[i] - x[i]*inv^3*mt` (mt = mean term).

### RoPE (Llama-style interleaved pairs)

Forward per pair (2i, 2i+1) posisi p:
```
y[2i]   = x[2i]*cos(θ) - x[2i+1]*sin(θ)
y[2i+1] = x[2i]*sin(θ) + x[2i+1]*cos(θ)
θ = p * base^(-2i/d), base=10000
```
Backward: inverse rotation oleh -θ.

### Attention (single head, no causal)

Forward:
```
scores[t,s] = (Q[t] · K[s]) / sqrt(hd)
probs = softmax(scores)
out[t] = sum_s probs[t,s] * V[s]
```
Backward:
- `gV[s,h] += sum_t probs[t,s] * gOut[t,h]`
- `gprobs[t,s] = sum_h gOut[t,h] * V[s,h]`
- `gscore[t,s] = probs[t,s] * (gprobs[t,s] - sum_s' probs[t,s'] * gprobs[t,s'])`
- `gQ[t,h] = sum_s gscore[t,s] * K[s,h] / sqrt(hd)`
- `gK[s,h] = sum_t gscore[t,s] * Q[t,h] / sqrt(hd)`

### SiLU GLU

Forward: `y = silu(gate) * up`, `silu(x) = x * sigmoid(x)`
Backward:
- `gGate = gY * up * silu'(gate)`, `silu'(x) = sigmoid(x) * (1 + x*(1-sigmoid(x)))`
- `gUp = gY * silu(gate)`

### Softmax + Cross-Entropy

Forward: `L = -log(softmax(logits)[target])`
Backward: `gLogits[i] = probs[i] - onehot[i]`

Bug fix: finite-diff float precision inadequate (cancellation error ~1e-3).
Solution: double-precision L accumulation untuk finite-diff reference.

## Code Change

- `src/backward.h` — grad_op enum, 6 op declarations, dispatch
- `src/backward.c` — ~530 LOC: 6 ops + helpers (rand_seed deterministik, f32_alloc, f32_fill_rand, max_abs_diff) + dispatch + compat shim `backward_matmul_grad_check`
- `src/studio.c` — `cmd_grad_check` rewrite: parse `--op`, dispatch, echo `op=<name>`, return rc by threshold
- `eval/grad_check_test.py` — 6 ops, regex `max_abs_error=([0-9.e+-]+)`, require `f"op={op}"` in output (guard silent matmul fallback)

## Why This Change

- Per-op grad check (bukan end-to-end): isolated failure diagnosis. Bila
  attention fails, kita tahu attention backward salah, bukan整个 pipeline.
- Deterministic RNG (rand_seed(42)): reproducible. Setiap op punya config
  kecil (matmul 4×3×5, rmsnorm n=16, rope 2 heads × 8 hd, attention 4 tokens × 8 hd,
  silu_glu n=16, softmax_ce vocab=32) — finite-diff cepat.
- Double precision finite-diff untuk softmax_ce: float32 cancellation pada
  (Lp - Lm) / 2eps tidak adequate; butuh double untuk reference.
- Compat shim `backward_matmul_grad_check`: existing tests (phase 1) tidak break.

## Logic / Pseudocode

```
grad_check_matmul(m, n, k, eps):
    X[m,k], W[k,n], Y[m,n], gY[m,n] = rand(...)
    gX_a[m,k], gW_a[k,n] = matmul_backward(X, W, gY)
    for i in m*k:
        X[i] += eps; fwd(); Lp = dot(Y, gY)
        X[i] -= 2*eps; fwd(); Lm = dot(Y, gY)
        X[i] += eps  # restore
        gX_n[i] = (Lp - Lm) / (2*eps)
    # same for W
    return max(max_abs_diff(gX_a, gX_n), max_abs_diff(gW_a, gW_n))
```

## Test Simulation & Tracing

### Matmul (4×3×5, eps=1e-4)
Expected: max_abs_error ≈ 1e-5 (eps^2 truncation dominated, no precision issue).

### RMSNorm (n=16, eps=1e-4, eps_rms=1e-5)
Expected: max_abs_error ≈ 5e-4 (slightly worse due to chain rule complexity).

### Softmax CE (vocab=32, eps=1e-4)
Initial failure: 1.5e-3 (float cancellation). Fix: double Lp, Lm. Result:
1.7e-5.

### Rope (T=4, hd=8, base=10000)
Expected: ~1e-4 (cosf precision dominates).

## Manual Testing Plan

```bash
make

# Per-op
for op in matmul rmsnorm rope attention silu_glu softmax_ce; do
    ./smollm2 studio grad-check --op $op
done
# Expect: all max_abs_error < 1e-3

# Full suite
python3 eval/grad_check_test.py   # expect 6/6 pass
```

## Status

- [x] Spec written
- [x] Implementation
- [x] Verified (6/6 grad_check_test pass, all errors < 1e-3)
- [x] Handoff written (0009_backprop_done.md)
