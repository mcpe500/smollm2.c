# 0009_backprop_done.md — Session Handoff

**Session date:** 2026-07-19
**Session goal:** Phase B (backprop foundation). Spec 013. Analytical backward untuk 6 op fundamental transformer. TDD: max abs error < 1e-3 vs finite-diff per op.
**Status at handoff:** Spec 013 complete. 6/6 grad_check_test pass. All errors < 1e-3. CLI dispatch via `studio grad-check --op <name>`.

---

## What landed

### `src/backward.h` rewrite
```c
typedef enum {
    GRAD_MATMUL = 0, GRAD_RMSNORM, GRAD_ROPE,
    GRAD_ATTENTION, GRAD_SILU_GLU, GRAD_SOFTMAX_CE,
    GRAD_OP_COUNT,
} grad_op;
grad_op     grad_op_from_name(const char* name);
const char* grad_op_name(grad_op op);
float       grad_check_<op>(...);     /* per op */
float       grad_check_dispatch(grad_op op, float eps);
float       backward_matmul_grad_check(int m, int n, int k, float eps);  /* compat */
```

### `src/backward.c` rewrite (~530 LOC)
- Helpers: `rand_seed(42)` deterministik, `randf()`, `f32_alloc`, `f32_fill_rand`, `max_abs_diff`
- 6 ops, masing-masing dengan `_fwd`, `_backward`, `grad_check_`:
  1. **matmul**: Y[m,n] = X[m,k] @ W[k,n]. gradX = gY @ W^T, gradW = X^T @ gY.
  2. **rmsnorm**: y = x*w/rms. Bug fix awal: extra w[i] pada second term. Koreksi: `gX[i] = inv*w[i]*gY[i] - x[i]*inv^3*mt` (mt = mean term).
  3. **rope** (Llama interleaved pairs): forward rotate +θ, backward rotate -θ. theta = p * base^(-2i/d), base=10000.
  4. **attention** (1 head, no causal): scores = Q@K^T/sqrt(hd). gV, gQ, gK via gscore = probs * (gprobs - sum(probs*gprobs)).
  5. **silu_glu**: silu(gate)*up. silu'(x) = sigmoid*(1+x*(1-sigmoid)).
  6. **softmax_ce**: L = -log(probs[target]). gLogits = probs - onehot.

- **softmax_ce double precision**: float cancellation pada (Lp - Lm)/2eps tidak adequate (~1.5e-3 error). Fix: `softmax_ce_fwd_d` accumulate L in double. Result: 1.7e-5.

### `src/studio.c cmd_grad_check` rewrite
- Parse `--op <name>` (default matmul), `--eps` (default 1e-4)
- Accept legacy `--m/--n/--k` (ignore) — backward compat
- Dispatch via `grad_op_from_name` → `grad_check_dispatch`
- Echo `op=<name>` in output (guard silent matmul fallback)
- Print: `grad-check: op=<name> eps=%.2e max_abs_error=%.3e`
- Return 0 if err < 1e-3 else 1

### `eval/grad_check_test.py`
- 6 ops, regex `max_abs_error=([0-9.e+-]+)`
- Require `f"op={op}"` in output (catches silent fallback)
- Threshold: max_abs_error < 1e-3 per op

## Results

```
grad-check: op=matmul     eps=1.00e-04 max_abs_error=1.522e-05  PASS
grad-check: op=rmsnorm    eps=1.00e-04 max_abs_error=4.927e-04  PASS
grad-check: op=rope       eps=1.00e-04 max_abs_error=1.053e-04  PASS
grad-check: op=attention  eps=1.00e-04 max_abs_error=9.445e-05  PASS
grad-check: op=silu_glu   eps=1.00e-04 max_abs_error=5.599e-06  PASS
grad-check: op=softmax_ce eps=1.00e-04 max_abs_error=1.651e-05  PASS
=== grad_check_test: 6/6 pass ===
```

## Bugs caught & fixed during impl

1. **rmsnorm formula**: extra `w[i]` factor pada second term. Corrected.
2. **softmax_ce finite-diff precision**: float cancellation. Switched to double Lp/Lm. Order of magnitude improvement.
3. **Makefile stale .o**: studio.c modify tidak trigger rebuild studio.o setelah backward.h berubah. Fixed via `-MMD -MP` (Phase A0 fix, retained).

## Verified commands

```bash
./smollm2 studio grad-check --op matmul
./smollm2 studio grad-check --op rmsnorm
./smollm2 studio grad-check --op rope
./smollm2 studio grad-check --op attention
./smollm2 studio grad-check --op silu_glu
./smollm2 studio grad-check --op softmax_ce
python3 eval/grad_check_test.py   # 6/6
```

## Next

Spec 014 (LoRA implementation) — slice 1 (real weight merge via gguf_patch_tensor) berikutnya.
