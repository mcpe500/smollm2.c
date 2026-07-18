#!/usr/bin/env python3
"""TDD for Phase B backprop. Each op: analytical vs finite-diff, max err < 1e-3.

Ops:
  matmul     Y[m,n] = X[m,k] @ W[k,n]
  rmsnorm    y = x * w / sqrt(mean(x^2)+eps)
  rope       rotate pairs by position-dependent angle
  attention  Q@K^T softmax weighted V (single head, no causal)
  silu_glu   silu(gate)*up
  softmax_ce L = -log p[target]
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "smollm2"


def run(*args, timeout=60):
    return subprocess.run([str(BIN), *args],
                          capture_output=True, timeout=timeout, cwd=str(ROOT))


def grad_check(op, eps="1e-4"):
    r = run("studio", "grad-check", "--op", op, "--eps", eps)
    out = r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")
    # Require op name in output — guards against silent fallback to matmul.
    if f"op={op}" not in out:
        return None, f"op name not echoed in output: {out[:300]}"
    m = re.search(r"max_abs_error=([0-9.e+-]+)", out)
    if not m:
        return None, f"no max_abs_error in output: {out[:300]}"
    return float(m.group(1)), out


TESTS_OPS = ["matmul", "rmsnorm", "rope", "attention", "silu_glu", "softmax_ce"]


def main():
    if not BIN.exists():
        print(f"FAIL: {BIN} missing — run `make` first", file=sys.stderr)
        return 1
    results = []
    for op in TESTS_OPS:
        err, detail = grad_check(op)
        ok = err is not None and err < 1e-3
        if err is not None:
            print(f"[{op}] {'PASS' if ok else 'FAIL'}: max_abs_error={err:.2e}")
        else:
            print(f"[{op}] FAIL: {detail}")
        results.append(ok)
    print(f"\n=== grad_check_test: {sum(results)}/{len(results)} pass ===")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
