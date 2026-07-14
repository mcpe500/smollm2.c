#!/usr/bin/env python3
"""Smoke test for --rope/--kv/--attn mode flags. Must FAIL before impl exists.

Asserts:
  1. `--help` lists --rope, --kv, --attn
  2. default f32/f32/naive produces valid output (sanity)
  3. each non-default combo runs without crash (smoke)
  4. invalid value rejected (--rope=qq)
  5. default still parity-green: argmax of 'Hello' ChatML is 19556 (Hello)
"""
import subprocess
import sys
from pathlib import Path

BINARY = Path(__file__).resolve().parent.parent / "smollm2"


def run(*args, timeout=120):
    return subprocess.run([str(BINARY), *args],
                          capture_output=True, timeout=timeout,
                          cwd=str(BINARY.parent))


def test_help_lists_modes() -> bool:
    r = run("--help")
    out = r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")
    ok = ("--rope" in out) and ("--kv" in out) and ("--attn" in out)
    print(f"[help] {'PASS' if ok else 'FAIL'}: rope={('--rope' in out)} "
          f"kv={('--kv' in out)} attn={('--attn' in out)}")
    return ok


def test_default_runs() -> bool:
    r = run("-p", "Hello, how are you?", "-n", "16", "--temp", "0")
    out = r.stdout.decode("utf-8", "replace")
    err = r.stderr.decode("utf-8", "replace")
    ok = (r.returncode == 0) and (len(out.strip()) > 0)
    print(f"[default] {'PASS' if ok else 'FAIL'}: rc={r.returncode} "
          f"len={len(out.strip())}")
    if not ok:
        print(f"  stderr: {err[:300]}")
    return ok


def test_smoke_combos() -> bool:
    """Each combo must complete without crash and produce non-empty output."""
    rope_vals = ["f32", "f16", "q8"]
    kv_vals = ["f32", "f16", "q8"]
    attn_vals = ["naive", "flash"]
    ok_all = True
    for rope in rope_vals:
        for kv in kv_vals:
            for attn in attn_vals:
                r = run("-p", "hi", "-n", "8", "--temp", "0",
                        "--rope", rope, "--kv", kv, "--attn", attn)
                out = r.stdout.decode("utf-8", "replace")
                err = r.stderr.decode("utf-8", "replace")
                ok = (r.returncode == 0) and (len(out.strip()) > 0)
                tag = "PASS" if ok else "FAIL"
                print(f"[combo {rope}/{kv}/{attn}] {tag}: "
                      f"rc={r.returncode} len={len(out.strip())}")
                if not ok:
                    print(f"  stderr: {err[:300]}")
                ok_all = ok_all and ok
    return ok_all


def test_invalid_value_rejected() -> bool:
    """Unknown --rope value must error out (non-zero rc or clear stderr)."""
    r = run("-p", "hi", "-n", "4", "--rope", "qq")
    err = r.stderr.decode("utf-8", "replace")
    ok = (r.returncode != 0) or ("invalid" in err.lower()) or ("unknown" in err.lower())
    print(f"[invalid] {'PASS' if ok else 'FAIL'}: rc={r.returncode}")
    return ok


def test_default_argmax_hello() -> bool:
    """Sanity: default modes preserve correctness — ChatML 'hello' argmax is Hello."""
    r = run("--logits", "system\n"
            "You are a helpful AI assistant named SmolLM, "
            "trained by Hugging Face\n"
            "user\nhello\nassistant\n")
    out = r.stdout.decode("utf-8", "replace")
    argmax = None
    for line in out.splitlines():
        if line.startswith("argmax:"):
            parts = line.split()
            if len(parts) >= 2:
                argmax = parts[1]
            break
    ok = argmax == "19556"
    print(f"[argmax] {'PASS' if ok else 'FAIL'}: got={argmax} want=19556")
    return ok


def main() -> int:
    if not BINARY.exists():
        print(f"FAIL: {BINARY} missing", file=sys.stderr)
        return 1
    results = [
        test_help_lists_modes(),
        test_default_runs(),
        test_smoke_combos(),
        test_invalid_value_rejected(),
        test_default_argmax_hello(),
    ]
    print(f"\n=== attn_matrix_test: {sum(results)}/{len(results)} pass ===")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())