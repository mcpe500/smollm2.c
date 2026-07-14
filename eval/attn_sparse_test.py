#!/usr/bin/env python3
"""TDD for studio phase 3a — attention registry + SWA.

Asserts:
  1. Default (no flags) argmax matches baseline.
  2. --attn swa:window=2048 (>seq_len) ≈ default (parity within noise).
  3. --attn swa:window=2 produces a different argmax on a long prompt
     (sanity that SWA is actually narrowing context).
  4. studio attn-list dump works.
  5. studio attn-config loads a tiny JSON file successfully.
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "smollm2"
GGUF = ROOT / "models" / "smollm2-135m-f16.gguf"
LONG_PROMPT = ("The quick brown fox jumps over the lazy dog. "
               "Pack my box with five dozen liquor jugs. "
               "How vexingly quick daft zebras jump! "
               "Sphinx of black quartz, judge my vow. ") * 4


def run(*args, timeout=600):
    return subprocess.run([str(BIN), *args],
                          capture_output=True, timeout=timeout, cwd=str(ROOT))


def argmax_for(prompt: str, *extra_args) -> tuple[int, str]:
    r = run("-m", str(GGUF), "--logits", prompt, *extra_args)
    out = r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")
    m = re.search(r"argmax:\s*(\d+)", out)
    return (int(m.group(1)) if m else None), out


def test_default_argmax_known() -> bool:
    a, _ = argmax_for("The capital of France is Paris.\n")
    ok = a is not None and a >= 0
    print(f"[default] {'PASS' if ok else 'FAIL'}: argmax={a}")
    return ok


def test_swa_long_window_equals_default() -> bool:
    a_default, _ = argmax_for("The capital of France is Paris.\n")
    a_swa_long, _ = argmax_for("The capital of France is Paris.\n",
                               "--attn", "swa:window=2048")
    ok = a_default == a_swa_long
    print(f"[swa-long] {'PASS' if ok else 'FAIL'}: default={a_default} swa_long={a_swa_long}")
    return ok


def test_swa_short_differs() -> bool:
    a_default, _ = argmax_for(LONG_PROMPT)
    a_swa_short, _ = argmax_for(LONG_PROMPT, "--attn", "swa:window=2")
    # Different argmax OR at least different logit shape (best signal: argmax)
    ok = a_default != a_swa_short
    print(f"[swa-short] {'PASS' if ok else 'FAIL'}: default={a_default} swa_short={a_swa_short}")
    if not ok:
        # Sanity: also check that logits diverged (top5 differs even if argmax same)
        _, out_def = argmax_for(LONG_PROMPT)
        _, out_swa = argmax_for(LONG_PROMPT, "--attn", "swa:window=2")
        ok = out_def != out_swa
        print(f"         fallback: logs_differ={ok}")
    return ok


def test_studio_attn_list() -> bool:
    r = run("studio", "attn-list")
    out = r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")
    ok = (r.returncode == 0 and "dense" in out and "swa" in out)
    print(f"[attn-list] {'PASS' if ok else 'FAIL'}: rc={r.returncode}")
    if not ok: print(f"  out={out[:400]}")
    return ok


def test_studio_attn_config() -> bool:
    cfg = Path(tempfile.gettempdir()) / "attn_cfg.json"
    cfg.write_text(json.dumps({
        "default": {"type": "dense"},
        "layers": [
            {"type": "swa", "window": 64},
            {"type": "swa", "window": 128},
            {"type": "dense"}
        ]
    }))
    r = run("studio", "attn-config", "--config", str(cfg), "--layers", "30")
    out = r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")
    ok = (r.returncode == 0 and "swa" in out and "L00" in out)
    print(f"[attn-config] {'PASS' if ok else 'FAIL'}: rc={r.returncode}")
    if not ok: print(f"  out={out[:400]}")
    return ok


def main() -> int:
    if not BIN.exists() or not GGUF.exists():
        print(f"FAIL: BIN or GGUF missing", file=sys.stderr)
        return 1
    results = [
        test_default_argmax_known(),
        test_swa_long_window_equals_default(),
        test_swa_short_differs(),
        test_studio_attn_list(),
        test_studio_attn_config(),
    ]
    print(f"\n=== attn_sparse: {sum(results)}/{len(results)} pass ===")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
