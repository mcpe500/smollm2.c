#!/usr/bin/env python3
"""Smoke test for --heavy pipeline. Must FAIL before impl exists.

Asserts:
  1. `--help` lists --heavy and --heavy-think-n
  2. `./smollm2 -p <Q> --heavy` stdout contains
     THINK, ANSWER, VERIFY section headers
  3. `./smollm2 -p <Q>` (no --heavy) stays single-pass
     (no THINK/ANSWER/VERIFY headers)
  4. REJECT path (if present) triggers RE-ANSWER section
"""
import re
import subprocess
import sys
from pathlib import Path

BINARY = Path(__file__).resolve().parent.parent / "smollm2"


def run(*args, timeout=120):
    return subprocess.run([str(BINARY), *args],
                          capture_output=True, timeout=timeout, cwd=str(BINARY.parent))


def test_help_lists_heavy() -> bool:
    r = run("--help")
    out = r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")
    ok = "--heavy" in out and "--heavy-think-n" in out
    print(f"[help] {'PASS' if ok else 'FAIL'}: "
          f"--heavy={'--heavy' in out} "
          f"--heavy-think-n={'--heavy-think-n' in out}")
    return ok


def test_heavy_sections_present() -> bool:
    r = run("-p", "What is 2+2?", "--heavy", "-n", "48",
            "--heavy-think-n", "32", "--heavy-verify-n", "24",
            "--temp", "0")
    out = r.stdout.decode("utf-8", "replace")
    has_think = "=== THINK ===" in out
    has_ans = "=== ANSWER ===" in out
    has_ver = "=== VERIFY ===" in out
    ok = has_think and has_ans and has_ver
    print(f"[heavy] {'PASS' if ok else 'FAIL'}: "
          f"think={has_think} ans={has_ans} ver={has_ver}")
    return ok


def test_plain_prompt_unchanged() -> bool:
    r = run("-p", "hello", "-n", "16", "--temp", "0")
    out = r.stdout.decode("utf-8", "replace")
    has_think = "=== THINK ===" in out
    has_ans = "=== ANSWER ===" in out
    has_ver = "=== VERIFY ===" in out
    ok = not (has_think or has_ans or has_ver)
    print(f"[plain] {'PASS' if ok else 'FAIL'}: "
          f"no-section={ok}")
    return ok


def test_reject_triggers_reanswer() -> bool:
    """REJECT path is probabilistic on 135M. We only verify that the
    RE-ANSWER section appears IF the verifier said REJECT."""
    r = run("-p", "What is the capital of France?", "--heavy", "-n", "48",
            "--heavy-think-n", "32", "--heavy-verify-n", "32",
            "--temp", "0")
    out = r.stdout.decode("utf-8", "replace")
    has_reject = bool(re.search(r"^\s*REJECT\b", out, re.M))
    has_reanswer = "=== RE-ANSWER ===" in out
    # Invariant: RE-ANSWER iff REJECT present (or verifier said neither — ok).
    if has_reject and not has_reanswer:
        print(f"[gate] FAIL: REJECT seen without RE-ANSWER")
        return False
    print(f"[gate] PASS: re_answer_consistent={not has_reject or has_reanswer}")
    return True


def main() -> int:
    if not BINARY.exists():
        print(f"FAIL: {BINARY} missing", file=sys.stderr)
        return 1
    results = [
        test_help_lists_heavy(),
        test_heavy_sections_present(),
        test_plain_prompt_unchanged(),
        test_reject_triggers_reanswer(),
    ]
    print(f"\n=== heavy_test: {sum(results)}/{len(results)} pass ===")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())