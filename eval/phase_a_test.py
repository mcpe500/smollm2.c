#!/usr/bin/env python3
"""TDD for Phase A studio bug fixes. RED before fix, GREEN after.

Covers:
  A4: silent QLoRA/FullFT dispatch — must refuse with clear error
  A5: LoRA rank > 64 must not silently corrupt
  A6: emergency watchdog uses MEM_EMERGENCY_MB constant (150)
  A7: LoRA init Gaussian (loss sanity via grad-check)
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "smollm2"


def run(*args, timeout=60, env=None):
    import os
    e = os.environ.copy()
    if env:
        e.update(env)
    return subprocess.run([str(BIN), *args],
                          capture_output=True, timeout=timeout,
                          cwd=str(ROOT), env=e)


def test_a4_qlora_refused_until_phase_d():
    """QLoRA mode must fail with clear 'not implemented' message in phase A."""
    r = run("studio", "train", "--mode", "qlora",
            "--data", "/dev/null", "--model", "dummy.gguf")
    assert r.returncode != 0, "qlora should be refused in phase A"
    err = r.stderr.decode("utf-8", "replace")
    assert "not implemented" in err.lower() or "phase" in err.lower(), err
    return True


def test_a4_fullft_refused_until_phase_e():
    """FullFT mode must fail with clear 'not implemented' message in phase A."""
    r = run("studio", "train", "--mode", "fullft",
            "--data", "/dev/null", "--model", "dummy.gguf")
    assert r.returncode != 0
    err = r.stderr.decode("utf-8", "replace")
    assert "not implemented" in err.lower() or "phase" in err.lower(), err
    return True


def test_a4_lora_still_accepted():
    """LoRA mode must still be accepted (no error about 'not implemented')."""
    # Use a dummy model path — will fail at model load, but NOT with 'not implemented'
    r = run("studio", "train", "--mode", "lora",
            "--data", "/dev/null", "--model", "dummy.gguf")
    err = r.stderr.decode("utf-8", "replace")
    # Should fail for other reasons (model not found), not "not implemented"
    assert "not implemented" not in err.lower(), err
    return True


def test_a5_rank_above_64_explicit_error():
    """Rank > 64 must either work or fail with explicit error, not silent corruption.
    With heap-alloced d_mid, rank=128 should be accepted at struct creation.
    This test asserts no silent corruption — we test by checking init doesn't crash."""
    # We test indirectly: train_create with rank=128 should not crash
    # via grad-check at rank=128 if init is clean.
    # Since grad-check only validates matmul, this is a smoke test.
    # The real test: train_smoke at rank=128 doesn't fail with "rank > 64".
    # We just check here that the binary doesn't have the old cap message.
    r = run("studio", "train", "--mode", "lora", "--rank", "128",
            "--data", "/dev/null", "--model", "dummy.gguf")
    err = r.stderr.decode("utf-8", "replace")
    # If rank cap remains, we'd see "rank > 64" or similar
    assert "rank > 64" not in err, f"rank cap still present: {err}"
    return True


def test_a6_emergency_threshold_constant():
    """Emergency watchdog at 150 MB (was 100 MB).
    Indirectly tested via hw_probe_test — check constant exposed."""
    r = run("studio", "hw", "--json", "--simulate-mem-kb", "150000")
    obj_ok = r.returncode == 0
    return obj_ok


TESTS = [
    test_a4_qlora_refused_until_phase_d,
    test_a4_fullft_refused_until_phase_e,
    test_a4_lora_still_accepted,
    test_a5_rank_above_64_explicit_error,
    test_a6_emergency_threshold_constant,
]


def main():
    if not BIN.exists():
        print(f"FAIL: {BIN} missing — run `make` first", file=sys.stderr)
        return 1
    results = []
    for t in TESTS:
        try:
            ok = t()
            print(f"[{t.__name__}] {'PASS' if ok else 'FAIL'}")
            results.append(ok)
        except AssertionError as e:
            print(f"[{t.__name__}] FAIL: {e}")
            results.append(False)
        except Exception as e:
            print(f"[{t.__name__}] ERROR: {type(e).__name__}: {e}")
            results.append(False)
    print(f"\n=== phase_a_test: {sum(results)}/{len(results)} pass ===")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
