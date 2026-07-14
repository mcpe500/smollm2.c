#!/usr/bin/env python3
"""TDD smoke for studio phase 1. Must FAIL before impl exists.

Asserts:
  1. `./smollm2 studio data-build --in raw.txt --out packed.bin --fmt auto`
     produces packed.bin with token count > 0 and detected format RAW.
  2. `./smollm2 studio data-build --in instruct.jsonl --out packed.bin
     --fmt auto` → detected INSTRUCT.
  3. `./smollm2 studio data-build --in sharegpt.jsonl --out packed.bin
     --fmt auto` → detected SHAREGPT.
  4. `./smollm2 studio gguf-rewrite --in base.gguf --out copy.gguf`
     followed by `./smollm2 -m copy.gguf --logits 'hello'` argmax = 19556.
  5. `./smollm2 studio grad-check` exits 0, output reports max abs error
     and number is < 1e-3.
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


def run(*args, timeout=300, cwd=None):
    return subprocess.run([str(BIN), *args],
                          capture_output=True, timeout=timeout,
                          cwd=str(cwd or ROOT))


def write_sample(name: str, body: str) -> Path:
    p = Path(tempfile.gettempdir()) / name
    p.write_text(body, encoding="utf-8")
    return p


def test_data_raw_auto() -> bool:
    sample = write_sample("studio_raw.txt",
                          "Hello world.\nThis is a raw text test.\n")
    out = Path(tempfile.gettempdir()) / "packed_raw.bin"
    if out.exists(): out.unlink()
    r = run("studio", "data-build",
            "--in", str(sample), "--out", str(out), "--fmt", "auto",
            "--model", str(GGUF))
    out_blob = r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")
    ok = (r.returncode == 0) and out.exists() and out.stat().st_size > 0
    fmt_ok = "RAW" in out_blob or "raw" in out_blob
    print(f"[raw] {'PASS' if ok and fmt_ok else 'FAIL'}: "
          f"rc={r.returncode} size={out.stat().st_size if out.exists() else 0} "
          f"fmt_ok={fmt_ok}")
    if not ok: print(f"  out: {out_blob[:400]}")
    return ok and fmt_ok


def test_data_instruct_auto() -> bool:
    sample = write_sample("studio_instruct.jsonl", "\n".join([
        json.dumps({"prompt": "Hello?", "completion": "Hi there!"}),
        json.dumps({"prompt": "Bye?",   "completion": "See you!"}),
    ]) + "\n")
    out = Path(tempfile.gettempdir()) / "packed_instruct.bin"
    if out.exists(): out.unlink()
    r = run("studio", "data-build",
            "--in", str(sample), "--out", str(out), "--fmt", "auto",
            "--model", str(GGUF))
    out_blob = r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")
    ok = (r.returncode == 0) and out.exists() and out.stat().st_size > 0
    fmt_ok = "INSTRUCT" in out_blob
    print(f"[instruct] {'PASS' if ok and fmt_ok else 'FAIL'}: "
          f"rc={r.returncode} size={out.stat().st_size if out.exists() else 0} "
          f"fmt_ok={fmt_ok}")
    if not ok: print(f"  out: {out_blob[:400]}")
    return ok and fmt_ok


def test_data_sharegpt_auto() -> bool:
    sample = write_sample("studio_sharegpt.jsonl", "\n".join([
        json.dumps({"messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]}),
    ]) + "\n")
    out = Path(tempfile.gettempdir()) / "packed_sharegpt.bin"
    if out.exists(): out.unlink()
    r = run("studio", "data-build",
            "--in", str(sample), "--out", str(out), "--fmt", "auto",
            "--model", str(GGUF))
    out_blob = r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")
    ok = (r.returncode == 0) and out.exists() and out.stat().st_size > 0
    fmt_ok = "SHAREGPT" in out_blob
    print(f"[sharegpt] {'PASS' if ok and fmt_ok else 'FAIL'}: "
          f"rc={r.returncode} size={out.stat().st_size if out.exists() else 0} "
          f"fmt_ok={fmt_ok}")
    if not ok: print(f"  out: {out_blob[:400]}")
    return ok and fmt_ok


def test_gguf_rewrite_roundtrip() -> bool:
    if not GGUF.exists():
        print(f"[gguf] SKIP: {GGUF} missing")
        return False
    out = Path(tempfile.gettempdir()) / "copy.gguf"
    if out.exists(): out.unlink()
    r = run("studio", "gguf-rewrite",
            "--in", str(GGUF), "--out", str(out))
    ok0 = (r.returncode == 0) and out.exists() and out.stat().st_size > 0
    if not ok0:
        print(f"[gguf] FAIL: rewrite rc={r.returncode}")
        return False

    chatml = ("system\nYou are a helpful AI assistant named SmolLM, "
              "trained by Hugging Face\nuser\nhello\nassistant\n")
    r0 = run("-m", str(GGUF), "--logits", chatml)
    r1 = run("-m", str(out), "--logits", chatml)
    a0 = re.search(r"argmax:\s*(\d+)", r0.stdout.decode("utf-8", "replace"))
    a1 = re.search(r"argmax:\s*(\d+)", r1.stdout.decode("utf-8", "replace"))
    argmax_orig = a0.group(1) if a0 else None
    argmax_copy = a1.group(1) if a1 else None
    ok = (argmax_orig is not None
          and argmax_copy is not None
          and argmax_orig == argmax_copy)
    print(f"[gguf] {'PASS' if ok else 'FAIL'}: "
          f"orig={argmax_orig} copy={argmax_copy}")
    return ok


def test_grad_check() -> bool:
    r = run("studio", "grad-check", "--m", "4", "--n", "3", "--k", "5")
    out_blob = r.stdout.decode("utf-8", "replace")
    err_blob = r.stderr.decode("utf-8", "replace")
    m = re.search(r"max_abs_error=([0-9.e+-]+)", out_blob + err_blob)
    if not m:
        print(f"[grad] FAIL: no max_abs_error in output\n"
              f"  out={out_blob[:200]}\n  err={err_blob[:200]}")
        return False
    err = float(m.group(1))
    ok = (r.returncode == 0) and (err < 1e-3)
    print(f"[grad] {'PASS' if ok else 'FAIL'}: rc={r.returncode} "
          f"max_abs_error={err:.2e}")
    return ok


def main() -> int:
    if not BIN.exists():
        print(f"FAIL: {BIN} missing", file=sys.stderr)
        return 1
    results = [
        test_data_raw_auto(),
        test_data_instruct_auto(),
        test_data_sharegpt_auto(),
        test_gguf_rewrite_roundtrip(),
        test_grad_check(),
    ]
    print(f"\n=== studio_smoke: {sum(results)}/{len(results)} pass ===")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())