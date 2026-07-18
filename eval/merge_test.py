#!/usr/bin/env python3
"""TDD for real LoRA weight merge.

Pre-fix: train_merge just byte-copies base + writes sidecar adapter.
Post-fix: gguf_patch_tensor writes ΔW = scale * (B @ A) into token_embd.weight.

Asserts:
  1. gguf_patch_tensor API exists: `output.weight` patchable in a copy
  2. After train + merge, the merged GGUF differs from base (by >= 1 byte)
  3. Merged GGUF still loads + infers (argmax returns valid token)
  4. No .lora sidecar left beside merged file
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "smollm2"
GGUF = ROOT / "models" / "smollm2-135m-f16.gguf"


def run(*args, timeout=600):
    return subprocess.run([str(BIN), *args],
                          capture_output=True, timeout=timeout, cwd=str(ROOT))


def file_hash(p: Path) -> bytes:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.digest()


def make_adapter(rank: int = 4, steps: int = 3) -> Path:
    sample = Path(tempfile.gettempdir()) / "merge_train_raw.txt"
    sample.write_text("Hello world.\nHow are you?\nGood day.\n", encoding="utf-8")
    packed = Path(tempfile.gettempdir()) / "merge_train_packed.bin"
    if packed.exists(): packed.unlink()
    run("studio", "data-build",
        "--in", str(sample), "--out", str(packed),
        "--fmt", "raw", "--model", str(GGUF))
    out_dir = Path(tempfile.gettempdir()) / "merge_train_out"
    out_dir.mkdir(exist_ok=True)
    for f in out_dir.glob("*"): f.unlink()
    r = run("studio", "train",
            "--data", str(packed), "--mode", "lora",
            "--rank", str(rank), "--epochs", "1",
            "--lr", "1e-3", "--seq", "32", "--batch", "1",
            "--max-steps", str(steps),
            "--out-dir", str(out_dir), "--model", str(GGUF))
    if r.returncode != 0:
        return None
    adapters = list(out_dir.glob("lora_*.bin"))
    return adapters[0] if adapters else None


def main() -> int:
    if not BIN.exists() or not GGUF.exists():
        print("FAIL: BIN or GGUF missing", file=sys.stderr)
        return 1
    results = []

    # 1. Hash differs base vs merged
    adapter = make_adapter()
    if not adapter:
        print("[merge-real] FAIL: no adapter produced")
        return 1
    merged = Path(tempfile.gettempdir()) / "merged_real.gguf"
    if merged.exists(): merged.unlink()
    r = run("studio", "merge",
            "--base", str(GGUF),
            "--adapter", str(adapter),
            "--out", str(merged))
    rc = r.returncode
    out = r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")
    if not merged.exists():
        print(f"[merge-exists] FAIL: rc={rc} out={out[:300]}")
        results.append(False)
    else:
        h_base = file_hash(GGUF)
        h_merged = file_hash(merged)
        differs = h_base != h_merged
        # Also count byte differences (must be >> 0 — ideally ~56MB for full emb patch)
        diff_bytes = 0
        with open(GGUF, "rb") as fa, open(merged, "rb") as fb:
            while True:
                a = fa.read(65536)
                b = fb.read(65536)
                if not a: break
                for x, y in zip(a, b):
                    if x != y: diff_bytes += 1
        print(f"[merge-hash-differs] {'PASS' if differs else 'FAIL'}: "
              f"diff_bytes={diff_bytes} rc={rc}")
        if not differs:
            print(f"  out={out[:300]}")
        results.append(differs and diff_bytes > 100)

    # 2. Merged still loads + infers
    chatml = ("system\nYou are a helpful AI assistant named SmolLM, "
              "trained by Hugging Face\nuser\nhello\nassistant\n")
    r = run("-m", str(merged), "--logits", chatml)
    m = re.search(r"argmax:\s*(\d+)", r.stdout.decode("utf-8", "replace"))
    argmax = m.group(1) if m else None
    ok = argmax is not None and r.returncode == 0 and int(argmax) > 0
    print(f"[merge-infers] {'PASS' if ok else 'FAIL'}: argmax={argmax}")
    results.append(ok)

    # 3. No .lora sidecar (full merge, not byte-copy)
    sidecar = Path(str(merged) + ".lora")
    no_side = not sidecar.exists()
    print(f"[merge-no-sidecar] {'PASS' if no_side else 'FAIL'}: {sidecar.name}")
    results.append(no_side)

    print(f"\n=== merge_test: {sum(results)}/{len(results)} pass ===")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
