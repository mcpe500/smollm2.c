#!/usr/bin/env python3
"""Sweep all rope x kv x attn combos; report tok/s + parity verdict.

For each (rope, kv, attn):
  - run ./smollm2 -p "Hello, how are you?" -n 50 --temp 0 --rope R --kv K --attn A
    (2 warmup, 3 measured, take median)
  - record first-token argmax for "hello" ChatML (expected 19556 for hard parity)
  - mark PASS soft (argmax in valid range / no crash), or FAIL on crash
  - write eval/results/attn_bench_<date>.md
"""
import re
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "smollm2"
OUTDIR = ROOT / "eval" / "results"

ROPES = ["f32", "f16", "q8"]
KVS   = ["f32", "f16", "q8"]
ATTNS = ["naive", "flash"]


def run(*args, timeout=120):
    return subprocess.run([str(BIN), *args],
                          capture_output=True, timeout=timeout, cwd=str(ROOT))


def tok_s_median(rope, kv, attn, n=50, trials=3, warmup=2):
    for _ in range(warmup):
        run("-p", "Hello, how are you?", "-n", str(n), "--temp", "0",
            "--rope", rope, "--kv", kv, "--attn", attn,
            timeout=180)
    vals = []
    for _ in range(trials):
        r = run("-p", "Hello, how are you?", "-n", str(n), "--temp", "0",
                "--rope", rope, "--kv", kv, "--attn", attn,
                timeout=180)
        out = r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")
        m = re.search(r"([0-9.]+) tok/s", out)
        vals.append(float(m.group(1)) if m else 0.0)
    vals.sort()
    return vals[len(vals) // 2]


def argmax_hello(rope, kv, attn):
    r = run("--logits",
            "system\nYou are a helpful AI assistant named SmolLM, "
            "trained by Hugging Face\nuser\nhello\nassistant\n",
            "--rope", rope, "--kv", kv, "--attn", attn,
            timeout=120)
    out = r.stdout.decode("utf-8", "replace")
    for line in out.splitlines():
        if line.startswith("argmax:"):
            parts = line.split()
            if len(parts) >= 2:
                return parts[1]
    return None


def main() -> int:
    if not BIN.exists():
        print(f"FAIL: {BIN} missing", file=sys.stderr)
        return 1
    OUTDIR.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    out_path = OUTDIR / f"attn_bench_{stamp}.md"

    rows = []
    for rope in ROPES:
        for kv in KVS:
            for attn in ATTNS:
                print(f"\n=== {rope}/{kv}/{attn} ===", flush=True)
                r = run("-p", "hi", "-n", "8", "--temp", "0",
                        "--rope", rope, "--kv", kv, "--attn", attn,
                        timeout=120)
                if r.returncode != 0:
                    rows.append((rope, kv, attn, "CRASH", 0.0, "?"))
                    print(f"  CRASH rc={r.returncode}")
                    continue
                ts = tok_s_median(rope, kv, attn)
                am = argmax_hello(rope, kv, attn)
                verdict = "PASS" if am == "19556" else "soft"
                if am is None:
                    verdict = "FAIL"
                rows.append((rope, kv, attn, verdict, ts, am))
                print(f"  tok/s={ts:.1f} argmax={am} verdict={verdict}")

    # Render markdown
    lines = ["# attn_bench " + stamp, "",
             "| rope | kv | attn | parity | tok/s | hello_argmax |",
             "|---|---|---|---|---|---|"]
    for rope, kv, attn, verdict, ts, am in rows:
        lines.append(f"| {rope} | {kv} | {attn} | {verdict} | {ts:.1f} | {am} |")
    # Winner = max tok/s among PASS
    pass_rows = [r for r in rows if r[3] == "PASS"]
    if pass_rows:
        winner = max(pass_rows, key=lambda r: r[4])
        lines.append("")
        lines.append(f"**Winner (PASS + max tok/s):** "
                     f"{winner[0]}/{winner[1]}/{winner[2]} = {winner[4]:.1f} tok/s")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())