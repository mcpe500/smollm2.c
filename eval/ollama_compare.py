#!/usr/bin/env python3
"""
Compare smollm2.c output vs Ollama SmolLM2:135m on 200 prompts.

Usage:
    python3 eval/ollama_compare.py             # full 200 prompts
    python3 eval/ollama_compare.py --limit 10  # quick test
    python3 eval/ollama_compare.py --limit 10 --verbose
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "smollm2:135m"
BINARY = "./smollm2"
PROMPTS_FILE = "eval/prompts.txt"
RESULTS_DIR = Path("eval/results")
MAX_TOKENS = 1000
CHAT_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

# Readability thresholds
PRINTABLE_THRESHOLD = 0.95
REPETITION_THRESHOLD = 0.30


def check_ollama() -> bool:
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def check_model_available() -> bool:
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3) as r:
            data = json.loads(r.read())
        models = [m["name"] for m in data.get("models", [])]
        return any(OLLAMA_MODEL in m for m in models)
    except Exception:
        return False


def ollama_generate(prompt: str) -> tuple[str, float]:
    """Returns (text, tok_per_sec). tok_per_sec=-1 on error."""
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": CHAT_TEMPLATE.format(prompt=prompt),
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": MAX_TOKENS,
            "top_k": 1,
        }
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        elapsed = time.time() - t0
        text = data.get("response", "")
        n_tok = data.get("eval_count", 0)
        tps = n_tok / elapsed if elapsed > 0 else 0.0
        return text, tps
    except Exception as e:
        return f"[ollama error: {e}]", -1.0


def smollm_generate(prompt: str) -> tuple[str, float]:
    """Returns (text, tok_per_sec). tok_per_sec=-1 on error."""
    cmd = [
        BINARY, "-p", prompt,
        "-n", str(MAX_TOKENS),
        "--temp", "0.0",
    ]
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=300
        )
        elapsed = time.time() - t0
        output = result.stdout.decode("utf-8", errors="replace")

        # Extract tok/s from timing line: [N tokens, Xs, Y tok/s]
        tps = -1.0
        timing_match = re.search(r"\[(\d+) tokens,\s*[\d.]+s,\s*([\d.]+) tok/s\]", output)
        if timing_match:
            n_tok = int(timing_match.group(1))
            tps = float(timing_match.group(2))

        # Strip timing line from text
        text = re.sub(r"\n?\[[\d]+ tokens,.*?\]\n?", "", output).strip()
        return text, tps
    except subprocess.TimeoutExpired:
        return "[smollm2 timeout]", -1.0
    except Exception as e:
        return f"[smollm2 error: {e}]", -1.0


def printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(1 for c in text if 32 <= ord(c) <= 126 or c in "\n\r\t")
    return printable / len(text)


def repetition_4gram(text: str) -> float:
    """Fraction of 4-grams that are the most-repeated one. 0=no repetition, 1=all same."""
    words = text.split()
    if len(words) < 4:
        return 0.0
    grams = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
    if not grams:
        return 0.0
    counts = Counter(grams)
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count / len(grams)


def is_coherent(text: str) -> bool:
    if not text or text.startswith("["):
        return False
    pr = printable_ratio(text)
    rr = repetition_4gram(text)
    return pr >= PRINTABLE_THRESHOLD and rr < REPETITION_THRESHOLD


def analyze(text: str) -> dict:
    return {
        "printable_ratio": round(printable_ratio(text), 4),
        "repetition_4gram": round(repetition_4gram(text), 4),
        "word_count": len(text.split()),
        "char_count": len(text),
        "coherent": is_coherent(text),
    }


def snippet(text: str, n: int = 80) -> str:
    s = text.replace("\n", " ").strip()
    return s[:n] + "..." if len(s) > n else s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Only run first N prompts (0=all)")
    parser.add_argument("--verbose", action="store_true", help="Print each prompt result")
    parser.add_argument("--skip-ollama", action="store_true", help="Only run smollm2.c")
    args = parser.parse_args()

    # Check binary exists
    if not Path(BINARY).exists():
        print(f"ERROR: {BINARY} not found. Run from project root and build first.")
        sys.exit(1)

    # Check Ollama
    use_ollama = not args.skip_ollama
    if use_ollama:
        if not check_ollama():
            print("ERROR: Ollama not running at localhost:11434.")
            print("  Start it with:  ollama serve")
            sys.exit(1)
        if not check_model_available():
            print(f"ERROR: Model {OLLAMA_MODEL} not found in Ollama.")
            print(f"  Pull it with:  ollama pull {OLLAMA_MODEL}")
            sys.exit(1)
        print(f"Ollama OK — model {OLLAMA_MODEL} available")

    # Load prompts
    prompts_path = Path(PROMPTS_FILE)
    if not prompts_path.exists():
        print(f"ERROR: {PROMPTS_FILE} not found.")
        sys.exit(1)
    prompts = [p.strip() for p in prompts_path.read_text().splitlines() if p.strip()]
    if args.limit > 0:
        prompts = prompts[:args.limit]

    print(f"Running {len(prompts)} prompts, max {MAX_TOKENS} tokens each (greedy)")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    result_file = RESULTS_DIR / f"{timestamp}.jsonl"

    smollm_coherent = 0
    ollama_coherent = 0
    smollm_tps_list = []
    failures_smollm = []
    failures_ollama = []

    with open(result_file, "w") as f:
        for i, prompt in enumerate(prompts, 1):
            print(f"[{i:3d}/{len(prompts)}] {prompt[:60]}", end="", flush=True)

            # Run smollm2.c
            smollm_text, smollm_tps = smollm_generate(prompt)
            smollm_stats = analyze(smollm_text)
            if smollm_stats["coherent"]:
                smollm_coherent += 1
            else:
                failures_smollm.append((i, prompt, smollm_text, smollm_stats))
            if smollm_tps > 0:
                smollm_tps_list.append(smollm_tps)

            # Run Ollama
            ollama_text, ollama_tps = ("", 0.0)
            ollama_stats = {}
            if use_ollama:
                ollama_text, ollama_tps = ollama_generate(prompt)
                ollama_stats = analyze(ollama_text)
                if ollama_stats["coherent"]:
                    ollama_coherent += 1
                else:
                    failures_ollama.append((i, prompt, ollama_text, ollama_stats))

            # Status indicator
            smollm_ok = "OK" if smollm_stats["coherent"] else "FAIL"
            ollama_ok = ("OK" if ollama_stats.get("coherent") else "FAIL") if use_ollama else ""
            tps_str = f" {smollm_tps:.1f}t/s" if smollm_tps > 0 else ""
            print(f"  smollm2={smollm_ok}{tps_str}" + (f"  ollama={ollama_ok}" if use_ollama else ""))

            if args.verbose:
                print(f"         smollm2: {snippet(smollm_text)}")
                if use_ollama:
                    print(f"         ollama:  {snippet(ollama_text)}")
                print()

            record = {
                "i": i,
                "prompt": prompt,
                "smollm2": {
                    "text": smollm_text,
                    "tok_per_sec": smollm_tps,
                    **smollm_stats,
                },
            }
            if use_ollama:
                record["ollama"] = {
                    "text": ollama_text,
                    "tok_per_sec": ollama_tps,
                    **ollama_stats,
                }
            f.write(json.dumps(record) + "\n")

    # Summary
    n = len(prompts)
    avg_tps = sum(smollm_tps_list) / len(smollm_tps_list) if smollm_tps_list else 0.0

    print()
    print("=" * 60)
    print(f"RESULTS — {n} prompts, max {MAX_TOKENS} tokens, greedy (temp=0)")
    print("=" * 60)
    print(f"smollm2.c  coherent: {smollm_coherent}/{n} ({100*smollm_coherent//n}%)  avg {avg_tps:.1f} tok/s")
    if use_ollama:
        print(f"ollama     coherent: {ollama_coherent}/{n} ({100*ollama_coherent//n}%)")
        gap = smollm_coherent - ollama_coherent
        if gap < -10:
            print(f"  => smollm2.c is {abs(gap)} prompts worse — likely inference bug")
        elif gap < 0:
            print(f"  => smollm2.c is {abs(gap)} prompts worse — minor divergence")
        else:
            print(f"  => smollm2.c matches or beats Ollama")

    if failures_smollm:
        print(f"\nsmollm2.c FAILURES ({len(failures_smollm)}):")
        for idx, p, text, stats in failures_smollm[:20]:
            print(f"  [{idx:3d}] {p[:50]!r}")
            print(f"         pr={stats['printable_ratio']:.2f} rep={stats['repetition_4gram']:.2f}  => {snippet(text, 70)!r}")

    if use_ollama and failures_ollama:
        print(f"\nOllama FAILURES ({len(failures_ollama)}):")
        for idx, p, text, stats in failures_ollama[:10]:
            print(f"  [{idx:3d}] {p[:50]!r}")
            print(f"         pr={stats['printable_ratio']:.2f} rep={stats['repetition_4gram']:.2f}  => {snippet(text, 70)!r}")

    print(f"\nFull results: {result_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
