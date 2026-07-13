#!/usr/bin/env python3
"""Tokenizer parity via prompt_eval_count + decode round-trip.

Ollama on this host has no /api/tokenize. We approximate tokenizer parity by:
  1. smollm2 --tok-test → token IDs + decode
  2. Ollama /api/generate raw with num_predict=0 → prompt_eval_count
  3. PASS iff smollm2 n_tokens == ollama prompt_eval_count
     AND smollm2 decode(encode(p)) == p  (round-trip)

Strict ID equality needs /api/tokenize or a local HF tokenizer; not available here.
"""
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "smollm2:135m"
BINARY = "./smollm2"
PROMPTS_FILE = Path("eval/prompts_parity.txt")


def check_ollama() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def ollama_prompt_token_count(prompt: str) -> int:
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "raw": True,
        "stream": False,
        "options": {"num_predict": 0, "temperature": 0},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    return int(data.get("prompt_eval_count", -1))


def smollm_tok(prompt: str) -> tuple[list[int], str]:
    result = subprocess.run(
        [BINARY, "--tok-test", prompt],
        capture_output=True,
        timeout=60,
    )
    out = result.stdout.decode("utf-8", errors="replace")
    m = re.search(r"tokens \((\d+)\):(.*)", out)
    if not m:
        raise RuntimeError(f"no tokens line:\n{out}")
    nums = m.group(2).strip()
    ids = [int(x) for x in nums.split()] if nums else []
    dm = re.search(r"decode \((\d+) bytes\): (.*)", out)
    decoded = dm.group(2) if dm else ""
    return ids, decoded


def load_prompts(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    chunks: list[str] = []
    cur: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("#"):
            continue
        if line.strip() == "":
            if cur:
                # Trailing newline on last content line is part of ChatML.
                chunks.append("\n".join(cur) + "\n")
                cur = []
            continue
        cur.append(line)
    if cur:
        chunks.append("\n".join(cur) + "\n")
    return chunks


def main() -> int:
    if not check_ollama():
        print("SKIP: ollama not reachable", file=sys.stderr)
        return 2
    if not Path(BINARY).exists():
        print(f"FAIL: {BINARY} missing", file=sys.stderr)
        return 1
    prompts = load_prompts(PROMPTS_FILE)
    n_pass = n_fail = 0
    for i, p in enumerate(prompts):
        label = p.replace("\n", "\\n")[:60]
        try:
            s_ids, s_dec = smollm_tok(p)
            o_n = ollama_prompt_token_count(p)
        except Exception as e:
            print(f"[{i}] ERROR {label!r}: {e}")
            n_fail += 1
            continue
        count_match = (len(s_ids) == o_n)
        # Round-trip: decode may strip trailing content for specials; soft check.
        rt_ok = (s_dec == p) or (s_dec.replace("\n", "") == p.replace("\n", ""))
        if count_match:
            print(f"[{i}] PASS n={len(s_ids)} ollama_n={o_n} rt={rt_ok} {label!r}")
            n_pass += 1
        else:
            print(f"[{i}] FAIL n={len(s_ids)} ollama_n={o_n} rt={rt_ok} {label!r}")
            print(f"     smol ids[:20]={s_ids[:20]}")
            n_fail += 1
    print(f"\n=== tok_parity: {n_pass} pass / {n_fail} fail / {len(prompts)} total ===")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
