#!/usr/bin/env python3
"""Logits parity: smollm2.c --logits-json vs Ollama /api/generate (raw+logprobs).

Compares by TOKEN BYTES (Ollama logprobs don't expose token IDs on this host):
  - prompt token COUNT (smollm2 n_tokens vs ollama prompt_eval_count)
  - top-1 token bytes match
  - top-5 token-bytes set Jaccard
  - reports smollm2 raw logits + ollama logprobs (different scales; ranking only)

Exit 0 iff every prompt PASSes. Exit 1 on FAIL. Exit 2 if Ollama unreachable.
"""
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "smollm2:135m"
BINARY = "./smollm2"
PROMPTS_FILE = Path("eval/prompts_parity.txt")
TOPK_JACCARD_MIN = 0.6  # top-5 bytes sets


def check_ollama() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def load_prompts(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    chunks: list[str] = []
    cur: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("#"):
            continue
        if line.strip() == "":
            if cur:
                chunks.append("\n".join(cur) + "\n")
                cur = []
            continue
        cur.append(line)
    if cur:
        chunks.append("\n".join(cur) + "\n")
    return chunks


def ollama_top(prompt: str, top_k: int = 10) -> dict:
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "raw": True,
        "stream": False,
        "logprobs": True,
        "top_logprobs": top_k,
        "options": {
            "temperature": 0,
            "num_predict": 1,
            "top_k": 1,
        },
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())

    topk = []
    logprobs = data.get("logprobs") or []
    if logprobs and isinstance(logprobs, list):
        last = logprobs[-1]
        tlp = last.get("top_logprobs") or []
        for item in tlp:
            b = item.get("bytes")
            if b is None:
                # Fall back to token string utf-8 bytes.
                tok = item.get("token", "")
                b = list(tok.encode("utf-8"))
            topk.append({
                "bytes": list(b),
                "token": item.get("token", ""),
                "logprob": float(item.get("logprob", 0.0)),
            })
    return {
        "response": data.get("response", ""),
        "prompt_eval_count": int(data.get("prompt_eval_count", -1)),
        "topk": topk,
        "argmax_bytes": topk[0]["bytes"] if topk else None,
        "argmax_token": topk[0]["token"] if topk else None,
    }


def smollm_logits(prompt: str) -> dict:
    result = subprocess.run(
        [BINARY, "--logits-json", prompt],
        capture_output=True,
        timeout=180,
    )
    out = result.stdout.decode("utf-8", errors="replace").strip()
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("{") and "prompt_tokens" in line:
            return json.loads(line)
    raise RuntimeError(
        f"no JSON line in --logits-json output:\n{out}\n"
        f"stderr:\n{result.stderr.decode()}"
    )


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def bytes_key(b) -> tuple:
    return tuple(int(x) for x in b)


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
            s = smollm_logits(p)
            o = ollama_top(p)
        except Exception as e:
            print(f"[{i}] ERROR {label!r}: {e}")
            n_fail += 1
            continue

        s_n = s.get("n_tokens", len(s.get("prompt_tokens", [])))
        o_n = o.get("prompt_eval_count", -1)
        count_match = (s_n == o_n)

        s_topk = s.get("topk", [])
        o_topk = o.get("topk", [])
        s_argmax_b = bytes_key(s_topk[0]["bytes"]) if s_topk else None
        o_argmax_b = bytes_key(o["argmax_bytes"]) if o.get("argmax_bytes") is not None else None
        argmax_match = (s_argmax_b is not None and s_argmax_b == o_argmax_b)

        s_set = {bytes_key(t["bytes"]) for t in s_topk[:5] if "bytes" in t}
        o_set = {bytes_key(t["bytes"]) for t in o_topk[:5]}
        jac = jaccard(s_set, o_set) if o_set else None

        reasons = []
        if not count_match:
            reasons.append(f"tok_count smol={s_n} ollama={o_n}")
        if not argmax_match:
            reasons.append(
                f"argmax_bytes smol={s_argmax_b} ollama={o_argmax_b}"
            )
        if jac is not None and jac < TOPK_JACCARD_MIN:
            reasons.append(f"top5_jaccard={jac:.2f}")

        status = "FAIL" if reasons else "PASS"
        if status == "PASS":
            n_pass += 1
        else:
            n_fail += 1

        # Human-readable argmax tokens
        s_tok = bytes(s_topk[0]["bytes"]).decode("utf-8", "replace") if s_topk else "?"
        o_tok = o.get("argmax_token") or (
            bytes(o["argmax_bytes"]).decode("utf-8", "replace") if o.get("argmax_bytes") else "?"
        )

        print(f"[{i}] {status} {label!r}")
        print(f"     tokens: smol={s_n} ollama={o_n} match={count_match}")
        print(f"     argmax: smol={s_tok!r}(id={s.get('argmax')}) "
              f"ollama={o_tok!r} match={argmax_match}")
        if s_topk:
            print(f"     smol  top5: "
                  f"{[(bytes(t['bytes']).decode('utf-8','replace'), round(t['logit'], 3)) for t in s_topk[:5]]}")
        if o_topk:
            print(f"     ollama top5: "
                  f"{[(t['token'], round(t['logprob'], 3)) for t in o_topk[:5]]}")
        if jac is not None:
            print(f"     top5_jaccard={jac:.2f}")
        if reasons:
            print(f"     reasons: {', '.join(reasons)}")

    print(f"\n=== parity: {n_pass} pass / {n_fail} fail / {len(prompts)} total ===")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
