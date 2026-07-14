#!/usr/bin/env python3
"""TDD for studio phase 4a — WebUI routes.

Asserts:
  1. GET /studio → 200 + 'Studio' in body
  2. GET /studio/hw → JSON with mem_avail
  3. GET /studio/attn → dense/swa listed
"""
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "smollm2"
GGUF = ROOT / "models" / "smollm2-135m-f16.gguf"
PORT = 18082


def free_port(port: int) -> bool:
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def http_get(path: str, timeout: float = 5.0) -> tuple[int, str]:
    url = f"http://127.0.0.1:{PORT}{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return 0, str(e)


def main() -> int:
    if not BIN.exists() or not GGUF.exists():
        print("FAIL: BIN or GGUF missing", file=sys.stderr)
        return 1
    if not free_port(PORT):
        print(f"FAIL: port {PORT} in use", file=sys.stderr)
        return 1

    proc = subprocess.Popen(
        [str(BIN), "studio", "web", "--port", str(PORT), "--model", str(GGUF)],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # Wait for listen
        for _ in range(40):
            time.sleep(0.25)
            code, body = http_get("/studio")
            if code == 200:
                break
        else:
            print("[studio] FAIL: server never responded")
            return 1

        results = []

        # 1. GET /studio
        ok = code == 200 and ("Studio" in body or "studio" in body)
        print(f"[studio-page] {'PASS' if ok else 'FAIL'}: code={code} len={len(body)}")
        results.append(ok)

        # 2. GET /studio/hw
        code, body = http_get("/studio/hw")
        try:
            j = json.loads(body)
        except Exception:
            j = {}
        ok = code == 200 and "mem_avail_kb" in j and j["mem_avail_kb"] > 0
        print(f"[studio-hw] {'PASS' if ok else 'FAIL'}: code={code} j={j}")
        results.append(ok)

        # 3. GET /studio/attn
        code, body = http_get("/studio/attn")
        ok = code == 200 and ("dense" in body and "swa" in body)
        print(f"[studio-attn] {'PASS' if ok else 'FAIL'}: code={code} body={body[:200]}")
        results.append(ok)

        print(f"\n=== studio_web: {sum(results)}/{len(results)} pass ===")
        return 0 if all(results) else 1
    finally:
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=3)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
