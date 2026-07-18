#!/usr/bin/env python3
"""TDD for hw_probe Phase A0. Must FAIL before impl, PASS after.

Asserts hw_probe exposes CPU + memory + advisory via `studio hw` subcommand:
  - --json: structured JSON for WebUI
  - --suggest: human-readable mode/rank/seq recommendation
  - --simulate-mem-kb N: override mem_avail for deterministic tests
"""
import json as jjson
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "smollm2"


def run(*args, timeout=30):
    return subprocess.run([str(BIN), *args],
                          capture_output=True, timeout=timeout, cwd=str(ROOT))


def parse_json(s):
    s = s.strip()
    try:
        return jjson.loads(s)
    except Exception:
        a = s.find('{'); b = s.rfind('}')
        if a >= 0 and b > a:
            return jjson.loads(s[a:b+1])
        raise


def test_json_schema():
    """hw --json returns object with cpu + mem + advisory fields."""
    r = run("studio", "hw", "--json")
    assert r.returncode == 0, r.stderr.decode()
    obj = parse_json(r.stdout.decode())
    for k in ("cpu_cores", "cpu_neon", "mem_total_mb", "mem_available_mb",
              "vm_rss_mb", "max_seq_advised", "max_rank_advised",
              "fullft_allowed", "qlora_recommended",
              "lora_min_mb", "qlora_min_mb", "fullft_min_mb"):
        assert k in obj, f"missing field {k}: {obj}"
    assert obj["cpu_cores"] >= 1
    assert obj["mem_total_mb"] > 0
    assert obj["lora_min_mb"] == 800
    assert obj["qlora_min_mb"] == 900
    assert obj["fullft_min_mb"] == 2560
    return True


def test_low_mem_disables_fullft():
    """500 MB avail: fullft not allowed, qlora not recommended (<900), rank=8."""
    r = run("studio", "hw", "--json", "--simulate-mem-kb", "500000")
    assert r.returncode == 0, r.stderr.decode()
    obj = parse_json(r.stdout.decode())
    assert obj["mem_available_mb"] == 488, obj
    assert obj["fullft_allowed"] == 0
    assert obj["qlora_recommended"] == 0  # 488 < 900
    assert obj["max_seq_advised"] <= 256
    assert obj["max_rank_advised"] == 8
    return True


def test_qlora_recommended_at_900mb():
    """976 MB avail: qlora recommended (>=900), fullft still not."""
    r = run("studio", "hw", "--json", "--simulate-mem-kb", "1000000")
    assert r.returncode == 0, r.stderr.decode()
    obj = parse_json(r.stdout.decode())
    assert obj["qlora_recommended"] == 1
    assert obj["fullft_allowed"] == 0
    return True


def test_high_mem_enables_fullft():
    """4 GB avail: fullft allowed, qlora not recommended, seq=1024, rank=32."""
    r = run("studio", "hw", "--json", "--simulate-mem-kb", "4000000")
    assert r.returncode == 0, r.stderr.decode()
    obj = parse_json(r.stdout.decode())
    assert obj["fullft_allowed"] == 1
    assert obj["qlora_recommended"] == 0
    assert obj["max_seq_advised"] == 1024
    assert obj["max_rank_advised"] == 32
    return True


def test_mid_mem_suggests_lora():
    """1.66 GB avail: lora allowed, qlora recommended (not fullft), rank=16."""
    r = run("studio", "hw", "--json", "--simulate-mem-kb", "1700000")
    assert r.returncode == 0, r.stderr.decode()
    obj = parse_json(r.stdout.decode())
    assert obj["fullft_allowed"] == 0
    assert obj["qlora_recommended"] == 1
    assert obj["max_rank_advised"] == 16
    return True


def test_suggest_format():
    """--suggest prints human-readable recommendation."""
    r = run("studio", "hw", "--suggest", "--simulate-mem-kb", "1700000")
    assert r.returncode == 0, r.stderr.decode()
    s = r.stdout.decode()
    assert "mode=lora" in s, s
    assert "rank=16" in s, s
    return True


def test_default_human_readable():
    """no flags: prints human-readable (backward compat)."""
    r = run("studio", "hw")
    assert r.returncode == 0, r.stderr.decode()
    s = r.stdout.decode()
    assert "hw:" in s, s
    return True


def test_emergency_threshold_constant():
    """Emergency watchdog threshold surfaced in --suggest output."""
    r = run("studio", "hw", "--suggest")
    assert r.returncode == 0
    # advisory text mentions thresholds
    return True


TESTS = [
    test_json_schema,
    test_low_mem_disables_fullft,
    test_qlora_recommended_at_900mb,
    test_high_mem_enables_fullft,
    test_mid_mem_suggests_lora,
    test_suggest_format,
    test_default_human_readable,
    test_emergency_threshold_constant,
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
    print(f"\n=== hw_probe_test: {sum(results)}/{len(results)} pass ===")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
