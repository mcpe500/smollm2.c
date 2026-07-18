# 012_phase_a_bugfixes.md

## Prompt

Benerin studio WebUI: shell-injection via popen, model reload per request,
recv 64KB truncation, silent QLoRA/FullFT dispatch, rank cap corruption,
RAM gate inconsistency, uniform init salah, file-open-per-sample. Tambah
hardware-aware foundation (hw_probe.c sebagai single source of truth).

## Goal

Studio WebUI harus:
1. Aman dari shell injection (fork+execvp, bukan popen).
2. Cepat (model cached globally, reload hanya jika rope/kv precision change).
3. Menerima POST body besar (growable heap buffer, cap 8MB).
4. Refuse QLoRA/FullFT dengan error jelas (belum implemented).
5. RAM gate konsisten (konstanta di hw_probe.h).
6. LoRA init Gaussian (Box-Muller), bukan uniform.
7. Packed file dibuka sekali per epoch.
8. HW probe lengkap (cpu_cores, neon, dotprod, sve, atomics, mem, vm_rss).

## Why

Studio phase 4a-4c landing separuh. `popen` shell injection — security hole.
Model reload per request — 600ms × N request, unacceptable. recv 64KB cap —
silently truncate large prompt. Silent dispatch — user tidak tahu QLoRA/FullFT
gagal. Uniform init — LoRA convergence buruk. File-open-per-sample — disk thrash.

Hardware-aware: SmolLM2 jalan di Termux Android, RAM 1-3GB typical. Tanpa
probe lengkap, training mode dispatch bisa OOM handphone.

## Codebase Context

- `src/hw_probe.{c,h}` — REWRITE dengan hw_caps extended + hw_advice + hw_json + hw_suggest
- `src/web.c` — fix A1 (popen→fork+execvp + path_safe), A2 (global cache g_fwd/g_tok/g_gctx), A3 (recv grow heap), A6 (hw_json route)
- `src/studio.c` — fix A4 (refuse QLoRA/FullFT), cmd_hw (--json --suggest --simulate-mem-kb), cmd_grad_check (--op dispatch)
- `src/train.c` — fix A5 (LORA_MAX_RANK=256 heap), A6 (MEM_EMERGENCY_MB konstanta), A7 (Box-Muller Gaussian), A8 (single fopen packed)
- `Makefile` — `-MMD -MP` untuk header dep tracking, target `hw-test`
- `eval/studio_web_test.py` — TDD: path injection, large recv, model cache ratio
- `eval/hw_probe_test.py` — TDD: cpu/mem fields, simulate-mem-kb
- `eval/phase_a_test.py` — TDD: A4/A5/A6 refuse + rank cap + RAM gate

## Logical Change

### A0 — hw_probe.c extended

`hw_caps` struct field baru: `cpu_cores`, `cpu_neon`, `cpu_dotprod`,
`cpu_sve`, `cpu_atomics`, `cpu_model[128]`, `cpu_freq_max_mhz`, `vm_rss_mb`,
`max_rank_advised`, `lora_min_mb`, `qlora_min_mb`, `fullft_min_mb`,
`emergency_min_mb`.

API: `hw_probe(c)`, `hw_advice(c)` fills derived, `hw_json(c)` returns
malloc'd JSON string, `hw_suggest(c)` returns human-readable recommendation,
`hw_print(c)` legacy pretty-print.

CPU detection: `getauxval(AT_HWCAP)` pada ARM (HWCAP_ASIMD=neon,
HWCAP_ASIMDDOT=dotprod, HWCAP_SVE=sve, HWCAP_ATOMICS=atomics). Fallback
parse `/proc/cpuinfo` Features.

`hw_advice`:
- `max_seq_advised`: 1024/512/256/128 by mem tier (≥3GB/≥2GB/≥1GB/<1GB).
- `max_rank_advised`: 32/16/8 by mem (≥2.5GB/≥1.5GB/<1.5GB).
- `max_batch_advised`: 1 (single-threaded, gradient accumulation).
- `fullft_allowed`: mem_available_mb ≥ 2560.
- `qlora_recommended`: !fullft_allowed && mem_available_mb ≥ 900.

Konstanta: `MEM_START_LORA_MB=800`, `MEM_START_QLORA_MB=900`,
`MEM_START_FULLFT_MB=2560`, `MEM_EMERGENCY_MB=150`.

Test override: env `SMOLLIM2_SIM_MEM_KB=N` mengganti `mem_avail_kb`.

### A1 — Shell injection fix

`path_safe(path)` helper: reject bila mengandung `;`, `|`, backtick, `$`,
`<`, `>`, `\n`, `\r`, `\`, `"`, `'`. Tambah ke web.c.

`run_capture(argv[], out_buf, out_cap)` helper: `fork()` + `execvp()` child,
`pipe()` untuk capture stdout/stderr parent-side. Tidak ada shell.

`handle_studio_data`, `handle_studio_train`, `handle_studio_merge` rewrite:
build argv array, validate paths via `path_safe`, call `run_capture`.

### A2 — Model cache

Global: `g_fwd`, `g_tok`, `g_gctx`, `g_model_loaded`, `g_cache_rope`,
`g_cache_kv`. Load sekali di `handle_generate` bila `!g_model_loaded`.
Invalidation: bila request rope/kv beda dari cache, `forward_free` +
reload. `forward_reset(fwd)` per request (clear KV saja, tidak free weights).

`web_run` free semua cache di shutdown.

### A3 — Recv growable

`recv_request(fd)` dengan struct `reqbuf { char* data; size_t len, cap; }`.
Grow heap via `realloc` sampai `\r\n\r\n` header terminator terlihat.
Parse `Content-Length`, continue recv sampai body complete. Hard cap
`MAX_REQ_BODY = 8 * 1024 * 1024`.

### A4 — QLoRA/FullFT refuse

`cmd_train` di studio.c: setelah parse mode, bila `p.mode == TRAIN_QLORA`
print `"train: qlora refused — not implemented in this build (planned: spec 015 / phase D)"`,
return 2. Sama untuk `TRAIN_FULLFT`. Dilakukan SEBELUM RAM gate, sebelum
forward_load. Akan di-relax di phase D/E.

### A5 — Rank cap heap

`#define LORA_MAX_RANK 256`. `lora_accum` cek `t->rank > LORA_MAX_RANK`,
print error eksplisit, return -1. Sebelumnya `d_mid[64]` stack array —
silent corruption bila rank > 64.

### A6 — RAM gate konsisten

Konstanta di `hw_probe.h` (bukan magic numbers di studio.c/train.c).
Emergency threshold bumped 100→150 MB (margin lebih aman).

### A7 — Gaussian init

Box-Muller:
```c
float u1 = ((float)rand() + 1) / ((float)RAND_MAX + 1);
float u2 = ((float)rand() + 1) / ((float)RAND_MAX + 1);
float z = sqrtf(-2 * logf(u1)) * cosf(6.28318530718 * u2);
A[i] = z * sqrtf(2.0f / dim);  // Kaiming std
```
B tetap 0 (LoRA convention: ΔW=0 at init).

### A8 — Single fopen

`load_sample(FILE* f, long offset, ...)` statik. `train_run` buka packed
file sekali, pass `FILE*` ke `load_sample`. Sebelumnya: fopen/fclose per
sample = O(N) opens per epoch.

## Code Change

- `src/hw_probe.{c,h}` — full rewrite
- `src/web.c` — tambah path_safe, run_capture, recv_request, rewrite 5 handler, global cache
- `src/studio.c` — cmd_hw extended, cmd_grad_check --op dispatch, cmd_train A4 refuse
- `src/train.c` — LORA_MAX_RANK, Box-Muller, MEM_EMERGENCY_MB, load_sample refactor
- `Makefile` — `-MMD -MP`, `hw-test` target
- `eval/studio_web_test.py` — 8 tests
- `eval/hw_probe_test.py` — 8 tests
- `eval/phase_a_test.py` — 5 tests

## Why This Change

- fork+execvp (bukan sanitization regex): satu jalur aman, tidak mungkin
  shell metacharacter lewat. Regex sanitization mudah salah.
- Global cache (bukan LRU): studio single-user, satu model cukup. Invalidation
  on rope/kv precision change sudah cukup granular.
- Growable heap (bukan larger fixed buffer): 8MB cap tidak alokasi 8MB per
  request; grow dari 4KB.
- Konstanta di hw_probe.h (bukan magic numbers): satu source of truth,
  mudah tuning tanpa grep。
- Box-Muller (bukan approx): distribusi Gaussian yang benar untuk Kaiming init.

## Test Simulation & Tracing

### A1 path injection
```
POST /studio/data {"out": "/tmp/x; rm -rf /", "text": "x"}
→ path_safe rejects ";" → 400 {"ok":false,"error":"unsafe path"}
```

### A2 model cache
```
1st /generate hello → model load (e.g. 3s) → response
2nd /generate hello → cache hit (e.g. 0.2s) → response
ratio 2nd/1st << 5
```

### A3 recv large
```
POST /generate {"prompt":"x"*200000, "n":1}
→ Content-Length 200021 → growable recv works → 200 (bukan 400 Bad Request)
```

### A4 refuse
```
studio train --mode qlora ...
→ stderr "qlora refused — not implemented in this build"
→ rc=2
```

## Manual Testing Plan

```bash
make clean && make && make studio

# HW probe
./smollm2 studio hw --json
./smollm2 studio hw --suggest
./smollm2 studio hw --simulate-mem-kb 500000 --json

# Studio web security
python3 eval/studio_web_test.py   # expect 8/8 pass

# HW probe tests
python3 eval/hw_probe_test.py     # expect 8/8 pass

# Phase A tests
python3 eval/phase_a_test.py      # expect 5/5 pass
```

## Status

- [x] Spec written
- [x] Implementation
- [x] Verified (21/21 tests pass: 8 studio_web + 8 hw_probe + 5 phase_a)
- [x] Handoff written (0008_phase_a_bugfixes_done.md)
