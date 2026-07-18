# 0008_phase_a_bugfixes_done.md — Session Handoff

**Session date:** 2026-07-18
**Session goal:** Phase A0 (HW-aware foundation) + Phase A (8 studio bugfixes). Spec 012.
**Status at handoff:** Spec 012 complete. All 8 bugs fixed. HW probe extended with cpu_cores/neon/dotprod/sve/atomics/mem/vm_rss + advisory + JSON output. 21/21 tests pass (8 studio_web + 8 hw_probe + 5 phase_a).

---

## What landed

### A0 — `src/hw_probe.{c,h}` extended
- `hw_caps` struct: `cpu_cores, cpu_neon, cpu_dotprod, cpu_sve, cpu_atomics, cpu_model[128], cpu_freq_max_mhz, vm_rss_mb, max_rank_advised, lora_min_mb, qlora_min_mb, fullft_min_mb, emergency_min_mb`
- Konstanta: `MEM_START_LORA_MB=800, MEM_START_QLORA_MB=900, MEM_START_FULLFT_MB=2560, MEM_EMERGENCY_MB=150`
- API: `hw_probe, hw_advice, hw_print, hw_json (malloc'd), hw_suggest`
- ARM detection: `getauxval(AT_HWCAP)` — HWCAP_ASIMD (neon), HWCAP_ASIMDDOT (dotprod), HWCAP_SVE, HWCAP_ATOMICS. Fallback `/proc/cpuinfo` Features.
- Test override: `SMOLLIM2_SIM_MEM_KB` env var

### A1 — Shell injection fix (`src/web.c`)
- `path_safe(path)` helper: reject `;`, `|`, backtick, `$`, `<`, `>`, `\n`, `\r`, `\`, `"`, `'`
- `run_capture(argv[], buf, cap)` helper: fork + execvp (no shell) + pipe capture
- Rewrite `handle_studio_data/train/merge`: argv array + path_safe + run_capture
- `/studio/data` injection → HTTP 400 `{"ok":false,"error":"unsafe path"}`

### A2 — Model cache (`src/web.c`)
- Globals: `g_fwd, g_tok, g_gctx, g_model_loaded, g_cache_rope, g_cache_kv`
- Load once in `handle_generate` bila `!g_model_loaded`
- Invalidation on rope/kv mode change: `forward_free` + reload
- Per-request `forward_reset(fwd)` (clear KV only)
- `web_run` frees cache at shutdown
- 5-request ratio: 0.58× first (down from 6.49× pre-fix)

### A3 — Recv growable (`src/web.c`)
- `recv_request(fd, reqbuf*)` struct with `data, len, cap`
- Growable via realloc sampai header `\r\n\r\n` + Content-Length body complete
- `MAX_REQ_BODY = 8 * 1024 * 1024`
- 200KB prompt test: HTTP 200 (previously 400 truncated)

### A4 — QLoRA/FullFT refuse (`src/studio.c`)
- `cmd_train`: setelah parse mode, intercept `TRAIN_QLORA`/`TRAIN_FULLFT` sebelum RAM gate
- Stderr: `"<mode> refused — not implemented in this build (planned: spec 015 / phase D)"` (atau spec 016 / phase E)
- Return 2
- Akan di-relax di phase D/E

### A5 — Rank cap (`src/train.c`)
- `#define LORA_MAX_RANK 256`
- `lora_accum` cek `t->rank > LORA_MAX_RANK`, error eksplisit, return -1
- Sebelumnya `d_mid[64]` stack array — silent corruption bila rank > 64

### A6 — RAM gate konsisten
- Konstanta di `hw_probe.h` (bukan magic numbers)
- Emergency 100→150 MB
- `train.c` watchdog pakai `(long)MEM_EMERGENCY_MB * 1024`

### A7 — Box-Muller Gaussian init (`src/train.c`)
```c
float u1 = ((float)rand() + 1) / ((float)RAND_MAX + 1);
float u2 = ((float)rand() + 1) / ((float)RAND_MAX + 1);
float z = sqrtf(-2 * logf(u1)) * cosf(6.28318530718 * u2);
A[i] = z * sqrtf(2.0f / dim);  // Kaiming std
```
B tetap 0 (LoRA convention).

### A8 — Single fopen (`src/train.c`)
- `static int load_sample(FILE* f, long offset, int n_tokens, int* out, int max_out)`
- `train_run` buka packed file sekali per run, pass `FILE*`

### Makefile
- `-MMD -MP` for header dep tracking (sebelumnya .o stale bila .h berubah)
- `clean` removes .d files
- `hw-test` target

## Tests

- `eval/studio_web_test.py` 8/8: page, hw, attn, data, data-inj, merge-inj, recv-large, model-cache
- `eval/hw_probe_test.py` 8/8: cpu/mem fields, simulate-mem-kb, advisory
- `eval/phase_a_test.py` 5/5: A4 refuse (qlora, fullft), A5 rank cap, A6 emergency gate

## Verified commands

```bash
./smollm2 studio hw --json
./smollm2 studio hw --suggest
./smollm2 studio hw --simulate-mem-kb 500000 --json
python3 eval/studio_web_test.py   # 8/8
python3 eval/hw_probe_test.py     # 8/8
python3 eval/phase_a_test.py      # 5/5
```

## Next

Spec 013 (backprop analytical) + spec 014 (LoRA implementation).
