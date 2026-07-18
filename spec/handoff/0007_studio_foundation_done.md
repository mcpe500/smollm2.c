# 0007_studio_foundation_done.md — Session Handoff

**Session date:** 2026-06-28 (retroactive)
**Session goal:** Studio foundation phase 1-3 + WebUI phase 4a-4c. Data packing, minimal LoRA trainer (lm_head), web server with `/generate` + `/studio/*` routes.
**Status at handoff:** Spec 008 complete. Spec 011 phase 4a-4c complete. Studio WebUI runs, training produces adapter files, merge sidecar. BUT phase A bugfixes pending (see spec 012 for the 8 bugs).

---

## What landed

### Data pipeline
- `src/data.{c,h}` — `data_build`, `data_inspect`
- Formats: `FMT_RAW, FMT_INSTRUCT, FMT_SHAREGPT, FMT_AUTO`
- Packed file: tokens F32 + `.idx` trailer with offset/n_tokens per sample
- Magic: `STUDIO`

### Training (minimal)
- `src/train.{c,h}` — LoRA on lm_head only (lm_head via tied embeddings)
- File format: `LORA0001` magic + dim/vocab/rank/step/scale + A + B
- Adam optimizer (m + v moments)
- `train_create/free/step/save/load/run/merge`
- Emergency watchdog (baca /proc/meminfo)
- Checkpoint per `--checkpoint-every`

### Studio CLI
- `studio data-build`, `data-inspect`, `gguf-rewrite`
- `studio grad-check` (numerical matmul only, Phase B will extend)
- `studio hw` (basic, no JSON/suggest yet)
- `studio train` (LoRA only)
- `studio merge` (byte-copy + sidecar)
- `studio attn-list`, `attn-config`
- `studio web --port 8082 --model <gguf>`

### Web UI (Phase 4a-4c)
- `src/web.c` — HTTP server, GET `/`, `/generate`, `/studio`, `/studio/hw`, `/studio/attn`
- POST `/studio/data` (popen to studio CLI), `/studio/train`, `/studio/merge`
- Embedded `kStudioHTML` (~10KB shell, basic dashboard)
- Sampling params in `/generate` JSON body
- Commit 741118a: phase 4a (WebUI shell + HW/Attn routes)
- Commit 3b498db: phase 4b (POST /studio/{data,train,merge} + README)
- Commit 59770e3: phase 4c (studio + Ollama auto-resolve)
- Commit 50fb040: rich sampling controls (sliders/dropdowns/checkboxes)
- Commit 4497254: align default sampling with CLI

### HW probe (initial)
- `src/hw_probe.{c,h}` — basic `mem_total_kb`, `mem_avail_kb`, `max_seq_advised`, `max_batch_advised`, `fullft_allowed`, `qlora_recommended`

## Known bugs at this handoff (fixed in Phase A, spec 012)

1. Shell injection via `popen` di web.c handle_studio_data/train/merge
2. Model reload per request (~600ms each, no global cache)
3. Recv 64KB truncation (`MAX_REQ = 65536`)
4. Silent QLoRA/FullFT dispatch (CLI flag accept, silently fall through to LoRA)
5. Rank cap hardcoded `d_mid[64]` (silent corruption bila rank > 64)
6. RAM gate inconsistent (different magic numbers di studio.c vs train.c)
7. Uniform init salah (should be Gaussian Box-Muller)
8. File-open per sample (slow)

## Next

Spec 012 (Phase A bugfixes) + Spec 013 (backprop analytical) + Spec 014 (LoRA real merge).
