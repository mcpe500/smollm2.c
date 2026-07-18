# 015_qlora.md

## Prompt

QLoRA: LoRA training di atas base weights terquantisasi (Q4_0 / Q4_K).
Target: peak RAM < 700 MB pada rank=8/16. Auto-suggest via `studio hw`
bila `qlora_recommended=1 && fullft_allowed=0`.

## Goal

```bash
./smollm2 studio train --mode qlora --base models/smollm2-135m-q4_k.gguf \
    --rank 8 --targets q,v --epochs 1 --out-dir qlora_adapters/
# expect: peak RSS < 700 MB, loss turun

./smollm2 studio hw --json
# expect: qlora_recommended reflects mem tier
```

## Why

LoRA di base F16 butuh 540 MB weights (135M × 4 byte F32 setelah dequant
untuk training forward). QLoRA simpan weights sebagai Q4 (70 MB) dan dequant
per-matmul ke scratch tile (~5 MB). Peak RAM ~250 MB, membuka training di
ponsel budget 1-2GB.

QAFT (Quantization-Aware Fine-Tuning) di-defer: spec 009 eksplisit "no INT4
grad". QLoRA = freeze base Q4 + train LoRA F32 di atas (grad flow hanya
through LoRA params).

## Codebase Context

- `src/forward_train.{c,h}` (slice 2 LoRA) — extend dengan Q4 dequant path
- `src/gguf.c` — `gguf_tensor_data` sudah expose Q4 bytes; add Q4 dequant helpers
- `src/hw_probe.{c,h}` — surface `qlora_recommended` in JSON (sudah ada)
- `src/studio.c` — relax A4 refuse untuk mode=qlora; auto-suggest flow
- `src/train.c` — QLoRA dispatch: dequant base + LoRA forward (reuse LoRA slice 2)
- `Makefile` — tetap, hanya depend pada slice 2 files

## Logical Change

### Q4 dequant

Tambah helper `q4_dequant_row(const void* q4_bytes, float* out, int n)` di
`forward_train.c`:
- Q4_0 layout: blocks of 32 elements. Setiap block: 1 F16 scale (2 bytes)
  + 16 int4 elements (8 bytes packed). Total 10 byte per 32 elemen.
- Q4_K layout lebih kompleks (super-block + sub-scales), defer bila perlu.

Per-matmul forward untuk QLoRA:
```c
/* For each output row: */
for (int o = 0; o < out_dim; o++) {
    /* Tile: dequant 32 elemen base weight sekaligus */
    float tile[32];
    q4_dequant_row(W_q4 + block_offset(o, in_dim), tile, 32);
    /* Dot product dengan input F32 */
    for (int i = 0; i < 32; i++)
        out[o] += tile[i] * x[block_start + i];
}
```

LoRA delta tetap F32 (mirip slice 2 LoRA):
```
Y = base_dequant(x) + scale * (x @ B) @ A
```

### HW gate

`studio.c cmd_train`: hapus refuse untuk `mode=qlora`. Ganti dengan:
```c
if (p.mode == TRAIN_QLORA) {
    long need = MEM_START_QLORA_MB * 1024;  /* 900 MB */
    if (caps.mem_avail_kb < need) {
        fprintf(stderr, "train: qlora refused — insufficient mem (avail=%ld MB, need>=900 MB)\n", ...);
        return 1;
    }
}
```

Auto-suggest di `studio hw --suggest`:
```
Given your HW (mem=1500 MB, dotprod=1):
  Recommended: mode=lora rank=16 seq=256 targets=q,v
  qlora_recommended=1 (fullft_allowed=0)
  For QLoRA: --base models/smollm2-135m-q4_k.gguf --rank 8 --targets q,v
```

### Forward_train QLoRA path

`forward_train.c` extend dengan `train_mode mode` di ctx:
- `mode == TRAIN_LORA`: base F32 (dequant F16 → F32 saat load, 540 MB)
- `mode == TRAIN_QLORA`: base Q4 (simpan raw bytes 70 MB, dequant per-matmul)

Activations cache identik. Backward chain identik. Hanya dequant logic beda.

## Code Change

- `src/forward_train.{c,h}` — `q4_dequant_row`, mode dispatch in forward pass
- `src/gguf.c` — ekspos `gguf_dtype_size` untuk Q4_0/Q4_K (sudah ada)
- `src/hw_probe.{c,h}` — `qlora_recommended` (sudah ada), add to JSON
- `src/studio.c` — relax A4 refuse qlora, suggest flow
- `src/train.c` — TRAIN_QLORA dispatch ke forward_train QLoRA path
- `eval/train_smoke.py` — extend: qlora loss-decrease test, RAM budget test via `--simulate-mem-kb`

## Why This Change

- Q4 dequant per-matmul tile (bukan dequant full upfront): hemat 470 MB
  (540 MB F32 → 70 MB Q4 + 5 MB tile scratch). Tile reuse via ring buffer.
- LoRA F32 atas Q4 base (bukan quantized LoRA): standard QLoRA paper approach.
  LoRA gradients tidak perlu di-quantize.
- HW gate via konstanta: same dengan LoRA, simple. Bila RAM insufficient,
  refuse dengan pesan jelas (bukan crash mid-training).

## Logic / Pseudocode

```
forward_train_step(x):
    for L in 0..n_layers:
        x_in[L] = x
        rmsnorm(x_norm, x_in[L], w_attn_norm[L])
        # Q: y = base_W_q4 @ x_norm + scale*(x_norm@B_q)@A_q
        q4_matmul_f32(q, W_q4_q[L], x_norm)   # tile dequant
        if LORA_Q: q += scale * (x_norm @ B_q[L]) @ A_q[L]
        # ... same for k, v, o, gate, up, down
        # attention, ffn (standard)
        x_mid = x_in[L] + attn_out
        x_out[L] = x_mid + ffn_out
        x = x_out[L]
    logits = base_W_lm_head_q4 @ final_norm
    if LORA_LM_HEAD: logits += scale * (final_norm @ B_lm) @ A_lm
```

## Test Simulation & Tracing

### Q4 dequant parity
```
F16 base → F32 → dequant to F32
Q4 base → dequant to F32
max abs error: ~0.01 (Q4 quantization noise, expected)
```

### QLoRA loss decrease
```
3 epochs × 5 samples, rank=8, lr=5e-3
median(first half losses) vs median(last half)
expect: last < first * 0.9
```

### QLoRA RAM budget
```
--simulate-mem-kb 700000  (700 MB available)
expect: qlora proceeds (need 900 MB threshold, but simulate-mem sets below)
Hmm — adjust: --simulate-mem-kb 900000, should proceed.

Actually test: peak RSS via /proc/self/status VmRSS during run
expect: VmRSS < 700 MB
```

## Manual Testing Plan

```bash
make
# Q4 GGUF dari sumber (HuggingFace atau gguf-quantize)
# Asumsi: models/smollm2-135m-q4_k.gguf tersedia (download terpisah)

./smollm2 studio hw --json | grep qlora_recommended
# expect: "qlora_recommended": 1 pada mem < 2.5GB

./smollm2 studio train --mode qlora --base models/smollm2-135m-q4_k.gguf \
    --data packed.bin --rank 8 --targets q,v --epochs 1 \
    --max-steps 10 --out-dir /tmp/qlora
# expect: VmRSS < 700 MB, loss decreasing
```

## Status

- [x] Spec written
- [ ] Implementation
- [ ] Verified
- [ ] Handoff written
