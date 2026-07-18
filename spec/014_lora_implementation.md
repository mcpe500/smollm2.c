# 014_lora_implementation.md

## Prompt

Implementasi LoRA (Low-Rank Adaptation) per spec 009. Dua slice:

**Slice 1 (DONE):** Real weight merge via `gguf_patch_tensor`. LoRA training
existing (lm_head only) tetap jalan, tapi merge sekarang patch tensor bytes
langsung (ΔW = scale · B^T @ A^T), bukan sidecar adapter.

**Slice 2 (PENDING):** 7-target LoRA (Q,K,V,O,GATE,UP,DOWN). Butuh
`forward_train.{c,h}` baru — F32 transformer dengan activations cache untuk
backprop penuh through semua layer.

## Goal

```bash
# Slice 1 (done)
./smollm2 studio train --data packed.bin --mode lora --rank 8 --out-dir adapters/
./smollm2 studio merge --base model.gguf --adapter adapters/lora_final.bin --out merged.gguf
./smollm2 -m merged.gguf -p "hello"   # argmax valid, no NaN

# Slice 2 (pending)
./smollm2 studio train --mode lora --targets q,v --rank 8 ...
./smollm2 studio train --mode lora --targets all --rank 16 ...
```

## Why

Slice 1: Sebelumnya `train_merge` hanya byte-copy base + sidecar adapter
Phase 2a). Inference jalan dengan base weights, adapter tidak benar-benar
terapan. User rat: "yang penting beneran finetuning, jangan penipuan".
Patch weight F16 langsung di GGUF = real merge.

Slice 2: Spec 009 sebenarnya minta LoRA pada 7 target linear (q_proj,
k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj). lm_head (yang
sekarang) hanya 1 dari 7. Untuk LoRA yang efektif membentuk perilaku,
perlu inject ΔW di setiap layer, bukan hanya output.

7-target LoRA butuh backprop penuh: dari loss melalui lm_head, layer L-1
sampai layer 0. Setiap layer backward butuh gradient w.r.t. input
(untuk chain ke layer sebelumnya). Maka activations cache wajib.

## Codebase Context

- `src/gguf_write.{c,h}` — EXTEND: `gguf_patch_tensor`, `gguf_patch_tensors`
- `src/train.c` — REWRITE `train_merge` body: real weight merge
- `src/lora.{c,h}` — BARU (slice 2): storage + target bitmask
- `src/lora_backward.{c,h}` — BARU (slice 2): gA, gB chain rule
- `src/forward_train.{c,h}` — BARU (slice 2): F32 transformer + activations cache
- `src/studio.c` — extend `cmd_train`: `--targets q|k|v|o|gate|up|down|all|q,v`
- `eval/merge_test.py` — TDD slice 1
- `eval/train_smoke.py` — 5 tests

## Logical Change

### Slice 1 — Real weight merge

`gguf_patch_tensor(src, dst, name, data, n_bytes)`:
1. `gguf_copy(src, dst)` byte-exact
2. `gguf_load(src)` dapatkan tensor info + `tensor_data` pointer
3. `file_off = (tensor_data - map) + t.offset`
4. `fopen(dst, "r+b")`, `fseek(file_off)`, `fwrite(data, n_bytes)`

`gguf_patch_tensors(src, dst, names[], data[], sizes[], count)`:
multi-tensor in one pass (avoid N copies).

`train_merge(base, adapter, out)` rewrite:
1. Parse adapter header (LORA0001 magic, dim, vocab, rank, step, scale, A, B)
2. `gguf_load(base)`, dapatkan `token_embd.weight` (tied = lm_head)
3. Verifikasi dtype F16 + shape [dim, vocab] match adapter
4. `lora_patched_weight(base_f16, A, B, dim, rank, vocab, scale)`:
   ```c
   for v in vocab:
       for d in dim:
           delta = scale * sum_r A[d*rank+r] * B[r*vocab+v]
           base_f = f16_to_f32(base_f16[v*dim+d])
           out_f16[v*dim+d] = f32_to_f16(base_f + delta)
   ```
5. `gguf_patch_tensor(base, out, "token_embd.weight", out_f16, n_bytes)`
6. Hapus sidecar logic (tidak ada file `.lora` lagi)

F16 conversion: inline helpers `merge_f32_to_f16` dan F16→F32 expansion
(local ke train.c, tidak share dengan forward.c yang optimasi path).

### Slice 2 — 7-target LoRA (PENDING)

`lora.h`:
```c
typedef enum {
    LORA_Q = 1<<0, LORA_K = 1<<1, LORA_V = 1<<2, LORA_O = 1<<3,
    LORA_GATE = 1<<4, LORA_UP = 1<<5, LORA_DOWN = 1<<6,
    LORA_LM_HEAD = 1<<7,
} lora_target;

typedef struct {
    int layer;
    lora_target target;
    int in_dim, out_dim, rank;
    float scale;
    float *A;  /* [in_dim, rank] */
    float *B;  /* [rank, out_dim] */
    /* Adam state */
    float *gA, *gB, *mA, *vA, *mB, *vB;
} lora_pair;

typedef struct {
    int magic;  /* LORA0002 */
    int n_layers, dim, vocab;
    int rank, alpha;
    int target_mask;  /* bitmask */
    lora_pair* pairs;  /* n_layers × popcount(target_mask) */
} lora_state;
```

File format v2 `LORA0002`:
- Header: magic, n_layers, dim, vocab, rank, alpha, target_mask
- Per (layer, target): A float[in*rank], B float[rank*out]
- Backward-compat: detect `LORA0001` (lm_head only, 1 pair)

`forward_train.{c,h}`: F32 transformer dengan activations cache per-layer:
- `x_in[L]` (input ke layer L, untuk residual + rmsnorm)
- `x_norm_a[L]` (post attn_norm)
- `q_pre[L], k_pre[L], v_pre[L]`, `q_post_rope[L], k_post_rope[L]`
- `attn_scores[L], attn_probs[L], attn_out[L]`
- `x_mid[L]` (residual after attn)
- `x_norm_f[L], gate_pre[L], up_out[L], silu_out[L], ffn_down[L]`
- `x_out[L]` (output layer L)
- `final_norm, logits`

Forward pass cermin `forward.c:935-1093` tapi F32-only, no INT8, no cache
reuse — kemudahan debug. Aktivasi semua disimpan.

`lora_backward.{c,h}`:
```c
/* Per target di layer L: */
/* Forward: Y = X @ W + scale * (X @ B) @ A
 * Backward:
 *   gA += scale * Z^T @ gY,   Z = X @ B (cached)
 *   gZ = scale * gY @ A^T
 *   gB += X^T @ gZ
 *   gX = gY @ (W + scale*B@A)^T   (atau gY @ W^T + gZ @ B^T, lebih efisien)
 */
```

Injection points (`forward.c:997-1067` cermin):
- `Y_q = x @ W_q + scale*(x@B_q)@A_q` (target LORA_Q)
- `Y_k = x @ W_k + ...` (LORA_K)
- `Y_v, Y_o, Y_gate, Y_up, Y_down` similar

Default targets `q,v` (RAM friendly, 2 of 7). `--targets all` → 7 targets
(~80 MB Adam state for rank 8 × 30 layers).

## Code Change

### Slice 1 (DONE)
- `src/gguf_write.h` — `gguf_patch_tensor`, `gguf_patch_tensors`
- `src/gguf_write.c` — copy + resolve + patch
- `src/train.c` — `train_merge` rewrite (ΔW compute + F16 patch + no sidecar), F16 helpers local
- `eval/merge_test.py` — 3 tests: hash-differs + infers + no-sidecar

### Slice 2 (PENDING)
- `src/lora.{c,h}` BARU — storage, init Gaussian, save/load format v2
- `src/lora_backward.{c,h}` BARU — gA, gB, gZ per target
- `src/forward_train.{c,h}` BARU — F32 transformer forward + cache + injection
- `src/train.c` — rewrite body jadi mode-dispatcher (LoRA mode calls into lora_state)
- `src/studio.c` — `cmd_train --targets` parse, default `lm_head`; relax A4 refuse for LoRA mode only
- `Makefile` — add 4 new SRC files
- `eval/train_smoke.py` — extend: --targets flag test, multi-target save format

## Why This Change

- Patch weight bytes (bukan rewrite GGUF): simple, header intact, reader
  accepts result. Inference tidak butuh adapter sidecar.
- Token_embd (bukan output.weight): tied embeddings di SmolLM2-135M.
  Patching token_embd = patching lm_head juga. Untuk model untied, perlu
  patch output.weight terpisah.
- LoRA backward via cached Z (bukan recompute X@B): hemat 1 matmul per step.
- File format v2 backward-compat: adapter lama (LORA0001) tetap load.

## Test Simulation & Tracing

### Slice 1
```
train 3 steps rank=4 → adapter.bin (size ~795KB)
merge → gguf_copy + patch 49152*576*2 = 56MB bytes
diff_bytes > 55,000,000 (most of token_embd)
argmax inference valid (no NaN, valid token id)
no .lora sidecar file
```

### Slice 2 (projected)
```
train --targets q,v rank=8 30 steps
loss decreasing trend (similar ke lm_head-only)
adapter size: 30 layers × 2 targets × (576*8 + 8*576) × 4 bytes = 1.1 MB
inference post-merge: argmax valid
```

## Manual Testing Plan

```bash
# Slice 1
make
python3 eval/merge_test.py        # 3/3 pass
python3 eval/train_smoke.py       # 5/5 pass

# Slice 2 (post-impl)
make
./smollm2 studio train --data packed.bin --mode lora --targets q,v \
    --rank 8 --epochs 3 --model models/smollm2-135m-f16.gguf
./smollm2 studio merge --base models/smollm2-135m-f16.gguf \
    --adapter adapters/lora_final.bin --out merged.gguf
./smollm2 -m merged.gguf -p "hello"
# Expect: argmax valid, loss curve descending
```

## Status

- [x] Spec written
- [x] Slice 1 implemented (real weight merge, 3/3 merge_test, 5/5 train_smoke)
- [x] Slice 1 verified
- [ ] Slice 2 implemented (7-target LoRA via forward_train.c)
- [ ] Slice 2 verified
- [x] Handoff (slice 1) written (0010_lora_merge_done.md)
