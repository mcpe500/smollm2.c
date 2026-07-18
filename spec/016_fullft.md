# 016_fullft.md

## Prompt

FullFT: full fine-tuning semua 135M parameter. Gradient checkpointing untuk
hemat RAM (cache hanya `x_in[L]` per layer, recompute internal saat backward).
AdamW optimizer per-tensor dengan m+v moments. Emergency dump saat RAM kritis.
Gate: MemAvailable ≥ 2.5 GB.

## Goal

```bash
./smollm2 studio train --mode fullft --seq 128 --epochs 1 \
    --base model.gguf --data packed.bin --out-dir fullft_out/
# expect: loss turun, emergency dump bila RAM < 150 MB

./smollm2 studio merge --base model.gguf --adapter fullft_out/weights.f32.bin \
    --out merged.gguf
# expect: patched token_embd + semua layer weights
```

## Why

FullFT = maximum capacity adjustment. Semua 135M params di-update. Tapi:
- Weight F32: 540 MB
- Adam m + v moments: 1080 MB (2 × 540)
- Activations + grads: ~200 MB
- Peak: ~1.8 GB

Perlu ≥ 2.5 GB MemAvailable margin. Hanya feasible di Android high-end atau
desktop. Tapi user minta "yang beneran" — FullFT adalah mode paling agresif.

Gradient checkpointing: cache hanya `x_in[L]` (17 MB untuk seq=256) dan
`x_out[L-1]`. Backward layer L: recompute internal activations dari `x_in[L]`,
jalankan backward, free, lanjut L-1. Tradeoff: 30% lebih banyak compute,
hemat 80% aktivasi memory.

## Codebase Context

- `src/forward_train.{c,h}` — checkpointed variant dari forward
- `src/adam.{c,h}` — BARU: AdamW per-tensor state
- `src/gguf_write.{c,h}` — extend: patch multiple tensors (sudah ada `gguf_patch_tensors`)
- `src/train.c` — TRAIN_FULLFT dispatch + emergency watchdog
- `src/studio.c` — relax A4 refuse untuk mode=fullft
- `src/hw_probe.{c,h}` — `fullft_allowed` (sudah ada), `MEM_START_FULLFT_MB` gate
- `eval/train_smoke.py` — fullft loss-decrease test, emergency-dump test

## Logical Change

### AdamW per-tensor

`adam.{c,h}`:
```c
typedef struct {
    float* m;  /* same shape as param */
    float* v;
    int    n;
    int    step;  /* shared per-tensor for bias correction */
} adam_state;

void adam_init(adam_state* s, int n);
void adam_free(adam_state* s);
/* AdamW update with decoupled weight decay: */
void adam_step(adam_state* s, float* params, const float* grads,
               int n, float lr, float beta1, float beta2,
               float eps, float weight_decay);
```

Tensors di-update secara terpisah via `gguf_patch_tensors` di save.

### Checkpointed forward

`forward_train_checkpointed(ctx, tokens, n)`:
- Cache hanya `x_in[L]` (input ke setiap layer) dan `x_out[L-1]` (output prev layer)
- Internal activations (`x_norm, q_pre, k_pre, v_pre, scores, probs, etc.`)
  di-recompute saat backward.

`backward_checkpointed(ctx, g_logits, ...)`:
- Iterasi L dari n_layers-1 ke 0:
  - Recompute internal activations dari `x_in[L]`
  - Jalankan backward untuk layer L (chain rule ke `g_x_in[L]`)
  - Akumulasi gW untuk layer L
  - Free internal recomputed activations
  - Pass `g_x_in[L]` ke layer L-1

### Emergency watchdog

`train_run` per-step check:
```c
hw_caps caps;
hw_probe(&caps);
if (caps.mem_avail_kb < MEM_EMERGENCY_MB * 1024) {
    /* Dump all weights via gguf_patch_tensors loop */
    char path[1024];
    snprintf(path, sizeof(path), "%s/emergency_%d.gguf", out_dir, step);
    train_dump_weights(t, base_gguf, path);  /* patch all tensors */
    fprintf(stderr, "train: emergency save at step %d\n", step);
    exit(0);
}
```

### Save merged

`train_save_fullft(t, base_gguf, out_gguf)`:
1. Loop semua tensors di base (token_embd, all blk.*.attn_*, ffn_*, norms)
2. Patch masing-masing dengan updated F32 weights → F16
3. Write via `gguf_patch_tensors`

Tidak ada adapter file separate — output adalah GGUF lengkap dengan weights updated.

## Code Change

- `src/adam.{c,h}` — BARU: AdamW state per-tensor
- `src/forward_train.{c,h}` — checkpointed variant
- `src/train.c` — TRAIN_FULLFT dispatch, emergency save, multi-tensor save
- `src/studio.c` — relax A4 refuse untuk fullft
- `Makefile` — add adam.c
- `eval/train_smoke.py` — fullft emergency + save tests

## Why This Change

- Gradient checkpointing (bukan full cache): standar HF/PyTorch untuk model
  >100M di RAM terbatas. Tradeoff compute-time acceptable.
- AdamW decoupled (bukan L2): decay tidak terakumulasi di grad, lebih stabil.
- Per-tensor Adam state (bukan satu vector 135M): mudah free saat emergency,
  patch tensor-by-tensor untuk save.
- Emergency save via gguf_patch (bukan sidecar): output adalah GGUF utuh,
  inference dapat load langsung.

## Logic / Pseudocode

```
fullft_train_step(tokens, n):
    forward_checkpointed(tokens, n)  # cache x_in[L] only
    g_logits = softmax_ce_backward(logits, target)
    g_x = matmul_backward(g_logits, ...)  # final projection
    for L in n_layers-1 .. 0:
        internal = recompute_internal(x_in[L])
        g_W[L], g_attn_norm[L], g_ffn_norm[L], g_x = backward_layer(internal, g_x)
        free(internal)
        adam_step(adam[L], W[L], g_W[L], ...)
    return loss
```

## Test Simulation & Tracing

### Loss decrease
```
3 steps, seq=64, lr=1e-4
expect: loss descending trend
```

### Emergency save
```
--simulate-mem-kb 100000  (100 MB)
expect: train starts, mem drops below 150 MB → emergency save, exit 0
emergency_N.gguf exists and is loadable
```

### Multi-tensor save
```
After 3 steps fullft, save → merged.gguf
./smollm2 -m merged.gguf -p "hello"
expect: argmax valid, no NaN
```

## Manual Testing Plan

```bash
make

# FullFT loss decrease (needs 2.5 GB free RAM)
free_mem=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
if [ $free_mem -gt 2500000 ]; then
    ./smollm2 studio train --mode fullft --base model.gguf \
        --data packed.bin --seq 128 --epochs 1 --max-steps 5 \
        --out-dir /tmp/fullft
    # expect: loss decreasing, weights.f32.bin atau merged.gguf saved
fi

# Emergency save (simulated low mem)
./smollm2 studio train --mode fullft --base model.gguf \
    --data packed.bin --seq 128 --epochs 1 --max-steps 5 \
    --simulate-mem-kb 100000 --out-dir /tmp/fullft
# expect: emergency save triggered, exit 0
ls /tmp/fullft/emergency_*.gguf
```

## Status

- [x] Spec written
- [ ] Implementation
- [ ] Verified
- [ ] Handoff written
