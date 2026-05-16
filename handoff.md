# smollm2.c Handoff - Decode Path Fixed (Real Weights Now Used)

## Status: 🚧 DECODE PATH FIXED - But C model output differs from Python reference

**Date:** May 17, 2026 01:00  
**Location:** `/data/data/com.termux/files/home/smollm2.c/`

## GOAL
Build a decode-first C inference engine for SmolLM2 (135M/360M/1.7B) yang jalan di 512MB VPS/Termux.

## Current State: ALL 270 LAYER WEIGHTS LOADED ✅

```
DEBUG: All weights loaded successfully
Processing 1 tokens...
Prefill done in 88.3 ms
Generated 3 tokens in 1400.2 ms (466.7 ms/token)
```

**Decode path now uses actual weights** (Q/K/V/O projections + SwiGLU FFN).

## Bugs Fixed (This Session)

### Bug #1: Decode path used identity placeholders ✅ FIXED
**Problem:** `sm2dl_decode.c` was using identity matrices instead of real weights.

**Fix in `src/decode/sm2dl_decode.c`:**
- Q projection: now uses `model->q_proj[q_off + j*dim + i]`
- K projection: now uses `model->k_proj[k_off + j*kv_dim + i]`
- V projection: now uses `model->v_proj[v_off + j*kv_dim + i]`
- O projection: now uses `model->o_proj[o_off + j*dim + i]`
- Post-attention RMSNorm: now applies proper normalization
- SwiGLU FFN: gate_proj, up_proj, down_proj all connected

### Bug #2: q_proj indexing ✅ FIXED
**Problem:** `q_proj[i*dim + j]` gives element [i][j] but matmul with transpose needs [j][i].

**Fix:** Changed to `q_proj[j * dim + i]` for correct transpose matmul.

### Bug #3: weights_offset header corruption ✅ WORKAROUND
**Problem:** Header at offset 96-103 shows `weights_offset=0, size=0` (corrupted).

**Workaround:** Hardcoded `weights_offset = 256 + 1178859 = 1179115` in `sm2_model.c`.

## Verified File Structure (Python)

```
smollm2-135m-v4.sm2 (270,211,307 bytes):
  - Header: 0-255 bytes
  - Tokenizer: 256 to 1,179,114
  - Weights at offset 1,179,115:
    - tok_embeddings header: rows=49152, cols=576
    - tok_embeddings data: 56,623,104 bytes (49152 * 576 * 2)
    - Layer 0 starts at: 57,802,227
      - input_layernorm: [1, 576]
      - q_proj: [576, 576]
      - k_proj: [192, 576]
      - v_proj: [192, 576]
      - o_proj: [576, 576]
      - post_attention_layernorm: [1, 576]
      - gate_proj: [1536, 576]
      - up_proj: [1536, 576]
      - down_proj: [576, 1536]
    - final_norm: at 270,208,527
```

## What's Working

- ✅ All 270 layer weight tensors loaded correctly
- ✅ tok_embeddings loads correctly (verified rows=49152, cols=576 at offset 1179115)
- ✅ Layer 0 input_layernorm verified at offset 57,802,227 (rows=1, cols=576)
- ✅ Q/K/V/O projections now use real weights in decode path
- ✅ SwiGLU FFN with gate_proj, up_proj, down_proj

## What's NOT Working

**C model output differs from Python reference:**
```
Python reference (token "H"):
  x_H[0]=0.082031
  ln[0]=0.013977
  rms=0.107922
  x_norm[0]=0.010624
  q_proj[0][0]=-0.075195
  q[0]=1.150287

C code (prefill, Layer 0):
  x[0..4]={0.0820,-0.0718,0.0491,0.0889,-0.0405} ✅ (matches)
  xb[0..4]={0.0011,-0.0017,-0.0010,-0.0027,0.0008} ❌ (should be different)
  q[0]=0.061322 ❌ (should be ~1.15)
```

**Issue:** RMSNorm result is wrong - xb values are ~1000x smaller than expected.

## Known Issues

### 1. RMSNorm computation differs
The RMSNorm in `sm2dl_decode.c` computes:
```c
float sum_sq = 0.0f;
for (int i = 0; i < model->dim; i++) {
    sum_sq += x[i] * x[i];
}
float rms = sqrtf(sum_sq / (float)model->dim + spec->rms_eps);
float scale = 1.0f / rms;

for (int i = 0; i < model->dim; i++) {
    xb[i] = x[i] * scale * sm2_f16_to_float(layer_ln[i]);
}
```

But Python's rmsnorm is: `x_norm = x * (1.0 / (sqrt(mean(x^2) + eps))) * weight`

The difference is that C computes `sqrt(sum(x^2)/dim + eps)` while Python computes `sqrt(sum(x^2)/dim + eps)` per the same formula. But the values differ.

### 2. NaN after Layer 20-21
When running with longer prompts, NaN appears at layers 20-21 in prefill.

## Test Commands

```bash
cd /data/data/com.termux/files/home/smollm2.c

# Build and run
make clean && make -j4
./smollm2-cli -m smollm2-135m-v4.sm2 -p "Hi" -n 3

# Debug output
./smollm2-cli -m smollm2-135m-v4.sm2 -p "Hi" -n 1 2>&1 | grep -E "Layer [0-2]:"
```

## Progress Log

### 01:00 - Decode path with real weights
- sm2dl_decode.c now uses actual weights for all projections
- SwiGLU FFN connected
- Output differs from Python reference (RMSNorm issue suspected)

### 00:30 - All 270 weights loaded
- o_proj size bug fixed
- q_proj indexing fixed
- All weights load in ~144ms

### 00:00 - Session start (May 17)
- Context compacted from previous session
- Target: fix matmul/RMSNorm discrepancy and load all layer weights

## Files Modified

- `src/sm2_context.c` - layer_forward with full weights, q_proj indexing fix
- `src/sm2_model.c` - weights loading for all 270 tensors
- `src/decode/sm2dl_decode.c` - decode path now uses real weights (REWRITE)
- `include/smollm2.h` - weight struct definitions
- `tools/smollm2-convert.py` - BF16 conversion fix