# 0006_attn_quant_done.md — Session Handoff

**Session date:** 2026-06-25 (retroactive)
**Session goal:** Attention registry with 6 variants + per-layer JSON config. Spec 007 + 010. Quantized attention path.
**Status at handoff:** Attention registry dispatches via function pointer. Default dense; SWA per-layer configurable. 6 variants registered (dense, swa, dilated, bigbird, glocal, mla) — dense + swa fully implemented, others stubs.

---

## What landed

### `src/attn_registry.{c,h}`
- `attn_type { ATTN_TYPE_DENSE, ATTN_TYPE_SWA, ATTN_TYPE_DILATED, ATTN_TYPE_BIGBIRD, ATTN_TYPE_GLOCAL, ATTN_TYPE_MLA }`
- `attn_set_default_spec(type, window, dilation, n_global, latent_dim)`
- `attn_set_layer_spec(L, ...)`
- `attn_s_start(L, t)` — used in forward.c prefill/decode
- `attn_reset()` — clear all layer overrides

### `src/forward.c`
- `attn_s_start(L, t)` replaces hardcoded causal `s_start = 0`
- Forward dispatch: dense uses `s=0..t`, SWA uses `max(0, t-window+1)..t`

### `src/studio.c`
- `studio attn-list` — list variants
- `studio attn-config --config layers.json [--layers N]` — load per-layer config

### `src/web.c`
- `/studio/attn` — JSON `{"n_layers":N,"variants":[...],"layers":[...]}`

### `eval/attn_matrix_test.py`, `eval/attn_sparse_test.py`, `eval/attn_bench.py`
- Matrix test: dense vs reference
- Sparse test: SWA window correctness
- Bench: throughput per variant

## Commits

```
(various commits implementing attn_registry, exact hashes in git log)
```

## Next

Spec 008 (studio foundation): data packing + LoRA on lm_head + web UI shell. Spec 011 (WebUI phase 4a-4c).
