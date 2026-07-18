# 0005_heavy_mode_done.md — Session Handoff

**Session date:** 2026-06-22 (retroactive)
**Session goal:** Heavy mode for inference speed. INT8 KV cache + per-row quantized matmul with ARM dotprod. Spec 006.
**Status at handoff:** Spec 006 complete. F32 → F16 → Q8 progression. KV cache INT8 path. Peak speed-up ~2.5x vs F32 baseline on Termux (Venus SM8350, dotprod=1).

---

## What landed

### `src/forward.c`
- `quantize_row_to_q8_tensor` — F32 row → INT8 + scale
- `matmul_q8_dot_perrow` — INT8 matmul via dotprod inline asm (ARM AARCH64)
- `matmul_q8_2_batch` — 2-token batch FFN
- KV cache modes: F32 (default), F16, Q8
- RoPE modes: F32, F16, Q8
- Forward path: prefill uses full INT8 + 2-batch FFN

### `src/forward.h`
- `enum rope_mode { ROPE_F32, ROPE_F16, ROPE_Q8 }`
- `enum kv_mode { KV_F32, KV_F16, KV_Q8 }`
- `enum attn_mode { ATTN_NAIVE, ATTN_FLASH }`
- `forward_set_modes(rope, kv, attn)`

### `src/main.c`
- CLI flags: `--rope f32|f16|q8`, `--kv f32|f16|q8`, `--attn naive|flash`
- Default: rope=f32, kv=f32, attn=naive (quality baseline)

### `src/web.c`
- `/generate` accepts `rope`, `kv`, `attn` JSON fields
- Cache invalidation on rope/kv mode change (handled per-request, later in Phase A2 made global)

### `eval/parity.py`
- Verify F32 vs F16 vs Q8 token parity on 50-prompt suite
- argmax drift acceptable (< 1% top-1 mismatch on chat prompts)

## Performance

Baseline F32: ~5-8 tok/s prefill, ~3-5 tok/s decode.
Heavy F16: ~10-15 tok/s.
Heavy Q8: ~20-25 tok/s decode (dotprod enabled).

## Commits

```
(multiple commits implementing heavy mode, exact hashes in git log)
```

## Next

Spec 007 (attn quant flash): flash attention kernel + sparse attention variants. Spec 008 (studio foundation): data packing, LoRA on lm_head, web UI shell.
