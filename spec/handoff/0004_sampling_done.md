# 0004_sampling_done.md — Session Handoff

**Session date:** 2026-06-17 (retroactive)
**Session goal:** Sampling quality for studio chat. Temperature, top-k, top-p, repetition penalty. Verified output coherent.
**Status at handoff:** Spec 005 complete. Sampling integrated into chat mode + studio web `/generate`. Build green.

---

## What landed

### `src/sampling.{c,h}`
- `sample_argmax` — greedy
- `sample_temperature` — temperature scaling + softmax
- `sample_top_k`, `sample_top_p` — filtering
- `sample_repetition_penalty` — discount recent tokens
- Default: temp=0.3, top_k=5, top_p=0.0 (matches CLI default after fix)

### `src/main.c`
- Sampling flags: `--temp`, `--top-k`, `--top-p`, `--rep-penalty`
- Chat mode default: rich sampling (not argmax)

### `src/web.c`
- `/generate` accepts sampling params in JSON body
- Default aligned with CLI (commit 4497254: temp=0.3 top_k=5 top_p=0)

### `eval/tok_parity.py`
- Tokenizer roundtrip test (encode → decode) vs reference

## Commits

```
4497254 fix(studio): align default sampling with CLI (temp=0.3 top_k=5 top_p=0)
```

## Next

Spec 006 (heavy mode): INT8 KV cache + matmul_q8 dotprod kernel via ARM dotprod. Forward pass benchmark target: 5-8 tok/s baseline → 20+ tok/s heavy.
