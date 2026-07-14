# 010 — Attention registry & sparse variants

## Why

SmolLM2-135M uses dense GQA (9 query heads / 3 KV, head_dim 64). Full
attention over 2047 tokens = O(n²·30·hd) which dominates prefill for long
contexts. Modern large models (DeepSeek V3, Mistral, Llama 3.x, Gemma) mix
per-layer attention flavors to amortize this cost. Studio must allow
swap-in of sparse variants without rewriting the transformer each time.

This phase lands the **registry** (pluggable attn flavors) and the first
useful variant: **sliding window attention (SWA)**. Per-layer mixers and
the remaining sparse flavors (dilated, bigbird, glocal, mla) follow in
3b.

## Constraints

- Drop-in: parity for default-GQA must be byte-identical (existing tests).
- Single layer-spec array, set at load time, not branched in hot loop.
- No new deps; vanilla C99.
- Per-layer `s_start(t)` only — no need to refactor head loop kernel.

## API

```c
/* attn_registry.h */
typedef struct {
    int   type;          /* ATTN_DENSE | ATTN_SWA | ATTN_DILATED | ... */
    int   window;        /* SWA window size in tokens */
    int   dilation;      /* stride for dilated (3b) */
    int   n_global;      /* number of global tokens (3b) */
    int   latent_dim;    /* MLA latent dim (3b) */
} attn_spec;

void attn_set_default(const attn_spec* s);                 /* all layers */
int  attn_set_per_layer(const attn_spec* specs, int n);    /* exact n_layers */
int  attn_load_config(const char* json_path);              /* JSON config */

/* Hot-path call: position-bounded attention start index. */
int  attn_s_start(int L, int t, int kv_len);   /* inclusive */
```

## Forward integration

The two attention hot loops (prefill at forward.c:1011, decode at
forward.c:1192) currently iterate `for (s = 0; s <= t; s++)`. A single
`int s_start = attn_s_start(L, t, t+1)` (or `(pos+1)` for decode) replaces
the literal `0`. With default spec (type=DENSE, window=0), `s_start = 0`
preserves bit-exact behavior.

SWA reduces window in O(1): `s_start = max(0, t+1 - window)`.

## CLI surface

```
./smollm2 -m model.gguf -p hi --attn dense                    # default
./smollm2 -m model.gguf -p hi --attn swa:window=256
./smollm2 -m model.gguf -p hi --attn-config layers.json
```

`--attn-config <path>` JSON:

```json
{ "default": {"type":"dense"},
  "layers":  [
      {"type":"swa","window":128},
      {"type":"swa","window":256},
      {"type":"dense"}
  ]}
```

Trailing layers reuse the default when the array is shorter than
`n_layers`.

## Memory + perf

Registry holds `attn_spec[n_layers]` only; no buffers, no malloc in hot
path. Lookup is `specs[L].window` — branchless in the common path
(spec->type == DENSE returns 0 immediately).

## Tests

`eval/attn_sparse_test.py`:

1. **swa_long_window_equals_dense**: window ≥ seq_len must produce logits
   equal to default (modulo numeric noise) for the same prompt.
2. **swa_short_window_differs**: window = 4 must produce a different
   argmax than dense on a long prompt.
3. **registry_roundtrip**: `attn_load_config` then dump via `--attn-config`
   listing produces same JSON.

## Files

| Path | New/Mod |
|---|---|
| `src/attn_registry.h` + `.c` | new |
| `src/forward.h` + `.c` | mod — call attn_s_start per layer |
| `src/main.c` | mod — parse `--attn dense|swa:window=N` and `--attn-config` |
| `src/studio.c` | mod — `studio attn-list`, `studio attn-config` |
| `eval/attn_sparse_test.py` | new |
| `spec/010_attn_registry.md` | new |

## Out of scope (3b)

- Dilated strided attention
- BigBird-style local+random+global
- Glocal global+local mix
- MLA latent compression
- LoRA backward through sparse masks
