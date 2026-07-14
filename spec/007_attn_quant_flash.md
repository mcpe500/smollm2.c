# 007_attn_quant_flash.md

## Prompt

Tambahkan CLI flag runtime-switchable untuk RoPE table (f32/f16/q8), KV cache
(f32/f16/q8), dan attention kernel (naive/flash). Sweep semua 18 kombinasi,
ukur tok/s, pilih winner yang parity-hijau.

## Goal

`./smollm2 --rope f16 --kv f16 --attn naive ...` berfungsi tanpa rebuild.
`eval/attn_bench.py` sweep 18 combo, output markdown table, tunjuk winner.

## Why

Decode Intel dominated KV cache DRAM bandwidth (94MB F32). RoPE table juga
hot. Quantization mengurangi bandwidth 2–4×; Q8 versi mungkin mengurangi
akurasi tapi kalau parity masih hijau, bisa menang. Flash attention prefill
mengurangi memory traffic di attention (sub-O(N) K-tile reuse) — beneficial
saat prompt panjang (>16 token).

## Codebase Context

- `src/forward.h` — tambah `enum rope_mode/kv_mode/attn_mode` + `forward_set_modes`.
- `src/forward.c` — struct forward_ctx punya tabel rope + KV cache 3 mode
  (F32/F16/Q8). Hanya mode aktif yang dialokasi; F32 table dilepas setelah
  konversi. `rope_apply` dispatch. `kv_store_row`/`kv_load_row` dispatch.
- `src/main.c` — `--rope/--kv/--attn` flags; `forward_set_modes` dipanggil
  sebelum `forward_load`. Invalid value → exit 1.
- `eval/attn_matrix_test.py` — harness smoke + parity green default.
- `eval/attn_bench.py` — sweep 18 combo.
- `spec/007_attn_quant_flash.md` — dokumen ini.

## Logical Change

### Enums (`forward.h`)

```c
enum rope_mode { ROPE_F32 = 0, ROPE_F16 = 1, ROPE_Q8 = 2 };
enum kv_mode   { KV_F32   = 0, KV_F16   = 1, KV_Q8   = 2 };
enum attn_mode { ATTN_NAIVE = 0, ATTN_FLASH = 1 };

void forward_set_modes(int rope, int kv, int attn);
```

### `struct forward_ctx`

Tambah:

- `int rope_mode, kv_mode, attn_mode` — salinan saat `forward_load`.
- `uint16_t* rope_cos_f16, rope_sin_f16`, `int8_t* rope_cos_q8, rope_sin_q8`
  (F32 table dilepas setelah konversi).
- `uint16_t* k_cache_f16, v_cache_f16`, `int8_t* k_cache_q8, v_cache_q8`,
  `float* k_cache_scale, v_cache_scale` (per-(L,pos) scale), `float* kv_tmp`.
- F32 KV (`k_cache`, `v_cache`) dipertahankan untuk mode F32.

### Allocation

Hanya precision aktif yang dialokasi. Cek OOM di `forward_load`. Default
(f32/f32/naive) identik dengan baseline pra-perubahan.

### Dispatchers

```c
static void rope_apply(float* v, int pos, int n_heads, int head_dim,
                       const forward_ctx* f);  // mode → rope_llama_table_*

static inline void kv_store_row(int kv_mode, int kv_dim,
                                float* cf32, uint16_t* cf16,
                                int8_t* cq8, float* cq8_scale,
                                int pos, int off, const float* src, int n);

static inline void kv_load_row(int kv_mode, int kv_dim,
                               const float* cf32, const uint16_t* cf16,
                               const int8_t* cq8, const float* cq8_scale,
                               int pos, int off, float* out, int n);
```

Decode + prefill attention: load K/V via `kv_load_row` ke stack buffer
(64 float = head_dim). Saat `kv_mode == KV_F32` helper `memcpy` cepat.

### Attention algo

`attn_mode == ATTN_FLASH` saat ini == `ATTN_NAIVE` (no-op stub). Decode
single-thread tidak diuntungkan split-KV; flash prefill dengan Q-block ×
K-block akan datang saat prompt panjang. Stub didokumentasikan, bukan silent
alias.

### CLI

| Flag | Nilai | Default |
|---|---|---|
| `--rope` | f32\|f16\|q8 | f32 |
| `--kv`   | f32\|f16\|q8 | f32 |
| `--attn` | naive\|flash | naive |

Invalid → exit 1 + usage hint. Dispatch via `forward_set_modes` sebelum
`forward_load` pertama.

## Acceptance

```bash
make
./smollm2 --help                                  # flag terlihat
python3 eval/attn_matrix_test.py                  # 5/5 PASS
python3 eval/parity.py                            # 5/5 PASS (default)
./smollm2 -p "hi" -n 8 --rope q8 --kv q8          # works, output non-empty
python3 eval/attn_bench.py                        # 18-combo markdown
```

## Benchmark Results

Diisi oleh `eval/attn_bench.py` (lihat `eval/results/attn_bench_<date>.md`).
Winner dipilih max tok/s di antara parity-hijau.

## Risk

| Risiko | Mitigasi |
|---|---|
| Q8 KV/RoPE shifts parity | soft gate; harness exclude dari winner |
| Dual F32+F16 KV balloon RSS | hanya allocate precision aktif |
| Decode flash no speedup single-thread | didokumentasikan, no-op stub |
| Decode attn Q8 memuat ulang per-tile | stack scratch 64 float; tidak heap |
| Invalid flag lupa di usage | main.c arg parse error → exit 1 + hint |

## Out of scope

- Multi-thread split-KV
- FA3 / non-linear softmax
- Per-tile K quantization (vdotq di head_dim)
- TUI/web flag plumbing
- Flash prefill (planned: saat n_tokens > 16)