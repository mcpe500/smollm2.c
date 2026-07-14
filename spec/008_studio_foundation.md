# 008_studio_foundation.md

## Prompt

Tambahkan fondasi "studio" ke `smollm2.c`: autograd minimal, penulis GGUF,
dan adapter dataset dengan auto-detect. Ini phase 1 dari 4 menuju studio
WebUI penuh (spec 009 training, 010 attention registry, 011 WebUI).

## Goal

```bash
make
./smollm2 studio data-build --in sample.txt --out packed.bin --fmt auto
./smollm2 studio data-build --in instruct.jsonl --out packed.bin --fmt auto
./smollm2 studio gguf-rewrite --in model.gguf --out copy.gguf   # roundtrip
python3 eval/studio_smoke.py                                     # 4/4 PASS
```

Fondasi ini tidak menghasilkan model yang berbeda — hanya infrastruktur
agar phase 2 (training) bisa jalan.

## Why

Inferensi murni tidak bisa memperbaiki kelemahan kapasitas 135M. Tambah
scaffold finetuning butuh: dataset loader (auto-detect), penulis GGUF
(export base+adapter hasil merge), dan backward pass (autograd). Phase 1
bangun ketiganya dalam bentuk minimum, dengan TDD.

## Codebase Context

- File baru:
  - `src/data.c` + `data.h` — adapter dataset
  - `src/gguf_write.c` + `gguf_write.h` — penulis GGUF (roundtrip)
  - `src/backward.c` + `backward.h` — autograd minimal (matmul, rmsnorm)
  - `src/studio.c` + `studio.h` — subcommand dispatcher
  - `spec/008_studio_foundation.md` (dokumen ini)
  - `eval/studio_smoke.py` — harness TDD
- File modifikasi:
  - `src/main.c` — dispatch `studio <cmd>` di awal `main`
  - `Makefile` — tambah `src/data.c src/gguf_write.c src/backward.c src/studio.c`
- Tidak menyentuh `forward.c` (yang akan di-refactor di phase 3 untuk
  attention registry).

## Logical Change

### `src/data.h`

```c
typedef enum { FMT_RAW, FMT_INSTRUCT, FMT_SHAREGPT, FMT_AUTO } data_fmt;

typedef struct {
    int  n_tokens;        /* jumlah token sample ini */
    int  offset;          /* posisi di file packed */
} sample_idx;

typedef struct {
    int         n_samples;
    long        total_tokens;
    sample_idx* index;        /* [n_samples] */
    char*       packed_path;  /* file berisi token IDs (int32) */
    data_fmt    detected_fmt;
} dataset;

data_fmt data_detect(const char* path);   /* sniff first JSON line */
int  data_build(const char* in_path, const char* out_path,
                data_fmt fmt, const char* gguf_path);
int  data_open (const char* packed_path, dataset** out);
void data_free (dataset* d);

/* Streaming iterator: panggil cb per-sample. cb return non-zero = abort. */
typedef int (*data_iter_cb)(const int* tokens, int n, void* user);
int  data_iter(const dataset* d, data_iter_cb cb, void* user);
```

**Auto-detect heuristic:**
- Baca baris pertama, coba parse JSON.
- Key `messages` → SHAREGPT.
- Key `prompt` + `completion` → INSTRUCT.
- Key `text` → RAW.
- Bukan JSON → RAW (stream seluruh file).

**Template:**
- RAW → tokenize langsung.
- INSTRUCT → ChatML: `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{completion}<|im_end|>\n`.
- SHAREGPT → loop role, system prefix default.

### `src/gguf_write.h`

```c
/* Penulis GGUF minimal — cukup untuk roundtrip tensor F16 dan write
   metadata. Tidak menangani quantization baru di phase 1. */
typedef struct gguf_writer gguf_writer;

gguf_writer* gw_create(const char* path);
void gw_close (gguf_writer* w);   /* flush header + pad + tensor info */

int gw_kv_string (gguf_writer* w, const char* key, const char* val);
int gw_kv_i32    (gguf_writer* w, const char* key, int v);
int gw_kv_f32    (gguf_writer* w, const char* key, float v);
int gw_kv_array_str(gguf_writer* w, const char* key,
                    const char* const* vals, int n);

int gw_tensor_f16(gguf_writer* w, const char* name,
                  int n_dims, const int64_t* dims,
                  const uint16_t* data);
int gw_tensor_f32(gguf_writer* w, const char* name,
                  int n_dims, const int64_t* dims,
                  const float* data);
```

Format sesuai spec GGUF v3: header magic+version+`n_kv`+`n_tensors`+
alignment 32 + tensor data + tensor info. Reader existing (`src/gguf.c`)
harus bisa load file yang ditulis.

### `src/backward.h`

```c
/* Autograd minimal — di phase 1 hanya untuk gradient check.
   Phase 2 akan extend ke LoRA backward. */

/* Numerical gradient check untuk satu matmul.
   Y = X @ W^T (X: m×k, W: n×k, Y: m×n)
   Verifies backward analytical grad against finite-difference.
   Returns max abs error. */
float backward_matmul_grad_check(int m, int n, int k, float eps);

/* RMSNorm forward+backward dengan symbolic grad. */
void rmsnorm_backward(const float* x, const float* g, const float* w,
                      float eps, int n,
                      float* dx, float* dw);
```

### `src/studio.c`

```c
int studio_dispatch(int argc, char** argv);  /* dipanggil jika argv[1]=="studio" */

/* Subcommand:
   studio data-build  --in X --out Y --fmt auto|raw|instruct|sharegpt
   studio gguf-rewrite --in base.gguf --out copy.gguf
   studio grad-check   # run backward_matmul_grad_check, print error
*/
```

### CLI dispatch (`src/main.c`)

```c
if (argc >= 2 && strcmp(argv[1], "studio") == 0) {
    return studio_dispatch(argc - 1, argv + 1);  /* shift */
}
```

Ditambahkan paling atas `main`, sebelum flag parse biasa.

### Makefile

```makefile
SRC += src/data.c src/gguf_write.c src/backward.c src/studio.c
```

## Acceptance

```bash
make
./smollm2 --help | head -5
# data adapter
./smollm2 studio data-build --in eval/prompts.txt --out packed.bin --fmt auto
./smollm2 studio data-build --in eval/sample_sharegpt.jsonl --out packed2.bin --fmt auto
# gguf roundtrip
./smollm2 studio gguf-rewrite --in models/smollm2-135m-f16.gguf --out /tmp/copy.gguf
./smollm2 -m /tmp/copy.gguf --inspect | head -3
./smollm2 -m /tmp/copy.gguf --logits 'hello' 2>&1 | grep argmax
# gradient check
./smollm2 studio grad-check    # max abs error < 1e-3
# TDD
python3 eval/studio_smoke.py    # 4/4 PASS
python3 eval/parity.py          # 5/5 PASS (baseline unchanged)
python3 eval/attn_matrix_test.py # 5/5 PASS
```

## TDD harness

`eval/studio_smoke.py` — 4 test:

1. `test_data_auto_detect` — 3 file sampel, format terdeteksi benar via
   `studio data-build --fmt auto` exit code dan `--inspect` output.
2. `test_data_roundtrip` — sample tokenize → packed → reload → token IDs
   match.
3. `test_gguf_rewrite_roundtrip` — tulis ulang model GGUF, reload, infer
   argmax `hello` = 19556 (Hello) — parity preserved.
4. `test_backward_grad_check` — `studio grad-check` exit 0, output
   mengandung "max_abs_error=" dan nilainya < 1e-3.

Semua test HARUS FAIL sebelum impl (binary belum ada subcommand `studio`).

## Limit

| Batas | Nilai |
|---|---|
| Sample tokens per baris | 2048 |
| Packed file size | bebas (streaming) |
| Token vocab | 49152 (SmolLM2) |
| GGUF dtype di phase 1 | F16, F32 |

## Out of scope

- Backward untuk rope/attention/FFN — phase 2 (LoRA)
- Writer Q4/Q8 quant — phase 3 (QLoRA)
- Streaming dataset untuk file >100MB (tetap full load di phase 1, dataset
  eval kecil)
- WebUI — phase 4
- Attention registry refactor — phase 3

## Risk

| Risk | Mitigasi |
|---|---|
| GGUF writer corrupt alignment | roundtrip test dngan `--inspect` |
| Auto-detect misclassify JSONL | manual override via `--fmt` |
| Backward numerical error besar | eps 1e-4, max iter 100, cek kondisi |
| Memory untuk dataset besar | cap 2048 token/sample; phase 4 streaming |
| ChatML template salah → wrong loss | sample inspection di harness |

## Out of scope (full spec 009-011)

Phase 2 (training modes): train.c dengan LoRA/QLoRA/FullFT, hw_probe,
checkpoint emergency.
Phase 3 (attention registry): attn_registry.c + 6 variants, per-layer JSON
config, refactor forward.c untuk dispatch via fn pointer.
Phase 4 (WebUI): studio routes di web.c, dashboard tabs, README update.