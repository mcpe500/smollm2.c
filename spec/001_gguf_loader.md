# 001_gguf_loader.md

## Prompt

Implementasikan GGUF reader untuk membaca model SmolLM2-135M dari storage Ollama (`~/.ollama/models/blobs/sha256-f535f83e...`, 270MB, GGUF v3).

## Goal

Parse file GGUF v3 (header + metadata KV + tensor info table + tensor data), expose API untuk akses tensor by name dan KV by key. Tensor data di-mmap, tidak di-copy. Fokus pada F16 dtype (SmolLM2 Ollama blob ~270MB = 135M params × 2 bytes — matches F16).

## Why

Tanpa GGUF reader yang benar, tidak ada cara untuk load model. Ini foundation untuk semua step selanjutnya (tokenizer butuh `tokenizer.ggml.tokens` array, forward pass butuh tensor weights, dst.).

Format GGUF adalah standar ecosystem (llama.cpp, Ollama, koboldcpp, dst.), sehingga reader yang benar akan kompatibel dengan model apa pun.

## Codebase Context

- File: `~/.ollama/models/blobs/sha256-f535f83e...` (270 MB, magic = "GGUF" v3)
- Manifest Ollama: `~/.ollama/models/manifests/registry.ollama.ai/library/smollm2/135m` (digest → layer mapping)
- Referensi format: <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>
- Akan dipakai oleh: `src/tokenizer.c` (Step 4), `src/forward.c` (Step 5)

## Logical Change

Tambah modul `src/gguf.{h,c}` yang:
1. Buka file GGUF, validasi magic + version.
2. Baca header: `n_tensors`, `n_kv`.
3. Baca `n_kv` metadata KV pairs (string key + typed value). Tipe value mencakup: uint8/16/32/64, int8/16/32/64, float32/64, bool, string, array.
4. Baca `n_tensors` tensor info (name, n_dims, dims[], dtype, offset).
5. mmap section tensor data (dengan alignment 32).
6. Expose accessor: `gguf_kv_get(key)`, `gguf_tensor_get(name)`, `gguf_tensor_data(info)`.

Tidak ada dequantization di sini. Tensor data dikembalikan sebagai pointer mentah; konversi F16→F32 terjadi di `src/forward.c` saat load weights.

## Code Change

### `src/gguf.h`
Tipe:
- `gguf_dtype` enum (F32, F16, dan placeholder untuk Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1, QK families).
- `gguf_vtype` enum (semua tipe value metadata).
- `gguf_kv` struct: key (string), type, value (union untuk scalar / string / array).
- `gguf_tensor_info` struct: name, n_dims, dims[8], dtype, offset.
- `gguf_ctx` struct: fd, mmap pointer, file size, version, n_tensors, n_kv, kv array, tensor info array, tensor data pointer, alignment.

API:
```c
int  gguf_load(const char* path, gguf_ctx* ctx);
void gguf_free(gguf_ctx* ctx);

const gguf_kv*           gguf_kv_get(const gguf_ctx*, const char* key);
int64_t                  gguf_kv_i64 (const gguf_ctx*, const char* key, int64_t def);
const char*              gguf_kv_str(const gguf_ctx*, const char* key);
const void*              gguf_kv_arr(const gguf_ctx*, const char* key,
                                     gguf_vtype* elem_type, uint64_t* n);

const gguf_tensor_info*  gguf_tensor_get (const gguf_ctx*, const char* name);
const void*              gguf_tensor_data(const gguf_ctx*, const gguf_tensor_info*);
```

### `src/gguf.c`
Implementasi dengan mmap via `mmap()` POSIX. Parsing little-endian (Termux ARM = LE, jadi direct read aman).

### `src/main.c` (minimal untuk Step 3)
Flag `--inspect`: load GGUF dari path (default resolve via Ollama manifest), print:
- version, n_tensors, n_kv
- 5 KV pertama (key + type)
- key config values: `general.architecture`, `<arch>.embedding_length`, `<arch>.block_count`, `<arch>.vocab_size`
- 5 tensor pertama (name + dtype + dims)
- konfirmasi tensor `token_embd.weight`, `output.weight`, `blk.0.attn_q.weight` ada

## Why This Change

- **mmap vs read**: mmap menghindari copy 270MB ke memori; OS page-in sesuai kebutuhan. Penting di perangkat dengan RAM terbatas.
- **No dequant di gguf.c**: separation of concerns. Reader hanya baca format; konversi adalah tanggung jawab konsumen. Mempermudah reuse jika nanti pakai quantized model.
- **5 KV pertama saja di inspect**: cukup untuk verifikasi parse berhasil tanpa spam ribuan baris.

## Logic / Pseudocode

```
gguf_load(path, ctx):
    fd = open(path, O_RDONLY)
    size = fstat(fd).st_size
    map = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0)
    p = map
    magic = read_u32(&p)        # expect 0x46554747
    version = read_u32(&p)      # expect 3
    n_tensors = read_u64(&p)
    n_kv = read_u64(&p)
    for i in n_kv:
        k = read_str(&p)
        t = read_u32(&p)
        v = read_value(t, &p)
        ctx->kv[i] = (k, t, v)
    for i in n_tensors:
        name = read_str(&p)
        n_dims = read_u32(&p)
        for d in n_dims: dims[d] = read_u64(&p)
        dtype = read_u32(&p)
        offset = read_u64(&p)
        ctx->tensors[i] = (name, dims, dtype, offset)
    # align to GGUF_DEFAULT_ALIGNMENT (32)
    tensor_data_start = align_up(p, 32)
    ctx->tensor_data = tensor_data_start
    return 0
```

## Test Simulation & Tracing

### Test: load + inspect
```
Input:  ~/.ollama/models/blobs/sha256-f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57

Expect:
  GGUF v3, n_tensors=?, n_kv=?
  general.architecture = "llama"   (SmolLM2 pakai arch llama)
  llama.embedding_length = 576
  llama.block_count = 30
  llama.attention.head_count = 9
  llama.attention.head_count_kv = 3
  llama.feed_forward_length = 1536
  tokenizer.ggml.tokens = array of 49152 strings
  tokenizer.ggml.model = "gpt2"   (BPE)
  Tensor[0]: token_embd.weight  dtype=F16  dims=[49152, 576]
  Tensor[1]: blk.0.attn_norm.weight ...
  ...
  Found: token_embd.weight, output.weight, blk.0.attn_q.weight  ✓
```

Total tensor count expect ≈ 363 (30 layers × ~12 tensors + 3 global).

## Manual Testing Plan

```bash
make
./smollm2 --inspect -m ~/.ollama/models/blobs/sha256-f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57
```

Output harus menampilkan config + 5 tensor pertama + konfirmasi key tensors ada.

## Status

- [ ] Spec written (this file)
- [ ] src/gguf.h
- [ ] src/gguf.c
- [ ] Minimal main.c with --inspect
- [ ] Verify against Ollama blob
