# 002_tokenizer.md

## Prompt

Implementasikan BPE tokenizer yang membaca vocab/merges dari metadata GGUF (sudah di-parse di Step 3). Output token IDs yang konsisten dengan Ollama untuk input string yang sama.

## Goal

Pipeline encode (`text` → `int[]` token IDs) dan decode (`int[]` → `text`) yang menghasilkan token IDs identik dengan `ollama run smollm2:135m` untuk input bahasa Inggris umum.

## Why

Tokenizer adalah satu-satunya cara mengubah prompt user menjadi input model. Jika encode menghasilkan token IDs yang berbeda dari referensi Ollama, logits akan sepenuhnya berbeda dan output tidak akan koheren — terlepas dari betapa benarnya forward pass.

Sebelumnya di repo lama, tokenizer BPE adalah sumber bug parah: special tokens seperti `<|im_start|>` di-encode sebagai 7 byte individu, bukan 1 token. Fix di Step 4 harus memastikan ini benar.

## Codebase Context

- Source data (di dalam GGUF, sudah ter-parse via `src/gguf.h`):
  - `tokenizer.ggml.model` = `"gpt2"` (BPE variant)
  - `tokenizer.ggml.tokens` — array 49152 string (vocabulary)
  - `tokenizer.ggml.merges` — array ~48900 string `"tokA tokB"` (merge operations)
  - `tokenizer.ggml.token_type` — array 49152 int32 (1=normal, 2=unknown, 3=control, 4=user_defined, 5=unused, 6=byte)
  - `tokenizer.ggml.scores` — array 49152 float32 (merge priority, lower = earlier merge)
- Special tokens SmolLM2 yang sering dipakai: `<|im_start|>` (id=1), `<|im_end|>` (id=2)
- Referensi algoritma: GPT-2 BPE (<https://github.com/openai/gpt-2/blob/master/src/encoder.py>)

## Logical Change

Tambah modul `src/tokenizer.{h,c}` yang:

1. **Build maps saat load**:
   - `vocab`: string → token_id (hash table)
   - `inv_vocab`: token_id → string
   - `token_type`: token_id → type enum
   - `merge_rank`: pair-string → rank (index dalam merges array)
   - `byte_to_unicode`: 256-byte → unicode char mapping (GPT-2 standard)
   - `special_tokens`: set dari token dengan type=CONTROL/USER_DEFINED

2. **Encode**:
   - Pre-tokenize input text dengan GPT-2 regex approximation (`'s|'t|'re|'ve|'m|'ll|'d| ?[letters]+| ?[digits]+| ?[other]+|[\s]+`).
   - Untuk setiap pre-token: konversi setiap byte ke unicode char via `byte_to_unicode`, lalu iteratively merge pasangan dengan rank terendah sampai tidak ada merge yang tersisa.
   - Special tokens di-match secara literal di level sebelum pre-tokenize (pisahkan input berdasarkan kemunculan special tokens).
   - Return array token IDs.

3. **Decode**:
   - Untuk setiap token_id: ambil string dari `inv_vocab`, reverse-map setiap char via `unicode_to_byte`, akumulasikan byte.
   - Skip special tokens (atau emit placeholder) — opsional, bisa di-customize caller.

## Code Change

### `src/tokenizer.h`

```c
typedef struct {
    int n_vocab;
    // ... opaque fields ...
} tokenizer;

int  tokenizer_load (tokenizer* t, const gguf_ctx* g);
void tokenizer_free (tokenizer* t);

// Encode: returns number of tokens, writes to out[] (caller allocates).
// max_out caps the count. Special tokens in input are recognized.
int  tokenizer_encode(const tokenizer* t, const char* text,
                      int* out, int max_out);

// Decode: token_id → bytes, written to buf (caller allocates).
// Returns number of bytes, or -1 if token unknown.
int  tokenizer_decode(const tokenizer* t, int token_id,
                      char* buf, int max_buf);

// Convenience: look up token by literal name (e.g. "<|im_start|>").
int  tokenizer_lookup(const tokenizer* t, const char* token_text);
```

### `src/tokenizer.c`

Struktur internal:
- Hash table (open addressing, FNV-1a hash) untuk `vocab` dan `merge_rank`.
- byte_to_unicode precomputed di `tokenizer_load`.

### Verifikasi via `--tok-test` flag di main.c

Tambah flag `--tok-test <text>`: encode text, print token IDs, lalu decode kembali dan bandingkan dengan input.

## Why This Change

- **BPE rank-based merge** menjamin token IDs identik dengan Ollama (yang juga pakai algoritma yang sama dari GGUF metadata).
- **Special token literal match** menghindari bug sebelumnya (special tokens tidak boleh dipecah menjadi byte).
- **byte_to_unicode mapping** adalah standard GPT-2 — memungkinkan byte 0–255 (termasuk control chars) direpresentasikan sebagai printable unicode, sehingga bisa di-merge sebagai string biasa.

## Logic / Pseudocode

```
byte_to_unicode():
    printable = bytes !..~ + ¡..¬ + ®..ÿ
    cs = printable[:]
    n = 0
    for b in 0..255:
        if b not in printable:
            cs.append(256 + n); n++
    return {b: chr(c) for b, c in zip(range(256), cs)}

encode(text):
    out = []
    # Split on special tokens first
    segments = split_on_specials(text)
    for seg_text, is_special in segments:
        if is_special:
            out.append(lookup(seg_text))
            continue
        # Pre-tokenize
        for chunk in pretokenize(seg_text):
            # Convert chunk bytes to unicode chars
            word = [byte_to_unicode[b] for b in chunk.encode('utf-8')]
            # Iteratively merge
            while len(word) > 1:
                best_rank = INF
                best_idx = -1
                for i in 0..len(word)-2:
                    key = word[i] + " " + word[i+1]
                    r = merge_rank.get(key, INF)
                    if r < best_rank:
                        best_rank = r; best_idx = i
                if best_idx < 0: break
                word = word[:best_idx] + [word[best_idx] + word[best_idx+1]] + word[best_idx+2:]
            # Look up
            for tok in word:
                id = vocab[tok]
                if id >= 0: out.append(id)
                # else: emit unknown token or fallback
    return out

pretokenize(text):
    # Approximation of GPT-2 regex
    chunks = []
    i = 0
    while i < len(text):
        c = text[i]
        if c == ' ':
            # Take the space + following run of letters/digits/other
            j = i + 1
            # Classify next char
            ... classify and consume
        elif isspace(c):
            # Run of whitespace
            ...
        else:
            # Run of letters / digits / other based on first char class
            ...
    return chunks

decode(token_id):
    s = inv_vocab[token_id]
    out = []
    for c in s:
        b = unicode_to_byte[c]
        out.append(b)
    return bytes(out)
```

## Test Simulation & Tracing

### Test 1: byte_to_unicode sanity
```
byte_to_unicode[' '] = 'Ġ' (0 torsion 0x104)  -- GPT-2 standard
byte_to_unicode['H'] = 'H'
byte_to_unicode['\n'] = 'Ċ'
```

### Test 2: encode "Hello"
```
Input: "Hello"
Pre-tok: ["Hello"]
Bytes: [H, e, l, l, o]
Unicode: [H, e, l, l, o]
Merges: 
  (H, e) → "He"? (check rank)
  (He, l) → "Hel"? 
  ... 
Final: ["Hello"] (likely single token in vocab)
Expected token_id: ~15496 (SmolLM2 GPT-2 BPE has 'Hello' as one token)
```

### Test 3: encode " hello" (leading space)
```
Pre-tok: [" hello"]
Bytes: [0x20, h, e, l, l, o]
Unicode: [Ġ, h, e, l, l, o]
Merges: Ġ+h→Ġh, Ġh+e→Ġhe, ... → "Ġhello"
token_id: ~13000s
```

### Test 4: special token
```
Input: "<|im_start|>"
Should be: [1] (single token, token_type=CONTROL)
```

## Manual Testing Plan

```bash
make

# Encode/decode test
./smollm2 --tok-test "Hello, world!"
./smollm2 --tok-test " hello"
./smollm2 --tok-test "<|im_start|>assistant\nHello<|im_end|>"

# Compare with Ollama
ollama show smollm2:135m  # may show tokenizer info
```

Expected: roundtrip (encode → decode) menghasilkan teks identik dengan input (untuk teks plain ASCII).

## Status

- [ ] Spec written (this file)
- [ ] src/tokenizer.h
- [ ] src/tokenizer.c
- [ ] --tok-test flag in main.c
- [ ] Verify roundtrip + compare with Ollama
