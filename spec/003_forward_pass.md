# 003_forward_pass.md

## Prompt

> Lanjutkan Step 5 dari `spec/000_reset_and_rebuild.md`: implementasikan forward pass transformer SmolLM2-135M dalam C murni. Terima `token_ids[]`, keluarkan `logits[vocab]`. Verifikasi: argmax logits untuk prompt `"Hello"` harus sama dengan token pertama yang dipancarkan `ollama run smollm2:135m "Hello"` pada greedy decoding.

## Goal

Implement forward pass SmolLM2-135M end-to-end dalam satu file `src/forward.{h,c}`: embedding lookup, 30 layer transformer (RMSNorm → QKV → RoPE → causal attention dengan KV cache & GQA → output projection → residual → RMSNorm → SwiGLU FFN → residual), final RMSNorm, dan proyeksi logits memakai embedding yang ter-tied (tidak ada tensor `output.weight`). Tambahkan mode CLI `--logits <prompt>` yang menjalankan prefill lalu mencetak token ID argmax beserta string hasil decode-nya, sebagai gate verifikasi sebelum Step 6 (sampling).

## Why

Tanpa forward pass, binary `smollm2` tidak bisa menghasilkan token baru — `-p` dan `-n` di `src/main.c:261–263` masih stub. Tokenizer (Step 4) hanya mengubah prompt jadi ID; transformer-lah yang mengubah ID jadi distribusi token berikutnya. Gate verifikasi sempit (argmax satu prompt vs Ollama) cukup untuk membuktikan kabel transformer terpasang benar — jika match, RoPE, RMSNorm, GQA, SwiGLU, dan tied embeddings semua bekerja. Jika tidak, salah satu detail itu yang mencelakai.

## Codebase Context

- `src/gguf.h` — sumber API untuk membaca tensor: `gguf_tensor_get(ctx, name)` → `gguf_tensor_info`, `gguf_tensor_data(ctx, info)` → pointer ke data mmap. `gguf_kv_i64(ctx, key, default)`, `gguf_kv_f32(ctx, key, default)` untuk konfigurasi. `gguf_dtype` enum: `GGUF_DT_F16` untuk bobot, `GGUF_DT_F32` untuk norm.
- `src/tokenizer.h` — sumber `tokenizer_encode` (prompt → IDs), `tokenizer_decode` (ID → bytes), `tokenizer_vocab_size`. Forward pass tidak memakai tokenizer internal, tapi CLI mode `--logits` memakainya.
- `src/main.c` — CLI driver. Akan ditambah mode baru `--logits <prompt>`.
- `Makefile` — perlu menambah `src/forward.c` ke `SRC`.
- `spec/handoff/0002_gguf_and_tokenizer_done.md` §8 — sumber keputusan: F16→F32 lazy at load, max_seq=2048, RoPE GPT-NeoX, tied embeddings.

## Logical Change

Tambah satu modul `forward.{h,c}`. Saat `forward_load` dipanggil dengan `gguf_ctx` dan `max_seq`:

1. Baca konfigurasi dari KV (`embedding_length`, `block_count`, `attention.head_count`, `attention.head_count_kv`, `feed_forward_length`, `attention.layer_norm_rms_epsilon`). Turunkan `head_dim = dim / n_heads`, `kv_dim = n_kv_heads * head_dim`.
2. Alokasikan satu buffer besar per kategori bobot: token embedding, output norm, per-layer norm weights, Q/K/V/O, gate/up/down, KV cache.
3. Dequantisasi F16 → F32 sekaligus saat load (lazy at load).
4. Alokasikan buffer aktivasi sementara (x_norm, q, k, v, attn_out, o_proj, ffn_gate, ffn_up, ffn_mid, scores) — di-reuse lintas token & layer.

`forward_prefill(tokens, n_tokens, logits_out)`:

1. Embed: untuk setiap t, `memcpy(x[t], w_token_embd[tokens[t]], dim)`.
2. Untuk setiap layer L (0..29), untuk setiap token t (0..n_tokens-1):
   - RMSNorm(x[t], w_attn_norm[L]) → x_norm.
   - q = Wq[L] @ x_norm (dim), k = Wk[L] @ x_norm (kv_dim), v = Wv[L] @ x_norm (kv_dim).
   - RoPE GPT-NeoX pada q & k, posisi t, head_dim=64.
   - Tulis k,v ke k_cache[L][t] / v_cache[L][t].
   - Untuk setiap q head h (0..8): kvh = h/3; hitung skor = q·k_cache[L][0..t][kvh] / sqrt(head_dim); softmax (causal otomatis karena cuma 0..t); jumlahkan v_cache[L][0..t][kvh] tertimbang ke attn_out[h].
   - o = Wo[L] @ attn_out; x[t] += o.
   - RMSNorm(x[t], w_ffn_norm[L]) → x_norm.
   - g = silu(Wg[L] @ x_norm) * (Wu[L] @ x_norm); down = Wd[L] @ g; x[t] += down.
3. Final RMSNorm(x[n-1], w_norm) → x_norm.
4. Untuk setiap v di [0,vocab): logits[v] = dot(x_norm, w_token_embd[v]).

## Code Change

- `src/forward.h` (new) — expose `forward_ctx` opaque struct, `forward_load`, `forward_free`, `forward_prefill`, `forward_vocab_size`.
- `src/forward.c` (new) — struct fields, F16→F32 helper, `rmsnorm`, `matmul` (row-major W[out,in] @ x[in] → y[out]), `rope_neox`, `softmax_inplace`, `silu`, loader, prefill loop.
- `Makefile` — tambah `src/forward.c` ke `SRC`.
- `src/main.c` — tambah argumen CLI `--logits <prompt>`: load GGUF, load tokenizer, load forward ctx (max_seq default 2048), encode prompt, panggil `forward_prefill`, cari argmax, decode, cetak. Tidak menyentuh `-p`/`-n`/`-i` (Step 6/7).

## Why This Change

- **Lazy F16→F32 at load** (bukan di hot loop): bobot F32 di-hot-loop akan menduplikasi konversi tiap prefill; load sekali, pakai berulang. Memori ±540 MB bobot + 90 MB KV cache, masih aman di Termux.
- **Token-by-token dalam layer** (bukan vector-over-sequence): lebih sederhana, buffer aktivasi kecil (per-token), tetap benar karena setiap token hanya butuh k/v posisi ≤ t yang sudah ditulis ke cache sebelumnya dalam iterasi yang sama. Loop luar layer → dalam token.
- **max_seq=2048** default: membatasi KV cache ke ±90 MB. Prompt ≥2048 token akan ditolak eksplisit (error), bukan diam-diam dipotong.
- **Tied embeddings**: jangan cari `output.weight`. Logits = x @ token_embd.weight^T. Salah di sini → output tampak masuk akal tapi salah total.
- **Mode `--logits` terpisah** (bukan langsung `-p`): Step 5 hanya verifikasi argmax. Sampling, autoregressive decode, dan CLI chat adalah Step 6–7. Menyuntikkan mereka sekarang akan mengaburkan gate verifikasi.
- **Matmul naif** (no BLAS, no NEON): target Step 5 adalah kebenaran, bukan performa. -O2 -march=native cukup untuk prefill beberapa token dalam hitungan detik. NEON adalah Step 9.

## Logic / Pseudocode

```
forward_load(out, g, max_seq):
    baca konfigurasi dari g (dim=576, n_layers=30, n_heads=9, n_kv_heads=3,
                              ffn_hidden=1536, rms_eps=1e-5, vocab=49152)
    head_dim = dim / n_heads = 64
    kv_dim = n_kv_heads * head_dim = 192

    alokasikan & dequant F16→F32:
        w_token_embd [vocab * dim]
        w_norm       [dim]
        per-layer arrays [n_layers * ...]:
            w_attn_norm [dim], w_q [dim*dim], w_k [kv_dim*dim],
            w_v [kv_dim*dim], w_o [dim*dim],
            w_ffn_norm [dim],
            w_gate [ffn*dim], w_up [ffn*dim], w_down [dim*ffn]
        k_cache, v_cache [n_layers * max_seq * kv_dim]
        aktivasi temp: x[max_seq*dim], x_norm[dim], q[dim],
                       k[kv_dim], v[kv_dim], attn_out[dim], o_proj[dim],
                       ffn_gate[ffn], ffn_up[ffn], ffn_mid[dim],
                       scores[max_seq]

rmsnorm(out, x, w, n, eps):
    ss = sum(x[i]^2 for i) / n
    inv = 1 / sqrt(ss + eps)
    out[i] = x[i] * inv * w[i]

matmul(y, x, W, in_dim, out_dim):
    for o in 0..out_dim:
        acc = 0
        for i in 0..in_dim:
            acc += W[o*in_dim + i] * x[i]
        y[o] = acc

rope_neox(v, pos, n_h, hd, theta=10000):
    for h in 0..n_h:
        vh = v + h*hd
        for i in 0..hd/2:
            freq = 1 / theta^(2*i / hd)
            angle = pos * freq
            c = cos(angle); s = sin(angle)
            a = vh[i]; b = vh[i + hd/2]
            vh[i]       = a*c - b*s
            vh[i+hd/2]  = a*s + b*c

forward_prefill(f, tokens, n_tokens, logits_out):
    assert 0 < n_tokens <= max_seq
    for t in 0..n_tokens:
        memcpy(x[t], w_token_embd + tokens[t]*dim, dim)

    for L in 0..n_layers:
        for t in 0..n_tokens:
            rmsnorm(x_norm, x[t], w_attn_norm[L], dim, eps)
            matmul(q, x_norm, w_q[L], dim, dim)
            matmul(k, x_norm, w_k[L], dim, kv_dim)
            matmul(v, x_norm, w_v[L], dim, kv_dim)
            rope_neox(q, t, n_heads, head_dim)
            rope_neox(k, t, n_kv_heads, head_dim)
            k_cache[L][t] <- k
            v_cache[L][t] <- v

            inv_sqrt = 1 / sqrt(head_dim)
            for h in 0..n_heads:
                kvh = h * n_kv_heads / n_heads
                max_s = -INF
                for s in 0..=t:
                    d = dot(q[h], k_cache[L][s][kvh], head_dim) * inv_sqrt
                    scores[s] = d; max_s = max(max_s, d)
                sum = 0
                for s in 0..=t:
                    scores[s] = exp(scores[s] - max_s); sum += scores[s]
                inv_sum = 1 / sum
                zero(attn_out[h])
                for s in 0..=t:
                    w = scores[s] * inv_sum
                    for i in 0..head_dim:
                        attn_out[h*head_dim + i] += w * v_cache[L][s][kvh*head_dim + i]

            matmul(o_proj, attn_out, w_o[L], dim, dim)
            x[t] += o_proj

            rmsnorm(x_norm, x[t], w_ffn_norm[L], dim, eps)
            matmul(ffn_gate, x_norm, w_gate[L], dim, ffn_hidden)
            matmul(ffn_up,   x_norm, w_up[L],   dim, ffn_hidden)
            for i in 0..ffn_hidden:
                g = ffn_gate[i]
                sig = 1 / (1 + exp(-g))
                ffn_gate[i] = g * sig * ffn_up[i]
            matmul(ffn_mid, ffn_gate, w_down[L], ffn_hidden, dim)
            x[t] += ffn_mid

    rmsnorm(x_norm, x[n_tokens-1], w_norm, dim, eps)
    for v in 0..vocab:
        logits_out[v] = dot(x_norm, w_token_embd + v*dim, dim)
```

## Test Simulation & Tracing

Prompt `"Hello"` → token ID tunggal `19556` (verified via `--tok-test`).

Prefill: n_tokens=1, jadi loop t hanya t=0.

- x[0] = w_token_embd[19556] (vector 576-d).
- Layer 0: RMSNorm → q,k,v (dim=576 + 2×kv_dim=192). RoPE pos 0 → identitas (sin 0 = 0, cos 0 = 1). KV cache diisi di slot 0. Attention: 1 token, softmax dari 1 skor = 1.0, attn_out = v[0]. o = Wo @ attn_out; x += o. FFN block; x += down.
- Ulang 30 layer.
- RMSNorm akhir; logits = x @ w_token_embd^T.
- argmax → harusnya token yang Ollama pancarkan setelah "Hello".

Predicted first-token dari `smollm2:135m` untuk prompt "Hello" biasanya token kosong-dipimpin seperti ` there` atau `!`. Akan diverifikasi saat menjalankan.

Sanity bound: jika logits semua NaN → bug di F16 decode atau RMSNorm. Jika argmax = 0 atau token ngaco (<|im_start|>, dsb.) → bug di RoPE / matmul / GQA mapping.

## Manual Testing Plan

```bash
cd /data/data/com.termux/files/home/smollm2.c

# 1. Build (no warnings)
make clean && make

# 2. Regression: GGUF & tokenizer masih jalan
./smollm2 --inspect | head -5
./smollm2 --tok-test "Hello"

# 3. Forward pass gate
./smollm2 --logits "Hello"
# Expected output:
#   prompt tokens (1): 19556
#   argmax: <ID>
#   decoded: "..."

# 4. Ground truth comparison
ollama run smollm2:135m "Hello"
# First generated token must match the argmax printed above

# 5. Multi-token prompt
./smollm2 --logits "Hello, how are you?"
ollama run smollm2:135m "Hello, how are you?"
```

## Status

- [x] Spec written
- [x] Implementation
- [x] Verified
- [x] Handoff written
