# 000_reset_and_rebuild.md

## Prompt

User: "kita reset semua, kita ulang semua yang sudah gagal, kita start from scratch dulu. jadi saya mau apa. terus juga (pake yang dari ollama biar nggak perlu training ulang)."

Tujuan: coherent chat output yang benar dulu → performance → clean reference code (paralel). Sumber model: Ollama `smollm2:135m` yang sudah ter-install lokal.

## Goal

Wipe total codebase lama di branch `main` (commit `370fbec`), ganti dengan implementasi C minimal yang membaca GGUF langsung dari storage Ollama (`~/.ollama/models/`). Priority urutan: (2) coherent chat output, (1) performance (target ≥ 100 tok/s, baseline scalar ≥ 5 tok/s), (3) clean reference code.

## Why

Codebase lama menumpuk 201 file: 107 file spec-driven-llm-wiki, 43 file src dengan banyak sub-modul experimental yang tidak jalan (dflash placeholder, server HTTP/SSE, kv pool/page/quant variants, paged attention, quant q4k/q5k, chat_web/tui, speculative), plus puluhan scratch debug binaries (`test_*.c`, `trace_*.c`, `check_*.c`) di root. Sulit dibaca, sulit di-trust. Banyak percobaan optimasi yang tidak verified membantu.

Dengan memakai model Ollama yang sudah ter-install (`smollm2:135m`, 270MB GGUF v3), kita:
- Tidak perlu training/conversi ulang.
- Punya ground truth: `ollama run smollm2:135m "<prompt>"` untuk validasi output.
- Tokenizer tertanam di GGUF (vocab + merges + scores + token_type).

## Codebase Context

### Model source (Ollama storage)
| Komponen | Path blob | Ukuran |
|----------|-----------|--------|
| GGUF weights | `~/.ollama/models/blobs/sha256-f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57` | 270 MB |
| Config JSON | `~/.ollama/models/blobs/sha256-b0f58c4c1a3ca56f34a7673b353d7d09b773a1abce00451a58d5ebd2541331cf` | 561 B |
| Chat template (Go-template) | `~/.ollama/models/blobs/sha256-d502d55c1d609104ae6127aee92eb940e51e15c56dfb26dbd067e2771ee746f1` | 675 B |
| System prompt | `~/.ollama/models/blobs/sha256-fbacade46b4da804e0398c339c64b944d4b954452adf77cf050b49420116129e` | 68 B |
| Params (`{"stop": ...}`) | `~/.ollama/models/blobs/sha256-f02dd72bb2423204352eabc5637b44d79d17f109fdb510a7c51455892aa2d216` | 59 B |
| Manifest | `~/.ollama/models/manifests/registry.ollama.ai/library/smollm2/135m` | mapping digest→layer |

GGUF magic verified: `4747 5546 0300 0000` = "GGUF" v3.

### SmolLM2-135M config (dari config.json + GGUF metadata)
- dim (embedding) = 576
- n_layers = 30
- n_heads = 9 (query)
- n_kv_heads = 3 (GQA, ratio 3:1)
- head_dim = 64 (= 576 / 9)
- vocab_size = 49152
- ffn_hidden = 1536 (SwiGLU, multiplier ~2.67)
- max_seq = 8192
- RMSNorm with eps = 1e-5
- RoPE base theta = 10000

## Logical Change

Bangun dari nol dengan pipeline minimal:

1. **GGUF reader** — parse header + metadata KV + tensor info table, mmap tensor data. Output: struct dengan pointer ke setiap tensor by name.
2. **Tokenizer BPE** — baca `tokenizer.ggml.{tokens,merges,scores,token_type,model}` dari GGUF metadata. Encode: byte-level → BPE merge. Decode: token IDs → bytes. Special tokens (`<|im_start|>`, `<|im_end|>`) di-match literal.
3. **Forward pass transformer** — embed → 30×[RMSNorm → QKV proj → RoPE → causal attention w/ KV cache → O proj → residual → RMSNorm → SwiGLU FFN → residual] → final RMSNorm → output proj → logits. F16 weights → F32 saat load.
4. **Sampling** — greedy (temp=0) sebagai default. Plus temperature, top-p, top-k, repetition penalty. Parse params blob untuk stop tokens.
5. **Chat CLI** — auto-resolve GGUF path dari manifest Ollama, apply chat template (hardcode assistant-turn untuk MVP), stream token.

## Code Change

### File structure setelah reset
```
smollm2.c/
├── Makefile                       # single target, ~30 lines
├── README.md                      # build + run + model auto-resolve
├── BEHAVIOUR.md                   # coding rules
├── .gitignore
├── spec/
│   ├── 000_reset_and_rebuild.md   # this file
│   ├── 001_gguf_loader.md
│   ├── 002_tokenizer.md
│   ├── 003_forward_pass.md
│   ├── 004_sampling.md
│   ├── prompts/INSTRUCTIONS.md
│   ├── handoff/0001_wipe_and_rebuild.md
│   └── templates/spec_template.md
└── src/
    ├── gguf.{h,c}                 # GGUF v3 reader
    ├── tokenizer.{h,c}            # BPE
    ├── forward.{h,c}              # transformer
    ├── sampling.{h,c}             # sampling
    └── main.c                     # CLI
```

### Tensor name mapping (GGUF → SmolLM2)
| GGUF tensor name | Shape | Used as |
|------------------|-------|---------|
| `token_embd.weight` | [vocab, dim] | embedding lookup |
| `blk.{n}.attn_norm.weight` | [dim] | pre-attention RMSNorm |
| `blk.{n}.attn_q.weight` | [dim, dim] | Q projection |
| `blk.{n}.attn_k.weight` | [dim, kv_dim] | K projection |
| `blk.{n}.attn_v.weight` | [dim, kv_dim] | V projection |
| `blk.{n}.attn_output.weight` | [dim, dim] | output projection |
| `blk.{n}.ffn_norm.weight` | [dim] | pre-FFN RMSNorm |
| `blk.{n}.ffn_gate.weight` | [ffn_hidden, dim] | SwiGLU gate |
| `blk.{n}.ffn_up.weight` | [ffn_hidden, dim] | SwiGLU up |
| `blk.{n}.ffn_down.weight` | [dim, ffn_hidden] | SwiGLU down |
| `output_norm.weight` | [dim] | final RMSNorm |
| `output.weight` | [vocab, dim] | unembed |

## Why This Change

- **GGUF langsung**: menghindari konversi `.sm2` custom yang error-prone (sumber bug sebelumnya). Pakai format standar ecosystem.
- **Pipeline minimal dalam 5 file**: menghilangkan duplikasi (sebelumnya ada `sm2_matmul_ref.c`, `sm2_matmul_fast.c`, `sm2_attn_prefill.c`, `sm2_attn_flash_prefill.c`, `sm2_attn_paged.c`, dst.). Satu forward pass, satu backend.
- **Spec dulu, code kemudian**: setiap komponen punya spec doc terpisah (`001`..`004`) untuk audit trail.
- **Coherent dulu**: prioritas keluaran koheren sebelum optimasi perf. Ground truth Ollama memungkinkan verifikasi.

## Logic / Pseudocode

### GGUF load
```
gguf_load(path):
    fd = open(path)
    read magic, version, n_tensors, n_kv
    for i in n_kv: read key, type, value → kv_map
    for i in n_tensors: read name, n_dims, dims, dtype, offset → tensor_table
    mmap data section
    return ctx
```

### Forward pass (single token with KV cache)
```
forward(model, token, pos, kv_cache):
    x = model.token_embd[token]                    # [dim]
    for layer in 0..n_layers:
        h = rmsnorm(x, layer.attn_norm)
        q = h @ layer.attn_q.T                      # [dim]
        k = h @ layer.attn_k.T                      # [kv_dim]
        v = h @ layer.attn_v.T                      # [kv_dim]
        rope_apply(q, k, pos)                       # rotate q, k
        kv_cache[layer].k[pos] = k
        kv_cache[layer].v[pos] = v
        attn = causal_attention(q, kv_cache[layer].k[:pos+1], kv_cache[layer].v[:pos+1])
        x = x + attn @ layer.attn_output.T
        h = rmsnorm(x, layer.ffn_norm)
        gate = h @ layer.ffn_gate.T
        up = h @ layer.ffn_up.T
        ffn = silu(gate) * up
        x = x + ffn @ layer.ffn_down.T
    x = rmsnorm(x, model.output_norm)
    logits = x @ model.output.T                     # [vocab]
    return logits
```

## Test Simulation & Tracing

### Test 1: GGUF load
```
Input:  ~/.ollama/models/blobs/sha256-f535f83e...
Expect: n_tensors ≈ 363 (30 layers × ~12 tensors + globals)
        n_kv > 50
        dim = 576, n_layers = 30, vocab = 49152
```

### Test 2: Tokenizer encode "Hello"
```
Input:  "Hello"
Expect: 1-2 tokens, e.g. [15496] ('Hello') — confirm via ollama show smollm2:135m
```

### Test 3: First-token argmax match
```
Input:  prompt "Hello"
        prefill → logits at last position
        argmax(logits) → token_id
Expect: token_id matches what ollama run smollm2:135m "Hello" returns as first generated token
```

### Test 4: 50-token greedy generation
```
Input:  prompt "Hello, how are you?"
        greedy decode 50 tokens
Expect: coherent English, comparable to Ollama output
```

## Manual Testing Plan

```bash
# Build
make

# GGUF inspect
./smollm2 --inspect -m ~/.ollama/models/blobs/sha256-f535f83e...

# Single prompt
./smollm2 -p "Hello, how are you?" -n 50

# Compare with ground truth
ollama run smollm2:135m "Hello, how are you?"

# Performance baseline
./smollm2 -p "Once upon a time" -n 50
# Expect: tok/s reported, baseline ≥ 5 tok/s scalar
```

## Status

- [x] Spec written (this file)
- [x] Wipe & reset commit (`370fbec`)
- [ ] Step 2: spec skeleton + meta files
- [ ] Step 3: GGUF reader (`001_gguf_loader.md`)
- [ ] Step 4: Tokenizer (`002_tokenizer.md`)
- [ ] Step 5: Forward pass (`003_forward_pass.md`)
- [ ] Step 6: Sampling (`004_sampling.md`)
- [ ] Step 7: Chat CLI
- [ ] Step 8: Build & verify coherence vs Ollama
- [ ] Step 9: NEON perf (stretch)
- [ ] Step 10: Handoff (`spec/handoff/0001_wipe_and_rebuild.md`)
