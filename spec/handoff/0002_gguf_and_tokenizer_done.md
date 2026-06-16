# 0002_gguf_and_tokenizer_done.md — Session Handoff

**Session date:** 2026-06-16
**Session goal:** Reset repo, build minimal C inference for SmolLM2-135M reading GGUF directly from Ollama storage. Reaches end of Step 4 of `spec/000_reset_and_rebuild.md`.
**Status at handoff:** Steps 1–4 complete and verified. Steps 5–10 pending. Build green. `./smollm2 --inspect` and `./smollm2 --tok-test "<text>"` both work.

---

## 1. What this project is

C inference engine for SmolLM2-135M. No training, no conversion — reads the GGUF blob that Ollama already has on disk. Goal priority order: (2) coherent chat output first → (1) performance (target ≥ 100 tok/s) → (3) clean reference code.

Why reset: old repo accumulated 201 files of dead-end experiments (dflash, server, paged attention, quant variants, chat_web/tui, spec-driven-llm-wiki). Wiped at commit `370fbec`. Git history preserved (`07b91d2` is the tip before reset; recoverable via `git checkout 07b91d2`).

## 2. Where the model lives

Model source is Ollama's local storage (no download, no conversion):

| Artifact | Path |
|---|---|
| GGUF weights (258 MB, F16) | `~/.ollama/models/blobs/sha256-f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57` |
| Manifest (digest → layer) | `~/.ollama/models/manifests/registry.ollama.ai/library/smollm2/135m` |
| System prompt blob | `sha256-fbacade46b4da804e0398c339c64b944d4b954452adf77cf050b49420116129e` (68 B) |
| Chat template blob (Go text/template) | `sha256-d502d55c1d609104ae6127aee92eb940e51e15c56dfb26dbd067e2771ee746f1` (675 B) |
| Params blob (`{"stop": [...]}`) | `sha256-f02dd72bb2423204352eabc5637b44d79d17f109fdb510a7c51455892aa2d216` (59 B) |
| Config blob (JSON) | `sha256-b0f58c4c1a3ca56f34a7673b353d7d09b773a1abce00451a58d5ebd2541331cf` (561 B) |

CLI auto-resolves the model blob path from the manifest. Look at `resolve_ollama_model_path()` in `src/main.c` — it does minimal JSON substring search for `"application/vnd.ollama.image.model"`, then takes the NEXT `"digest":` value (Ollama layer objects order fields as `mediaType, digest, size, from`). The digest `"sha256:HEX"` is rewritten to filename `"sha256-HEX"` for blob lookup.

## 3. SmolLM2-135M config (verified from GGUF metadata)

```
general.name             = "Smollm2 135M 8k Lc100K Mix1 Ep2"
general.architecture     = "llama"
llama.embedding_length   = 576
llama.block_count        = 30
llama.attention.head_count    = 9
llama.attention.head_count_kv = 3   (GQA, 3:1)
llama.feed_forward_length     = 1536
llama.attention.layer_norm_rms_epsilon = 1e-5
tokenizer.ggml.model     = "gpt2"   (BPE)
tokenizer.ggml.tokens    = 49152 entries
tokenizer.ggml.merges    = ~48900 entries
```

Tensor count = 272 = 1 (`token_embd.weight`) + 30 × 9 (per layer: `attn_norm`, `attn_q`, `attn_k`, `attn_v`, `attn_output`, `ffn_norm`, `ffn_gate`, `ffn_up`, `ffn_down`) + 1 (`output_norm.weight`).

**SmolLM2 ties embeddings**: there is no `output.weight` tensor. Logits = `x @ token_embd.weight^T`. Do not waste time looking for it.

Tensor dtypes: weights are F16 (`token_embd`, `attn_*`, `ffn_*`), norm weights are F32 (`attn_norm`, `ffn_norm`, `output_norm`).

## 4. Commits this session

```
370fbec  reset: wipe codebase, restart from scratch using Ollama GGUF
f92fde0  spec: add reset plan, spec skeleton, BEHAVIOUR.md, README, Makefile
0fc3342  feat(gguf): minimal GGUF v3 reader + inspect mode
8b21e43  feat(tokenizer): BPE tokenizer with roundtrip verification   ← HEAD
```

## 5. File layout

```
smollm2.c/
├── .gitignore                    # ignores CLAUDE.md, *.o, smollm2, *.sm2, etc.
├── BEHAVIOUR.md                  # coding rules (think / simplify / surgical / goal-driven)
├── Makefile                      # SRC = src/gguf.c src/tokenizer.c src/main.c
├── README.md                     # build + run + model auto-resolve
├── CLAUDE.md                     # UNTRACKED (22 KB prompt-style block, gitignored — replace if needed)
├── smollm2                       # built binary (gitignored)
├── spec/
│   ├── 000_reset_and_rebuild.md  # full rebuild plan, 10 steps
│   ├── 001_gguf_loader.md        # GGUF reader spec
│   ├── 002_tokenizer.md          # BPE tokenizer spec
│   ├── prompts/INSTRUCTIONS.md   # agent operating guide (read spec/handoff first, etc.)
│   ├── handoff/
│   │   ├── 0001_wipe_and_rebuild.md       # placeholder
│   │   └── 0002_gguf_and_tokenizer_done.md # ← THIS FILE
│   └── templates/spec_template.md         # XXX_<task>.md template
└── src/
    ├── gguf.h, gguf.c            # GGUF v3 reader (mmap, no dequant)
    ├── tokenizer.h, tokenizer.c  # GPT-2 BPE encode/decode
    └── main.c                    # CLI: --inspect, --tok-test
```

## 6. Verified working (re-run to confirm after fresh clone)

```bash
make
./smollm2 --inspect
# Expected first lines:
#   GGUF v3, n_tensors=272, n_kv=33, size=258.3 MB
#   architecture: llama
#     embedding_length : 576
#     block_count      : 30
#     head_count       : 9
#     head_count_kv    : 3
#     ffn_hidden       : 1536
#     rms_eps          : 1e-05
#   tokenizer model: gpt2
#   vocab size: 49152 (elem_type=8)

./smollm2 --tok-test "Hello"
# Expected:
#   input  (5 bytes): Hello
#   tokens (1): 19556
#   decode (5 bytes): Hello

./smollm2 --tok-test "Hello, world!"
# Expected:
#   tokens (4): 19556 28 905 17   = 'Hello' ',' ' world' '!'

./smollm2 --tok-test "<|im_start|>assistant
Hello<|im_end|>"
# Expected (special tokens matched literally, not as bytes):
#   tokens (6): 1 520 9531 198 19556 2
#   = '<|im_start|>' 'ass' 'istant' '\n' 'Hello' '<|im_end|>'
```

If any of those drift, the regression is in `src/gguf.c` (parse) or `src/tokenizer.c` (BPE algorithm). Inspect and tok-test are the two fast feedback loops — keep them green.

## 7. What's already done — implementation notes

### GGUF reader (`src/gguf.{h,c}`)
- mmap-based, no copy of tensor data. Tensor pointers are offsets into the mmap.
- KV parser handles all GGUF metadata value types (uint/int 8/16/32/64, float32/64, bool, string, array).
- Array of strings is heap-allocated as `char**` (used for `tokenizer.ggml.tokens` and `tokenizer.ggml.merges`).
- Accessors: `gguf_kv_get`, `gguf_kv_i64`, `gguf_kv_f32`, `gguf_kv_str`, `gguf_kv_arr`, `gguf_tensor_get`, `gguf_tensor_data`.
- `gguf_free` releases mmap + all heap KV data. Safe to call on partial init.

### Tokenizer (`src/tokenizer.{h,c}`)
- GPT-2 BPE with standard `byte_to_unicode` mapping (space → U+0120 'Ġ', newline → U+010A 'Ċ', etc.).
- Vocab hash: open-addressing FNV-1a, grows automatically. Same structure for `merges` (key = `"tokA tokB"`, value = rank index in merges array).
- Pre-tokenizer is hand-rolled ASCII approximation of the GPT-2 regex (`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`). UTF-8 multibyte chars are treated as letters (continuation bytes fold into the run). This matches SmolLM2 tokenization for ordinary English; may need refinement later for emoji/heavy Unicode.
- Special tokens (token_type=3 CONTROL or 4 USER_DEFINED) are matched literally against input before pre-tokenization, so `<|im_start|>` survives as one token id instead of being split into bytes.
- `tokenizer_decode` reverses `byte_to_unicode`: parses the UTF-8 of each token string back to codepoints, then maps each codepoint to its raw byte.
- Known minor limitation: when a BPE fragment is not in vocab (shouldn't happen for SmolLM2 since every byte has a single-codepoint token), we fall back to `<|unk|>` or id 0.

### CLI (`src/main.c`)
- `-m <path>` overrides auto-resolve.
- `--inspect` prints config + 5 first tensors + presence check for the 12 tensor names the forward pass will need.
- `--tok-test "<text>"` does encode → print IDs → decode → hex + per-token strings. Use this as the regression test whenever touching the tokenizer.
- `-p` and `-n` flags are parsed but not yet implemented (Step 7).

## 8. What's next — ordered, with acceptance checks

Follow `spec/000_reset_and_rebuild.md` steps 5–10. Each step should land its own commit and spec doc (`spec/003_*.md` onward).

### Step 5: forward pass (`spec/003_forward_pass.md` → `src/forward.{h,c}`)
- Read tensor pointers via `gguf_tensor_get`. Weights are F16; convert to F32 lazily at load time (recommended) or via lookup in the matmul hot loop (faster first load, slower per-token).
- Pipeline: `embed → 30×[RMSNorm → QKV proj → RoPE → causal attention w/ KV cache → O proj → residual → RMSNorm → SwiGLU FFN → residual] → final RMSNorm → x @ token_embd.weight^T → logits`.
- Tensor layouts: GGUF stores `dims=[out, in]` (column-major / row-major depends on tensor — llama.cpp convention is `ne[0]` = innermost/fastest = `in`, `ne[1]` = `out`). Verify against `dims` printed by `--inspect` (`token_embd.weight` is `[576, 49152]` → 576 = in/dim, 49152 = vocab).
- KV cache: per-layer K and V buffers, `n_layers × max_seq × kv_dim × sizeof(float)`. For max_seq=8192, kv_dim=192 (3 heads × 64), 30 layers → 8192 × 192 × 4 × 2 × 30 ≈ 754 MB. **Reduce default max_seq to ~2048 for Termux** to stay under typical RAM limits.
- Acceptance: prefill `"Hello"`, argmax the logits at the last position, compare with `ollama run smollm2:135m "Hello"` (use `--verbose` or check Ollama logs to see the first generated token id). They must match for greedy decoding.
- Reference for picking the right RoPE: SmolLM2 uses GPT-NeoX style (rotate on head_dim=64, theta base 10000). `llama.rope.dimension_count` should be 64 in the GGUF KV — verify.

### Step 6: sampling (`spec/004_sampling.md` → `src/sampling.{h,c}`)
- Greedy (temp=0) first. That alone is enough to verify coherence.
- Then temperature, top-p, top-k, repetition penalty.
- Parse `~/.ollama/models/blobs/sha256-f02dd72b...` for Ollama's default stop tokens (`{"stop": ["<|im_start|>", "<|im_end|>", ...]}`).
- Acceptance: `ollama run smollm2:135m "Hello"` and our `./smollm2 -p "Hello" -n 50 --temp 0` produce the same token sequence (or nearly, given RNG differences for temp>0).

### Step 7: chat CLI (`src/main.c` extend)
- Implement `-p` and `-n`. Build the chat prompt manually for the assistant turn:
  - `<|im_start|>system\n<system prompt from system blob>\n<|im_end|>\n<|im_start|>user\n<user prompt>\n<|im_end|>\n<|im_start|>assistant\n`
  - The system prompt is in blob `sha256-fbacade4...` (68 bytes, starts with `"You are"`).
  - For MVP: hardcode the template. Evaluating the Go text/template in blob `sha256-d502d55c...` is a follow-up.
- Stream tokens to stdout as they decode.

### Step 8: build + verify coherence vs Ollama
- 3-prompt coherence battery:
  ```
  ./smollm2 -p "Hello, how are you?" -n 50
  ./smollm2 -p "What is 2+2?" -n 50
  ./smollm2 -p "Tell me a joke." -n 80
  ```
- Compare each against `ollama run smollm2:135m "<same prompt>"`. Output should be comparable English (doesn't need to match byte-for-byte; both should be coherent, on-topic, not gibberish).
- Print tok/s. Scalar baseline expected ≥ 5 tok/s on the device.

### Step 9 (stretch): NEON perf
- `src/backend_neon.c` with NEON intrinsics for F16→F32 + dot product.
- Matmul is the bottleneck (SmolLM2-135M is small-dim, lots of layer iterations).
- Target ≥ 30 tok/s with NEON; full target 100 tok/s may also need weight quantization (Q4_0/Q8_0) which is a much larger change.

### Step 10: write the next handoff (`spec/handoff/0003_*.md`)
- Update with what landed, current tok/s, any new known issues.

## 9. Useful background facts (don't re-derive)

- `tie_word_embeddings` KV is missing (`-1`); output IS tied to `token_embd.weight` regardless. The GGUF just omits the redundant tensor.
- Tokenizer byte_to_unicode mapping must match GPT-2 exactly. The non-printable byte ordering is: iterate b=0..255 in order; for each non-printable byte (where printable = 33–126, 161–173, 174–255), assign codepoint 256+n with n incrementing. Space (b=32) lands at U+0120 ('Ġ'). Newline (b=10) lands at U+010A ('Ċ').
- `gguf_dtype_size()` for F16 returns 2 bytes — sanity-check any `offset` math against `gguf_tensor_data()` if you suspect alignment issues.
- Makefile currently lists `SRC = src/gguf.c src/tokenizer.c src/main.c`. Add `src/forward.c` and `src/sampling.c` to that list when those files exist.
- The `encode_state` struct in `src/tokenizer.c` is initialized field-by-field (not via initializer list) to avoid `-Wmissing-field-initializers`.
- Build is C99, `-O2 -march=native -Wall -Wextra`. No warnings expected.
- `CLAUDE.md` at repo root is a 22 KB prompt-style block, NOT real project instructions. It's gitignored. Replace with actual project memory if desired.

## 10. Common gotchas already hit (don't repeat)

- **Manifest digest parsing**: Ollama layer objects put `mediaType` BEFORE `digest`. Scanning backwards from `"application/vnd.ollama.image.model"` finds the wrong digest (the config object's). Scan FORWARD from the mediaType to the next `"digest":`. Fixed in `resolve_ollama_model_path()`.
- **Blob filename**: digest `"sha256:HEX..."` is stored on disk as `sha256-HEX...` (colon → dash).
- **Decode index off-by-256**: `g_b2u_unicode_to_byte` is indexed by full codepoint (0..511), not codepoint-minus-256. Easy to get backwards. The array is sized 512 specifically so direct codepoint indexing works.
- **No `output.weight`**: SmolLM2 ties embeddings. Don't fail the load on its absence; just use `token_embd.weight` for the unembed.

## 11. Quick start for the next session

```bash
cd /data/data/com.termux/files/home/smollm2.c
git log --oneline -5                       # confirm HEAD = 8b21e43
git status                                 # should be clean
ls spec/ src/                              # confirm layout matches section 5
make                                       # build should succeed with no warnings
./smollm2 --inspect                        # confirm model loads
./smollm2 --tok-test "Hello, world!"       # confirm tokenizer

# Then start Step 5: write spec/003_forward_pass.md, then src/forward.{h,c}.
# Reference: spec/000_reset_and_rebuild.md section "Step 5 — Forward pass".
```

When in doubt about intended behavior, the ground truth is `ollama run smollm2:135m`. Keep comparing.
