# 006_heavy_mode.md

## Prompt

Tambahkan CLI flag `--heavy` ke `smollm2` yang menjalankan multi-pass pipeline
think → answer → verify (gate) dalam satu proses. Tujuannya memberi model
"waktu berpikir" sebelum menjawab, lalu quality-gate satu putaran sebelum
output final.

## Goal

`./smollm2 -p "<PROMPT>" --heavy -n 48 --temp 0` menjalankan 3-4 prefills pada
**satu** `forward_load`:
1. THINK — system prompt minta step-by-step notes di `<think>...</think>`, stop
   pada `</think>` atau budget `--heavy-think-n` (default 128).
2. ANSWER — system prompt "answer from notes", sertakan notes di user body,
   stop pada `im_end` atau `-n`.
3. VERIFY — model stamp `VERIFIED: <reason>` atau `REJECT: <critique>`. Stop
   pada `im_end` atau `--heavy-verify-n` (default 64).
4. RE-ANSWER (opsional) — hanya jika VERIFY output dimulai dengan `REJECT`
   (prefix match setelah skip spasi). Tidak ada VERIFY kedua.

Output stdout diberi label section `=== THINK ===`, `=== ANSWER ===`,
`=== VERIFY ===`, dan `=== RE-ANSWER ===` (jika ada). Dihasilkan juga baris
summary `[heavy: think=N ans=N ver=N [re=N] tokens, T.Ts, X.X tok/s overall]`.

## Why

SmolLM2 135M sering menjawab langsung tanpa "berpikir". Scaffold sederhana ini
memberi prompting terstruktur tanpa menambah dependency atau model. Gate VERIFY
menangkap jawaban salah yang lolos di pass ANSWER.

Tidak menggantikan model yang lebih besar — ini scaffolding, bukan keajaiban.

## Codebase Context

- File dimodifikasi:
  - `src/main.c` — flag parse, `gen_once()` helper, `do_heavy()`, dispatch di
    `main`, update `usage()`.
  - `eval/heavy_test.py` — harness TDD.
- File baru:
  - `spec/006_heavy_mode.md` (dokumen ini).
- Tidak ada perubahan ke `forward.c`, `tokenizer.c`, `sampling.c`,
  `tui.c`, `web.c`.
- Special tokens: `<|im_start|>` (id=1), `<|im_end|>` (id=2) hardcoded di
  `main.c`/TUI/web. `<think>` / `</think>` adalah **plain BPE text**, bukan
  token kontrol.

## Logical Change

### Flags (di parse loop `main`)

| Flag | Default | Arti |
|---|---|---|
| `--heavy` | off | aktifkan pipeline multi-pass |
| `--heavy-think-n <int>` | 128 | budget token THINK |
| `--heavy-verify-n <int>` | 64 | budget token VERIFY |
| `-n` | 200 | budget token ANSWER & RE-ANSWER |

`--temp`, `--top-p`, `--top-k`, `--rep-penalty` berlaku untuk semua pass.

Dispatch: `if (heavy && prompt)` → `do_heavy()`, else `do_generate()`.

### `gen_once()` (static helper di `main.c`)

Faktor dari loop `do_generate` existing. Signature:

```c
static int gen_once(forward_ctx* fwd, tokenizer* tok,
                    const char* prompt, int max_new,
                    const sample_params* sp,
                    const char* stop_sub,
                    char* out_buf, int out_max, int stream);
```

Stop condition: `next==1 || next==2` (im_start/im_end), atau
`stop_sub` ditemukan di `out_buf` (substring scan setelah setiap decode),
atau `gen_n >= max_new`, atau `pos >= 2047`. `stream=1` → `fwrite` decode
bytes ke stdout seketika (seperti CLI biasa). Mengembalikan token count,
atau `-1` jika encode/prefill gagal.

### `do_heavy()`

1. Load gguf/tok/fwd sekali (sama dengan `do_generate`).
2. Alokasi heap: `prompt` (12KB), `think_buf`/`ans_buf`/`ver_buf`/`re_buf`
   (masing-masing 8KB).
3. Empat pass dengan prompt ChatML terstruktur (lihat di bawah).
4. Gate: scan prefix `ver_buf` setelah skip whitespace; jika `REJECT` →
   RE-ANSWER sekali, tidak ada VERIFY kedua.
5. Print summary, free buffers, return 0.

### ChatML per pass (hardcoded C strings)

**THINK system:** `You are a careful reasoner. Write step-by-step notes inside <think>...</think> only. Focus on what the user asked. End with </think>.`
Assistant prefix: `<think>\n` (model diharapkan langsung menulis notes di sini).

**ANSWER system:** `Answer the user using the notes. Be direct and correct. No <think> tags.`
User body: `Question: <Q>\nNotes: <think_text>`

**VERIFY system:** `Check the draft answer against the question. Reply with exactly one line: VERIFIED: <short reason> or REJECT: <what is wrong>.`
User body: `Question: <Q>\nDraft: <ans_text>`

**RE-ANSWER system:** `The draft was rejected. Write a corrected final answer only. No tags.`
User body: `Question: <Q>\nDraft: <ans_text>\nCritique: <ver_text>`

### Limit

| Batas | Nilai |
|---|---|
| encode ids per pass | 1536 |
| prompt char | 12288 heap |
| out text per pass | 8192 |
| max_seq | 2048 (tidak berubah) |

Jika encode/prefill gagal → print error untuk pass itu, lanjut pass
berikutnya (tapi pipeline sudah rusak; faktanya abort total lebih aman,
saat ini kita `return -1` dan program berhenti).

## Acceptance

```bash
make
./smollm2 --help                              # --heavy terlihat
python3 eval/heavy_test.py                    # 4/4 PASS
python3 eval/parity.py                        # masih 5/5 PASS
./smollm2 -p "What is 2+2?" --heavy -n 48 --heavy-think-n 32 --heavy-verify-n 24 --temp 0
# stdout: THINK/ANSWER/VERIFY sections + summary line
./smollm2 -p hello -n 16 --temp 0             # unchanged, no sections
```

Smoke check manual: think phase berhenti di `</think>` atau budget, tidak
runaway. Plain `-p` tetap single-pass (tidak ada section header).

## Out of scope

- Heavy mode di TUI/web (rencana berikut, terpisah)
- Chain-of-thought specials baru / tokenizer changes
- Streaming JSON API
- Multi-model / parallel passes
- Auto-push

## Risk

| Risiko | Mitigasi |
|---|---|
| 135M tidak pernah emit `</think>` | hard budget `heavy_think_n` |
| Think overflow | cap `think_buf` 8KB; truncate notes di answer prompt |
| VERIFY tidak bilang VERIFIED/REJECT | tidak masuk REJECT → tidak ada re-answer, tidak loop |
| Lambat (3-4 prefills) | expected; summary overall tok/s |
| System prompt menurunkan kualitas | system singkat; ChatML shape sama dgn default |