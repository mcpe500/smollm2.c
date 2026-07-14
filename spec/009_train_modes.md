# 009_train_modes.md

## Prompt

Tambah training (LoRA / QLoRA / FullFT) ke studio phase 2 dengan
hardware-aware: refuse training yang OOM, watch MemAvailable, save
checkpoint emergency, support batching + streaming dataset.

## Goal

```bash
./smollm2 studio train --data packed.bin --mode lora --rank 8 \
    --epochs 1 --lr 1e-4 --seq 256 --batch 1
# expect: loss turun ~epoch, no OOM, adapter saved ke adapters/lora_step_NNN.bin

./smollm2 studio hw                        # print hw_caps
./smollm2 studio train --mode fullft       # auto-reject kalau <2.5GB
./smollm2 studio merge --base model.gguf --adapter lora.bin --out merged.gguf
./smollm2 -m merged.gguf -p "hello"        # argmax 19556
```

## Why

SmolLM2-135M lemah untuk task generic. LoRA finetuning bisa menyetir
perilaku spesifik (style, format) tanpa menambah parameter besar. QLoRA
buka opsi training model lebih besar (3B+) pada hardware kecil. FullFT
jarang feasible di Termux, tapi harus tetap bisa di-trigger dengan safety.

## Codebase Context

- File baru:
  - `src/train.c` + `train.h` — trainer (LoRA, QLoRA, FullFT dispatch)
  - `src/hw_probe.c` + `hw_probe.h` — `/proc/meminfo` reader
- File modifikasi:
  - `src/studio.c` — subcommand `train`, `merge`, `hw`
  - `src/studio.h` — exports (sudah minimal)
  - `src/backward.c` — extend untuk LoRA backward (analytical + checkpointing)
  - `Makefile` — tambah src/train.c, src/hw_probe.c
- Tidak menyentuh: `forward.c` (inference tetap utuh)

## Logical Change

### `src/hw_probe.h`

```c
typedef struct {
    long mem_total_kb;
    long mem_avail_kb;        /* /proc/meminfo MemAvailable */
    int  max_seq_advised;     /* heuristic, lihat tabel */
    int  max_batch_advised;
    int  fullft_allowed;      /* 1 iff mem_avail >= 2.5GB */
    int  qlora_recommended;
} hw_caps;

void hw_probe(hw_caps* c);
void hw_print(const hw_caps* c);
```

Heuristic (untuk SmolLM2-135M):
| Mode | peak RAM | min MemAvailable |
|---|---|---|
| LoRA r=8 | ~500MB | 800MB |
| LoRA r=16 | ~520MB | 850MB |
| QLoRA r=8 | ~600MB | 900MB |
| QLoRA r=16 | ~620MB | 950MB |
| FullFT seq=128 | ~1.8GB | 2.5GB |
| FullFT seq=64 | ~1.4GB | 2.0GB |

### `src/train.h`

```c
typedef enum { TRAIN_LORA = 0, TRAIN_QLORA, TRAIN_FULLFT } train_mode;

typedef struct {
    train_mode mode;
    int  lora_rank;          /* 0 for full FT */
    int  lora_alpha;         /* scaling, default 2*rank */
    int  target_mask;        /* bitmask: bit(Q)=1, K=2, V=4, O=8, GATE=16, UP=32, DOWN=64 */
    int  seq_max;
    int  batch;              /* gradient accumulation */
    float lr;
    int  epochs;
    int  checkpoint_every;   /* save adapter setiap N step */
    int  max_steps;          /* 0 = unlimited */
    int  seed;               /* RNG seed (default 42) */
} train_params;

typedef struct train_state train_state;

train_state* train_create(const forward_ctx* f, const train_params* p);
void         train_free  (train_state* t);

/* Single step: forward + backward + optimizer. Returns loss. */
float train_step(train_state* t, const int* tokens, int n_tokens);

/* Save/load adapter. Format: magic + per-tensor F32 blob (small). */
int  train_save (const train_state* t, const char* path);
int  train_load (train_state* t, const char* path);

/* Main loop driver. Watchdog: cek mem_avail setiap step; kalau <100MB,
   save emergency + exit 0. */
int  train_run  (train_state* t, const dataset* d, const train_params* p,
                 const char* out_dir);
```

### LoRA adapter storage

Per-target (Q/K/V/O/Gate/Up/Down per layer): A: dim × rank F32, B: rank × dim F32.
Total ~30 layers × 7 targets × (576+576)*8 floats × 4 bytes = ~10MB.

```c
typedef struct {
    float* A;   /* [in_dim, rank] */
    float* B;   /* [rank, out_dim] */
    float* gA;  /* grad */
    float* gB;  /* grad */
    float* mA;  /* Adam moment 1 */
    float* vA;  /* Adam moment 2 */
    float* mB, *vB;
} lora_pair;
```

### `studio` subcommands

```c
studio train --data X --mode lora --rank N --epochs N --lr F --seq N --batch N
studio merge  --base X.gguf --adapter Y.bin --out Z.gguf
studio hw
```

## Crash safety

1. `hw_probe` di awal; tolak mode yang exceed budget.
2. Watchdog via `getrusage` atau `/proc/self/status` setiap step.
3. MemAvailable < 100MB → save emergency ke `adapters/emergency_<step>.bin`,
   print ringkasan, exit 0.
4. Gradient checkpointing FullFT: recompute forward di backward pass.

## Streaming dataset

`train_run` consume `dataset*` yang berisi `sample_idx[]` + packed.bin.
Iterasi per-sample, batch sesuai `--batch`. Untuk dataset >100MB, tidak
hold seluruh activations — checkpointed backward.

## Acceptance

```bash
make
./smollm2 studio hw
./smollm2 studio train --data packed.bin --mode lora --rank 8 \
    --epochs 1 --lr 1e-4 --seq 128 --batch 1 --max-steps 50
# expect: loss turun dari ~12 ke < 11, adapter saved
./smollm2 studio train --mode fullft --seq 256
# expect: refused (RAM <2.5GB) atau proceed + checkpoint
python3 eval/train_smoke.py    # TDD red → green
python3 eval/studio_smoke.py   # 5/5 still pass
```

## TDD harness

`eval/train_smoke.py`:

1. `test_hw_probe_reports_caps` — `studio hw` exit 0, output mengandung
   `mem_avail=` dan `fullft_allowed=`.
2. `test_lora_train_step_loss_decreases` — bikin packed dataset kecil
   (50 token), jalankan 10 step, assert `loss[-1] < loss[0]`.
3. `test_lora_save_load_roundtrip` — train 5 step, save, train 5 step
   lagi, save, compare two adapter files structure sama (size > 0,
   magic match).
4. `test_fullft_refused_low_mem` — run dengan fake `MemAvailable` < 2.5GB
   (test dengan `--simulate-mem-kb` flag), expect exit non-zero dan stderr
   mention "refused".
5. `test_merge_lora_roundtrip` — train 1 step, save, merge ke copy GGUF,
   load copy, argmax hello tetap 19556.

## Limit

| Batas | Nilai |
|---|---|
| LoRA rank | 1-32 |
| seq_max | ≤ 1024 (cap dari forward.c max_seq=2047) |
| batch | 1-8 |
| Adapter targets | Q, K, V, O, Gate, Up, Down (7 mask) |
| Adam β1, β2 | 0.9, 0.999 (hardcoded) |

## Out of scope (deferred)

- DPO / RLHF / preference optimization
- Multi-GPU / distributed
- Gradient accumulation across data shards (in-process only)
- Quantization-aware training di luar Q4 base (no INT4 grad)
- Auto-LR scheduler

## Risk

| Risk | Mitigasi |
|---|---|
| OOM crash (SIGSEGV) | hw_probe + watchdog + emergency checkpoint |
| NaN loss dari LR terlalu besar | gradient clip + early stop jika loss > 100 |
| Adapter file corrupt | magic header + checksum |
| Backward pass bug → wrong grad | extend grad check harness ke semua ops |
| MemAvailable unreliable di Termux | fallback ke VmRSS self + buffer |