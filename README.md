# smollm2.c

Minimal C inference engine for SmolLM2-135M. Reads the GGUF model file directly from Ollama's local storage — no conversion, no training.

## Build

```bash
make
```

## Run

```bash
# Default: auto-resolve model from Ollama storage
./smollm2 -p "Hello, how are you?" -n 50

# Explicit model path
./smollm2 -m ~/.ollama/models/blobs/sha256-f535f83e... -p "Hello" -n 20

# Interactive chat
./smollm2 -i
```

## Model

Uses Ollama's `smollm2:135m` model from `~/.ollama/models/`. The GGUF blob is resolved via the manifest at `~/.ollama/models/manifests/registry.ollama.ai/library/smollm2/135m`.

Install the model via Ollama if not already present:

```bash
ollama pull smollm2:135m
```

## Studio (LoRA / attention / WebUI)

Single CLI subcommand umbrella + WebUI dashboard.

```bash
# Build packed dataset (raw / instruct / sharegpt auto-detect)
./smollm2 studio data-build --in raw.txt --out packed.bin \
    --fmt auto --model models/smollm2-135m-f16.gguf

# Hardware probe
./smollm2 studio hw

# LoRA on lm_head (multi-token CE, AdamW, watchdog)
./smollm2 studio train --data packed.bin --model models/smollm2-135m-f16.gguf \
    --mode lora --rank 4 --epochs 1 --lr 1e-3 --max-steps 50 \
    --out-dir adapters

# Merge LoRA adapter (phase 2a: base + .lora sidecar)
./smollm2 studio merge --base models/smollm2-135m-f16.gguf \
    --adapter adapters/lora_final.bin --out merged.gguf

# Sparse attention (registry — see spec/010)
./smollm2 -m models/smollm2-135m-f16.gguf -p hi --attn swa:window=256
./smollm2 studio attn-config --config layers.json --layers 30

# Gradient check (autograd sanity)
./smollm2 studio grad-check --m 4 --n 3 --k 5

# Tabbed WebUI: Infer / HW / Attn / Data / Train / Merge
./smollm2 studio web --port 8082 --model models/smollm2-135m-f16.gguf
# → open http://localhost:8082/studio
```

Tests (TDD):

```bash
make studio-test         # studio_smoke + attn_matrix + heavy + parity
python3 eval/train_smoke.py
python3 eval/attn_sparse_test.py
python3 eval/studio_web_test.py
```

Refs: `spec/008` foundation · `spec/009` train modes · `spec/010`
attn registry · `spec/011` WebUI.

## Spec & Handoff

- `spec/` — design specs numbered `XXX_<task>.md`
- `spec/handoff/` — session-to-session handoffs
- `spec/prompts/INSTRUCTIONS.md` — agent operating guide
- `BEHAVIOUR.md` — coding rules

See `spec/000_reset_and_rebuild.md` for the full rebuild plan.
