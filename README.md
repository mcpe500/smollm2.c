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

## Spec & Handoff

- `spec/` — design specs numbered `XXX_<task>.md`
- `spec/handoff/` — session-to-session handoffs
- `spec/prompts/INSTRUCTIONS.md` — agent operating guide
- `BEHAVIOUR.md` — coding rules

See `spec/000_reset_and_rebuild.md` for the full rebuild plan.
