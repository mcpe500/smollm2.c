# SmolLM2 C Inference Engine

High-performance C implementation of SmolLM2-135M for mobile and edge devices.

## Features

- **Multiple interfaces**: CLI, TUI (text UI), and WebUI
- **Optimized for mobile**: Precomputed F32 weights, loop unrolling, LTO
- **No dependencies**: Pure C99, compiles with gcc/clang
- **Memory efficient**: Preallocated buffers, no runtime malloc in decode path

## Build

```bash
# Clean build
make clean && make

# Or with LTO for extra optimization
make LTO=1
```

## Quick Start

```bash
# Interactive CLI chat
./smollm2-cli -m smollm2-135m.sm2

# Single prompt
./smollm2-cli -m smollm2-135m.sm2 -p "Hello, how are you?"

# Generate 50 tokens
./smollm2-cli -m smollm2-135m.sm2 -p "What is 2+2?" -n 50
```

## Running Modes

### CLI Mode (Interactive Chat)

```bash
./smollm2-cli -m smollm2-135m.sm2 --mode cli
```

Features:
- Interactive text-based chat
- Type `quit` or `exit` to stop
- Type `clear` to clear chat history

### TUI Mode (Terminal UI)

```bash
./smollm2-cli -m smollm2-135m.sm2 --mode tui
```

Features:
- Modern terminal UI with colors
- Real-time token generation display
- Token speed indicator (tok/s)
- Persistent chat history within session

### WebUI Mode (HTTP Server)

```bash
./smollm2-cli -m smollm2-135m.sm2 --mode web
```

Then open: http://127.0.0.1:7331

Features:
- Web interface accessible from any browser
- Chat with the model in real-time

### WebUI with Custom Port

```bash
./smollm2-cli -m smollm2-135m.sm2 --mode web --port 8080
```

## Generation Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model` | Model file (.sm2) | required |
| `-p, --prompt` | Input prompt | "Hello" |
| `-n, --max-output` | Max tokens to generate | 50 |
| `-t, --temp` | Temperature (0=greedy, 0.7=balanced, 1.0=creative) | 0.7 |
| `-q, --top-p` | Top-p nucleus sampling | 90 |
| `-k, --top-k` | Top-k sampling (0=disabled) | 0 |
| `-r, --rep-penalty` | Repetition penalty (1.0=off, 1.2=default) | 1.3 |

### Example: Creative Mode

```bash
./smollm2-cli -m smollm2-135m.sm2 -p "Write a story" -t 1.0 -n 100
```

### Example: Precise/Deterministic

```bash
./smollm2-cli -m smollm2-135m.sm2 -p "What is the capital of France?" -t 0.0 -n 20
```

## Model File

Download the model file `smollm2-135m.sm2` from releases or create one from HuggingFace weights.

Required model format: `.sm2` binary file with:
- 256-byte header
- Tokenizer data (vocab + merges)
- F16/F32 weight tensors

## Performance

| Device | Speed |
|--------|-------|
| Desktop (x86_64) | ~50 tok/s |
| Mobile (ARM64) | ~5 tok/s |
| Low-end device | ~2 tok/s |

### Optimization Tips

1. **Use LTO**: `make LTO=1`
2. **Close other apps**: Free up memory/CPU
3. **Use lower temperature**: Reduces sampling overhead

## Architecture

```
smollm2-cli
├── src/
│   ├── smollm2.c          # Main entry point
│   ├── sm2_context.c      # Inference loop (hot path)
│   ├── sm2_model.c        # Model loading
│   ├── sm2_rope.c         # RoPE position encoding
│   ├── chat_cli.c         # CLI chat mode
│   ├── chat_tui.c         # TUI mode
│   └── chat_web.c         # WebUI mode
├── include/
│   └── smollm2.h          # API definitions
└── experiments/
    └── experiment_log.md # Performance experiments
```

## Troubleshooting

### "Cannot open model.sm2"
- Check file path is correct
- Ensure file has read permissions

### Slow generation
- Normal on mobile (~5 tok/s)
- Check device thermal throttling
- Try closing background apps

### Garbled output
- Model file may be corrupted
- Try re-downloading the model

## Development

```bash
# Build with debug symbols
make DEBUG=1

# Check syntax only
make check

# Full clean rebuild
make clean && make
```

## License

MIT