# Wiki Index

## Overview
- [Overview](overview.md) - living project overview

## Components
- [smollm2c-core-runtime](components/smollm2c-core-runtime.md) - Core C runtime
- [smollm2dl-decode-layer](components/smollm2dl-decode-layer.md) - Decode optimization layer
- [smollm2d-server-daemon](components/smollm2d-server-daemon.md) - HTTP server daemon
- [sm2-kv-cache](components/sm2-kv-cache.md) - Paged KV cache management
- [sm2-file-format](components/sm2-file-format.md) - Custom .sm2 binary format
- [sm2-backend-ref](components/sm2-backend-ref.md) - Portable matmul backend
- [smollm2c-tokenizer](components/smollm2c-tokenizer.md) - HF tokenizer wrapper

## Decisions
- [ADR-001: Decode-First Architecture](decisions/decode-first-architecture.md)
- [ADR-002: .sm2 File Format](decisions/sm2-file-format.md)
- [ADR-003: Paged KV Cache](decisions/paged-kv.md)
- [ADR-004: KV Turbo Quantization](decisions/kv-turbo-quant.md)
- [ADR-005: DFlash Integration](decisions/dflash-integration.md)
- [ADR-006: Weight Quantization](decisions/weight-quantization.md)

## Patterns
- [GQA Attention](patterns/gqa-attention.md) - Grouped Query Attention pattern
- [Low-Memory VPS Mode](patterns/low-memory-mode.md) - 512 MB VPS deployment
- [Speculative Decoding](patterns/speculative-decoding.md) - Draft-verifier pattern

## Syntheses
- [Overview](overview.md) - project overview

## Specs
- [SPEC 001: smollm2c-master-blueprint](../spec/docs/001.smollm2c-master-blueprint.md)
