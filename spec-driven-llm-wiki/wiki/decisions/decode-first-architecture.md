# ADR-001: Decode-First Architecture

**Status:** Accepted  
**Date:** 2026-05-16  
**Spec:** [[spec:001]]

## Context

We need to build smollm2.c inference engine. Two approaches:

1. **Forward-pass-first:** Build generic forward pass, decode is just repeated forward
2. **Decode-first:** Build specialized decode path optimized for single-token generation

## Decision

**Decode-first architecture.** smollm2dl as separate decode layer with dedicated optimization.

## Rationale

1. **Memory allocation:** Generic forward pass allocates per-call. decode_next called thousands of times. malloc in hot path kills performance.

2. **Flash Decode:** Stanford Flash-Decoding shows 8x speedup by splitting KV length into chunks. Requires specialized path, not generic attention.

3. **Paged KV:** Request lengths vary. Contiguous KV cache wastes memory. Paged KV needs decode-specific read path.

4. **Streaming:** Tokens must stream immediately. Can't buffer full output. Decode path handles this natively.

## Consequences

**Positive:**
- <50ms/token achievable on 135M
- VPS 512 MB mode possible
- Server scales with batching

**Negative:**
- Two attention implementations (prefill/decode)
- More complex codebase
- Harder to maintain

## References
- [[smollm2dl-decode-layer]]
- [[sm2dl-flash-decode]]
- Stanford CRFM Flash-Decoding