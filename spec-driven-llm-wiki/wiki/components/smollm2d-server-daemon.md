---
title: "smollm2d-server-daemon"
type: component
tags: [server, http, api, daemon]
last_updated: 2026-05-16
---

# smollm2d-server-daemon

HTTP server daemon for smollm2.c with OpenAI-compatible API and continuous batching.

## Parent
- [[smollm2c-core-runtime]]

## Features
- OpenAI-compatible /v1/chat/completions
- SSE (Server-Sent Events) streaming
- Continuous batching scheduler
- Prometheus metrics
- Health checks
- Low-memory mode for VPS 512 MB

## API Endpoints

```http
POST /v1/chat/completions    - Chat completions (streaming supported)
POST /v1/completions          - Text completions
GET  /health                  - Health check
GET  /metrics                 - Prometheus metrics
```

## Request States

```c
typedef enum {
    SM2_REQ_WAITING,   // Queued
    SM2_REQ_PREFILL,   // Processing prompt
    SM2_REQ_DECODE,    // Generating tokens
    SM2_REQ_VERIFY,    // Speculative verification
    SM2_REQ_DONE,      // Complete
    SM2_REQ_CANCELLED  // Client disconnect
} sm2_req_state;
```

## Continuous Batching Scheduler

```
while running:
  accept new requests
  batch prefill if possible
  move active -> decode pool
  run one decode step for batch
  sample tokens
  stream outputs via SSE
  release finished KV pages
```

## Example: Batch Decode Step

```
Request A pos 100
Request B pos 700
Request C pos 20
Request D pos 1400

All active in decode pool:
  1. Forward all requests (share model weights)
  2. Sample tokens
  3. Stream via SSE
  4. Check done/cancelled
```

## Low-Memory Mode (VPS 512 MB)

```bash
./smollm2d \
  --model smollm2-135m-q4.sm2 \
  --host 127.0.0.1 --port 7331 \
  --threads 1 --ctx 1024 \
  --kv-dtype q8 --low-mem \
  --max-output 128 --max-parallel 1
```

Hard limits:
- model: 135M only
- quant: Q4/Q5 only
- threads: 1
- max_parallel_requests: 1
- queue_size: 2
- ctx max: 2048

## Health Response

```json
{
  "ok": true,
  "model": "smollm2-135m",
  "quant": "q4_k",
  "ctx": 1024,
  "kv": "q8",
  "backend": "portable-c",
  "rss_mb": 280
}
```

## SSE Streaming Format

```text
data: {"token":"Halo","id":123}
data: {"token":" dunia","id":456}
data: [DONE]
```

## Files

```
src/server/
  smollm2d.c          - main daemon
  sm2_http.c          - HTTP handling
  sm2_sse.c           - SSE streaming
  sm2_scheduler.c     - request scheduler
  sm2_metrics.c        - Prometheus metrics
  sm2_agent.c         - sm2-agent tiny proxy
```

## Dependencies
- [[smollm2dl-decode-layer]] - decode engine
- [[sm2-kv-cache]] - KV page management
- [[smollm2c-tokenizer]] - tokenizer

## Status
- [[spec:001]] - Phase 6 implementation
- Critical for SaaS deployment
- VPS 512 MB mode supported
