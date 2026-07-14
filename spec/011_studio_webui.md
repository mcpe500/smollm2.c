# 011 — Studio WebUI

## Goal

Single-binary dashboard for Infer / Train / Data / Attn / Merge / HW.
Vanilla JS, no framework. Extends existing `src/web.c`.

## Routes

| Method | Path | Body / Query | Response |
|---|---|---|---|
| GET | `/` | — | chat UI (existing) |
| GET | `/studio` | — | studio SPA |
| GET | `/studio/hw` | — | JSON hw_caps |
| GET | `/studio/attn` | — | JSON registry dump |
| POST | `/generate` | `{prompt,n}` | SSE tokens (existing) |
| POST | `/studio/infer` | `{prompt,n}` | SSE tokens (alias) |
| POST | `/studio/data` | `{text,fmt,out}` | `{ok,path,n_samples}` |
| POST | `/studio/train` | `{data,mode,rank,epochs,lr,max_steps}` | SSE `step/loss` |
| POST | `/studio/merge` | `{base,adapter,out}` | `{ok,path,sidecar}` |

## CLI

```
./smollm2 studio web [--port 8082] [--model <gguf>]
```

Also still works: `./smollm2 --web --port 8080`.

## Constraints

- No new deps.
- Train refuses FullFT when hw_probe.fullft_allowed=0.
- Emergency mem watchdog still applies (train.c).
- Single-threaded accept loop; one request at a time.

## Tests

`eval/studio_web_test.py`:
1. Start server bg, GET /studio → 200 + "Studio"
2. GET /studio/hw → JSON with mem_avail
3. GET /studio/attn → dense/swa listed
4. POST /studio/data with tiny raw → ok
5. Kill server cleanly
