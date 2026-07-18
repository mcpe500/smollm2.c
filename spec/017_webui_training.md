# 017_webui_training.md

## Prompt

Wire studio WebUI ke training: POST /studio/train langsung call `train_run`
dengan SSE (Server-Sent Events) callback per step. POST /studio/merge call
`train_merge`. GET /studio/hw surface `fullft_allowed`, `qlora_recommended`,
`max_seq_advised`, etc. Front-end tab Train/Merge/HW (vanilla JS).

## Goal

```bash
# Open browser http://localhost:8082/studio
# Tab Train: pilih mode (lora/qlora/fullft), rank, targets, dataset, klik Start
# SSE stream: data: {"step":N,"loss":X,"lr":Y,"ram_mb":Z,"epoch":E,"sample":S}
# When done: "data: {\"done\":true,\"adapter\":\"adapters/lora_final.bin\"}"

curl -N -X POST http://localhost:8082/studio/train \
    -H "Content-Type: application/json" \
    -d '{"mode":"lora","rank":8,"targets":"q,v","data":"packed.bin", \
         "epochs":1,"model":"models/smollm2-135m-f16.gguf"}'
# expect: SSE stream of step/loss lines
```

## Why

Studio WebUI phase 4a-4c sudah ada GET /studio/hw, /studio/attn, /studio/data,
POST /studio/{data,train,merge}. Tapi POST train/merge pakai `popen` ke CLI
(Phase A fix sudah direct call via `run_capture`). Tetap tidak streaming —
whole-output capture, tidak ada progress feedback ke browser.

SSE adalah standar HTML5 untuk server-push. Browser native support via
`EventSource`. Tidak butuh WebSocket library. Cocok untuk 1 connection per
training run.

## Codebase Context

- `src/web.c` — extend `handle_studio_train` (SSE), `handle_studio_merge` (direct call), extend `handle_studio_hw` (JSON fields sudah ada via hw_json)
- `src/train.c` — `train_run` extend dengan `on_step` callback
- `src/train.h` — `typedef void (*step_cb)(int step, int epoch, int sample, float loss, float lr, long ram_mb, void* user)`
- Front-end embedded HTML di `web.c` (`kStudioHTML`) — extend dengan tab Train
- `eval/ui.php` atau `eval/studio_web_test.py` — TDD SSE parsing test

## Logical Change

### SSE response format

`handle_studio_train`:
1. Parse JSON body: `mode`, `rank`, `targets`, `data`, `epochs`, `lr`, `seq`, `model`, `max_steps`
2. Validate paths via `path_safe`
3. Build `train_params` struct
4. Send HTTP 200 + headers:
   ```
   Content-Type: text/event-stream
   Cache-Control: no-cache
   Connection: close
   Transfer-Encoding: chunked
   ```
5. Direct call `train_run` with `on_step` callback
6. Callback writes chunked SSE: `data: {step,loss,lr,ram_mb,epoch,sample}\n\n`
7. On done: `data: {"done":true,"adapter":"path"}\n\n` + `data: [DONE]\n\n` + zero-length chunk

### train_run callback

`train.h`:
```c
typedef void (*step_cb)(const step_info* info, void* user);

typedef struct {
    int step;
    int epoch;
    int sample;
    float loss;
    float lr;
    long ram_mb;
    const char* adapter_path;  /* null until final */
} step_info;

int train_run(t, packed_path, p, out_dir, step_cb cb, void* user);
```

`train.c`: di loop training, panggil `cb` setelah `train_step`. Final call
dengan `adapter_path` filled.

### /studio/merge direct

`handle_studio_merge`:
1. Parse JSON: `base`, `adapter`, `out`
2. Validate paths via `path_safe`
3. Direct call `train_merge(base, adapter, out)` — NOT `run_capture`
4. Return JSON: `{"ok":true,"out":"path","size":N}` atau `{"ok":false,"error":"..."}`

Sudah partial: Phase A fix gunakan `run_capture` (fork+execvp). Phase F
pindahkan ke direct call (no fork overhead, in-process).

### /studio/hw JSON

Sudah ada via `hw_json` (Phase A0). Pastikan field lengkap:
`mem_total_mb, mem_available_mb, vm_rss_mb, cpu_cores, cpu_neon, cpu_dotprod,
cpu_sve, cpu_atomics, cpu_model, cpu_freq_max_mhz, max_seq_advised,
max_batch_advised, max_rank_advised, fullft_allowed, qlora_recommended,
lora_min_mb, qlora_min_mb, fullft_min_mb, emergency_min_mb`.

### Front-end tab Train

`kStudioHTML` extend dengan tab:
```html
<div class="tab" data-tab="train">Train</div>
<div class="tab" data-tab="merge">Merge</div>
<div class="tab" data-tab="hw">HW</div>
```

Vanilla JS, no framework:
```js
function startTrain() {
    const params = {
        mode: sel('.train-mode').value,
        rank: num('.train-rank').value,
        targets: sel('.train-targets').value,
        data: sel('.train-data').value,
        epochs: num('.train-epochs').value,
        model: sel('.train-model').value,
    };
    const es = new EventSource('/studio/train?' + new URLSearchParams(params));
    es.onmessage = (e) => {
        const d = JSON.parse(e.data);
        if (d.done) { es.close(); alert('Done: ' + d.adapter); }
        else chart.addPoint(d.step, d.loss);
    };
}
```

Tidak pakai framework. CSS inline. Compact (~5KB).

## Code Change

- `src/web.c`:
  - `handle_studio_train` rewrite: SSE response, direct call
  - `handle_studio_merge` rewrite: direct call, no fork
  - `handle_studio_hw`: verify fields, add advisory section to JSON
  - `kStudioHTML` extend: tab Train/Merge/HW, SSE consumer JS
- `src/train.{c,h}`: `train_run` signature add `step_cb`, `step_info` struct
- `src/studio.c`: `cmd_train` (CLI) pass NULL cb (no streaming)
- `eval/studio_web_test.py` — extend: SSE parsing test (read /studio/train, verify step/loss lines)

## Why This Change

- SSE (bukan WebSocket): HTML5 native, no library. Browser `EventSource` API.
  Cocok untuk 1-way push (training progress).
- Direct call (bukan fork): `train_run` in-process — no fork overhead,
  callback writes ke socket fd langsung. ~100× faster per-step message.
- Vanilla JS (bukan React/Vue): embed HTML string in C, no build step,
  minimal footprint (~5KB).
- Callback signature (bukan global): `step_info` struct extensible tanpa
  break ABI.

## Logic / Pseudocode

```
handle_studio_train(fd, body):
    params = parse_json_train_params(body)
    validate_paths(params.data, params.model)
    send_sse_headers(fd)
    train_state* t = train_create(...)
    train_run(t, params.data, params, params.out_dir,
              on_step_cb, (void*)(intptr_t)fd)
    /* on_step_cb writes SSE chunk to fd */

on_step_cb(info, user):
    int fd = (int)(intptr_t)user;
    char buf[256];
    if (info->adapter_path) {
        snprintf(buf, sizeof(buf), "data: {\"done\":true,\"adapter\":\"%s\"}\n\n",
                 info->adapter_path);
    } else {
        snprintf(buf, sizeof(buf),
            "data: {\"step\":%d,\"epoch\":%d,\"sample\":%d,\"loss\":%.4f,\"lr\":%.2e,\"ram_mb\":%ld}\n\n",
            info->step, info->epoch, info->sample, info->loss, info->lr, info->ram_mb);
    }
    send_chunk(fd, buf, strlen(buf));
```

## Test Simulation & Tracing

### SSE stream
```
POST /studio/train {"mode":"lora","rank":4,"max_steps":3,...}
expect stream:
    data: {"step":1,"epoch":0,"sample":0,"loss":2.34,"lr":1e-4,"ram_mb":150}
    data: {"step":2,...}
    data: {"step":3,...}
    data: {"done":true,"adapter":"adapters/lora_final.bin"}
    data: [DONE]
```

### Direct merge
```
POST /studio/merge {"base":"m.gguf","adapter":"a.bin","out":"o.gguf"}
expect: {"ok":true,"out":"o.gguf","size":270885952}
no fork/execvp
```

## Manual Testing Plan

```bash
make

# Start server
./smollm2 studio web --port 8082 --model models/smollm2-135m-f16.gguf &

# Browser
open http://localhost:8082/studio
# Click tab Train, fill form, click Start
# Expect: SSE loss chart updates per step

# CLI test
curl -N -X POST http://localhost:8082/studio/train \
    -H "Content-Type: application/json" \
    -d '{"mode":"lora","rank":4,"targets":"lm_head","max_steps":3, \
         "data":"/tmp/packed.bin","model":"models/smollm2-135m-f16.gguf"}'
# Expect: SSE lines stream

# Test suite
python3 eval/studio_web_test.py  # extend with SSE test
```

## Status

- [x] Spec written
- [ ] Implementation
- [ ] Verified
- [ ] Handoff written
