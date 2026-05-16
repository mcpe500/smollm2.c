# Pattern: Low-Memory VPS Mode

**For:** [[spec:001]], [[smollm2d-server-daemon]]  
**Type:** Deployment Pattern

## Hard Limits for 512 MB VPS

```bash
./smollm2d \
  --model smollm2-135m-q4.sm2 \
  --host 127.0.0.1 --port 7331 \
  --threads 1 \
  --ctx 1024 \
  --kv-dtype q8 \
  --low-mem \
  --max-output 128 \
  --max-parallel 1
```

## Resource Limits

| Resource | Limit | Rationale |
|----------|-------|-----------|
| Model | 135M only | Memory budget |
| Quantization | Q4/Q5 only | ~70MB vs 135MB F16 |
| Threads | 1 | Single vCPU |
| max_parallel | 1 | No concurrent requests |
| queue_size | 2 | Minimal queuing |
| ctx | 1024 default, 2048 max | Memory |
| max_output | 128 | Streaming-first |
| max_body | 256 KB | Parse limits |

## Memory Breakdown (135M Q4_K, ctx 1024, KV Q8)

```
Weights (Q4_K):     ~70 MB
KV F16:             ~23 MB
KV Q8:              ~12 MB
Scratch buffers:    ~5 MB
Tokenizer:          ~1 MB
HTTP stack:          ~2 MB
----------------------------
Total RSS:          ~90 MB (fits in 512 MB with margin)
```

## Forbidden in Low-Mem Mode

- Full prompt logging
- Response buffering (stream immediately)
- Multi-request concurrency
- GPU acceleration
- Speculative decode

## Implementation

```c
typedef struct {
    // Preallocate ALL buffers at startup
    float *scratch[NUM_SCRATCH];
    sm2_kv_pool *kv_pool;
    uint8_t *tokenizer_buf;
} sm2_low_mem_context;

// Hard-coded limits - no dynamic allocation
#define SM2_LM_MAX_CTX 2048
#define SM2_LM_MAX_OUTPUT 128
#define SM2_LM_QUEUE_SIZE 2
```

## See Also
- [[smollm2d-server-daemon]]
- [[kv-turbo-quant]]
- [[smollm2c-core-runtime]]
