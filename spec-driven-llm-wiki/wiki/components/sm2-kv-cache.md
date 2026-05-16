---
title: "sm2-kv-cache"
type: component
tags: [memory, cache, kv, paged]
last_updated: 2026-05-16
---

# sm2-kv-cache

Paged and quantized Key-Value cache management system for smollm2.c.

## Overview

KV cache stores intermediate key (K) and value (V) activations for attention computation. In inference, each new token attends to all previous tokens, so KV cache grows with sequence length.

## Architecture

```
sm2-kv-cache
  sm2_kv_pool      - global pool of KV pages
  sm2_kv_page      - individual page (16 tokens)
  sm2_kv_table     - per-sequence page mapping
  sm2_kv_dtype     - quantization mode (F16/Q8/Q4/TURBO2/MIXED)
```

## Key Structures

### KV Page

```c
#define SM2_KV_PAGE_TOKENS 16

typedef struct {
    uint32_t id;           // page identifier
    uint16_t used;         // tokens currently in use
    uint16_t refcount;     // reference count
    uint32_t next_free;    // free list link
    
    uint8_t *k_data;      // key cache data (quantized)
    uint8_t *v_data;      // value cache data (quantized)
    uint8_t *k_scale;      // key quantization scales
    uint8_t *v_scale;      // value quantization scales
} sm2_kv_page;
```

### KV Pool

```c
typedef struct {
    int n_layers;          // 30 (135M) / 32 (360M) / 24 (1.7B)
    int n_kv_heads;         // 3 (135M/360M GQA) / 32 (1.7B MHA)
    int head_dim;           // 64 (fixed for all variants)
    int page_tokens;        // 16 (default)
    sm2_kv_dtype dtype;    // quantization mode
    
    int max_pages;
    sm2_kv_page *pages;    // page array
    uint32_t *free_stack;  // free list stack
    int free_top;
} sm2_kv_pool;
```

### Per-Sequence Page Table

```c
typedef struct {
    uint64_t seq_id;       // unique sequence ID
    int seq_len;           // current sequence length
    int n_pages;           // number of pages allocated
    uint32_t page_ids[256]; // max ~4K tokens per sequence
} sm2_kv_table;
```

## Quantization Modes

| Mode | K Quant | V Quant | Memory | Quality |
|------|---------|---------|--------|---------|
| F16 | none | none | 100% | 100% |
| Q8 | per-token | per-token | 50% | ~99% |
| Q4 | per-token | per-token | 25% | ~95% |
| TURBO2 | per-channel | per-token | 12.5% | ~90% |
| MIXED | recent Q8 | recent Q8 | ~30% | ~97% |

## Memory Calculation (135M)

```
KV memory per token = n_layers x 2 x n_kv_heads x head_dim x bytes_per_element
                     = 30 x 2 x 3 x 64 x 2 (F16)
                     = 23,040 bytes/token = 22.5 KB/token

ctx 1024:  23 MB (F16), 12 MB (Q8), 6 MB (TURBO2)
ctx 2048:  46 MB (F16), 24 MB (Q8), 12 MB (TURBO2)
```

## Operations

### Allocate Page

```c
uint32_t sm2_kv_alloc_page(sm2_kv_pool *pool) {
    if (pool->free_top == 0) return 0;
    return pool->free_stack[--pool->free_top];
}
```

### Append Token

```c
int sm2_kv_append(sm2_kv_pool *pool, sm2_kv_table *table,
                 int layer, int kv_head, const float *k, const float *v) {
    int last_page_idx = table->n_pages - 1;
    sm2_kv_page *page = &pool->pages[table->page_ids[last_page_idx]];
    
    if (page->used >= pool->page_tokens) {
        uint32_t new_id = sm2_kv_alloc_page(pool);
        if (new_id == 0) return -1;
        table->page_ids[table->n_pages++] = new_id;
        page = &pool->pages[new_id];
    }
    
    sm2_kv_write_token(page, page->used, kv_head, k, v);
    page->used++;
    table->seq_len++;
    return 0;
}
```

### Read KV Block

```c
int sm2_kv_read_block(sm2_kv_pool *pool, sm2_kv_table *table,
                     int layer, int kv_head, int start, int len,
                     float *k_out, float *v_out) {
    for (int i = 0; i < len; i++) {
        int token_idx = start + i;
        int page_idx = token_idx / pool->page_tokens;
        int offset = token_idx % pool->page_tokens;
        uint32_t page_id = table->page_ids[page_idx];
        sm2_kv_page *page = &pool->pages[page_id];
        sm2_kv_read_token(page, offset, kv_head, &k_out[i], &v_out[i]);
    }
    return 0;
}
```

### Free Sequence

```c
void sm2_kv_free_seq(sm2_kv_pool *pool, sm2_kv_table *table) {
    for (int i = 0; i < table->n_pages; i++) {
        uint32_t page_id = table->page_ids[i];
        pool->pages[page_id].used = 0;
        pool->free_stack[pool->free_top++] = page_id;
    }
    table->seq_len = 0;
    table->n_pages = 0;
}
```

## Page Size Tradeoffs

| Page Size | Use Case | Pros | Cons |
|----------|----------|------|------|
| 16 | Server/GPU | Low waste, good parallelism | More allocations |
| 32 | Local 135M | Balanced | Moderate waste |
| 64 | Long context | Fewer allocations | Higher waste |

## Dependencies
- [[smollm2dl-decode-layer]] - uses KV cache for attention
- [[kv-turbo-quant]] - quantization logic

## Status
- [[spec:001]] - Phase 4 implementation
- Critical for server scalability and VPS memory efficiency