# ADR-003: Paged KV Cache

**Status:** Accepted  
**Date:** 2026-05-16  
**Spec:** [[spec:001]]

## Context

Server serves multiple requests with varying lengths. Contiguous KV cache wastes memory and causes fragmentation.

## Decision

Implement paged KV cache with fixed page size (16 tokens/page default).

## KV Page Structure
```c
#define SM2_KV_PAGE_TOKENS 16

typedef struct {
    uint32_t id;
    uint16_t used;      // tokens in use
    uint16_t refcount;  // for refcounting
    uint32_t next_free;
    
    uint8_t *k_data;    // quantized K
    uint8_t *v_data;    // quantized V
    uint8_t *k_scale;
    uint8_t *v_scale;
} sm2_kv_page;
```

## KV Pool
```c
typedef struct {
    int n_layers, n_kv_heads, head_dim;
    int page_tokens;
    sm2_kv_dtype dtype;
    
    int max_pages;
    sm2_kv_page *pages;
    uint32_t *free_stack;
    int free_top;
} sm2_kv_pool;
```

## Per-Request Page Table
```c
typedef struct {
    uint64_t seq_id;
    int seq_len;
    int n_pages;
    uint32_t page_ids[MAX_SEQ_PAGES];
} sm2_kv_table;
```

## Page Size Tradeoffs
| Page Size | Use Case | Waste |
|----------|----------|-------|
| 16 | Server/GPU | Low |
| 32 | Local 135M | Medium |
| 64 | Long context | High |

## Consequences

**Positive:**
- Memory efficient for varied request lengths
- No fragmentation
- Prefix cache possible

**Negative:**
- More complex allocation
- Indirect access (slower than contiguous)
- Page table overhead