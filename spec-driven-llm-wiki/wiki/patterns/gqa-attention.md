# Pattern: Grouped Query Attention (GQA)

**For:** [[spec:001]], [[smollm2dl-decode-layer]]  
**Type:** Attention Pattern

## GQA in SmolLM2

Grouped Query Attention reduces KV heads vs Query heads. Each query head group shares one KV head.

## Variants

| Model | Q Heads | KV Heads | Group Size | Type |
|-------|---------|----------|------------|------|
| 135M | 9 | 3 | 3 | GQA |
| 360M | 15 | 5 | 3 | GQA |
| 1.7B | 32 | 32 | 1 | MHA (full) |

## Formula

```c
// Group mapping: which KV head for each Q head
kv_head = q_head / group_size;
// For 135M: q_head 0,1,2 -> kv_head 0; q_head 3,4,5 -> kv_head 1; etc.

// Attention: each Q head attends to its KV head
for each q_head in 0..n_heads-1:
    kv_idx = q_head / group_size;  // same kv_idx for group
    attn_score = dot(q[q_head], k[kv_idx]) / sqrt(head_dim);
```

## Implementation Pattern

```c
// sm2_forward_qkv: compute Q, K, V for one token
void sm2_forward_qkv(float *q, float *k, float *v,
                     const float *x, const sm2_layer *layer) {
    // Q: dim -> dim (all n_heads)
    sm2_matmul(q, x, layer->wq, dim, dim);
    
    // K: dim -> dim_kv (less output for GQA)
    sm2_matmul(k, x, layer->wk, dim_kv, dim);
    
    // V: dim -> dim_kv
    sm2_matmul(v, x, layer->wv, dim_kv, dim);
}

// Flash decode attention loop
void sm2_flash_attention(float *out, const float *q,
                        const float *k_cache, const float *v_cache,
                        int n_heads, int n_kv_heads, int group_size) {
    for int qh = 0; qh < n_heads; qh++:
        int kvh = qh / group_size;  // GQA: share KV
        // compute attention for qh using kvh
        ...
}
```

## Memory Savings

| Model | MHA KV Size | GQA KV Size | Savings |
|-------|-------------|-------------|---------|
| 135M | 9×64×layers | 3×64×layers | 67% |
| 360M | 15×64×layers | 5×64×layers | 67% |

## Edge Cases

1. **1.7B:** group_size = 1, so kvh = qh (no sharing, MHA equivalent)
2. **Head dim fixed at 64** for all variants - simplifies SIMD
3. **Preallocate KV cache** sized for n_kv_heads, not n_heads

## See Also
- [[smollm2dl-decode-layer]]
- [[paged-kv]]
- [[kv-turbo-quant]]
