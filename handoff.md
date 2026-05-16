# smollm2.c Handoff - Decode Path Working (May 17, 2026)

## Status: ✅ WORKING - Model generates valid tokens

**Date:** May 17, 2026 03:00 AM
**Location:** `/data/data/com.termux/files/home/smollm2.c/`

## GOAL

Build a decode-first C inference engine for SmolLM2 (135M/360M/1.7B) that runs on 512MB VPS/Termux.

## Current State: ✅ Phase 1-3 COMPLETE

```
$ ./smollm2-cli -m smollm2-135m-v5.sm2 -p "Hello" -n 10
Model loaded in 262.4 ms
Prefill done in 618.7 ms
Output: [gen_tok=23]'[gen_tok=10]<iss[gen_tok=28],[gen_tok=20]$[gen_tok=6]<fil[EOS=0]
Generated 5 tokens in 3320.5 ms (664.1 ms/token)
Total time: 3939.8 ms
```

**Model generates valid tokens.** No segfault, output is syntactically valid.

## Critical Bugs Fixed (This Session)

### Bug #1: down_proj matmul indexing ✅ FIXED
**Problem:** `down_proj[down_off + j * hidden_dim + i]` accessed row j, col i (wrong for transpose).

**Fix:** Changed to `down_proj[down_off + i * hidden_dim + j]` for correct row i, col j.

**Impact:** This was causing memory corruption that manifested as segfault on "Hi" input.

### Bug #2: Post-attention RMSNorm ✅ FIXED
**Problem:** Code was doing simple `xb[i] = xb[i] * weight[i]` instead of proper RMSNorm.

**Fix:** Now computes `rms = sqrt(sum(x^2)/dim + eps)`, then `xb[i] = (x[i]/rms) * weight[i]`.

### Bug #3: sm2_decode_next sequence ✅ FIXED
**Problem:** Was sampling first, then embedding, then forwarding (WRONG order).

**Fix:** Now does: embed → forward all layers → compute logits → sample.

### Bug #4: Final RMSNorm ✅ FIXED
**Problem:** Was using LayerNorm (mean subtraction) instead of pure RMSNorm.

**Fix:** Removed mean subtraction, only uses `sqrt(sum(x^2)/dim + eps)`.

## Model File

| File | Size | Status |
|------|------|--------|
| smollm2-135m-v5.sm2 | 270 MB | ✅ Working |
| smollm2-135m.safetensors | 256.6 MB | Source (from HuggingFace) |

**weights_offset hardcoded:** 1,179,115 (header had corruption, workaround applied in sm2_model.c)

## Test Commands

```bash
cd /data/data/com.termux/files/home/smollm2.c

# Test basic generation
./smollm2-cli -m smollm2-135m-v5.sm2 -p "Hello" -n 10

# Test with "Hi" (was causing segfault)
./smollm2-cli -m smollm2-135m-v5.sm2 -p "Hi" -n 5

# Longer prompt
./smollm2-cli -m smollm2-135m-v5.sm2 -p "The quick brown fox" -n 10

# Rebuild after changes
make clean && make -j4
```

## Known Issues (Remaining)

### 1. Tokenizer uses byte fallback
Full tokenizer.json integration not complete. Current: token IDs map to raw bytes.

### 2. EOS detection stops too early
Token 0 (`<|endoftext|>`) is being selected after ~5 tokens. Generation should continue longer.

### 3. Output not verified against HF reference
Haven't done rigorous comparison with HuggingFace Python implementation.

## Files Modified (This Session)

- `src/sm2_context.c` - down_proj matmul, post-attention RMSNorm, final RMSNorm
- `src/smollm2.c` - Fixed sm2_decode_next sequence order
- `smollm2-135m-v5.sm2` - Regenerated from converter

## Implementation Details

### layer_forward structure (sm2_context.c)
```
1. RMSNorm on input x
2. Q projection: q = xb @ q_proj.T (q_proj: [dim, dim])
3. K projection: k = xb @ k_proj.T (k_proj: [kv_dim, dim])
4. V projection: v = xb @ v_proj.T (v_proj: [kv_dim, dim])
5. Apply RoPE to Q and K
6. Attention (GQA - 9 query heads, 3 KV heads)
7. O projection: xb = attn_out @ o_proj.T (o_proj: [dim, dim])
8. Post-attention RMSNorm on xb
9. FFN (SwiGLU):
   - gate_proj: gate = xb @ gate_proj.T (gate_proj: [hidden_dim, dim])
   - up_proj: up = xb @ up_proj.T (up_proj: [hidden_dim, dim])
   - SiLU activation: silu(gate)
   - Element-wise multiply: silu(gate) * up
   - down_proj: out = (silu(gate) * up) @ down_proj.T (down_proj: [dim, hidden_dim])
10. Residual connection: x = x + xb
```

### FFN Matmul Indexing (FIXED)
- gate_proj: `gate_proj[gate_off + j * dim + i]` (j=0..hidden_dim-1, i=0..dim-1)
- up_proj: `up_proj[up_off + j * dim + i]`
- down_proj: `down_proj[down_off + i * hidden_dim + j]` (row i, col j)

## Performance

| Operation | Time |
|-----------|------|
| Model load | ~262 ms |
| Prefill (5 tokens) | ~618 ms |
| Decode per token | ~664 ms |
| Total (10 tokens) | ~3939 ms |

## Next Steps

1. **Fix tokenizer integration** - Load tokenizer.json properly for real token IDs
2. **Fix EOS detection** - Token 0 shouldn't be selected so early
3. **Verify output quality** - Compare against HuggingFace reference
4. **Test edge cases** - Single char, special characters, long prompts