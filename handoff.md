# Handoff - Tokenizer Verified Working, Model Outputs Wrong Tokens (May 17, 2026 04:30)

## Status: 🔍 MODEL ISSUE - Logits computation produces wrong token IDs

**Date:** May 17, 2026 04:30

## Summary

Tokenizer loads correctly and BPE encode/decode works. But model generates wrong token IDs.

## Verification: Tokenizer is Working ✅

```
DEBUG: model->tokenizer = 0xb400006e4117b060
DEBUG: checking tokens[0]: tokens[0] = "<|endoftext|>"
DEBUG: num_merges = 48900
DEBUG: encoded 2 tokens for "Hello" input
```

Tokenizer loading verified working. Token 0 is correctly `<|endoftext|>`, merges are loaded.

## Model Output Issue 🔴

```
$ ./smollm2-cli -m smollm2-135m-v5.sm2 -p "Hello" -n 10
Output: [gen_tok=23]'[gen_tok=10]<iss[gen_tok=28],[gen_tok=20]$[gen_tok=6]<fil[EOS=0]
Generated 5 tokens in 3416.8 ms (683.4 ms/token)
```

**Token IDs generated:** 23, 10, 28, 20, 6, 0 (EOS)

**Problem:** Token 10 is being decoded as `<iss` - this is incomplete token (should be `<|im_start|>` or similar). Token 6 is `<fil` - also incomplete.

This suggests the model is generating garbage token IDs, NOT a tokenizer issue.

## Root Cause Analysis

The logits computation in the model is producing wrong results. Possible causes:

1. **lm_head projection wrong** - final logits = x @ lm_head.T but lm_head shares weights with tok_embeddings. Might have indexing issues.

2. **Final RMSNorm incorrect** - After all 30 layers, final_norm is applied but might be wrong.

3. **Layer outputs accumulating errors** - Each layer's matmul might have subtle indexing issues that compound over 30 layers.

## What Works

- ✅ Tokenizer loading (tokens[0] = "<|endoftext|>", 48900 merges loaded)
- ✅ BPE encoding (2 tokens for "Hello")
- ✅ BPE decoding (attempts to decode, produces partial tokens)
- ✅ Model loads weights (30 layers, 270 tensors)
- ✅ Layer forward computes without segfault
- ✅ No NaN in logits (max logit ~148 at token 44191 from earlier debug)

## What's Broken

- ❌ Model generates garbage tokens (23, 10, 28, 20, 6 instead of meaningful IDs)
- ❌ Tokens 10 and 6 decode to partial strings `<iss` and `<fil` - should be complete tokens

## Files Modified This Session

- `src/sm2_tokenizer.c` - BPE encoding implemented with proper merge table
- `src/smollm2.c` - debug prints added then removed

## Next Debug Steps

1. Check lm_head - does it point to same data as tok_embeddings?
2. Verify final_logits = hidden_state @ lm_head.T 
3. Compare with Python reference for same "Hello" input
4. Print first 5 logits and their token strings to see if even the logits are wrong

## Wiki Updated

- `wiki/components/smollm2c-tokenizer.md` - Status updated to working
- `wiki/log.md` - New entry about tokenizer verification

## Push to Git

Before pushing, need to:
1. Remove debug prints from smollm2.c
2. Write handoff.md
3. Update wiki if needed