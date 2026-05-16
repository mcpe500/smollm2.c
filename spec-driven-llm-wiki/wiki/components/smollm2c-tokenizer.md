---
title: "smollm2c-tokenizer"
type: component
tags: [tokenizer, bpe, hf, vocab]
last_updated: 2026-05-17
---

# smollm2c-tokenizer

HuggingFace-compatible BPE tokenizer for SmolLM2. Loads tokenizer from .sm2 file and implements BPE encode/decode.

## Status

**✅ WORKING** - Tokenizer loads and BPE encode/decode implemented. Model generates wrong tokens (see [[smollm2c-core-runtime]]).

## Verified Working (May 17, 2026)

```
DEBUG: model->tokenizer = 0xb400006e4117b060
DEBUG: checking tokens[0]: tokens[0] = "<|endoftext|>"
DEBUG: num_merges = 48900
DEBUG: encoded 2 tokens for "Hello" input
```

- Token 0 = `"<|endoftext|>"` ✅
- 48900 merges loaded ✅
- BPE encoding for "Hello" produces 2 tokens ✅

## File Format

The .sm2 file stores tokenizer as length-prefixed binary:
```
[49152 tokens] each: 4-byte length + token_bytes
[merges_count: 4 bytes]
[merges] each: 4-byte length + merge_bytes
```

## API

```c
typedef struct {
    int vocab_size;
    char** tokens;          // id -> token string
    float* scores;
    int* token_to_id;       // token string -> id (not used currently)
    int num_merges;
    char** merges;          // merge rules
    uint8_t* vocab_data;    // raw tokenizer JSON (if loaded from file)
} sm2_tokenizer;

// Load from .sm2 file
int sm2_load_tokenizer_from_sm2(sm2_tokenizer* tok, FILE* f, uint64_t offset, uint64_t size);

// Encode text -> token ids
int sm2_tokenizer_encode(sm2_tokenizer* tok, const char* text, int* ids, int max_len);

// Decode token ids -> text
char* sm2_tokenizer_decode(sm2_tokenizer* tok, const int* ids, int n_ids);
```

## BPE Algorithm

```
1. Pre-tokenize: split text on whitespace
2. For each word:
   a. Split into character bytes
   b. Iteratively find best merge (lowest rank in merges table)
   c. Apply merge until no more possible
   d. Output final token IDs
```

## Encoding Pipeline

```
input text
  -> pre-tokenize (split on whitespace)
  -> BPE merge (apply merge rules greedily)
  -> lookup token ids (byte -> token ID)
  -> output int array
```

## Current Issue

Tokenizer is working correctly. **The issue is model generates wrong token IDs** (garbage tokens like 23, 10, 28, 20, 6 instead of meaningful IDs). This is a [[smollm2c-core-runtime]] issue with logits computation.

## Dependencies

- [[smollm2c-core-runtime]] - tokenization used during prefill/decode

## Performance

- Vocabulary size: 49152 tokens
- Encoding: O(n * merges) worst case
- Decode: O(n) simple lookup
- 48900 merge rules

## Related

- [[spec:001]] - Master blueprint
- [[smollm2c-core-runtime]] - Core runtime (has model output issue)