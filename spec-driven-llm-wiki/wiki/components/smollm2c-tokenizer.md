---
title: "smollm2c-tokenizer"
type: component
tags: [tokenizer, bpe, hf, vocab]
last_updated: 2026-04-24
---

# smollm2c-tokenizer

HuggingFace-compatible BPE tokenizer for SmolLM2. Loads vocab.json and merges.txt, encodes/decodes text for model inference.

## Dependencies
- [[smollm2c-core-runtime]] - tokenization used during prefill/decode

## File Format
| File | Format | Description |
|------|--------|-------------|
| `vocab.json` | JSON dict | token -> id mapping |
| `merges.txt` | text, line-separated | BPE merge rules |

## API
```c
typedef struct {
    int vocab_size;
    char** tokens;
    float* scores;
    int* token_to_id;
    int num_merges;
    sm2_merge* merges;
} sm2_tokenizer;

// Encode text -> token ids
int sm2_tokenizer_encode(sm2_tokenizer* tok, const char* text, int* ids, int max_len);

// Decode token ids -> text
char* sm2_tokenizer_decode(sm2_tokenizer* tok, int* ids, int n_ids);
```

## Encoding Pipeline
```
input text
  -> pre-tokenize (split on whitespace/punct)
  -> BPE merge (apply merge rules greedily)
  -> lookup token ids
  -> output int array
```

## Performance
- Vocabulary size: ~50k tokens (SmolLM2)
- Encoding: O(n * merges) worst case
- Decode: O(n) simple lookup
- Cache: optional LRU for frequent strings

## Status
- [[spec:001]] - tokenizer interface defined
- Phase 1 implementation
- ⚠️ **IN PROGRESS**: Current implementation uses byte fallback, full tokenizer.json integration pending

**Tokenizer.json vs vocab.json+merges.txt:**
SmolLM2 uses tokenizer.json (HuggingFace's single-file format containing vocab + merges as JSON). The converter embeds this in the .sm2 file at tokenizer_offset.

**Current issue:** `sm2_tokenizer_encode()` not fully integrated, CLI falls back to byte-based encoding.
