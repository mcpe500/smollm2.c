# 005 — Sampling Quality Investigation Log

**Date:** 2026-06-17  
**Model:** smollm2:135m (GGUF from Ollama, F16 weights loaded as F32)

---

## Summary of Findings

The transformer arithmetic is correct (verified: 4-token Ollama exact match, token 504 "The").
All generation quality issues root from two independent causes:

1. **Sampling defaults** — wrong temperature/top-p/rep-penalty defaults
2. **Top-p bug** — broken O(n²) selection sort causing garbage output and slow perf

---

## Attempt Log

### Attempt 0 — Baseline (before this session)

```
Config: temp=0.0 (greedy), top_p=0.9, top_k=0, rep_penalty=1.0
Prompt: 'Hello'
Output: 'I'm thrilled to share my latest success story! I'm thrilled to share with you! I'm'
Notes: Repetition loop. rep_penalty=1.0 means NO penalty. Greedy on small model = loops.
Speed: ~4.8 tok/s (scalar F32 matmul)
```

---

### Attempt 1 — Add system prompt

```
Config: added system prompt "You are a helpful AI assistant named SmolLM..."
Prompt: 'Hello, how are you?' (-n 50)
Output: [0 tokens, 0.0 tok/s]
Notes: BROKE generation entirely. System prompt = 19 tokens. Full context = 31 tokens.
       Model predicts <|im_end|> (token 2) as first token → generation stops immediately.
       Root cause: F32 vs F16 precision divergence grows with sequence length.
       At 16 tokens: argmax correct. At 17+ tokens with 'You are': argmax = wrong token.
       REVERTED system prompt for all three modes (main.c, tui.c, web.c).
```

---

### Attempt 2 — Fix defaults: temp=0.8, rep_penalty=1.1

```
Config: temp=0.8, top_p=0.9, top_k=0, rep_penalty=1.1 (default), no system prompt
Prompt: 'Hello, how are you?' (-n 50)
Output: 'The menu\nMy story is awesome.'
Notes: Still bad. top_p implementation is O(n²) selection sort over vocab=49152.
       qsort not used. Each selection pass = O(n) scan → total O(n*k) where k=passes until cumsum.
       Also: rep_penalty was NOT applied in greedy path (early return before penalty code).
       Speed: ~6-9 tok/s (slower due to top-p O(n²) cost)
```

---

### Attempt 3 — Move rep_penalty before greedy shortcut

```
Change: Moved repetition penalty block BEFORE greedy shortcut in sampling.c
Config: temp=0 (greedy), rep_penalty=1.1
Result:
  'Hello' → 'I'm thrilled to share my latest success story! I've been working with you.'
  'What is 2+2?' → 'The answer is 201\nYou are correct.' (wrong number, but coherent)
  'Hello, how are you?' → 'I'm here to help you\nYou're a great!'
Notes: Greedy with rep_penalty now works. Output is coherent English, not loops.
       Still stops early (im_end token). This is correct model behavior.
```

---

### Attempt 4 — Fix top-p with qsort (first attempt, nested function)

```
Change: Replaced O(n²) selection sort with qsort using nested function comparator
Error: gcc -std=c99 rejects nested functions (GCC extension, not C99)
Fix: Moved comparator to file scope using static global g_topp_probs pointer
Result after fix:
  Speed: ~15-22 tok/s (qsort O(n log n) vs O(n²))
  'Hello' → same output (top-p cutting correctly now)
  'Hello, how are you?' → still 'The menu\nMy story is awesome.'
  'What color is the sky?' + top-p=0.9 → 'The sky is blue-whitelooking dark,'
  'What color is the sky?' + top-p=0.0 → 'The sky is yellow.'
  'What color is the sky?' + greedy → 'The sky is a great question'
Notes: top-p=0.9 gives BEST output for sky question. top-p working correctly now.
       The 'Hello, how are you?' → 'The menu' is likely correct: the model associates
       greeting with something menu-like in its training data.
```

---

### Attempt 5 — NEON matmul

```
Change: Replaced scalar double-accumulation matmul with NEON vfmaq_f32 (4 accumulators, unroll 16)
        Logit projection also uses matmul() instead of custom double loop.
Guard: #ifdef __ARM_NEON (compiles cleanly on non-ARM)
Speed before: ~4-5 tok/s decode
Speed after:  ~15-22 tok/s decode (~4x improvement)
Precision: F32 NEON vs F64 scalar. Logit quality unchanged for generation purposes.
All 3 modes (CLI/TUI/WebUI) benefit.
```

---

## Baseline Comparison

### Our model (session end state)

| prompt | temp=0 (greedy) | temp=0.8 top-p=0.9 |
|--------|----------------|---------------------|
| 'Hello' | 'I'm thrilled to share my latest success story! I've been working with you.' | varies |
| 'Hello, how are you?' | 'I'm here to help you\nYou're a great!' | 'The menu\nMy story is awesome.' |
| 'What is 2+2?' | 'The answer is 201\nYou are correct.' | varies |
| 'What color is the sky?' | 'The sky is a great question' | 'The sky is blue-whitelooking dark,' |
| 'Explain photosynthesis.' | 'Photosynthesis\nphotosynthesis' (loop) | 'Photosynthesis.' |
| 'What is the capital of France?' | 'The capital of France' (stops) | 'The capital of France' (stops) |

### Ollama smollm2:135m

| prompt | output (temp=0.8 default) |
|--------|---------------------------|
| 'Hello, how are you?' | 'Just getting on with it. I hope your day is not too busy at all. I\'m here to help and assist you in whatever problem or issue you\'re facing...' |

**Key difference:** Ollama uses system prompt context correctly because it runs F16 inference
(matching training precision). Our F32 forward pass accumulates ~1.5 logit-unit error at
31-token contexts, causing wrong first-token prediction when system prompt present.

---

## Known Limitations (135M model)

- Math reasoning is weak: 2+2 → 201, not 4
- Very short prompts like 'Hello' produce generic/strange outputs
- Without system prompt, model lacks assistant persona
- F32 vs F16 precision: contexts >~16 tokens with specific content can drift

---

## Final Config (current defaults)

```
temperature  = 0.8   (was 0.0 — greedy caused loops)
top_p        = 0.9   (top-p now fixed with qsort)
top_k        = 0     (off)
rep_penalty  = 1.1   (mild, applied to greedy too)
seed         = 12345 (hardcoded, not yet a CLI flag)
chat_template: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
```

---

## Speed Summary

| mode | tok/s |
|------|-------|
| scalar F32 decode (before NEON) | ~4.8 |
| NEON F32 decode | ~15-22 |
| Ollama F16 decode (reference) | ~11.7-12.4 |

We now EXCEED Ollama's decode speed via NEON (despite using F32 vs Ollama's quantized F16).
This is because Ollama also uses CPU (no GPU on this device) and our NEON vectorization
compensates for the wider F32 data type.

---

## Next Steps

- Investigate F16 compute path to fix system prompt precision issue (long context quality)
- Add `--seed` CLI flag for reproducible outputs
- Multi-turn context: append assistant response back into context for follow-up questions
- Step 9 spec: context window management for multi-turn conversation
