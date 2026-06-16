// tokenizer.h — BPE tokenizer reading vocab/merges from GGUF

#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "gguf.h"

typedef struct tokenizer tokenizer;

// Load tokenizer metadata from a parsed GGUF context.
int  tokenizer_load (tokenizer** out, const gguf_ctx* g);
void tokenizer_free (tokenizer* t);

// Encode text → token IDs. Writes into out[] (caller-allocated).
// Returns number of tokens written; never exceeds max_out.
// Special tokens in input (e.g. <|im_start|>) are matched literally.
int  tokenizer_encode(const tokenizer* t, const char* text,
                      int* out, int max_out);

// Decode a single token into bytes. Returns number of bytes, or -1.
int  tokenizer_decode(const tokenizer* t, int token_id,
                      char* buf, int max_buf);

// Look up token_id by literal text (e.g. "<|im_start|>"). Returns -1 if absent.
int  tokenizer_lookup(const tokenizer* t, const char* token_text);

// Number of vocab entries.
int  tokenizer_vocab_size(const tokenizer* t);

#endif // TOKENIZER_H
