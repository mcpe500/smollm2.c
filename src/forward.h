// forward.h — SmolLM2 transformer forward pass

#ifndef FORWARD_H
#define FORWARD_H

#include "gguf.h"

typedef struct forward_ctx forward_ctx;

// Load model weights from parsed GGUF into F32 buffers + allocate KV cache.
// max_seq caps how many tokens a single prefill can process.
// Returns 0 on success, -1 on error.
int  forward_load (forward_ctx** out, const gguf_ctx* g, int max_seq);

void forward_free (forward_ctx* f);

// Prefill: run transformer over tokens[0..n_tokens-1], write last-position
// logits into logits_out (caller-allocated, size >= forward_vocab_size).
// Returns 0 on success, -1 on error (n_tokens out of range, bad token id).
int  forward_prefill(forward_ctx* f,
                     const int* tokens, int n_tokens,
                     float* logits_out);

int  forward_vocab_size(const forward_ctx* f);

// Decode one new token at absolute position `pos`.
// Expects KV cache populated for positions 0..pos-1 by forward_prefill.
// Returns 0 on success, -1 on error.
int  forward_decode(forward_ctx* f, int token, int pos, float* logits_out);

// Clear KV cache. Call before starting a new conversation.
void forward_reset(forward_ctx* f);

#endif // FORWARD_H
