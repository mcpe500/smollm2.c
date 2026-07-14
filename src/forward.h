// forward.h — SmolLM2 transformer forward pass

#ifndef FORWARD_H
#define FORWARD_H

#include "gguf.h"

typedef struct forward_ctx forward_ctx;

/* Runtime mode axes (default = current baseline). Set before forward_load. */
enum rope_mode { ROPE_F32 = 0, ROPE_F16 = 1, ROPE_Q8 = 2 };
enum kv_mode   { KV_F32   = 0, KV_F16   = 1, KV_Q8   = 2 };
enum attn_mode { ATTN_NAIVE = 0, ATTN_FLASH = 1 };

void forward_set_modes(int rope, int kv, int attn);

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

// Studio training: expose final hidden state (post rmsnorm, pre lm_head).
// Valid only immediately after forward_prefill; pointer owned by ctx.
const float* forward_last_hidden(const forward_ctx* f);
int          forward_dim(const forward_ctx* f);
int          forward_n_layers(const forward_ctx* f);

#endif // FORWARD_H
