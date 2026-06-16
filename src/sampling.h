// sampling.h — token sampling from logit distribution

#ifndef SAMPLING_H
#define SAMPLING_H

typedef struct {
    float    temperature;   // 0.0 = greedy argmax
    float    top_p;         // nucleus: 0.0 or 1.0 = off
    int      top_k;         // 0 = off
    float    rep_penalty;   // 1.0 = off
    unsigned seed;          // RNG seed
} sample_params;

// Sample next token from logits[0..vocab-1].
// history/hist_len used only when rep_penalty > 1.0.
// Returns token id.
int sample_token(float* logits, int vocab, const sample_params* p,
                 const int* history, int hist_len);

#endif // SAMPLING_H
