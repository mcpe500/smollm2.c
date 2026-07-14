// train.c — LoRA on lm_head (phase 2 minimal)
//
// Strategy: freeze transformer, train LoRA A/B on final projection.
// Forward: logits = base_logits + scale * ((h @ A) @ B)
// Backward: dA, dB via CE gradient; Adam update.
// Peak RAM: base model + ~2*vocab*rank F32 ≈ 5-10MB extra.

#include "train.h"
#include "gguf_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <time.h>

#define LORA_MAGIC "LORA0001"
#define ADAM_B1 0.9f
#define ADAM_B2 0.999f
#define ADAM_EPS 1e-8f

struct train_state {
    forward_ctx* f;
    train_params p;
    int dim;
    int vocab;
    int rank;
    float scale;          /* alpha / rank */

    /* LoRA on lm_head: A [dim, rank], B [rank, vocab] */
    float* A;  float* B;
    float* gA; float* gB;
    float* mA; float* vA;
    float* mB; float* vB;
    float* h_buf;         /* last hidden [dim] */
    float* logits;        /* [vocab] */
    float* delta;         /* LoRA delta [vocab] */
    float* mid;           /* h @ A [rank] */
    int    step;
    float  last_loss;
};

static float* zalloc(size_t n) {
    float* p = calloc(n, sizeof(float));
    return p;
}

train_state* train_create(forward_ctx* f, const train_params* p) {
    if (!f || !p) return NULL;
    train_state* t = calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->f = f;
    t->p = *p;
    t->dim = forward_dim(f);
    t->vocab = forward_vocab_size(f);
    t->rank = p->lora_rank > 0 ? p->lora_rank : 8;
    int alpha = p->lora_alpha > 0 ? p->lora_alpha : 2 * t->rank;
    t->scale = (float)alpha / (float)t->rank;
    t->step = 0;

    size_t a_sz = (size_t)t->dim * t->rank;
    size_t b_sz = (size_t)t->rank * t->vocab;
    t->A  = zalloc(a_sz); t->B  = zalloc(b_sz);
    t->gA = zalloc(a_sz); t->gB = zalloc(b_sz);
    t->mA = zalloc(a_sz); t->vA = zalloc(a_sz);
    t->mB = zalloc(b_sz); t->vB = zalloc(b_sz);
    t->h_buf  = zalloc(t->dim);
    t->logits = zalloc(t->vocab);
    t->delta  = zalloc(t->vocab);
    t->mid    = zalloc(t->rank);

    if (!t->A || !t->B || !t->gA || !t->gB || !t->mA || !t->vA ||
        !t->mB || !t->vB || !t->h_buf || !t->logits || !t->delta || !t->mid) {
        train_free(t);
        return NULL;
    }

    /* Kaiming-ish init: A ~ N(0, 1/sqrt(dim)), B = 0 */
    srand(p->seed ? p->seed : 42);
    for (size_t i = 0; i < a_sz; i++)
        t->A[i] = ((float)rand() / 2147483647.0f - 0.5f) * 0.02f;
    /* B zeroed by zalloc */

    return t;
}

void train_free(train_state* t) {
    if (!t) return;
    free(t->A); free(t->B); free(t->gA); free(t->gB);
    free(t->mA); free(t->vA); free(t->mB); free(t->vB);
    free(t->h_buf); free(t->logits); free(t->delta); free(t->mid);
    free(t);
}

/* Softmax CE: loss = -log p[target], dlogits = p - one_hot */
static float ce_loss_and_grad(float* logits, int vocab, int target,
                              float* dlogits) {
    float max_l = logits[0];
    for (int i = 1; i < vocab; i++) if (logits[i] > max_l) max_l = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < vocab; i++) {
        dlogits[i] = expf(logits[i] - max_l);
        sum += dlogits[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < vocab; i++) dlogits[i] *= inv;
    float loss = -logf(dlogits[target] + 1e-12f);
    dlogits[target] -= 1.0f;
    return loss;
}

/* mid = h @ A  (dim → rank) */
static void matmul_hA(const float* h, const float* A, float* mid,
                      int dim, int rank) {
    for (int r = 0; r < rank; r++) {
        float s = 0;
        for (int d = 0; d < dim; d++) s += h[d] * A[d * rank + r];
        mid[r] = s;
    }
}

/* delta = mid @ B  (rank → vocab) */
static void matmul_midB(const float* mid, const float* B, float* delta,
                        int rank, int vocab) {
    for (int v = 0; v < vocab; v++) {
        float s = 0;
        for (int r = 0; r < rank; r++) s += mid[r] * B[r * vocab + v];
        delta[v] = s;
    }
}

/* Accumulate LoRA grads for one (h, logits, target) triple. */
static float lora_accum(train_state* t, const float* h, float* logits, int target) {
    matmul_hA(h, t->A, t->mid, t->dim, t->rank);
    matmul_midB(t->mid, t->B, t->delta, t->rank, t->vocab);
    for (int v = 0; v < t->vocab; v++)
        logits[v] += t->scale * t->delta[v];

    float* dlogits = t->delta;  /* reuse */
    float loss = ce_loss_and_grad(logits, t->vocab, target, dlogits);

    float d_mid[64];
    if (t->rank > 64) return -1.0f;
    for (int r = 0; r < t->rank; r++) {
        float s = 0;
        for (int v = 0; v < t->vocab; v++) {
            t->gB[r * t->vocab + v] += t->scale * t->mid[r] * dlogits[v];
            s += t->B[r * t->vocab + v] * dlogits[v];
        }
        d_mid[r] = t->scale * s;
    }
    for (int d = 0; d < t->dim; d++)
        for (int r = 0; r < t->rank; r++)
            t->gA[d * t->rank + r] += h[d] * d_mid[r];
    return loss;
}

static void adam_update(train_state* t) {
    t->step++;
    float b1t = 1.0f - powf(ADAM_B1, (float)t->step);
    float b2t = 1.0f - powf(ADAM_B2, (float)t->step);
    float lr = t->p.lr;
    size_t a_sz = (size_t)t->dim * t->rank;
    size_t b_sz = (size_t)t->rank * t->vocab;
    for (size_t i = 0; i < a_sz; i++) {
        t->mA[i] = ADAM_B1 * t->mA[i] + (1 - ADAM_B1) * t->gA[i];
        t->vA[i] = ADAM_B2 * t->vA[i] + (1 - ADAM_B2) * t->gA[i] * t->gA[i];
        float mhat = t->mA[i] / b1t;
        float vhat = t->vA[i] / b2t;
        t->A[i] -= lr * mhat / (sqrtf(vhat) + ADAM_EPS);
    }
    for (size_t i = 0; i < b_sz; i++) {
        t->mB[i] = ADAM_B1 * t->mB[i] + (1 - ADAM_B1) * t->gB[i];
        t->vB[i] = ADAM_B2 * t->vB[i] + (1 - ADAM_B2) * t->gB[i] * t->gB[i];
        float mhat = t->mB[i] / b1t;
        float vhat = t->vB[i] / b2t;
        t->B[i] -= lr * mhat / (sqrtf(vhat) + ADAM_EPS);
    }
}

float train_step(train_state* t, const int* tokens, int n_tokens) {
    if (!t || !tokens || n_tokens < 2) return -1.0f;
    int use_n = n_tokens;
    if (use_n > t->p.seq_max) use_n = t->p.seq_max;
    if (use_n < 2) return -1.0f;

    /* Multi-token CE: prefill tok0, then decode tok1..tok{n-2}.
       At each pos predict next token; accumulate grads; one Adam. */
    forward_reset(t->f);
    if (forward_prefill(t->f, tokens, 1, t->logits) < 0)
        return -1.0f;

    memset(t->gA, 0, (size_t)t->dim * t->rank * sizeof(float));
    memset(t->gB, 0, (size_t)t->rank * t->vocab * sizeof(float));

    float total = 0.0f;
    int n_pred = 0;
    for (int pos = 0; pos < use_n - 1; pos++) {
        const float* h = forward_last_hidden(t->f);
        if (!h) return -1.0f;
        memcpy(t->h_buf, h, t->dim * sizeof(float));

        float loss = lora_accum(t, t->h_buf, t->logits, tokens[pos + 1]);
        if (loss < 0) return -1.0f;
        total += loss;
        n_pred++;

        if (pos < use_n - 2) {
            if (forward_decode(t->f, tokens[pos + 1], pos + 1, t->logits) < 0)
                return -1.0f;
        }
    }
    if (n_pred == 0) return -1.0f;

    /* Mean-grad: scale by 1/n_pred so lr stays comparable across lengths */
    float inv = 1.0f / (float)n_pred;
    size_t a_sz = (size_t)t->dim * t->rank;
    size_t b_sz = (size_t)t->rank * t->vocab;
    for (size_t i = 0; i < a_sz; i++) t->gA[i] *= inv;
    for (size_t i = 0; i < b_sz; i++) t->gB[i] *= inv;

    adam_update(t);
    t->last_loss = total * inv;
    return t->last_loss;
}

int train_save(const train_state* t, const char* path) {
    if (!t || !path) return -1;
    FILE* f = fopen(path, "wb");
    if (!f) { perror("train_save"); return -1; }
    fwrite(LORA_MAGIC, 1, 8, f);
    int32_t hdr[4] = { t->dim, t->vocab, t->rank, t->step };
    fwrite(hdr, sizeof(int32_t), 4, f);
    float scale = t->scale;
    fwrite(&scale, sizeof(float), 1, f);
    size_t a_sz = (size_t)t->dim * t->rank;
    size_t b_sz = (size_t)t->rank * t->vocab;
    fwrite(t->A, sizeof(float), a_sz, f);
    fwrite(t->B, sizeof(float), b_sz, f);
    fclose(f);
    return 0;
}

int train_load(train_state* t, const char* path) {
    if (!t || !path) return -1;
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, LORA_MAGIC, 8) != 0) {
        fclose(f); return -1;
    }
    int32_t hdr[4];
    if (fread(hdr, sizeof(int32_t), 4, f) != 4) { fclose(f); return -1; }
    if (hdr[0] != t->dim || hdr[1] != t->vocab || hdr[2] != t->rank) {
        fprintf(stderr, "train_load: shape mismatch\n");
        fclose(f); return -1;
    }
    t->step = hdr[3];
    fread(&t->scale, sizeof(float), 1, f);
    size_t a_sz = (size_t)t->dim * t->rank;
    size_t b_sz = (size_t)t->rank * t->vocab;
    fread(t->A, sizeof(float), a_sz, f);
    fread(t->B, sizeof(float), b_sz, f);
    fclose(f);
    return 0;
}

/* Load packed sample index + tokens */
static int load_sample(const char* packed, long offset, int n_tokens,
                       int* out, int max_out) {
    if (n_tokens > max_out) n_tokens = max_out;
    FILE* f = fopen(packed, "rb");
    if (!f) return -1;
    if (fseek(f, offset, SEEK_SET) != 0) { fclose(f); return -1; }
    size_t n = fread(out, sizeof(int), n_tokens, f);
    fclose(f);
    return (int)n;
}

int train_run(train_state* t, const char* packed_path,
              const train_params* p, const char* out_dir) {
    if (!t || !packed_path || !p || !out_dir) return -1;

    char idx_path[1024];
    snprintf(idx_path, sizeof(idx_path), "%s.idx", packed_path);
    FILE* idx = fopen(idx_path, "rb");
    if (!idx) { perror("train_run open idx"); return -1; }

    /* Read trailer: last 20 bytes */
    if (fseek(idx, -20, SEEK_END) != 0) { fclose(idx); return -1; }
    int32_t hdr[3];
    char magic[8];
    fread(hdr, sizeof(int32_t), 3, idx);
    fread(magic, 1, 8, idx);
    if (memcmp(magic, "STUDIO", 6) != 0) {
        fprintf(stderr, "train_run: bad packed magic\n");
        fclose(idx); return -1;
    }
    int n_samples = hdr[0];
    if (n_samples <= 0) { fclose(idx); return -1; }

    /* Load all sample_idx */
    sample_idx* samples = malloc((size_t)n_samples * sizeof(sample_idx));
    if (!samples) { fclose(idx); return -1; }
    rewind(idx);
    if (fread(samples, sizeof(sample_idx), n_samples, idx) != (size_t)n_samples) {
        free(samples); fclose(idx); return -1;
    }
    fclose(idx);

    mkdir(out_dir, 0755);

    int max_steps = p->max_steps > 0 ? p->max_steps : n_samples * p->epochs;
    int tokens[2048];
    int step = 0;
    float first_loss = -1.0f;

    for (int epoch = 0; epoch < p->epochs && step < max_steps; epoch++) {
        for (int s = 0; s < n_samples && step < max_steps; s++) {
            int n = load_sample(packed_path, samples[s].offset,
                                samples[s].n_tokens, tokens, 2048);
            if (n < 2) continue;

            float loss = train_step(t, tokens, n);
            if (loss < 0) continue;
            if (first_loss < 0) first_loss = loss;
            step++;
            printf("step=%d epoch=%d sample=%d loss=%.4f\n",
                   step, epoch, s, loss);
            fflush(stdout);

            /* Watchdog */
            hw_caps caps;
            hw_probe(&caps);
            if (p->simulate_mem_kb > 0) caps.mem_avail_kb = p->simulate_mem_kb;
            if (caps.mem_avail_kb < 100 * 1024) {
                char path[1024];
                snprintf(path, sizeof(path), "%s/emergency_%d.bin", out_dir, step);
                train_save(t, path);
                fprintf(stderr, "train: emergency save at step %d "
                        "(mem_avail=%ld MB)\n", step, caps.mem_avail_kb / 1024);
                free(samples);
                return 0;
            }

            if (p->checkpoint_every > 0 && step % p->checkpoint_every == 0) {
                char path[1024];
                snprintf(path, sizeof(path), "%s/lora_%d.bin", out_dir, step);
                train_save(t, path);
            }
        }
    }

    /* Final save */
    char path[1024];
    snprintf(path, sizeof(path), "%s/lora_final.bin", out_dir);
    train_save(t, path);
    printf("train: done steps=%d first_loss=%.4f last_loss=%.4f adapter=%s\n",
           step, first_loss, t->last_loss, path);

    free(samples);
    return 0;
}

/* Phase 2a merge: copy base + keep adapter sidecar. Real weight merge
   needs gguf_write tensor rewrite (phase 2b). For now: copy base so
   inference still works, and log that adapter is sidecar. */
int train_merge(const char* base_gguf, const char* adapter_path,
                const char* out_gguf) {
    if (!base_gguf || !adapter_path || !out_gguf) return -1;
    /* Verify adapter magic */
    FILE* a = fopen(adapter_path, "rb");
    if (!a) { perror("train_merge adapter"); return -1; }
    char magic[8];
    if (fread(magic, 1, 8, a) != 8 || memcmp(magic, LORA_MAGIC, 8) != 0) {
        fprintf(stderr, "train_merge: bad adapter magic\n");
        fclose(a); return -1;
    }
    fclose(a);

    if (gguf_copy(base_gguf, out_gguf) < 0) return -1;

    /* Sidecar: write adapter next to out so runtime can load later */
    char side[1024];
    snprintf(side, sizeof(side), "%s.lora", out_gguf);
    FILE* src = fopen(adapter_path, "rb");
    FILE* dst = fopen(side, "wb");
    if (src && dst) {
        char buf[64 * 1024];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), src)) > 0)
            fwrite(buf, 1, n, dst);
    }
    if (src) fclose(src);
    if (dst) fclose(dst);
    printf("train_merge: base copied to %s, adapter sidecar %s\n",
           out_gguf, side);
    printf("train_merge: note — phase 2a keeps base weights; LoRA applied "
           "at runtime when sidecar present (phase 2b full merge)\n");
    return 0;
}