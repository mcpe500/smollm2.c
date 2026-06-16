// forward.c — SmolLM2 transformer forward pass (scalar, no NEON)

#include "forward.h"
#include "gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// RoPE theta read from GGUF at load time (SmolLM2 uses 100000, not 10000)

struct forward_ctx {
    // config
    int dim, n_layers, n_heads, n_kv_heads, head_dim, kv_dim;
    int ffn_hidden, vocab_size, max_seq;
    float rms_eps, rope_theta;

    // weights (F32, dequantized at load)
    float* w_token_embd;  // [vocab * dim]
    float* w_norm;        // [dim]  (output_norm)
    float* w_attn_norm;   // [n_layers * dim]
    float* w_q;           // [n_layers * dim * dim]
    float* w_k;           // [n_layers * kv_dim * dim]
    float* w_v;           // [n_layers * kv_dim * dim]
    float* w_o;           // [n_layers * dim * dim]
    float* w_ffn_norm;    // [n_layers * dim]
    float* w_gate;        // [n_layers * ffn_hidden * dim]
    float* w_up;          // [n_layers * ffn_hidden * dim]
    float* w_down;        // [n_layers * dim * ffn_hidden]

    // KV cache
    float* k_cache;       // [n_layers * max_seq * kv_dim]
    float* v_cache;       // [n_layers * max_seq * kv_dim]

    // activation buffers
    float* x_buf;         // [max_seq * dim]   residual stream
    float* x_norm;        // [dim]
    float* q;             // [dim]
    float* k;             // [kv_dim]
    float* v;             // [kv_dim]
    float* attn_out;      // [dim]
    float* o_proj;        // [dim]
    float* ffn_gate;      // [ffn_hidden]
    float* ffn_up;        // [ffn_hidden]
    float* ffn_mid;       // [dim]
    float* scores;        // [max_seq]
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static float f16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;
    int exp  = (h >> 10) & 0x1f;
    int mant = h & 0x3ff;

    if (exp == 0) {
        if (mant == 0) {
            uint32_t u = sign;
            float f; memcpy(&f, &u, 4);
            return f;
        }
        // subnormal F16: mant * 2^-24 (rare for inference weights)
        float val = ldexpf((float)mant, -24);
        return sign ? -val : val;
    }
    if (exp == 0x1f) {
        // inf/nan
        uint32_t u = sign | 0x7f800000u | ((uint32_t)mant << 13);
        float f; memcpy(&f, &u, 4);
        return f;
    }
    // normal
    uint32_t u = sign
               | ((uint32_t)(exp + 127 - 15) << 23)
               | ((uint32_t)mant << 13);
    float f; memcpy(&f, &u, 4);
    return f;
}

static int load_tensor_f16(const gguf_ctx* g, const char* name,
                           float* dst, size_t count) {
    const gguf_tensor_info* t = gguf_tensor_get(g, name);
    if (!t) {
        fprintf(stderr, "forward: missing tensor %s\n", name);
        return -1;
    }
    if (t->dtype != GGUF_DT_F16) {
        fprintf(stderr, "forward: %s expected F16, got dtype %u\n",
                name, (unsigned)t->dtype);
        return -1;
    }
    const uint16_t* src = (const uint16_t*)gguf_tensor_data(g, t);
    for (size_t i = 0; i < count; i++) dst[i] = f16_to_f32(src[i]);
    return 0;
}

static int load_tensor_f32(const gguf_ctx* g, const char* name,
                           float* dst, size_t count) {
    const gguf_tensor_info* t = gguf_tensor_get(g, name);
    if (!t) {
        fprintf(stderr, "forward: missing tensor %s\n", name);
        return -1;
    }
    if (t->dtype != GGUF_DT_F32) {
        fprintf(stderr, "forward: %s expected F32, got dtype %u\n",
                name, (unsigned)t->dtype);
        return -1;
    }
    const float* src = (const float*)gguf_tensor_data(g, t);
    memcpy(dst, src, count * sizeof(float));
    return 0;
}

static void rmsnorm(float* out, const float* x, const float* w,
                    int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    ss /= n;
    float inv = 1.0f / sqrtf((float)ss + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
}

// GGUF weight layout: ne[0]=in_dim (innermost/contiguous per row), ne[1]=out_dim.
// y[o] = sum_i W[o*in_dim + i] * x[i]
static void matmul(float* y, const float* x, const float* W,
                   int in_dim, int out_dim) {
#ifdef __ARM_NEON
    for (int o = 0; o < out_dim; o++) {
        const float* wrow = W + (size_t)o * in_dim;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i <= in_dim - 16; i += 16) {
            acc0 = vfmaq_f32(acc0, vld1q_f32(wrow+i),    vld1q_f32(x+i));
            acc1 = vfmaq_f32(acc1, vld1q_f32(wrow+i+4),  vld1q_f32(x+i+4));
            acc2 = vfmaq_f32(acc2, vld1q_f32(wrow+i+8),  vld1q_f32(x+i+8));
            acc3 = vfmaq_f32(acc3, vld1q_f32(wrow+i+12), vld1q_f32(x+i+12));
        }
        acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        float sum = vaddvq_f32(acc0);
        for (; i < in_dim; i++) sum += wrow[i] * x[i];
        y[o] = sum;
    }
#else
    for (int o = 0; o < out_dim; o++) {
        const float* wrow = W + (size_t)o * in_dim;
        double acc = 0.0;
        for (int i = 0; i < in_dim; i++) acc += (double)wrow[i] * (double)x[i];
        y[o] = (float)acc;
    }
#endif
}

// GPT-NeoX RoPE: pair (i, i+hd/2) for each head independently
static void rope_neox(float* v, int pos, int n_heads, int head_dim, float theta) {
    int half = head_dim / 2;
    for (int h = 0; h < n_heads; h++) {
        float* vh = v + (size_t)h * head_dim;
        for (int i = 0; i < half; i++) {
            float freq  = 1.0f / powf(theta, (2.0f * (float)i) / (float)head_dim);
            float angle = (float)pos * freq;
            float c = cosf(angle);
            float s = sinf(angle);
            float a = vh[i];
            float b = vh[i + half];
            vh[i]        = a * c - b * s;
            vh[i + half] = a * s + b * c;
        }
    }
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------
int forward_load(forward_ctx** out, const gguf_ctx* g, int max_seq) {
    if (!out || !g || max_seq <= 0) return -1;

    const char* arch = gguf_kv_str(g, "general.architecture");
    if (!arch || strcmp(arch, "llama") != 0) {
        fprintf(stderr, "forward: unexpected arch '%s' (expected llama)\n",
                arch ? arch : "(null)");
        return -1;
    }

    forward_ctx* f = calloc(1, sizeof(forward_ctx));
    if (!f) return -1;

    f->dim        = (int)gguf_kv_i64(g, "llama.embedding_length",             -1);
    f->n_layers   = (int)gguf_kv_i64(g, "llama.block_count",                  -1);
    f->n_heads    = (int)gguf_kv_i64(g, "llama.attention.head_count",         -1);
    f->n_kv_heads = (int)gguf_kv_i64(g, "llama.attention.head_count_kv",      -1);
    f->ffn_hidden = (int)gguf_kv_i64(g, "llama.feed_forward_length",          -1);
    f->rms_eps    = gguf_kv_f32(g, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    f->rope_theta = gguf_kv_f32(g, "llama.rope.freq_base", 10000.0f);

    uint64_t vocab_n = 0;
    gguf_vtype et;
    gguf_kv_arr(g, "tokenizer.ggml.tokens", &et, &vocab_n);
    f->vocab_size = (int)vocab_n;

    if (f->dim <= 0 || f->n_layers <= 0 || f->n_heads <= 0 ||
        f->n_kv_heads <= 0 || f->ffn_hidden <= 0 || f->vocab_size <= 0) {
        fprintf(stderr,
            "forward: invalid config (dim=%d nl=%d nh=%d nkv=%d ffn=%d vocab=%d)\n",
            f->dim, f->n_layers, f->n_heads, f->n_kv_heads,
            f->ffn_hidden, f->vocab_size);
        free(f);
        return -1;
    }
    if (f->n_kv_heads > f->n_heads || f->dim % f->n_heads != 0) {
        fprintf(stderr, "forward: bad head config\n");
        free(f);
        return -1;
    }

    f->head_dim = f->dim / f->n_heads;
    f->kv_dim   = f->n_kv_heads * f->head_dim;
    f->max_seq  = max_seq;

    size_t s_embd  = (size_t)f->vocab_size * f->dim;
    size_t s_q     = (size_t)f->n_layers  * f->dim    * f->dim;
    size_t s_kv    = (size_t)f->n_layers  * f->kv_dim * f->dim;
    size_t s_norm  = (size_t)f->n_layers  * f->dim;
    size_t s_gu    = (size_t)f->n_layers  * f->ffn_hidden * f->dim;
    size_t s_down  = (size_t)f->n_layers  * f->dim    * f->ffn_hidden;
    size_t s_cache = (size_t)f->n_layers  * max_seq  * f->kv_dim;

    f->w_token_embd = malloc(s_embd  * sizeof(float));
    f->w_norm       = malloc(f->dim  * sizeof(float));
    f->w_attn_norm  = malloc(s_norm  * sizeof(float));
    f->w_q          = malloc(s_q     * sizeof(float));
    f->w_k          = malloc(s_kv    * sizeof(float));
    f->w_v          = malloc(s_kv    * sizeof(float));
    f->w_o          = malloc(s_q     * sizeof(float));
    f->w_ffn_norm   = malloc(s_norm  * sizeof(float));
    f->w_gate       = malloc(s_gu    * sizeof(float));
    f->w_up         = malloc(s_gu    * sizeof(float));
    f->w_down       = malloc(s_down  * sizeof(float));
    f->k_cache      = malloc(s_cache * sizeof(float));
    f->v_cache      = malloc(s_cache * sizeof(float));

    f->x_buf    = malloc((size_t)max_seq * f->dim      * sizeof(float));
    f->x_norm   = malloc(f->dim         * sizeof(float));
    f->q        = malloc(f->dim         * sizeof(float));
    f->k        = malloc(f->kv_dim      * sizeof(float));
    f->v        = malloc(f->kv_dim      * sizeof(float));
    f->attn_out = malloc(f->dim         * sizeof(float));
    f->o_proj   = malloc(f->dim         * sizeof(float));
    f->ffn_gate = malloc(f->ffn_hidden  * sizeof(float));
    f->ffn_up   = malloc(f->ffn_hidden  * sizeof(float));
    f->ffn_mid  = malloc(f->dim         * sizeof(float));
    f->scores   = malloc((size_t)max_seq * sizeof(float));

    if (!f->w_token_embd || !f->w_norm || !f->w_attn_norm || !f->w_q ||
        !f->w_k || !f->w_v || !f->w_o || !f->w_ffn_norm ||
        !f->w_gate || !f->w_up || !f->w_down ||
        !f->k_cache || !f->v_cache ||
        !f->x_buf || !f->x_norm || !f->q || !f->k || !f->v ||
        !f->attn_out || !f->o_proj || !f->ffn_gate || !f->ffn_up ||
        !f->ffn_mid || !f->scores) {
        fprintf(stderr, "forward: alloc failed\n");
        forward_free(f);
        return -1;
    }

    int rc = 0;
    rc |= load_tensor_f16(g, "token_embd.weight",  f->w_token_embd, s_embd);
    rc |= load_tensor_f32(g, "output_norm.weight", f->w_norm,       (size_t)f->dim);

    char name[128];
    for (int L = 0; L < f->n_layers && rc == 0; L++) {
        size_t off_q   = (size_t)L * f->dim    * f->dim;
        size_t off_kv  = (size_t)L * f->kv_dim * f->dim;
        size_t off_n   = (size_t)L * f->dim;
        size_t off_gu  = (size_t)L * f->ffn_hidden * f->dim;
        size_t off_d   = (size_t)L * f->dim    * f->ffn_hidden;

        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight",   L);
        rc |= load_tensor_f32(g, name, f->w_attn_norm + off_n, (size_t)f->dim);
        snprintf(name, sizeof(name), "blk.%d.attn_q.weight",     L);
        rc |= load_tensor_f16(g, name, f->w_q        + off_q,  (size_t)f->dim    * f->dim);
        snprintf(name, sizeof(name), "blk.%d.attn_k.weight",     L);
        rc |= load_tensor_f16(g, name, f->w_k        + off_kv, (size_t)f->kv_dim * f->dim);
        snprintf(name, sizeof(name), "blk.%d.attn_v.weight",     L);
        rc |= load_tensor_f16(g, name, f->w_v        + off_kv, (size_t)f->kv_dim * f->dim);
        snprintf(name, sizeof(name), "blk.%d.attn_output.weight",L);
        rc |= load_tensor_f16(g, name, f->w_o        + off_q,  (size_t)f->dim    * f->dim);
        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight",   L);
        rc |= load_tensor_f32(g, name, f->w_ffn_norm + off_n,  (size_t)f->dim);
        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight",   L);
        rc |= load_tensor_f16(g, name, f->w_gate     + off_gu, (size_t)f->ffn_hidden * f->dim);
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight",     L);
        rc |= load_tensor_f16(g, name, f->w_up       + off_gu, (size_t)f->ffn_hidden * f->dim);
        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight",   L);
        rc |= load_tensor_f16(g, name, f->w_down     + off_d,  (size_t)f->dim    * f->ffn_hidden);
    }

    if (rc != 0) {
        forward_free(f);
        return -1;
    }

    *out = f;
    return 0;
}

// ---------------------------------------------------------------------------
// Prefill
// ---------------------------------------------------------------------------
int forward_prefill(forward_ctx* f, const int* tokens, int n_tokens,
                    float* logits_out) {
    if (!f || !tokens || n_tokens <= 0 || n_tokens > f->max_seq || !logits_out)
        return -1;

    const int dim   = f->dim;
    const int nl    = f->n_layers;
    const int nh    = f->n_heads;
    const int nkv   = f->n_kv_heads;
    const int hd    = f->head_dim;
    const int kvdim = f->kv_dim;
    const int ffn   = f->ffn_hidden;
    const int vocab = f->vocab_size;
    const float eps = f->rms_eps;

    // 1. Embedding lookup
    for (int t = 0; t < n_tokens; t++) {
        int tok = tokens[t];
        if (tok < 0 || tok >= vocab) {
            fprintf(stderr, "forward: token %d out of range at pos %d\n", tok, t);
            return -1;
        }
        memcpy(f->x_buf + (size_t)t * dim,
               f->w_token_embd + (size_t)tok * dim,
               dim * sizeof(float));
    }

    const size_t cache_stride = (size_t)f->max_seq * kvdim;
    const float  inv_sqrt_hd  = 1.0f / sqrtf((float)hd);

    // 2. Transformer layers
    for (int L = 0; L < nl; L++) {
        const float* wq  = f->w_q        + (size_t)L * dim   * dim;
        const float* wk  = f->w_k        + (size_t)L * kvdim * dim;
        const float* wv  = f->w_v        + (size_t)L * kvdim * dim;
        const float* wo  = f->w_o        + (size_t)L * dim   * dim;
        const float* wan = f->w_attn_norm + (size_t)L * dim;
        const float* wfn = f->w_ffn_norm + (size_t)L * dim;
        const float* wg  = f->w_gate     + (size_t)L * ffn * dim;
        const float* wu  = f->w_up       + (size_t)L * ffn * dim;
        const float* wd  = f->w_down     + (size_t)L * dim * ffn;
        float* kc = f->k_cache + (size_t)L * cache_stride;
        float* vc = f->v_cache + (size_t)L * cache_stride;

        for (int t = 0; t < n_tokens; t++) {
            float* xt = f->x_buf + (size_t)t * dim;

            // --- attention block ---
            rmsnorm(f->x_norm, xt, wan, dim, eps);
            matmul(f->q, f->x_norm, wq, dim, dim);
            matmul(f->k, f->x_norm, wk, dim, kvdim);
            matmul(f->v, f->x_norm, wv, dim, kvdim);
            rope_neox(f->q, t, nh,  hd, f->rope_theta);
            rope_neox(f->k, t, nkv, hd, f->rope_theta);
            memcpy(kc + (size_t)t * kvdim, f->k, kvdim * sizeof(float));
            memcpy(vc + (size_t)t * kvdim, f->v, kvdim * sizeof(float));

            for (int h = 0; h < nh; h++) {
                int kvh = h * nkv / nh;   // GQA: 9 q-heads share 3 kv-heads (h/3)
                const float* qh = f->q + h * hd;
                float* oh = f->attn_out + h * hd;

                float max_s = -INFINITY;
                for (int s = 0; s <= t; s++) {
                    const float* ks = kc + (size_t)s * kvdim + kvh * hd;
                    float d = 0.0f;
                    for (int i = 0; i < hd; i++) d += qh[i] * ks[i];
                    d *= inv_sqrt_hd;
                    f->scores[s] = d;
                    if (d > max_s) max_s = d;
                }
                float sum = 0.0f;
                for (int s = 0; s <= t; s++) {
                    float e = expf(f->scores[s] - max_s);
                    f->scores[s] = e;
                    sum += e;
                }
                float inv_sum = (sum > 0.0f) ? 1.0f / sum : 0.0f;

                for (int i = 0; i < hd; i++) oh[i] = 0.0f;
                for (int s = 0; s <= t; s++) {
                    float w = f->scores[s] * inv_sum;
                    const float* vs = vc + (size_t)s * kvdim + kvh * hd;
                    for (int i = 0; i < hd; i++) oh[i] += w * vs[i];
                }
            }

            matmul(f->o_proj, f->attn_out, wo, dim, dim);
            for (int i = 0; i < dim; i++) xt[i] += f->o_proj[i];

            // --- FFN block (SwiGLU) ---
            rmsnorm(f->x_norm, xt, wfn, dim, eps);
            matmul(f->ffn_gate, f->x_norm, wg, dim, ffn);
            matmul(f->ffn_up,   f->x_norm, wu, dim, ffn);
            for (int i = 0; i < ffn; i++) {
                float g = f->ffn_gate[i];
                float sig = 1.0f / (1.0f + expf(-g));
                f->ffn_gate[i] = g * sig * f->ffn_up[i];
            }
            matmul(f->ffn_mid, f->ffn_gate, wd, ffn, dim);
            for (int i = 0; i < dim; i++) xt[i] += f->ffn_mid[i];
        }
    }

    // 3. Final RMSNorm + tied-embedding logits
    rmsnorm(f->x_norm,
            f->x_buf + (size_t)(n_tokens - 1) * dim,
            f->w_norm, dim, eps);
    matmul(logits_out, f->x_norm, f->w_token_embd, dim, vocab);

    return 0;
}

int forward_vocab_size(const forward_ctx* f) {
    return f ? f->vocab_size : 0;
}

void forward_reset(forward_ctx* f) {
    if (!f) return;
    memset(f->k_cache, 0,
           (size_t)f->n_layers * f->max_seq * f->kv_dim * sizeof(float));
    memset(f->v_cache, 0,
           (size_t)f->n_layers * f->max_seq * f->kv_dim * sizeof(float));
}

int forward_decode(forward_ctx* f, int token, int pos, float* logits_out) {
    if (!f || token < 0 || token >= f->vocab_size ||
        pos < 0 || pos >= f->max_seq || !logits_out)
        return -1;

    const int dim   = f->dim;
    const int nl    = f->n_layers;
    const int nh    = f->n_heads;
    const int nkv   = f->n_kv_heads;
    const int hd    = f->head_dim;
    const int kvdim = f->kv_dim;
    const int ffn   = f->ffn_hidden;
    const int vocab = f->vocab_size;
    const float eps = f->rms_eps;
    const size_t cache_stride = (size_t)f->max_seq * kvdim;
    const float  inv_sqrt_hd  = 1.0f / sqrtf((float)hd);

    float* x = f->x_buf;  /* reuse first slot */
    memcpy(x, f->w_token_embd + (size_t)token * dim, dim * sizeof(float));

    for (int L = 0; L < nl; L++) {
        const float* wq  = f->w_q        + (size_t)L * dim   * dim;
        const float* wk  = f->w_k        + (size_t)L * kvdim * dim;
        const float* wv  = f->w_v        + (size_t)L * kvdim * dim;
        const float* wo  = f->w_o        + (size_t)L * dim   * dim;
        const float* wan = f->w_attn_norm + (size_t)L * dim;
        const float* wfn = f->w_ffn_norm + (size_t)L * dim;
        const float* wg  = f->w_gate     + (size_t)L * ffn * dim;
        const float* wu  = f->w_up       + (size_t)L * ffn * dim;
        const float* wd  = f->w_down     + (size_t)L * dim * ffn;
        float* kc = f->k_cache + (size_t)L * cache_stride;
        float* vc = f->v_cache + (size_t)L * cache_stride;

        rmsnorm(f->x_norm, x, wan, dim, eps);
        matmul(f->q, f->x_norm, wq, dim, dim);
        matmul(f->k, f->x_norm, wk, dim, kvdim);
        matmul(f->v, f->x_norm, wv, dim, kvdim);
        rope_neox(f->q, pos, nh,  hd, f->rope_theta);
        rope_neox(f->k, pos, nkv, hd, f->rope_theta);
        memcpy(kc + (size_t)pos * kvdim, f->k, kvdim * sizeof(float));
        memcpy(vc + (size_t)pos * kvdim, f->v, kvdim * sizeof(float));

        for (int h = 0; h < nh; h++) {
            int kvh = h * nkv / nh;
            const float* qh = f->q + h * hd;
            float* oh = f->attn_out + h * hd;

            float max_s = -INFINITY;
            for (int s = 0; s <= pos; s++) {
                const float* ks = kc + (size_t)s * kvdim + kvh * hd;
                float d = 0.0f;
                for (int i = 0; i < hd; i++) d += qh[i] * ks[i];
                d *= inv_sqrt_hd;
                f->scores[s] = d;
                if (d > max_s) max_s = d;
            }
            float sum = 0.0f;
            for (int s = 0; s <= pos; s++) {
                float e = expf(f->scores[s] - max_s);
                f->scores[s] = e;
                sum += e;
            }
            float inv_sum = (sum > 0.0f) ? 1.0f / sum : 0.0f;
            for (int i = 0; i < hd; i++) oh[i] = 0.0f;
            for (int s = 0; s <= pos; s++) {
                float w = f->scores[s] * inv_sum;
                const float* vs = vc + (size_t)s * kvdim + kvh * hd;
                for (int i = 0; i < hd; i++) oh[i] += w * vs[i];
            }
        }

        matmul(f->o_proj, f->attn_out, wo, dim, dim);
        for (int i = 0; i < dim; i++) x[i] += f->o_proj[i];

        rmsnorm(f->x_norm, x, wfn, dim, eps);
        matmul(f->ffn_gate, f->x_norm, wg, dim, ffn);
        matmul(f->ffn_up,   f->x_norm, wu, dim, ffn);
        for (int i = 0; i < ffn; i++) {
            float g = f->ffn_gate[i];
            float sig = 1.0f / (1.0f + expf(-g));
            f->ffn_gate[i] = g * sig * f->ffn_up[i];
        }
        matmul(f->ffn_mid, f->ffn_gate, wd, ffn, dim);
        for (int i = 0; i < dim; i++) x[i] += f->ffn_mid[i];
    }

    rmsnorm(f->x_norm, x, f->w_norm, dim, eps);
    matmul(logits_out, f->x_norm, f->w_token_embd, dim, vocab);
    return 0;
}

void forward_free(forward_ctx* f) {
    if (!f) return;
    free(f->w_token_embd);
    free(f->w_norm);
    free(f->w_attn_norm);
    free(f->w_q);
    free(f->w_k);
    free(f->w_v);
    free(f->w_o);
    free(f->w_ffn_norm);
    free(f->w_gate);
    free(f->w_up);
    free(f->w_down);
    free(f->k_cache);
    free(f->v_cache);
    free(f->x_buf);
    free(f->x_norm);
    free(f->q);
    free(f->k);
    free(f->v);
    free(f->attn_out);
    free(f->o_proj);
    free(f->ffn_gate);
    free(f->ffn_up);
    free(f->ffn_mid);
    free(f->scores);
    free(f);
}
