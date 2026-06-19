// forward.c — SmolLM2 transformer forward pass (F16 weights, NEON matmul)

#include "forward.h"
#include "gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// RoPE theta read from GGUF at load time (SmolLM2 uses 100000, not 10000)

struct forward_ctx {
    // config
    int dim, n_layers, n_heads, n_kv_heads, head_dim, kv_dim;
    int ffn_hidden, vocab_size, max_seq;
    float rms_eps, rope_theta;

    // weights: token_embd/norms stored F32; heavy matmul weights kept as F16
    float*    w_token_embd;     // [vocab * dim]  F32 — used for embedding lookup
    uint16_t* w_token_embd_f16; // [vocab * dim]  F16 — kept as fallback
    int8_t*   q8_embd;          // [vocab * dim]  INT8 — used for logit projection
    float*    q8_sembd;         // [vocab]         per-row scale for logit INT8
    float*    w_norm;        // [dim]                F32
    float*    w_attn_norm;   // [n_layers * dim]     F32
    uint16_t* w_q;           // [n_layers * dim * dim]        F16 (kept for prefill)
    uint16_t* w_k;           // [n_layers * kv_dim * dim]     F16
    uint16_t* w_v;           // [n_layers * kv_dim * dim]     F16
    uint16_t* w_o;           // [n_layers * dim * dim]        F16
    float*    w_ffn_norm;    // [n_layers * dim]     F32
    uint16_t* w_gate;        // [n_layers * ffn_hidden * dim] F16
    uint16_t* w_up;          // [n_layers * ffn_hidden * dim] F16
    uint16_t* w_down;        // [n_layers * dim * ffn_hidden] F16

    /* INT8 quantized weights + per-row scales for fast vdotq_s32 decode */
    int8_t*  q8_q;      float* q8_sq;    // [n_layers * dim * dim]
    int8_t*  q8_k;      float* q8_sk;    // [n_layers * kv_dim * dim]
    int8_t*  q8_v;      float* q8_sv;    // [n_layers * kv_dim * dim]
    int8_t*  q8_o;      float* q8_so;    // [n_layers * dim * dim]
    int8_t*  q8_gate;   float* q8_sgate; // [n_layers * ffn_hidden * dim]
    int8_t*  q8_up;     float* q8_sup;   // [n_layers * ffn_hidden * dim]
    int8_t*  q8_down;   float* q8_sdown; // [n_layers * dim * ffn_hidden]
    int8_t*  xq_buf;    // [max(dim, ffn_hidden)] activation quantization buffer

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

#define Q8_BLOCK 64  /* per-block INT8 quantization block size */

/* Fast sigmoid via Schraudolph exp bit trick: ~10x faster than libm expf.
   Error: ~1.7% max, negligible for SiLU gating in FFN. */
static inline float fast_sigmoid(float x) {
    union { float f; int32_t i; } u;
    u.i = (int32_t)(12102203.1875f * (-x) + 1065353216.0f);
    float eg = u.f; /* e^(-x) approximation */
    return 1.0f / (1.0f + eg);
}

/* Quantize a float row to int8 per-block-64 for activation quantization.
   dst_scales[b] = amax(block b) / 127.0f
   Returns number of blocks. */
#ifdef __ARM_NEON
static int quantize_row_to_q8_blocked(int8_t* dst, float* dst_scales,
                                       const float* src, int n) {
    int n_blocks = (n + Q8_BLOCK - 1) / Q8_BLOCK;
    for (int b = 0; b < n_blocks; b++) {
        int start = b * Q8_BLOCK;
        /* Block always exactly Q8_BLOCK elements (zero-padded at alloc); safe to read */
        const float* s = src + start;
        /* NEON abs-max over 64 floats = 16 vld + 16 vabs + 15 vmax + 1 horizontal */
        float32x4_t mx0 = vdupq_n_f32(0), mx1 = vdupq_n_f32(0);
        float32x4_t mx2 = vdupq_n_f32(0), mx3 = vdupq_n_f32(0);
        int nleft = n - start; if (nleft > Q8_BLOCK) nleft = Q8_BLOCK;
        int i = 0;
        for (; i <= nleft - 16; i += 16) {
            mx0 = vmaxq_f32(mx0, vabsq_f32(vld1q_f32(s+i)));
            mx1 = vmaxq_f32(mx1, vabsq_f32(vld1q_f32(s+i+4)));
            mx2 = vmaxq_f32(mx2, vabsq_f32(vld1q_f32(s+i+8)));
            mx3 = vmaxq_f32(mx3, vabsq_f32(vld1q_f32(s+i+12)));
        }
        float32x4_t mx = vmaxq_f32(vmaxq_f32(mx0,mx1), vmaxq_f32(mx2,mx3));
        float amax = vmaxvq_f32(mx);
        for (; i < nleft; i++) { float a = s[i] < 0 ? -s[i] : s[i]; if (a > amax) amax = a; }
        float scale = (amax > 0.0f) ? amax / 127.0f : 1.0f;
        float inv   = (amax > 0.0f) ? 127.0f / amax : 0.0f;
        dst_scales[b] = scale;
        /* Quantize: round-to-nearest, clamp to [-127, 127] */
        int8_t* d = dst + start;
        float32x4_t vinv = vdupq_n_f32(inv);
        int j = 0;
        for (; j <= nleft - 16; j += 16) {
            float32x4_t v0 = vmulq_f32(vld1q_f32(s+j),    vinv);
            float32x4_t v1 = vmulq_f32(vld1q_f32(s+j+4),  vinv);
            float32x4_t v2 = vmulq_f32(vld1q_f32(s+j+8),  vinv);
            float32x4_t v3 = vmulq_f32(vld1q_f32(s+j+12), vinv);
            /* Round to nearest int */
            int32x4_t i0 = vcvtnq_s32_f32(v0); int32x4_t i1 = vcvtnq_s32_f32(v1);
            int32x4_t i2 = vcvtnq_s32_f32(v2); int32x4_t i3 = vcvtnq_s32_f32(v3);
            /* Clamp to [-127,127] */
            int32x4_t lo = vdupq_n_s32(-127), hi = vdupq_n_s32(127);
            i0 = vmaxq_s32(lo, vminq_s32(hi, i0)); i1 = vmaxq_s32(lo, vminq_s32(hi, i1));
            i2 = vmaxq_s32(lo, vminq_s32(hi, i2)); i3 = vmaxq_s32(lo, vminq_s32(hi, i3));
            /* Pack int32 -> int16 -> int8 */
            int16x8_t s01 = vcombine_s16(vmovn_s32(i0), vmovn_s32(i1));
            int16x8_t s23 = vcombine_s16(vmovn_s32(i2), vmovn_s32(i3));
            vst1q_s8(d+j, vcombine_s8(vmovn_s16(s01), vmovn_s16(s23)));
        }
        for (; j < nleft; j++) {
            int q = (int)(s[j] * inv + 0.5f);
            if (q > 127) q = 127; if (q < -127) q = -127;
            d[j] = (int8_t)q;
        }
        for (int k = nleft; k < Q8_BLOCK; k++) d[k] = 0;
    }
    return n_blocks;
}
#else
static int quantize_row_to_q8_blocked(int8_t* dst, float* dst_scales,
                                       const float* src, int n) {
    int n_blocks = (n + Q8_BLOCK - 1) / Q8_BLOCK;
    for (int b = 0; b < n_blocks; b++) {
        int start = b * Q8_BLOCK;
        int end   = start + Q8_BLOCK < n ? start + Q8_BLOCK : n;
        float amax = 0.0f;
        for (int i = start; i < end; i++) {
            float a = src[i] < 0 ? -src[i] : src[i];
            if (a > amax) amax = a;
        }
        float scale = (amax > 0.0f) ? amax / 127.0f : 1.0f;
        float inv   = (amax > 0.0f) ? 127.0f / amax : 0.0f;
        dst_scales[b] = scale;
        for (int i = start; i < end; i++) {
            int q = (int)(src[i] * inv + 0.5f);
            if (q >  127) q =  127;
            if (q < -127) q = -127;
            dst[i] = (int8_t)q;
        }
        for (int i = end; i < start + Q8_BLOCK; i++) dst[i] = 0;
    }
    return n_blocks;
}
#endif

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
                           uint16_t* dst, size_t count) {
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
    memcpy(dst, gguf_tensor_data(g, t), count * sizeof(uint16_t));
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
#ifdef __ARM_NEON
    float32x4_t vss0 = vdupq_n_f32(0), vss1 = vdupq_n_f32(0);
    float32x4_t vss2 = vdupq_n_f32(0), vss3 = vdupq_n_f32(0);
    int i = 0;
    for (; i <= n - 16; i += 16) {
        float32x4_t v0 = vld1q_f32(x+i),   v1 = vld1q_f32(x+i+4);
        float32x4_t v2 = vld1q_f32(x+i+8), v3 = vld1q_f32(x+i+12);
        vss0 = vfmaq_f32(vss0, v0, v0); vss1 = vfmaq_f32(vss1, v1, v1);
        vss2 = vfmaq_f32(vss2, v2, v2); vss3 = vfmaq_f32(vss3, v3, v3);
    }
    float ss = vaddvq_f32(vaddq_f32(vaddq_f32(vss0,vss1),vaddq_f32(vss2,vss3)));
    for (; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    float32x4_t vinv = vdupq_n_f32(inv);
    i = 0;
    for (; i <= n - 16; i += 16) {
        vst1q_f32(out+i,    vmulq_f32(vmulq_f32(vld1q_f32(x+i),    vld1q_f32(w+i)),    vinv));
        vst1q_f32(out+i+4,  vmulq_f32(vmulq_f32(vld1q_f32(x+i+4),  vld1q_f32(w+i+4)),  vinv));
        vst1q_f32(out+i+8,  vmulq_f32(vmulq_f32(vld1q_f32(x+i+8),  vld1q_f32(w+i+8)),  vinv));
        vst1q_f32(out+i+12, vmulq_f32(vmulq_f32(vld1q_f32(x+i+12), vld1q_f32(w+i+12)), vinv));
    }
    for (; i < n; i++) out[i] = x[i] * w[i] * inv;
#else
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    ss /= n;
    float inv = 1.0f / sqrtf((float)ss + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
#endif
}

// Matmul with F32 weights (used for tied embedding logit projection)
static void matmul_f32(float* y, const float* x, const float* W,
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

// Matmul with F16 weights: y[o] = sum_i f16_to_f32(W[o*in_dim+i]) * x[i]
// Keeps weights in F16 storage, converts on-the-fly matching training precision.
static void matmul(float* y, const float* x, const uint16_t* W,
                   int in_dim, int out_dim) {
#ifdef __ARM_NEON
    for (int o = 0; o < out_dim; o++) {
        const uint16_t* wrow = W + (size_t)o * in_dim;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i <= in_dim - 16; i += 16) {
            acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vld1_f16((const __fp16*)(wrow+i))),    vld1q_f32(x+i));
            acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vld1_f16((const __fp16*)(wrow+i+4))),  vld1q_f32(x+i+4));
            acc2 = vfmaq_f32(acc2, vcvt_f32_f16(vld1_f16((const __fp16*)(wrow+i+8))),  vld1q_f32(x+i+8));
            acc3 = vfmaq_f32(acc3, vcvt_f32_f16(vld1_f16((const __fp16*)(wrow+i+12))), vld1q_f32(x+i+12));
        }
        acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        float sum = vaddvq_f32(acc0);
        for (; i < in_dim; i++) sum += f16_to_f32(wrow[i]) * x[i];
        y[o] = sum;
    }
#else
    for (int o = 0; o < out_dim; o++) {
        const uint16_t* wrow = W + (size_t)o * in_dim;
        double acc = 0.0;
        for (int i = 0; i < in_dim; i++) acc += (double)f16_to_f32(wrow[i]) * (double)x[i];
        y[o] = (float)acc;
    }
#endif
}

/* Quantize a F16 weight matrix to INT8 per-row with scale per row.
   Called once at load time for each weight tensor.
   scale[o] = max_abs_in_row / 127.0f so that w_f = w_q8 * scale */
/* Quantize a F16 weight matrix to INT8 per-block-64 (in the input dimension).
   dst_scale: [out_rows * n_blocks] where n_blocks = ceil(in_cols / Q8_BLOCK) */
static void quantize_f16_rows_to_q8(int8_t* dst_q, float* dst_scale,
                                    const uint16_t* src_f16,
                                    int out_rows, int in_cols) {
    int n_blocks = (in_cols + Q8_BLOCK - 1) / Q8_BLOCK;
    for (int o = 0; o < out_rows; o++) {
        const uint16_t* row = src_f16 + (size_t)o * in_cols;
        int8_t* drow = dst_q + (size_t)o * in_cols;
        float* dscale = dst_scale + (size_t)o * n_blocks;
        for (int b = 0; b < n_blocks; b++) {
            int start = b * Q8_BLOCK;
            int end   = start + Q8_BLOCK < in_cols ? start + Q8_BLOCK : in_cols;
            float amax = 0.0f;
            for (int i = start; i < end; i++) {
                float v = f16_to_f32(row[i]);
                float a = v < 0 ? -v : v;
                if (a > amax) amax = a;
            }
            float scale = (amax > 0.0f) ? amax / 127.0f : 1.0f;
            float inv   = (amax > 0.0f) ? 127.0f / amax : 0.0f;
            dscale[b] = scale;
            for (int i = start; i < end; i++) {
                float v = f16_to_f32(row[i]);
                int q = (int)(v * inv + 0.5f);
                if (q >  127) q =  127;
                if (q < -127) q = -127;
                drow[i] = (int8_t)q;
            }
            for (int i = end; i < start + Q8_BLOCK; i++) drow[i] = 0;
        }
    }
}

/* INT8 x INT8 -> float matmul using vdotq_s32, per-block-64 scaling.
   xq: [in_dim] INT8 activation quantized per-block-64
   x_scales: [ceil(in_dim/64)] float scale per block
   W: [out_dim * in_dim] INT8, wscales: [out_dim * n_blocks] float
   result[o] = sum_b( dot(W[o][b*64..], xq[b*64..]) * x_scales[b] * wscales[o*nblk+b] ) */
__attribute__((target("+dotprod")))
static void matmul_q8_dot(float* y, const int8_t* xq, const float* x_scales,
                          const int8_t* W, const float* wscales,
                          int in_dim, int out_dim) {
    int n_blocks = (in_dim + Q8_BLOCK - 1) / Q8_BLOCK;
#ifdef __ARM_NEON
    for (int o = 0; o < out_dim; o++) {
        const int8_t* wr = W + (size_t)o * in_dim;
        const float*  ws = wscales + (size_t)o * n_blocks;
        float result = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            int off = b * Q8_BLOCK;
            const int8_t* wb = wr + off;
            const int8_t* xb = xq + off;
            int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0), acc3 = vdupq_n_s32(0);
            acc0 = vdotq_s32(acc0, vld1q_s8(wb),    vld1q_s8(xb));
            acc1 = vdotq_s32(acc1, vld1q_s8(wb+16), vld1q_s8(xb+16));
            acc2 = vdotq_s32(acc2, vld1q_s8(wb+32), vld1q_s8(xb+32));
            acc3 = vdotq_s32(acc3, vld1q_s8(wb+48), vld1q_s8(xb+48));
            int32_t sum32 = vaddvq_s32(vaddq_s32(vaddq_s32(acc0,acc1),vaddq_s32(acc2,acc3)));
            result += (float)sum32 * x_scales[b] * ws[b];
        }
        y[o] = result;
    }
#else
    for (int o = 0; o < out_dim; o++) {
        const int8_t* wr = W + (size_t)o * in_dim;
        const float*  ws = wscales + (size_t)o * n_blocks;
        float result = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            int off = b * Q8_BLOCK, end_b = off + Q8_BLOCK < in_dim ? off + Q8_BLOCK : in_dim;
            int32_t sum32 = 0;
            for (int i = off; i < end_b; i++) sum32 += (int32_t)wr[i] * (int32_t)xq[i];
            result += (float)sum32 * x_scales[b] * ws[b];
        }
        y[o] = result;
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
    int xq_size    = f->ffn_hidden > f->dim ? f->ffn_hidden : f->dim;

    f->w_token_embd     = malloc(s_embd * sizeof(float));
    f->w_token_embd_f16 = malloc(s_embd * sizeof(uint16_t));
    f->q8_embd          = malloc(s_embd * sizeof(int8_t));
    int nb_d_embd = (f->dim + Q8_BLOCK - 1) / Q8_BLOCK;
    f->q8_sembd         = malloc((size_t)f->vocab_size * nb_d_embd * sizeof(float));
    f->w_norm       = malloc(f->dim  * sizeof(float));
    f->w_attn_norm  = malloc(s_norm  * sizeof(float));
    f->w_q          = malloc(s_q     * sizeof(uint16_t));
    f->w_k          = malloc(s_kv    * sizeof(uint16_t));
    f->w_v          = malloc(s_kv    * sizeof(uint16_t));
    f->w_o          = malloc(s_q     * sizeof(uint16_t));
    f->w_ffn_norm   = malloc(s_norm  * sizeof(float));
    f->w_gate       = malloc(s_gu    * sizeof(uint16_t));
    f->w_up         = malloc(s_gu    * sizeof(uint16_t));
    f->w_down       = malloc(s_down  * sizeof(uint16_t));
    /* INT8 quantized weights + per-row scales for vdotq_s32 decode */
    /* Per-block-64 scale: n_blocks = ceil(in_cols / Q8_BLOCK) per row */
    int nb_dim = (f->dim        + Q8_BLOCK - 1) / Q8_BLOCK; /* =9 for dim=576 */
    int nb_ffn = (f->ffn_hidden + Q8_BLOCK - 1) / Q8_BLOCK; /* =24 for ffn=1536 */
    size_t sq_sc   = (size_t)f->n_layers * f->dim        * nb_dim;
    size_t skv_sc  = (size_t)f->n_layers * f->kv_dim     * nb_dim;
    size_t sgu_sc  = (size_t)f->n_layers * f->ffn_hidden * nb_dim;
    size_t sdn_sc  = (size_t)f->n_layers * f->dim        * nb_ffn;
    f->q8_q    = malloc(s_q    * sizeof(int8_t));  f->q8_sq    = malloc(sq_sc  * sizeof(float));
    f->q8_k    = malloc(s_kv   * sizeof(int8_t));  f->q8_sk    = malloc(skv_sc * sizeof(float));
    f->q8_v    = malloc(s_kv   * sizeof(int8_t));  f->q8_sv    = malloc(skv_sc * sizeof(float));
    f->q8_o    = malloc(s_q    * sizeof(int8_t));  f->q8_so    = malloc(sq_sc  * sizeof(float));
    f->q8_gate = malloc(s_gu   * sizeof(int8_t));  f->q8_sgate = malloc(sgu_sc * sizeof(float));
    f->q8_up   = malloc(s_gu   * sizeof(int8_t));  f->q8_sup   = malloc(sgu_sc * sizeof(float));
    f->q8_down = malloc(s_down * sizeof(int8_t));  f->q8_sdown = malloc(sdn_sc  * sizeof(float));
    f->xq_buf  = malloc(xq_size * sizeof(int8_t));
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

    if (!f->w_token_embd || !f->w_token_embd_f16 || !f->q8_embd || !f->q8_sembd ||
        !f->w_norm || !f->w_attn_norm || !f->w_q ||
        !f->w_k || !f->w_v || !f->w_o || !f->w_ffn_norm ||
        !f->w_gate || !f->w_up || !f->w_down ||
        !f->q8_q || !f->q8_sq || !f->q8_k || !f->q8_sk || !f->q8_v || !f->q8_sv ||
        !f->q8_o || !f->q8_so || !f->q8_gate || !f->q8_sgate ||
        !f->q8_up || !f->q8_sup || !f->q8_down || !f->q8_sdown || !f->xq_buf ||
        !f->k_cache || !f->v_cache ||
        !f->x_buf || !f->x_norm || !f->q || !f->k || !f->v ||
        !f->attn_out || !f->o_proj || !f->ffn_gate || !f->ffn_up ||
        !f->ffn_mid || !f->scores) {
        fprintf(stderr, "forward: alloc failed\n");
        forward_free(f);
        return -1;
    }

    int rc = 0;
    /* token_embd: keep F32 for embedding lookup, also keep F16 for logit projection (half bandwidth) */
    {
        const gguf_tensor_info* t = gguf_tensor_get(g, "token_embd.weight");
        if (!t) { fprintf(stderr, "forward: missing token_embd.weight\n"); rc = -1; }
        else {
            const uint16_t* src = (const uint16_t*)gguf_tensor_data(g, t);
            memcpy(f->w_token_embd_f16, src, s_embd * sizeof(uint16_t));
            for (size_t i = 0; i < s_embd; i++) f->w_token_embd[i] = f16_to_f32(src[i]);
            /* Also quantize token_embd to INT8 for fast logit projection */
            quantize_f16_rows_to_q8(f->q8_embd, f->q8_sembd, src, f->vocab_size, f->dim);
        }
    }
    rc |= load_tensor_f32(g, "output_norm.weight", f->w_norm,       (size_t)f->dim);

    char name[128];
    for (int L = 0; L < f->n_layers && rc == 0; L++) {
        size_t off_q   = (size_t)L * f->dim    * f->dim;
        size_t off_kv  = (size_t)L * f->kv_dim * f->dim;
        size_t off_n   = (size_t)L * f->dim;
        size_t off_gu  = (size_t)L * f->ffn_hidden * f->dim;
        size_t off_d   = (size_t)L * f->dim    * f->ffn_hidden;

        int nb_d = (f->dim        + Q8_BLOCK - 1) / Q8_BLOCK;
        int nb_f = (f->ffn_hidden + Q8_BLOCK - 1) / Q8_BLOCK;
        size_t off_q_sc  = (size_t)L * f->dim        * nb_d;
        size_t off_kv_sc = (size_t)L * f->kv_dim     * nb_d;
        size_t off_gu_sc = (size_t)L * f->ffn_hidden * nb_d;
        size_t off_dn_sc = (size_t)L * f->dim        * nb_f;
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight",   L);
        rc |= load_tensor_f32(g, name, f->w_attn_norm + off_n, (size_t)f->dim);
        snprintf(name, sizeof(name), "blk.%d.attn_q.weight",     L);
        rc |= load_tensor_f16(g, name, f->w_q        + off_q,  (size_t)f->dim    * f->dim);
        if (!rc) quantize_f16_rows_to_q8(f->q8_q+off_q, f->q8_sq+off_q_sc, f->w_q+off_q, f->dim, f->dim);
        snprintf(name, sizeof(name), "blk.%d.attn_k.weight",     L);
        rc |= load_tensor_f16(g, name, f->w_k        + off_kv, (size_t)f->kv_dim * f->dim);
        if (!rc) quantize_f16_rows_to_q8(f->q8_k+off_kv, f->q8_sk+off_kv_sc, f->w_k+off_kv, f->kv_dim, f->dim);
        snprintf(name, sizeof(name), "blk.%d.attn_v.weight",     L);
        rc |= load_tensor_f16(g, name, f->w_v        + off_kv, (size_t)f->kv_dim * f->dim);
        if (!rc) quantize_f16_rows_to_q8(f->q8_v+off_kv, f->q8_sv+off_kv_sc, f->w_v+off_kv, f->kv_dim, f->dim);
        snprintf(name, sizeof(name), "blk.%d.attn_output.weight",L);
        rc |= load_tensor_f16(g, name, f->w_o        + off_q,  (size_t)f->dim    * f->dim);
        if (!rc) quantize_f16_rows_to_q8(f->q8_o+off_q, f->q8_so+off_q_sc, f->w_o+off_q, f->dim, f->dim);
        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight",   L);
        rc |= load_tensor_f32(g, name, f->w_ffn_norm + off_n,  (size_t)f->dim);
        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight",   L);
        rc |= load_tensor_f16(g, name, f->w_gate     + off_gu, (size_t)f->ffn_hidden * f->dim);
        if (!rc) quantize_f16_rows_to_q8(f->q8_gate+off_gu, f->q8_sgate+off_gu_sc, f->w_gate+off_gu, f->ffn_hidden, f->dim);
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight",     L);
        rc |= load_tensor_f16(g, name, f->w_up       + off_gu, (size_t)f->ffn_hidden * f->dim);
        if (!rc) quantize_f16_rows_to_q8(f->q8_up+off_gu, f->q8_sup+off_gu_sc, f->w_up+off_gu, f->ffn_hidden, f->dim);
        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight",   L);
        rc |= load_tensor_f16(g, name, f->w_down     + off_d,  (size_t)f->dim    * f->ffn_hidden);
        if (!rc) quantize_f16_rows_to_q8(f->q8_down+off_d, f->q8_sdown+off_dn_sc, f->w_down+off_d, f->dim, f->ffn_hidden);
    }

    if (rc != 0) {
        forward_free(f);
        return -1;
    }

    /* Free F16 weight arrays — no longer needed after INT8 quantization.
       Saves ~202 MB RAM and improves cache locality for inference. */
    free(f->w_q);     f->w_q     = NULL;
    free(f->w_k);     f->w_k     = NULL;
    free(f->w_v);     f->w_v     = NULL;
    free(f->w_o);     f->w_o     = NULL;
    free(f->w_gate);  f->w_gate  = NULL;
    free(f->w_up);    f->w_up    = NULL;
    free(f->w_down);  f->w_down  = NULL;
    free(f->w_token_embd_f16); f->w_token_embd_f16 = NULL;

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
    int nb_d = (dim  + Q8_BLOCK - 1) / Q8_BLOCK;
    int nb_f = (ffn + Q8_BLOCK - 1) / Q8_BLOCK;
    for (int L = 0; L < nl; L++) {
        const int8_t* q8q   = f->q8_q    + (size_t)L * dim   * dim;
        const float*  sq    = f->q8_sq   + (size_t)L * dim   * nb_d;
        const int8_t* q8k   = f->q8_k    + (size_t)L * kvdim * dim;
        const float*  sk    = f->q8_sk   + (size_t)L * kvdim * nb_d;
        const int8_t* q8v   = f->q8_v    + (size_t)L * kvdim * dim;
        const float*  sv    = f->q8_sv   + (size_t)L * kvdim * nb_d;
        const int8_t* q8o   = f->q8_o    + (size_t)L * dim   * dim;
        const float*  so    = f->q8_so   + (size_t)L * dim   * nb_d;
        const float*   wan  = f->w_attn_norm + (size_t)L * dim;
        const float*   wfn  = f->w_ffn_norm  + (size_t)L * dim;
        const int8_t* q8g   = f->q8_gate + (size_t)L * ffn  * dim;
        const float*  sg    = f->q8_sgate+ (size_t)L * ffn  * nb_d;
        const int8_t* q8u   = f->q8_up   + (size_t)L * ffn  * dim;
        const float*  su    = f->q8_sup  + (size_t)L * ffn  * nb_d;
        const int8_t* q8d   = f->q8_down + (size_t)L * dim  * ffn;
        const float*  sd    = f->q8_sdown+ (size_t)L * dim  * nb_f;
        float* kc = f->k_cache + (size_t)L * cache_stride;
        float* vc = f->v_cache + (size_t)L * cache_stride;

        for (int t = 0; t < n_tokens; t++) {
            float* xt = f->x_buf + (size_t)t * dim;

            // --- attention block ---
            rmsnorm(f->x_norm, xt, wan, dim, eps);
            float xn_sc[32]; quantize_row_to_q8_blocked(f->xq_buf, xn_sc, f->x_norm, dim);
            matmul_q8_dot(f->q, f->xq_buf, xn_sc, q8q, sq, dim, dim);
            matmul_q8_dot(f->k, f->xq_buf, xn_sc, q8k, sk, dim, kvdim);
            matmul_q8_dot(f->v, f->xq_buf, xn_sc, q8v, sv, dim, kvdim);
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

            float ao_sc_p[32]; quantize_row_to_q8_blocked(f->xq_buf, ao_sc_p, f->attn_out, dim);
            matmul_q8_dot(f->o_proj, f->xq_buf, ao_sc_p, q8o, so, dim, dim);
            for (int i = 0; i < dim; i++) xt[i] += f->o_proj[i];

            // --- FFN block (SwiGLU) ---
            rmsnorm(f->x_norm, xt, wfn, dim, eps);
            float xf_sc[32]; quantize_row_to_q8_blocked(f->xq_buf, xf_sc, f->x_norm, dim);
            matmul_q8_dot(f->ffn_gate, f->xq_buf, xf_sc, q8g, sg, dim, ffn);
            matmul_q8_dot(f->ffn_up,   f->xq_buf, xf_sc, q8u, su, dim, ffn);
            for (int i = 0; i < ffn; i++) {
                float g = f->ffn_gate[i];
                f->ffn_gate[i] = g * fast_sigmoid(g) * f->ffn_up[i];
            }
            float fg_sc[32]; quantize_row_to_q8_blocked(f->xq_buf, fg_sc, f->ffn_gate, ffn);
            matmul_q8_dot(f->ffn_mid, f->xq_buf, fg_sc, q8d, sd, ffn, dim);
            for (int i = 0; i < dim; i++) xt[i] += f->ffn_mid[i];
        }
    }

    // 3. Final RMSNorm + tied-embedding logits
    rmsnorm(f->x_norm,
            f->x_buf + (size_t)(n_tokens - 1) * dim,
            f->w_norm, dim, eps);
    {
        float xls[32]; quantize_row_to_q8_blocked(f->xq_buf, xls, f->x_norm, dim);
        matmul_q8_dot(logits_out, f->xq_buf, xls, f->q8_embd, f->q8_sembd, dim, vocab);
    }

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

    int nb_d = (dim  + Q8_BLOCK - 1) / Q8_BLOCK;  /* =9 for dim=576 */
    int nb_f = (ffn  + Q8_BLOCK - 1) / Q8_BLOCK;  /* =24 for ffn=1536 */
    for (int L = 0; L < nl; L++) {
        const int8_t*  q8q  = f->q8_q    + (size_t)L * dim   * dim;
        const float*   sq   = f->q8_sq   + (size_t)L * dim   * nb_d;
        const int8_t*  q8k  = f->q8_k    + (size_t)L * kvdim * dim;
        const float*   sk   = f->q8_sk   + (size_t)L * kvdim * nb_d;
        const int8_t*  q8v  = f->q8_v    + (size_t)L * kvdim * dim;
        const float*   sv   = f->q8_sv   + (size_t)L * kvdim * nb_d;
        const int8_t*  q8o  = f->q8_o    + (size_t)L * dim   * dim;
        const float*   so   = f->q8_so   + (size_t)L * dim   * nb_d;
        const float*   wan  = f->w_attn_norm + (size_t)L * dim;
        const float*   wfn  = f->w_ffn_norm  + (size_t)L * dim;
        const int8_t*  q8g  = f->q8_gate + (size_t)L * ffn * dim;
        const float*   sg   = f->q8_sgate+ (size_t)L * ffn  * nb_d;
        const int8_t*  q8u  = f->q8_up   + (size_t)L * ffn * dim;
        const float*   su   = f->q8_sup  + (size_t)L * ffn  * nb_d;
        const int8_t*  q8d  = f->q8_down + (size_t)L * dim * ffn;
        const float*   sd   = f->q8_sdown+ (size_t)L * dim  * nb_f;
        float* kc = f->k_cache + (size_t)L * cache_stride;
        float* vc = f->v_cache + (size_t)L * cache_stride;

        rmsnorm(f->x_norm, x, wan, dim, eps);
        /* Quantize x_norm per-block-64, reuse for q/k/v */
        float xn_sc[32]; quantize_row_to_q8_blocked(f->xq_buf, xn_sc, f->x_norm, dim);
        matmul_q8_dot(f->q, f->xq_buf, xn_sc, q8q, sq, dim, dim);
        matmul_q8_dot(f->k, f->xq_buf, xn_sc, q8k, sk, dim, kvdim);
        matmul_q8_dot(f->v, f->xq_buf, xn_sc, q8v, sv, dim, kvdim);
        rope_neox(f->q, pos, nh,  hd, f->rope_theta);
        rope_neox(f->k, pos, nkv, hd, f->rope_theta);
        memcpy(kc + (size_t)pos * kvdim, f->k, kvdim * sizeof(float));
        memcpy(vc + (size_t)pos * kvdim, f->v, kvdim * sizeof(float));

        for (int h = 0; h < nh; h++) {
            int kvh = h * nkv / nh;
            const float* qh = f->q + h * hd;
            float* oh = f->attn_out + h * hd;

            float max_s = -INFINITY;
#ifdef __ARM_NEON
            for (int s = 0; s <= pos; s++) {
                const float* ks = kc + (size_t)s * kvdim + kvh * hd;
                float32x4_t a0=vdupq_n_f32(0),a1=vdupq_n_f32(0);
                float32x4_t a2=vdupq_n_f32(0),a3=vdupq_n_f32(0);
                for (int i = 0; i <= hd-16; i += 16) {
                    a0=vfmaq_f32(a0,vld1q_f32(qh+i),   vld1q_f32(ks+i));
                    a1=vfmaq_f32(a1,vld1q_f32(qh+i+4), vld1q_f32(ks+i+4));
                    a2=vfmaq_f32(a2,vld1q_f32(qh+i+8), vld1q_f32(ks+i+8));
                    a3=vfmaq_f32(a3,vld1q_f32(qh+i+12),vld1q_f32(ks+i+12));
                }
                float d = vaddvq_f32(vaddq_f32(vaddq_f32(a0,a1),vaddq_f32(a2,a3))) * inv_sqrt_hd;
                f->scores[s] = d; if (d > max_s) max_s = d;
            }
#else
            for (int s = 0; s <= pos; s++) {
                const float* ks = kc + (size_t)s * kvdim + kvh * hd;
                float d = 0.0f;
                for (int i = 0; i < hd; i++) d += qh[i] * ks[i];
                d *= inv_sqrt_hd; f->scores[s] = d;
                if (d > max_s) max_s = d;
            }
#endif
            float sum = 0.0f;
            for (int s = 0; s <= pos; s++) {
                float e = expf(f->scores[s] - max_s);
                f->scores[s] = e;
                sum += e;
            }
            float inv_sum = (sum > 0.0f) ? 1.0f / sum : 0.0f;
            for (int i = 0; i < hd; i++) oh[i] = 0.0f;
#ifdef __ARM_NEON
            for (int s = 0; s <= pos; s++) {
                float w = f->scores[s] * inv_sum;
                const float* vs = vc + (size_t)s * kvdim + kvh * hd;
                float32x4_t vw = vdupq_n_f32(w);
                for (int i = 0; i <= hd-16; i += 16) {
                    vst1q_f32(oh+i,    vfmaq_f32(vld1q_f32(oh+i),    vw, vld1q_f32(vs+i)));
                    vst1q_f32(oh+i+4,  vfmaq_f32(vld1q_f32(oh+i+4),  vw, vld1q_f32(vs+i+4)));
                    vst1q_f32(oh+i+8,  vfmaq_f32(vld1q_f32(oh+i+8),  vw, vld1q_f32(vs+i+8)));
                    vst1q_f32(oh+i+12, vfmaq_f32(vld1q_f32(oh+i+12), vw, vld1q_f32(vs+i+12)));
                }
            }
#else
            for (int s = 0; s <= pos; s++) {
                float w = f->scores[s] * inv_sum;
                const float* vs = vc + (size_t)s * kvdim + kvh * hd;
                for (int i = 0; i < hd; i++) oh[i] += w * vs[i];
            }
#endif
        }

        /* Quantize attn_out for output projection */
        float ao_sc[32]; quantize_row_to_q8_blocked(f->xq_buf, ao_sc, f->attn_out, dim);
        matmul_q8_dot(f->o_proj, f->xq_buf, ao_sc, q8o, so, dim, dim);
        for (int i = 0; i < dim; i++) x[i] += f->o_proj[i];

        rmsnorm(f->x_norm, x, wfn, dim, eps);
        /* Quantize x_norm per-block-64 for gate+up */
        float xf_sc[32]; quantize_row_to_q8_blocked(f->xq_buf, xf_sc, f->x_norm, dim);
        matmul_q8_dot(f->ffn_gate, f->xq_buf, xf_sc, q8g, sg, dim, ffn);
        matmul_q8_dot(f->ffn_up,   f->xq_buf, xf_sc, q8u, su, dim, ffn);
        for (int i = 0; i < ffn; i++) {
            float g = f->ffn_gate[i];
            f->ffn_gate[i] = g * fast_sigmoid(g) * f->ffn_up[i];
        }
        /* Quantize ffn_gate (post-SwiGLU) per-block-64 for down projection */
        float fg_sc[32]; quantize_row_to_q8_blocked(f->xq_buf, fg_sc, f->ffn_gate, ffn);
        matmul_q8_dot(f->ffn_mid, f->xq_buf, fg_sc, q8d, sd, ffn, dim);
        for (int i = 0; i < dim; i++) x[i] += f->ffn_mid[i];
    }

    rmsnorm(f->x_norm, x, f->w_norm, dim, eps);
    {
        float xl_sc[32]; quantize_row_to_q8_blocked(f->xq_buf, xl_sc, f->x_norm, dim);
        matmul_q8_dot(logits_out, f->xq_buf, xl_sc, f->q8_embd, f->q8_sembd, dim, vocab);
    }
    return 0;
}

void forward_free(forward_ctx* f) {
    if (!f) return;
    free(f->w_token_embd);
    free(f->w_norm);
    free(f->w_attn_norm);
    free(f->w_q);     free(f->q8_q);    free(f->q8_sq);
    free(f->w_k);     free(f->q8_k);    free(f->q8_sk);
    free(f->w_v);     free(f->q8_v);    free(f->q8_sv);
    free(f->w_o);     free(f->q8_o);    free(f->q8_so);
    free(f->w_ffn_norm);
    free(f->w_gate);  free(f->q8_gate); free(f->q8_sgate);
    free(f->w_up);    free(f->q8_up);   free(f->q8_sup);
    free(f->w_down);  free(f->q8_down); free(f->q8_sdown);
    free(f->xq_buf);
    free(f->k_cache);
    free(f->v_cache);
    free(f->w_token_embd_f16);
    free(f->q8_embd); free(f->q8_sembd);
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
