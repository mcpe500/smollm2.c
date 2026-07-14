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

/* Process-wide mode defaults; applied at forward_load. */
static int g_rope_mode = ROPE_F32;
static int g_kv_mode   = KV_F32;
static int g_attn_mode = ATTN_NAIVE;

void forward_set_modes(int rope, int kv, int attn) {
    g_rope_mode = rope;
    g_kv_mode   = kv;
    g_attn_mode = attn;
}

struct forward_ctx {
    // config
    int dim, n_layers, n_heads, n_kv_heads, head_dim, kv_dim;
    int ffn_hidden, vocab_size, max_seq;
    float rms_eps, rope_theta;
    int rope_mode, kv_mode, attn_mode;

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

    // KV cache — only one precision active (selected by kv_mode)
    float*    k_cache;       // F32
    float*    v_cache;
    uint16_t* k_cache_f16;   // F16
    uint16_t* v_cache_f16;
    int8_t*   k_cache_q8;    // Q8 + per-(L,pos) scale
    int8_t*   v_cache_q8;
    float*    k_cache_scale; // [n_layers * max_seq]
    float*    v_cache_scale;
    float*    kv_tmp;        // [kv_dim] dequant scratch

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
    /* Second slot for 2-token batched prefill FFN */
    int8_t* xq_buf2;      // [max(dim, ffn_hidden)] second activation buffer
    float*  ffn_gate2;    // [ffn_hidden]  second gate output
    float*  ffn_up2;      // [ffn_hidden]  second up output
    float* scores;        // [max_seq]
    float* rope_cos;      // [max_seq * head_dim/2]  precomputed RoPE cosines
    float* rope_sin;      // [max_seq * head_dim/2]  precomputed RoPE sines
    uint16_t* rope_cos_f16;
    uint16_t* rope_sin_f16;
    int8_t*   rope_cos_q8;
    int8_t*   rope_sin_q8;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#define Q8_BLOCK 64  /* per-block INT8 quantization block size */

/* Fast exp via Schraudolph bit trick: ~10x faster than libm expf.
   Error: ~1.7% max, acceptable for sigmoid gating and attention softmax. */
/* libm expf — Schraudolph distorts attention softmax across 30 layers. */
static inline float fast_expf(float x) {
    return expf(x);
}
/* libm sigmoid — Schraudolph fails for |x|>10 which FFN activations hit. */
static inline float fast_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* Quantize a float row to int8 per-tensor: single amax scale for the whole vector.
   dst_scale = amax / 127.0f (single float)
   Weights keep per-block-64 scale; activations use per-tensor for lower overhead. */
#ifdef __ARM_NEON
static float quantize_row_to_q8_tensor(int8_t* dst, const float* src, int n) {
    /* Phase 1: global abs-max scan */
    float32x4_t mx0 = vdupq_n_f32(0), mx1 = vdupq_n_f32(0);
    float32x4_t mx2 = vdupq_n_f32(0), mx3 = vdupq_n_f32(0);
    int i = 0;
    for (; i <= n - 16; i += 16) {
        mx0 = vmaxq_f32(mx0, vabsq_f32(vld1q_f32(src+i)));
        mx1 = vmaxq_f32(mx1, vabsq_f32(vld1q_f32(src+i+4)));
        mx2 = vmaxq_f32(mx2, vabsq_f32(vld1q_f32(src+i+8)));
        mx3 = vmaxq_f32(mx3, vabsq_f32(vld1q_f32(src+i+12)));
    }
    float amax = vmaxvq_f32(vmaxq_f32(vmaxq_f32(mx0,mx1), vmaxq_f32(mx2,mx3)));
    for (; i < n; i++) { float a = src[i] < 0 ? -src[i] : src[i]; if (a > amax) amax = a; }
    float scale = (amax > 0.0f) ? amax / 127.0f : 1.0f;
    float inv   = (amax > 0.0f) ? 127.0f / amax : 0.0f;
    float32x4_t vinv = vdupq_n_f32(inv);
    int32x4_t lo = vdupq_n_s32(-127), hi = vdupq_n_s32(127);
    i = 0;
    for (; i <= n - 16; i += 16) {
        int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(src+i),    vinv));
        int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(src+i+4),  vinv));
        int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(src+i+8),  vinv));
        int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(src+i+12), vinv));
        i0 = vmaxq_s32(lo, vminq_s32(hi, i0)); i1 = vmaxq_s32(lo, vminq_s32(hi, i1));
        i2 = vmaxq_s32(lo, vminq_s32(hi, i2)); i3 = vmaxq_s32(lo, vminq_s32(hi, i3));
        int16x8_t s01 = vcombine_s16(vmovn_s32(i0), vmovn_s32(i1));
        int16x8_t s23 = vcombine_s16(vmovn_s32(i2), vmovn_s32(i3));
        vst1q_s8(dst+i, vcombine_s8(vmovn_s16(s01), vmovn_s16(s23)));
    }
    for (; i < n; i++) {
        int q = (int)(src[i] * inv + 0.5f);
        if (q > 127) q = 127; if (q < -127) q = -127;
        dst[i] = (int8_t)q;
    }
    /* Zero-pad to next Q8_BLOCK boundary for safe NEON reads in matmul */
    for (int k = n; k < ((n + Q8_BLOCK - 1) & ~(Q8_BLOCK-1)); k++) dst[k] = 0;
    return scale;
}
#else
static float quantize_row_to_q8_tensor(int8_t* dst, const float* src, int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) { float a = src[i]<0?-src[i]:src[i]; if(a>amax)amax=a; }
    float scale = (amax>0.0f)?amax/127.0f:1.0f;
    float inv   = (amax>0.0f)?127.0f/amax:0.0f;
    for (int i = 0; i < n; i++) {
        int q = (int)roundf(src[i]*inv); if(q>127)q=127; if(q<-127)q=-127; dst[i]=(int8_t)q;
    }
    return scale;
}
#endif

/* per-block activation quantize: dead code, excluded from build */
#if 0
static int quantize_row_to_q8_blocked(int8_t* dst, float* dst_scales,
                                       const float* src, int n) {
    int n_blocks = (n + Q8_BLOCK - 1) / Q8_BLOCK;
    for (int b = 0; b < n_blocks; b++) {
        int start = b * Q8_BLOCK;
        const float* s = src + start;
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
#endif /* #if 0 block */

static uint16_t f32_to_f16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint16_t sign = (uint16_t)((u >> 16) & 0x8000u);
    int exp = (int)((u >> 23) & 0xff) - 127 + 15;
    uint32_t mant = u & 0x7fffffu;
    if (exp <= 0)  return sign;
    if (exp >= 31) return sign | 0x7c00u;
    return (uint16_t)(sign | (uint16_t)(exp << 10) | (uint16_t)(mant >> 13));
}

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
    if (t->dtype == GGUF_DT_F16) {
        memcpy(dst, gguf_tensor_data(g, t), count * sizeof(uint16_t));
        return 0;
    }
    if (t->dtype == GGUF_DT_Q8_0) {
        /* Q8_0: 32-element blocks with F16 scale + int8 weights.
           Convert to F16 for compatibility with existing INT8 quantization path. */
        const uint8_t* src = (const uint8_t*)gguf_tensor_data(g, t);
        size_t n_blocks = (count + 31) / 32;
        for (size_t b = 0; b < n_blocks; b++) {
            uint16_t scale_f16;
            memcpy(&scale_f16, src, 2);
            float scale = f16_to_f32(scale_f16);
            const int8_t* q = (const int8_t*)(src + 2);
            size_t block_size = (b == n_blocks - 1) ? (count - b * 32) : 32;
            for (size_t i = 0; i < block_size; i++) {
                float val = (float)q[i] * scale;
                dst[b * 32 + i] = f32_to_f16(val);
            }
            src += 34;  /* 2 bytes scale + 32 bytes int8 */
        }
        return 0;
    }
    fprintf(stderr, "forward: %s expected F16, got unsupported dtype %u\n",
            name, (unsigned)t->dtype);
    return -1;
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

/* Quantize a F16 weight matrix to INT8 per-row: one scale per output row.
   dst_scale: [out_rows]
   Faster at inference than per-block-64 (no per-block hadd/scale in matmul). */
static void quantize_f16_rows_to_q8_perrow(int8_t* dst_q, float* dst_scale,
                                            const uint16_t* src_f16,
                                            int out_rows, int in_cols) {
    for (int o = 0; o < out_rows; o++) {
        const uint16_t* row = src_f16 + (size_t)o * in_cols;
        float amax = 0.0f;
        for (int i = 0; i < in_cols; i++) {
            float v = f16_to_f32(row[i]);
            float a = v < 0 ? -v : v; if (a > amax) amax = a;
        }
        float scale = (amax > 0.0f) ? amax / 127.0f : 1.0f;
        float inv   = (amax > 0.0f) ? 127.0f / amax : 0.0f;
        dst_scale[o] = scale;
        int8_t* drow = dst_q + (size_t)o * in_cols;
        for (int i = 0; i < in_cols; i++) {
            float v = f16_to_f32(row[i]);
            int q = (int)(v * inv + 0.5f);
            if (q > 127) q = 127; if (q < -127) q = -127;
            drow[i] = (int8_t)q;
        }
    }
}



/* Per-row weight scale matmul: W uses single scale per row (faster than per-block-64).
   Accumulates ALL elements in one tight loop: no per-block hadd/scale overhead.
   W must be quantized with per-row scale: wscales[o] = max_abs(W[o]) / 127.0f */
__attribute__((target("+dotprod")))
static void matmul_q8_dot_perrow(float* y, const int8_t* xq, float x_scale,
                                  const int8_t* W, const float* wscales,
                                  int in_dim, int out_dim) {
#ifdef __ARM_NEON
    for (int o = 0; o < out_dim; o++) {
        const int8_t* wr = W + (size_t)o * in_dim;
        int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0), acc3 = vdupq_n_s32(0);
        int i = 0;
        /* Unroll: process 128 bytes per iteration to hide loop overhead */
        for (; i <= in_dim - 128; i += 128) {
            acc0 = vdotq_s32(acc0, vld1q_s8(wr+i),     vld1q_s8(xq+i));
            acc1 = vdotq_s32(acc1, vld1q_s8(wr+i+16),  vld1q_s8(xq+i+16));
            acc2 = vdotq_s32(acc2, vld1q_s8(wr+i+32),  vld1q_s8(xq+i+32));
            acc3 = vdotq_s32(acc3, vld1q_s8(wr+i+48),  vld1q_s8(xq+i+48));
            acc0 = vdotq_s32(acc0, vld1q_s8(wr+i+64),  vld1q_s8(xq+i+64));
            acc1 = vdotq_s32(acc1, vld1q_s8(wr+i+80),  vld1q_s8(xq+i+80));
            acc2 = vdotq_s32(acc2, vld1q_s8(wr+i+96),  vld1q_s8(xq+i+96));
            acc3 = vdotq_s32(acc3, vld1q_s8(wr+i+112), vld1q_s8(xq+i+112));
        }
        for (; i <= in_dim - 64; i += 64) {
            acc0 = vdotq_s32(acc0, vld1q_s8(wr+i),    vld1q_s8(xq+i));
            acc1 = vdotq_s32(acc1, vld1q_s8(wr+i+16), vld1q_s8(xq+i+16));
            acc2 = vdotq_s32(acc2, vld1q_s8(wr+i+32), vld1q_s8(xq+i+32));
            acc3 = vdotq_s32(acc3, vld1q_s8(wr+i+48), vld1q_s8(xq+i+48));
        }
        int32_t sum32 = vaddvq_s32(vaddq_s32(vaddq_s32(acc0,acc1),vaddq_s32(acc2,acc3)));
        y[o] = (float)sum32 * x_scale * wscales[o];
    }
#else
    for (int o = 0; o < out_dim; o++) {
        const int8_t* wr = W + (size_t)o * in_dim;
        int32_t sum32 = 0;
        for (int i = 0; i < in_dim; i++) sum32 += (int32_t)wr[i] * (int32_t)xq[i];
        y[o] = (float)sum32 * x_scale * wscales[o];
    }
#endif
}

/* Batched matmul: processes 2 activation vectors per W pass.
   Loads W[o] once, computes dot with both xq0 and xq1 simultaneously.
   Halves W-read overhead vs calling matmul_q8_dot_perrow twice.
   For 6-token prefill called in 3 pairs: W read 3x instead of 6x. */
__attribute__((target("+dotprod")))
static void matmul_q8_2batch(float* y0, float* y1,
                              const int8_t* xq0, float xs0,
                              const int8_t* xq1, float xs1,
                              const int8_t* W, const float* wscales,
                              int in_dim, int out_dim) {
#ifdef __ARM_NEON
    for (int o = 0; o < out_dim; o++) {
        const int8_t* wr = W + (size_t)o * in_dim;
        /* 8 accumulators: 4 for t0, 4 for t1 */
        int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
        int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
        int32x4_t b0 = vdupq_n_s32(0), b1 = vdupq_n_s32(0);
        int32x4_t b2 = vdupq_n_s32(0), b3 = vdupq_n_s32(0);
        int i = 0;
        /* 64-byte inner loop: 4 W regs + 8 acc regs = 12 regs, no spills */
        for (; i <= in_dim - 64; i += 64) {
            int8x16_t w0 = vld1q_s8(wr+i),    w1 = vld1q_s8(wr+i+16);
            int8x16_t w2 = vld1q_s8(wr+i+32), w3 = vld1q_s8(wr+i+48);
            /* Token 0 */
            a0 = vdotq_s32(a0, w0, vld1q_s8(xq0+i));   a1 = vdotq_s32(a1, w1, vld1q_s8(xq0+i+16));
            a2 = vdotq_s32(a2, w2, vld1q_s8(xq0+i+32)); a3 = vdotq_s32(a3, w3, vld1q_s8(xq0+i+48));
            /* Token 1 */
            b0 = vdotq_s32(b0, w0, vld1q_s8(xq1+i));   b1 = vdotq_s32(b1, w1, vld1q_s8(xq1+i+16));
            b2 = vdotq_s32(b2, w2, vld1q_s8(xq1+i+32)); b3 = vdotq_s32(b3, w3, vld1q_s8(xq1+i+48));
        }
        float ws_o = wscales[o];
        y0[o] = (float)vaddvq_s32(vaddq_s32(vaddq_s32(a0,a1),vaddq_s32(a2,a3))) * xs0 * ws_o;
        y1[o] = (float)vaddvq_s32(vaddq_s32(vaddq_s32(b0,b1),vaddq_s32(b2,b3))) * xs1 * ws_o;
    }
#else
    matmul_q8_dot_perrow(y0, xq0, xs0, W, wscales, in_dim, out_dim);
    matmul_q8_dot_perrow(y1, xq1, xs1, W, wscales, in_dim, out_dim);
#endif
}

/* LLaMA RoPE (interleaved pairs): rotate (2i, 2i+1) per head.
   SmolLM2 GGUF is arch=llama — NOT GPT-NeoX half-split (i, i+hd/2).
   Wrong layout flips Hello/I ranking on ChatML prompts. */
static void rope_llama_table(float* v, int pos, int n_heads, int head_dim,
                             const float* rope_cos, const float* rope_sin) {
    int half = head_dim / 2;
    const float* cv = rope_cos + (size_t)pos * half;
    const float* sv = rope_sin + (size_t)pos * half;
    for (int h = 0; h < n_heads; h++) {
        float* vh = v + (size_t)h * head_dim;
        for (int i = 0; i < half; i++) {
            float a = vh[2 * i], b = vh[2 * i + 1];
            vh[2 * i]     = a * cv[i] - b * sv[i];
            vh[2 * i + 1] = a * sv[i] + b * cv[i];
        }
    }
}

static void rope_llama_table_f16(float* v, int pos, int n_heads, int head_dim,
                                 const uint16_t* rope_cos,
                                 const uint16_t* rope_sin) {
    int half = head_dim / 2;
    const uint16_t* cv = rope_cos + (size_t)pos * half;
    const uint16_t* sv = rope_sin + (size_t)pos * half;
    for (int h = 0; h < n_heads; h++) {
        float* vh = v + (size_t)h * head_dim;
        for (int i = 0; i < half; i++) {
            float c = f16_to_f32(cv[i]);
            float s = f16_to_f32(sv[i]);
            float a = vh[2 * i], b = vh[2 * i + 1];
            vh[2 * i]     = a * c - b * s;
            vh[2 * i + 1] = a * s + b * c;
        }
    }
}

static void rope_llama_table_q8(float* v, int pos, int n_heads, int head_dim,
                                const int8_t* rope_cos,
                                const int8_t* rope_sin) {
    int half = head_dim / 2;
    const int8_t* cv = rope_cos + (size_t)pos * half;
    const int8_t* sv = rope_sin + (size_t)pos * half;
    const float INV = 1.0f / 127.0f;
    for (int h = 0; h < n_heads; h++) {
        float* vh = v + (size_t)h * head_dim;
        for (int i = 0; i < half; i++) {
            float c = (float)cv[i] * INV;
            float s = (float)sv[i] * INV;
            float a = vh[2 * i], b = vh[2 * i + 1];
            vh[2 * i]     = a * c - b * s;
            vh[2 * i + 1] = a * s + b * c;
        }
    }
}

static void rope_apply(float* v, int pos, int n_heads, int head_dim,
                       const forward_ctx* f) {
    switch (f->rope_mode) {
    case ROPE_F16:
        rope_llama_table_f16(v, pos, n_heads, head_dim,
                             f->rope_cos_f16, f->rope_sin_f16);
        break;
    case ROPE_Q8:
        rope_llama_table_q8(v, pos, n_heads, head_dim,
                            f->rope_cos_q8, f->rope_sin_q8);
        break;
    default:
        rope_llama_table(v, pos, n_heads, head_dim, f->rope_cos, f->rope_sin);
        break;
    }
}

/* ---- KV store/load helpers.
   Q8 scale is per-(L,pos); data offset within layer = pos*kv_dim + off. ---- */
static inline void kv_store_row(int kv_mode, int kv_dim,
                                float* cf32, uint16_t* cf16,
                                int8_t* cq8, float* cq8_scale,
                                int pos, int off, const float* src, int n) {
    size_t didx = (size_t)pos * kv_dim + off;
    if (kv_mode == KV_F16) {
        uint16_t* d = cf16 + didx;
        for (int i = 0; i < n; i++) d[i] = f32_to_f16(src[i]);
    } else if (kv_mode == KV_Q8) {
        float amax = 0.0f;
        for (int i = 0; i < n; i++) {
            float a = src[i] < 0 ? -src[i] : src[i];
            if (a > amax) amax = a;
        }
        float sc  = (amax > 0.0f) ? amax / 127.0f : 1.0f;
        float inv = (amax > 0.0f) ? 127.0f / amax : 0.0f;
        cq8_scale[pos] = sc;
        int8_t* d = cq8 + didx;
        for (int i = 0; i < n; i++) {
            int v = (int)(src[i] * inv + (src[i] >= 0 ? 0.5f : -0.5f));
            if (v > 127) v = 127; if (v < -127) v = -127;
            d[i] = (int8_t)v;
        }
    } else {
        memcpy(cf32 + didx, src, n * sizeof(float));
    }
}

static inline void kv_load_row(int kv_mode, int kv_dim,
                               const float* cf32, const uint16_t* cf16,
                               const int8_t* cq8, const float* cq8_scale,
                               int pos, int off, float* out, int n) {
    size_t didx = (size_t)pos * kv_dim + off;
    if (kv_mode == KV_F16) {
        const uint16_t* s = cf16 + didx;
        for (int i = 0; i < n; i++) out[i] = f16_to_f32(s[i]);
    } else if (kv_mode == KV_Q8) {
        float sc = cq8_scale[pos];
        const int8_t* s = cq8 + didx;
        for (int i = 0; i < n; i++) out[i] = (float)s[i] * sc;
    } else {
        memcpy(out, cf32 + didx, n * sizeof(float));
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
    f->rope_mode = g_rope_mode;
    f->kv_mode   = g_kv_mode;
    f->attn_mode = g_attn_mode;

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
    /* Per-row logit weights: one scale per vocab row (faster inference) */
    f->q8_embd          = malloc(s_embd * sizeof(int8_t));
    f->q8_sembd         = malloc((size_t)f->vocab_size * sizeof(float));
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
    /* Per-row scale: 1 float per output row (simpler, faster than per-block-64) */
    size_t sq_sc   = (size_t)f->n_layers * f->dim;         /* one scale per q-row */
    size_t skv_sc  = (size_t)f->n_layers * f->kv_dim;      /* one scale per k/v-row */
    size_t sgu_sc  = (size_t)f->n_layers * f->ffn_hidden;  /* one scale per gate/up row */
    size_t sdn_sc  = (size_t)f->n_layers * f->dim;         /* one scale per down row */
    f->q8_q    = malloc(s_q    * sizeof(int8_t));  f->q8_sq    = malloc(sq_sc  * sizeof(float));
    f->q8_k    = malloc(s_kv   * sizeof(int8_t));  f->q8_sk    = malloc(skv_sc * sizeof(float));
    f->q8_v    = malloc(s_kv   * sizeof(int8_t));  f->q8_sv    = malloc(skv_sc * sizeof(float));
    f->q8_o    = malloc(s_q    * sizeof(int8_t));  f->q8_so    = malloc(sq_sc  * sizeof(float));
    f->q8_gate = malloc(s_gu   * sizeof(int8_t));  f->q8_sgate = malloc(sgu_sc * sizeof(float));
    f->q8_up   = malloc(s_gu   * sizeof(int8_t));  f->q8_sup   = malloc(sgu_sc * sizeof(float));
    f->q8_down = malloc(s_down * sizeof(int8_t));  f->q8_sdown = malloc(sdn_sc  * sizeof(float));
    f->xq_buf  = malloc(xq_size * sizeof(int8_t));
    /* KV cache: only allocate the active precision. */
    if (f->kv_mode == KV_F16) {
        f->k_cache_f16 = malloc(s_cache * sizeof(uint16_t));
        f->v_cache_f16 = malloc(s_cache * sizeof(uint16_t));
        f->kv_tmp      = malloc(f->kv_dim * sizeof(float));
    } else if (f->kv_mode == KV_Q8) {
        f->k_cache_q8    = malloc(s_cache * sizeof(int8_t));
        f->v_cache_q8    = malloc(s_cache * sizeof(int8_t));
        f->k_cache_scale = malloc((size_t)f->n_layers * max_seq * sizeof(float));
        f->v_cache_scale = malloc((size_t)f->n_layers * max_seq * sizeof(float));
        f->kv_tmp        = malloc(f->kv_dim * sizeof(float));
    } else {
        f->k_cache = malloc(s_cache * sizeof(float));
        f->v_cache = malloc(s_cache * sizeof(float));
    }

    f->x_buf    = malloc((size_t)max_seq * f->dim      * sizeof(float));
    f->x_norm   = malloc(f->dim         * sizeof(float));
    f->q        = malloc(f->dim         * sizeof(float));
    f->k        = malloc(f->kv_dim      * sizeof(float));
    f->v        = malloc(f->kv_dim      * sizeof(float));
    f->attn_out = malloc(f->dim         * sizeof(float));
    f->o_proj   = malloc(f->dim         * sizeof(float));
    f->ffn_gate = malloc(f->ffn_hidden  * sizeof(float));
    f->ffn_up   = malloc(f->ffn_hidden  * sizeof(float));
    f->ffn_mid   = malloc(f->dim         * sizeof(float));
    f->xq_buf2   = malloc(xq_size        * sizeof(int8_t));
    f->ffn_gate2 = malloc(f->ffn_hidden  * sizeof(float));
    f->ffn_up2   = malloc(f->ffn_hidden  * sizeof(float));
    f->scores   = malloc((size_t)max_seq * sizeof(float));
    int half_hd = f->head_dim / 2;
    /* Always build F32 table; convert to F16/Q8 if selected. */
    f->rope_cos = malloc((size_t)max_seq * half_hd * sizeof(float));
    f->rope_sin = malloc((size_t)max_seq * half_hd * sizeof(float));
    if (f->rope_mode == ROPE_F16) {
        f->rope_cos_f16 = malloc((size_t)max_seq * half_hd * sizeof(uint16_t));
        f->rope_sin_f16 = malloc((size_t)max_seq * half_hd * sizeof(uint16_t));
    } else if (f->rope_mode == ROPE_Q8) {
        f->rope_cos_q8 = malloc((size_t)max_seq * half_hd * sizeof(int8_t));
        f->rope_sin_q8 = malloc((size_t)max_seq * half_hd * sizeof(int8_t));
    }

    int kv_ok = 0;
    if (f->kv_mode == KV_F16)
        kv_ok = f->k_cache_f16 && f->v_cache_f16 && f->kv_tmp;
    else if (f->kv_mode == KV_Q8)
        kv_ok = f->k_cache_q8 && f->v_cache_q8 &&
                f->k_cache_scale && f->v_cache_scale && f->kv_tmp;
    else
        kv_ok = f->k_cache && f->v_cache;

    int rope_ok = f->rope_cos && f->rope_sin;
    if (f->rope_mode == ROPE_F16) rope_ok = rope_ok && f->rope_cos_f16 && f->rope_sin_f16;
    if (f->rope_mode == ROPE_Q8)  rope_ok = rope_ok && f->rope_cos_q8  && f->rope_sin_q8;

    if (!f->w_token_embd || !f->w_token_embd_f16 || !f->q8_embd || !f->q8_sembd ||
        !f->w_norm || !f->w_attn_norm || !f->w_q ||
        !f->w_k || !f->w_v || !f->w_o || !f->w_ffn_norm ||
        !f->w_gate || !f->w_up || !f->w_down ||
        !f->q8_q || !f->q8_sq || !f->q8_k || !f->q8_sk || !f->q8_v || !f->q8_sv ||
        !f->q8_o || !f->q8_so || !f->q8_gate || !f->q8_sgate ||
        !f->q8_up || !f->q8_sup || !f->q8_down || !f->q8_sdown || !f->xq_buf ||
        !kv_ok ||
        !f->x_buf || !f->x_norm || !f->q || !f->k || !f->v ||
        !f->attn_out || !f->o_proj || !f->ffn_gate || !f->ffn_up ||
        !f->ffn_mid || !f->xq_buf2 || !f->ffn_gate2 || !f->ffn_up2 ||
        !f->scores || !rope_ok) {
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
            quantize_f16_rows_to_q8_perrow(f->q8_embd, f->q8_sembd, src,
                                           f->vocab_size, f->dim);
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
        size_t off_q_sc  = (size_t)L * f->dim;
        size_t off_kv_sc = (size_t)L * f->kv_dim;
        size_t off_gu_sc = (size_t)L * f->ffn_hidden;
        size_t off_dn_sc = (size_t)L * f->dim;
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight",   L);
        rc |= load_tensor_f32(g, name, f->w_attn_norm + off_n, (size_t)f->dim);
        snprintf(name, sizeof(name), "blk.%d.attn_q.weight",     L);
        rc |= load_tensor_f16(g, name, f->w_q        + off_q,  (size_t)f->dim    * f->dim);
        if (!rc) quantize_f16_rows_to_q8_perrow(f->q8_q+off_q, f->q8_sq+off_q_sc, f->w_q+off_q, f->dim, f->dim);
        snprintf(name, sizeof(name), "blk.%d.attn_k.weight",     L);
        rc |= load_tensor_f16(g, name, f->w_k        + off_kv, (size_t)f->kv_dim * f->dim);
        if (!rc) quantize_f16_rows_to_q8_perrow(f->q8_k+off_kv, f->q8_sk+off_kv_sc, f->w_k+off_kv, f->kv_dim, f->dim);
        snprintf(name, sizeof(name), "blk.%d.attn_v.weight",     L);
        rc |= load_tensor_f16(g, name, f->w_v        + off_kv, (size_t)f->kv_dim * f->dim);
        if (!rc) quantize_f16_rows_to_q8_perrow(f->q8_v+off_kv, f->q8_sv+off_kv_sc, f->w_v+off_kv, f->kv_dim, f->dim);
        snprintf(name, sizeof(name), "blk.%d.attn_output.weight",L);
        rc |= load_tensor_f16(g, name, f->w_o        + off_q,  (size_t)f->dim    * f->dim);
        if (!rc) quantize_f16_rows_to_q8_perrow(f->q8_o+off_q, f->q8_so+off_q_sc, f->w_o+off_q, f->dim, f->dim);
        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight",   L);
        rc |= load_tensor_f32(g, name, f->w_ffn_norm + off_n,  (size_t)f->dim);
        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight",   L);
        rc |= load_tensor_f16(g, name, f->w_gate     + off_gu, (size_t)f->ffn_hidden * f->dim);
        if (!rc) quantize_f16_rows_to_q8_perrow(f->q8_gate+off_gu, f->q8_sgate+off_gu_sc, f->w_gate+off_gu, f->ffn_hidden, f->dim);
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight",     L);
        rc |= load_tensor_f16(g, name, f->w_up       + off_gu, (size_t)f->ffn_hidden * f->dim);
        if (!rc) quantize_f16_rows_to_q8_perrow(f->q8_up+off_gu, f->q8_sup+off_gu_sc, f->w_up+off_gu, f->ffn_hidden, f->dim);
        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight",   L);
        rc |= load_tensor_f16(g, name, f->w_down     + off_d,  (size_t)f->dim    * f->ffn_hidden);
        if (!rc) quantize_f16_rows_to_q8_perrow(f->q8_down+off_d, f->q8_sdown+off_dn_sc, f->w_down+off_d, f->dim, f->ffn_hidden);
    }

    if (rc != 0) {
        forward_free(f);
        return -1;
    }

    /* Precompute RoPE cos/sin tables for all positions up to max_seq. */
    {
        int half = f->head_dim / 2;
        size_t n = (size_t)max_seq * half;
        for (int pos = 0; pos < max_seq; pos++) {
            for (int i = 0; i < half; i++) {
                float freq = 1.0f / powf(f->rope_theta, (2.0f * (float)i) / (float)f->head_dim);
                float angle = (float)pos * freq;
                f->rope_cos[pos * half + i] = cosf(angle);
                f->rope_sin[pos * half + i] = sinf(angle);
            }
        }
        if (f->rope_mode == ROPE_F16) {
            for (size_t k = 0; k < n; k++) {
                f->rope_cos_f16[k] = f32_to_f16(f->rope_cos[k]);
                f->rope_sin_f16[k] = f32_to_f16(f->rope_sin[k]);
            }
            free(f->rope_cos); f->rope_cos = NULL;
            free(f->rope_sin); f->rope_sin = NULL;
        } else if (f->rope_mode == ROPE_Q8) {
            for (size_t k = 0; k < n; k++) {
                f->rope_cos_q8[k] = (int8_t)(f->rope_cos[k] * 127.0f + (f->rope_cos[k] >= 0 ? 0.5f : -0.5f));
                f->rope_sin_q8[k] = (int8_t)(f->rope_sin[k] * 127.0f + (f->rope_sin[k] >= 0 ? 0.5f : -0.5f));
            }
            free(f->rope_cos); f->rope_cos = NULL;
            free(f->rope_sin); f->rope_sin = NULL;
        }
    }

    /* Free F16 after quant — dual residency thrash (~600MB RSS → ~300MB).
       Prefill + decode both INT8. F32 embd kept for lookup + prefill logits. */
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

    /* Prefill: full INT8 + 2-batch FFN. F32 logits (safer ranking). */
    for (int L = 0; L < nl; L++) {
        const int8_t* q8q = f->q8_q    + (size_t)L * dim   * dim;
        const float*  sq  = f->q8_sq   + (size_t)L * dim;
        const int8_t* q8k = f->q8_k    + (size_t)L * kvdim * dim;
        const float*  sk  = f->q8_sk   + (size_t)L * kvdim;
        const int8_t* q8v = f->q8_v    + (size_t)L * kvdim * dim;
        const float*  sv  = f->q8_sv   + (size_t)L * kvdim;
        const int8_t* q8o = f->q8_o    + (size_t)L * dim   * dim;
        const float*  so  = f->q8_so   + (size_t)L * dim;
        const float*  wan = f->w_attn_norm + (size_t)L * dim;
        const float*  wfn = f->w_ffn_norm  + (size_t)L * dim;
        const int8_t* q8g = f->q8_gate + (size_t)L * ffn  * dim;
        const float*  sg  = f->q8_sgate+ (size_t)L * ffn;
        const int8_t* q8u = f->q8_up   + (size_t)L * ffn  * dim;
        const float*  su  = f->q8_sup  + (size_t)L * ffn;
        const int8_t* q8d = f->q8_down + (size_t)L * dim  * ffn;
        const float*  sd  = f->q8_sdown+ (size_t)L * dim;
        float* kc = f->k_cache ? f->k_cache + (size_t)L * cache_stride : NULL;
        float* vc = f->v_cache ? f->v_cache + (size_t)L * cache_stride : NULL;
        uint16_t* kc_f16 = f->k_cache_f16 ? f->k_cache_f16 + (size_t)L * cache_stride : NULL;
        uint16_t* vc_f16 = f->v_cache_f16 ? f->v_cache_f16 + (size_t)L * cache_stride : NULL;
        int8_t*   kc_q8  = f->k_cache_q8  ? f->k_cache_q8  + (size_t)L * cache_stride : NULL;
        int8_t*   vc_q8  = f->v_cache_q8  ? f->v_cache_q8  + (size_t)L * cache_stride : NULL;
        float*    ks     = f->k_cache_scale ? f->k_cache_scale + (size_t)L * f->max_seq : NULL;
        float*    vs     = f->v_cache_scale ? f->v_cache_scale + (size_t)L * f->max_seq : NULL;

        for (int t = 0; t < n_tokens; t++) {
            float* xt = f->x_buf + (size_t)t * dim;

            rmsnorm(f->x_norm, xt, wan, dim, eps);
            float xn_sc = quantize_row_to_q8_tensor(f->xq_buf, f->x_norm, dim);
            matmul_q8_dot_perrow(f->q, f->xq_buf, xn_sc, q8q, sq, dim, dim);
            matmul_q8_dot_perrow(f->k, f->xq_buf, xn_sc, q8k, sk, dim, kvdim);
            matmul_q8_dot_perrow(f->v, f->xq_buf, xn_sc, q8v, sv, dim, kvdim);
            rope_apply(f->q, t, nh,  hd, f);
            rope_apply(f->k, t, nkv, hd, f);
            kv_store_row(f->kv_mode, kvdim, kc, kc_f16, kc_q8, ks, t, 0, f->k, kvdim);
            kv_store_row(f->kv_mode, kvdim, vc, vc_f16, vc_q8, vs, t, 0, f->v, kvdim);

            for (int h = 0; h < nh; h++) {
                int kvh = h * nkv / nh;
                const float* qh = f->q + h * hd;
                float* oh = f->attn_out + h * hd;

                float max_s = -INFINITY;
                for (int s = 0; s <= t; s++) {
                    float ks_local[64];
                    kv_load_row(f->kv_mode, kvdim, kc, kc_f16, kc_q8, ks,
                                s, kvh * hd, ks_local, hd);
                    float d = 0.0f;
                    for (int i = 0; i < hd; i++) d += qh[i] * ks_local[i];
                    d *= inv_sqrt_hd;
                    f->scores[s] = d;
                    if (d > max_s) max_s = d;
                }
                float sum = 0.0f;
                for (int s = 0; s <= t; s++) {
                    float e = fast_expf(f->scores[s] - max_s);
                    f->scores[s] = e;
                    sum += e;
                }
                float inv_sum = (sum > 0.0f) ? 1.0f / sum : 0.0f;

                for (int i = 0; i < hd; i++) oh[i] = 0.0f;
                for (int s = 0; s <= t; s++) {
                    float w = f->scores[s] * inv_sum;
                    float vs_local[64];
                    kv_load_row(f->kv_mode, kvdim, vc, vc_f16, vc_q8, vs,
                                s, kvh * hd, vs_local, hd);
                    for (int i = 0; i < hd; i++) oh[i] += w * vs_local[i];
                }
            }

            float ao_sc = quantize_row_to_q8_tensor(f->xq_buf, f->attn_out, dim);
            matmul_q8_dot_perrow(f->o_proj, f->xq_buf, ao_sc, q8o, so, dim, dim);
            for (int i = 0; i < dim; i++) xt[i] += f->o_proj[i];
        }

        for (int t = 0; t < n_tokens; t += 2) {
            float* xt0 = f->x_buf + (size_t)t * dim;
            if (t + 1 < n_tokens) {
                float* xt1 = f->x_buf + (size_t)(t + 1) * dim;
                rmsnorm(f->x_norm, xt0, wfn, dim, eps);
                float xs0 = quantize_row_to_q8_tensor(f->xq_buf,  f->x_norm, dim);
                rmsnorm(f->x_norm, xt1, wfn, dim, eps);
                float xs1 = quantize_row_to_q8_tensor(f->xq_buf2, f->x_norm, dim);
                matmul_q8_2batch(f->ffn_gate, f->ffn_gate2,
                                 f->xq_buf, xs0, f->xq_buf2, xs1,
                                 q8g, sg, dim, ffn);
                matmul_q8_2batch(f->ffn_up, f->ffn_up2,
                                 f->xq_buf, xs0, f->xq_buf2, xs1,
                                 q8u, su, dim, ffn);
                for (int i = 0; i < ffn; i++) {
                    float g0 = f->ffn_gate[i];
                    f->ffn_gate[i]  = g0 * fast_sigmoid(g0) * f->ffn_up[i];
                    float g1 = f->ffn_gate2[i];
                    f->ffn_gate2[i] = g1 * fast_sigmoid(g1) * f->ffn_up2[i];
                }
                float fg0 = quantize_row_to_q8_tensor(f->xq_buf,  f->ffn_gate,  ffn);
                float fg1 = quantize_row_to_q8_tensor(f->xq_buf2, f->ffn_gate2, ffn);
                matmul_q8_2batch(f->ffn_mid, f->ffn_gate2,
                                 f->xq_buf, fg0, f->xq_buf2, fg1,
                                 q8d, sd, ffn, dim);
                for (int i = 0; i < dim; i++) xt0[i] += f->ffn_mid[i];
                for (int i = 0; i < dim; i++) xt1[i] += f->ffn_gate2[i];
            } else {
                rmsnorm(f->x_norm, xt0, wfn, dim, eps);
                float xf_sc = quantize_row_to_q8_tensor(f->xq_buf, f->x_norm, dim);
                matmul_q8_dot_perrow(f->ffn_gate, f->xq_buf, xf_sc, q8g, sg, dim, ffn);
                matmul_q8_dot_perrow(f->ffn_up,   f->xq_buf, xf_sc, q8u, su, dim, ffn);
                for (int i = 0; i < ffn; i++) {
                    float g = f->ffn_gate[i];
                    f->ffn_gate[i] = g * fast_sigmoid(g) * f->ffn_up[i];
                }
                float fg_sc = quantize_row_to_q8_tensor(f->xq_buf, f->ffn_gate, ffn);
                matmul_q8_dot_perrow(f->ffn_mid, f->xq_buf, fg_sc, q8d, sd, ffn, dim);
                for (int i = 0; i < dim; i++) xt0[i] += f->ffn_mid[i];
            }
        }
    }

    rmsnorm(f->x_norm,
            f->x_buf + (size_t)(n_tokens - 1) * dim,
            f->w_norm, dim, eps);
    matmul_f32(logits_out, f->x_norm, f->w_token_embd, dim, vocab);
    return 0;
}

int forward_vocab_size(const forward_ctx* f) {
    return f ? f->vocab_size : 0;
}

const float* forward_last_hidden(const forward_ctx* f) {
    return f ? f->x_norm : NULL;
}

int forward_dim(const forward_ctx* f) {
    return f ? f->dim : 0;
}

void forward_reset(forward_ctx* f) {
    if (!f) return;
    size_t cache_n = (size_t)f->n_layers * f->max_seq * f->kv_dim;
    size_t scale_n = (size_t)f->n_layers * f->max_seq;
    if (f->kv_mode == KV_F16) {
        memset(f->k_cache_f16, 0, cache_n * sizeof(uint16_t));
        memset(f->v_cache_f16, 0, cache_n * sizeof(uint16_t));
    } else if (f->kv_mode == KV_Q8) {
        memset(f->k_cache_q8, 0, cache_n * sizeof(int8_t));
        memset(f->v_cache_q8, 0, cache_n * sizeof(int8_t));
        memset(f->k_cache_scale, 0, scale_n * sizeof(float));
        memset(f->v_cache_scale, 0, scale_n * sizeof(float));
    } else {
        memset(f->k_cache, 0, cache_n * sizeof(float));
        memset(f->v_cache, 0, cache_n * sizeof(float));
    }
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

    /* Decode: full INT8 matmul (parity only gates prefill first-token). */
    for (int L = 0; L < nl; L++) {
        const int8_t* q8q = f->q8_q    + (size_t)L * dim   * dim;
        const float*  sq  = f->q8_sq   + (size_t)L * dim;
        const int8_t* q8k = f->q8_k    + (size_t)L * kvdim * dim;
        const float*  sk  = f->q8_sk   + (size_t)L * kvdim;
        const int8_t* q8v = f->q8_v    + (size_t)L * kvdim * dim;
        const float*  sv  = f->q8_sv   + (size_t)L * kvdim;
        const int8_t* q8o = f->q8_o    + (size_t)L * dim   * dim;
        const float*  so  = f->q8_so   + (size_t)L * dim;
        const float*  wan = f->w_attn_norm + (size_t)L * dim;
        const float*  wfn = f->w_ffn_norm  + (size_t)L * dim;
        const int8_t* q8g = f->q8_gate + (size_t)L * ffn * dim;
        const float*  sg  = f->q8_sgate+ (size_t)L * ffn;
        const int8_t* q8u = f->q8_up   + (size_t)L * ffn * dim;
        const float*  su  = f->q8_sup  + (size_t)L * ffn;
        const int8_t* q8d = f->q8_down + (size_t)L * dim * ffn;
        const float*  sd  = f->q8_sdown+ (size_t)L * dim;
        float* kc = f->k_cache + (size_t)L * cache_stride;
        float* vc = f->v_cache + (size_t)L * cache_stride;
        uint16_t* kc_f16 = f->k_cache_f16 + (size_t)L * cache_stride;
        uint16_t* vc_f16 = f->v_cache_f16 + (size_t)L * cache_stride;
        int8_t*   kc_q8  = f->k_cache_q8  + (size_t)L * cache_stride;
        int8_t*   vc_q8  = f->v_cache_q8  + (size_t)L * cache_stride;
        float*    ks     = f->k_cache_scale + (size_t)L * f->max_seq;
        float*    vs     = f->v_cache_scale + (size_t)L * f->max_seq;

        rmsnorm(f->x_norm, x, wan, dim, eps);
        float xa_sc = quantize_row_to_q8_tensor(f->xq_buf, f->x_norm, dim);
        matmul_q8_dot_perrow(f->q, f->xq_buf, xa_sc, q8q, sq, dim, dim);
        matmul_q8_dot_perrow(f->k, f->xq_buf, xa_sc, q8k, sk, dim, kvdim);
        matmul_q8_dot_perrow(f->v, f->xq_buf, xa_sc, q8v, sv, dim, kvdim);
        rope_apply(f->q, pos, nh,  hd, f);
        rope_apply(f->k, pos, nkv, hd, f);
        kv_store_row(f->kv_mode, kvdim, kc, kc_f16, kc_q8, ks, pos, 0, f->k, kvdim);
        kv_store_row(f->kv_mode, kvdim, vc, vc_f16, vc_q8, vs, pos, 0, f->v, kvdim);

        for (int h = 0; h < nh; h++) {
            int kvh = h * nkv / nh;
            const float* qh = f->q + h * hd;
            float* oh = f->attn_out + h * hd;

            float max_s = -INFINITY;
            for (int s = 0; s <= pos; s++) {
                float ks_local[64];
                kv_load_row(f->kv_mode, kvdim, kc, kc_f16, kc_q8, ks,
                            s, kvh * hd, ks_local, hd);
#ifdef __ARM_NEON
                float32x4_t a0=vdupq_n_f32(0),a1=vdupq_n_f32(0);
                float32x4_t a2=vdupq_n_f32(0),a3=vdupq_n_f32(0);
                for (int i = 0; i <= hd-16; i += 16) {
                    a0=vfmaq_f32(a0,vld1q_f32(qh+i),   vld1q_f32(ks_local+i));
                    a1=vfmaq_f32(a1,vld1q_f32(qh+i+4), vld1q_f32(ks_local+i+4));
                    a2=vfmaq_f32(a2,vld1q_f32(qh+i+8), vld1q_f32(ks_local+i+8));
                    a3=vfmaq_f32(a3,vld1q_f32(qh+i+12),vld1q_f32(ks_local+i+12));
                }
                float d = vaddvq_f32(vaddq_f32(vaddq_f32(a0,a1),vaddq_f32(a2,a3))) * inv_sqrt_hd;
#else
                float d = 0.0f;
                for (int i = 0; i < hd; i++) d += qh[i] * ks_local[i];
                d *= inv_sqrt_hd;
#endif
                f->scores[s] = d; if (d > max_s) max_s = d;
            }
            float sum = 0.0f;
            for (int s = 0; s <= pos; s++) {
                float e = fast_expf(f->scores[s] - max_s);
                f->scores[s] = e;
                sum += e;
            }
            float inv_sum = (sum > 0.0f) ? 1.0f / sum : 0.0f;
            for (int i = 0; i < hd; i++) oh[i] = 0.0f;
            for (int s = 0; s <= pos; s++) {
                float w = f->scores[s] * inv_sum;
                float vs_local[64];
                kv_load_row(f->kv_mode, kvdim, vc, vc_f16, vc_q8, vs,
                            s, kvh * hd, vs_local, hd);
#ifdef __ARM_NEON
                float32x4_t vw = vdupq_n_f32(w);
                for (int i = 0; i <= hd-16; i += 16) {
                    vst1q_f32(oh+i,    vfmaq_f32(vld1q_f32(oh+i),    vw, vld1q_f32(vs_local+i)));
                    vst1q_f32(oh+i+4,  vfmaq_f32(vld1q_f32(oh+i+4),  vw, vld1q_f32(vs_local+i+4)));
                    vst1q_f32(oh+i+8,  vfmaq_f32(vld1q_f32(oh+i+8),  vw, vld1q_f32(vs_local+i+8)));
                    vst1q_f32(oh+i+12, vfmaq_f32(vld1q_f32(oh+i+12), vw, vld1q_f32(vs_local+i+12)));
                }
#else
                for (int i = 0; i < hd; i++) oh[i] += w * vs_local[i];
#endif
            }
        }

        float ao_sc = quantize_row_to_q8_tensor(f->xq_buf, f->attn_out, dim);
        matmul_q8_dot_perrow(f->o_proj, f->xq_buf, ao_sc, q8o, so, dim, dim);
        for (int i = 0; i < dim; i++) x[i] += f->o_proj[i];

        rmsnorm(f->x_norm, x, wfn, dim, eps);
        float xf_sc = quantize_row_to_q8_tensor(f->xq_buf, f->x_norm, dim);
        matmul_q8_dot_perrow(f->ffn_gate, f->xq_buf, xf_sc, q8g, sg, dim, ffn);
        matmul_q8_dot_perrow(f->ffn_up,   f->xq_buf, xf_sc, q8u, su, dim, ffn);
        for (int i = 0; i < ffn; i++) {
            float g = f->ffn_gate[i];
            f->ffn_gate[i] = g * fast_sigmoid(g) * f->ffn_up[i];
        }
        float fg_sc = quantize_row_to_q8_tensor(f->xq_buf, f->ffn_gate, ffn);
        matmul_q8_dot_perrow(f->ffn_mid, f->xq_buf, fg_sc, q8d, sd, ffn, dim);
        for (int i = 0; i < dim; i++) x[i] += f->ffn_mid[i];
    }

    rmsnorm(f->x_norm, x, f->w_norm, dim, eps);
    /* Decode logits: q8 embd (halves 113MB F32 bandwidth per step). */
    float xl_sc = quantize_row_to_q8_tensor(f->xq_buf, f->x_norm, dim);
    matmul_q8_dot_perrow(logits_out, f->xq_buf, xl_sc,
                         f->q8_embd, f->q8_sembd, dim, vocab);
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
    free(f->k_cache_f16);  free(f->v_cache_f16);
    free(f->k_cache_q8);   free(f->v_cache_q8);
    free(f->k_cache_scale); free(f->v_cache_scale);
    free(f->kv_tmp);
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
    free(f->xq_buf2); free(f->ffn_gate2); free(f->ffn_up2);
    free(f->scores); free(f->rope_cos); free(f->rope_sin);
    free(f->rope_cos_f16); free(f->rope_sin_f16);
    free(f->rope_cos_q8);  free(f->rope_sin_q8);
    free(f);
}
