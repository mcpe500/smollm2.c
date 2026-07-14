// attn_registry.c — per-layer attention flavor lookup
#include "attn_registry.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Default: dense / full attention across all positions. */
static attn_spec g_specs[256];
static int       g_n_layers = 0;
static attn_spec g_default  = { ATTN_TYPE_DENSE, 0, 1, 0, 0 };
static int       g_inited   = 0;

static void ensure_inited(void) {
    if (g_inited) return;
    g_inited = 1;
    /* g_specs filled lazily by attn_set_per_layer or on demand. */
}

void attn_reset(void) {
    g_n_layers = 0;
    g_default.type = ATTN_TYPE_DENSE;
    g_default.window = 0;
    g_default.dilation = 1;
    g_default.n_global = 0;
    g_default.latent_dim = 0;
    g_inited = 0;
}

void attn_set_default_spec(int type, int window, int dilation,
                           int n_global, int latent_dim) {
    g_default.type       = type;
    g_default.window     = window;
    g_default.dilation   = dilation > 0 ? dilation : 1;
    g_default.n_global   = n_global;
    g_default.latent_dim = latent_dim;
}

int attn_set_per_layer(const attn_spec* specs, int n) {
    if (n <= 0 || n > (int)(sizeof(g_specs)/sizeof(g_specs[0]))) return -1;
    ensure_inited();
    for (int i = 0; i < n; i++) g_specs[i] = specs[i];
    /* Fill remaining with default. */
    for (int i = n; i < g_n_layers; i++) g_specs[i] = g_default;
    g_n_layers = n;
    return n;
}

int attn_get_spec(int L, attn_spec* out) {
    if (!out) return -1;
    ensure_inited();
    if (L < 0 || L >= g_n_layers) { *out = g_default; return -1; }
    *out = g_specs[L];
    return 0;
}

int attn_n_layers(void) { ensure_inited(); return g_n_layers; }

/* Hot path: compute first valid attention index for position t at layer L. */
int attn_s_start(int L, int t) {
    ensure_inited();
    attn_spec s;
    if (L < 0 || L >= g_n_layers) s = g_default;
    else s = g_specs[L];
    int total = t + 1;
    int start = 0;
    switch (s.type) {
    case ATTN_TYPE_SWA: {
        int w = s.window > 0 ? s.window : 0;
        start = total - w;
        if (start < 0) start = 0;
        if (start > t) start = t;  /* degenerates to single-token self */
        break;
    }
    case ATTN_TYPE_DILATED: {
        /* Phase 3b: attend at stride. For 3a, keep dense semantics. */
        start = 0;
        break;
    }
    case ATTN_TYPE_BIGBIRD:
    case ATTN_TYPE_GLOCAL:
    case ATTN_TYPE_MLA:
        start = 0;  /* Phase 3b */
        break;
    case ATTN_TYPE_DENSE:
    default:
        start = 0;
        break;
    }
    return start;
}

/* Tiny JSON config loader — expects:
   {
     "default": {"type":"dense"|"swa", ...},
     "layers":  [{"type":"swa", "window":128}, ...]
   }
   Conservative parser; skips whitespace + brace/bracket noise. */
int attn_load_config(const char* json_path, int n_layers_expected) {
    if (!json_path) return -1;
    FILE* f = fopen(json_path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz > 1024 * 1024) { fclose(f); return -1; }
    char* buf = malloc(sz + 1);
    if (!buf) { fclose(f); return -1; }
    fread(buf, 1, sz, f);
    buf[sz] = '\0';
    fclose(f);

    /* Parse "default": {"type":"...", "window":N} (best-effort). */
    char* p = strstr(buf, "\"default\"");
    int parsed_default = 0;
    if (p) {
        p = strchr(p, '{');
        if (p) {
            p++;
            char* t_str = strstr(p, "\"type\"");
            int  type_v = ATTN_TYPE_DENSE;
            if (t_str) {
                char* q = strchr(t_str, ':'); if (q) q = strchr(q, '"');
                if (q) {
                    q++;
                    if (strncmp(q, "swa", 3) == 0)            type_v = ATTN_TYPE_SWA;
                    else if (strncmp(q, "dilated", 7) == 0)  type_v = ATTN_TYPE_DILATED;
                    else if (strncmp(q, "bigbird", 7) == 0)  type_v = ATTN_TYPE_BIGBIRD;
                    else if (strncmp(q, "glocal", 6) == 0)   type_v = ATTN_TYPE_GLOCAL;
                    else if (strncmp(q, "mla", 3) == 0)      type_v = ATTN_TYPE_MLA;
                    else                                      type_v = ATTN_TYPE_DENSE;
                }
            }
            int win_v = 0, dil_v = 1, glob_v = 0, lat_v = 0;
            char* wn = strstr(p, "\"window\"");
            if (wn) { char* q = strchr(wn, ':'); if (q) win_v = atoi(q + 1); }
            char* dl = strstr(p, "\"dilation\"");
            if (dl) { char* q = strchr(dl, ':'); if (q) dil_v = atoi(q + 1); }
            char* gl = strstr(p, "\"n_global\"");
            if (gl) { char* q = strchr(gl, ':'); if (q) glob_v = atoi(q + 1); }
            char* lt = strstr(p, "\"latent_dim\"");
            if (lt) { char* q = strchr(lt, ':'); if (q) lat_v = atoi(q + 1); }
            attn_set_default_spec(type_v, win_v, dil_v, glob_v, lat_v);
            parsed_default = 1;
        }
    }

    /* Parse "layers": [ {...}, {...} ] — count "{...}" objects in array. */
    char* arr = strstr(buf, "\"layers\"");
    int n = 0;
    attn_spec tmp[256];
    memset(tmp, 0, sizeof(tmp));
    if (arr) {
        char* bp = strchr(arr, '[');
        if (bp) {
            bp++;
            while (*bp && *bp != ']' && n < (int)(sizeof(tmp)/sizeof(tmp[0]))) {
                while (*bp && *bp != '{') bp++;
                if (!*bp || *bp == ']') break;
                char* obj_s = bp;
                int depth = 0;
                while (*bp) {
                    if (*bp == '{') depth++;
                    else if (*bp == '}') { depth--; if (depth == 0) { bp++; break; } }
                    bp++;
                }
                /* Parse fields in [obj_s, bp). */
                int type_v = ATTN_TYPE_DENSE;
                int win_v = 0, dil_v = 1, glob_v = 0, lat_v = 0;
                char* t_str = strstr(obj_s, "\"type\"");
                if (t_str && t_str < bp) {
                    char* q = strchr(t_str, ':'); if (q && q < bp) q = strchr(q, '"');
                    if (q && q < bp) {
                        q++;
                        if (strncmp(q, "swa", 3) == 0)            type_v = ATTN_TYPE_SWA;
                        else if (strncmp(q, "dilated", 7) == 0)  type_v = ATTN_TYPE_DILATED;
                        else if (strncmp(q, "bigbird", 7) == 0)  type_v = ATTN_TYPE_BIGBIRD;
                        else if (strncmp(q, "glocal", 6) == 0)   type_v = ATTN_TYPE_GLOCAL;
                        else if (strncmp(q, "mla", 3) == 0)      type_v = ATTN_TYPE_MLA;
                        else                                      type_v = ATTN_TYPE_DENSE;
                    }
                }
                char* wn = strstr(obj_s, "\"window\"");
                if (wn && wn < bp) { char* q = strchr(wn, ':'); if (q && q < bp) win_v = atoi(q + 1); }
                char* dl = strstr(obj_s, "\"dilation\"");
                if (dl && dl < bp) { char* q = strchr(dl, ':'); if (q && q < bp) dil_v = atoi(q + 1); }
                char* gl = strstr(obj_s, "\"n_global\"");
                if (gl && gl < bp) { char* q = strchr(gl, ':'); if (q && q < bp) glob_v = atoi(q + 1); }
                char* lt = strstr(obj_s, "\"latent_dim\"");
                if (lt && lt < bp) { char* q = strchr(lt, ':'); if (q && q < bp) lat_v = atoi(q + 1); }
                tmp[n].type       = type_v;
                tmp[n].window     = win_v;
                tmp[n].dilation   = dil_v > 0 ? dil_v : 1;
                tmp[n].n_global   = glob_v;
                tmp[n].latent_dim = lat_v;
                n++;
            }
        }
    }

    free(buf);
    if (parsed_default) {
        /* Apply default to all layers then override per-layer. */
        for (int i = 0; i < n_layers_expected; i++) g_specs[i] = g_default;
        g_n_layers = n_layers_expected;
        for (int i = 0; i < n && i < n_layers_expected; i++) g_specs[i] = tmp[i];
        return n_layers_expected;
    }
    if (n > 0) return attn_set_per_layer(tmp, n);
    return 0;
}
