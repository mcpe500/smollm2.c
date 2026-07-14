// attn_registry.h — per-layer attention flavor registry
//
// Phase 3a lands SWA. Other variants (DILATED, BIGBIRD, GLOCAL, MLA) are
// stub types here; their behaviors land in 3b.
#ifndef ATTN_REGISTRY_H
#define ATTN_REGISTRY_H

typedef enum {
    ATTN_TYPE_DENSE   = 0,
    ATTN_TYPE_SWA     = 1,
    ATTN_TYPE_DILATED = 2,
    ATTN_TYPE_BIGBIRD = 3,
    ATTN_TYPE_GLOCAL  = 4,
    ATTN_TYPE_MLA     = 5
} attn_type;

typedef struct {
    int type;
    int window;        /* SWA window in tokens (0 = full / dense) */
    int dilation;      /* stride for dilated (3b) */
    int n_global;      /* number of leading "global" tokens (3b) */
    int latent_dim;    /* MLA latent compression dim (3b) */
} attn_spec;

/* Hot-path lookup. Pure const function — caller inlines if hot. */
int attn_s_start(int L, int t);

/* Configuration setters. Call once before forward_load (or any prefill). */
void attn_set_default_spec(int type, int window, int dilation,
                           int n_global, int latent_dim);
int  attn_set_per_layer(const attn_spec* specs, int n);
int  attn_load_config(const char* json_path, int n_layers_expected);

/* Inspect current registry (for --attn-info / studio attn-list). */
int  attn_get_spec(int L, attn_spec* out);
int  attn_n_layers(void);

/* Reset (mainly for tests). */
void attn_reset(void);

#endif // ATTN_REGISTRY_H
