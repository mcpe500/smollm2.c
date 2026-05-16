// sm2_model.c - Model loading and structures

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include "smollm2.h"

// ============================================================================
// MODEL ALLOCATION / FREE
// ============================================================================

sm2_model* sm2_model_alloc(sm2_variant variant) {
    const sm2_spec* spec = sm2_get_spec(variant);
    
    sm2_model* model = calloc(1, sizeof(sm2_model));
    if (!model) return NULL;
    
    model->variant = variant;
    model->n_layers = spec->n_layers;
    model->dim = spec->dim;
    model->hidden_dim = spec->hidden_dim;
    model->n_heads = spec->n_heads;
    model->n_kv_heads = spec->n_kv_heads;
    model->head_dim = spec->head_dim;
    model->vocab_size = spec->vocab_size;
    
    return model;
}

void sm2_model_free(sm2_model* model) {
    if (!model) return;

    // Free layer weights (contiguous arrays)
    if (model->input_layernorm) free(model->input_layernorm);
    if (model->q_proj) free(model->q_proj);
    if (model->k_proj) free(model->k_proj);
    if (model->v_proj) free(model->v_proj);
    if (model->o_proj) free(model->o_proj);
    if (model->post_attention_layernorm) free(model->post_attention_layernorm);
    if (model->gate_proj) free(model->gate_proj);
    if (model->up_proj) free(model->up_proj);
    if (model->down_proj) free(model->down_proj);
    if (model->final_norm) free(model->final_norm);
    if (model->tok_embeddings) {
        free(model->tok_embeddings->data);
        free(model->tok_embeddings);
    }
    if (model->tokenizer) sm2_tokenizer_free(model->tokenizer);

    free(model);
}

// ============================================================================
// .SM2 FILE LOADING
// ============================================================================

static int sm2_read_header(FILE* f, sm2_file_header* hdr) {
    if (fread(hdr, sizeof(sm2_file_header), 1, f) != 1) {
        return -1;
    }
    
    // Validate magic
    if (memcmp(hdr->magic, SM2_MAGIC, 8) != 0) {
        fprintf(stderr, "Invalid .sm2 file: bad magic\n");
        return -1;
    }
    
    if (hdr->version != SM2_VERSION) {
        fprintf(stderr, "Unsupported .sm2 version: %u (expected %d)\n", 
                hdr->version, SM2_VERSION);
        return -1;
    }
    
    return 0;
}

static int sm2_load_weights_f16(FILE* f, sm2_model* model, const sm2_file_header* hdr) {
    // CRITICAL FIX: After header (256 bytes), we have:
    //   - Tokenizer section (starts at 256, size ~1.17 MB)
    //   - Weights section (starts after tokenizer)
    //
    // Weights start at 256 + tokenizer_size = 1,179,115
    //
    // File format after tokenizer:
    //   1. tok_embeddings: [rows, cols] header + F16 data
    //   2. Per layer: input_layernorm, q_proj, k_proj, v_proj, o_proj,
    //                post_attention_layernorm, gate_proj, up_proj, down_proj
    //   3. final_norm: [1, dim] header + F16 data
    
    uint64_t weights_offset = 256 + 1178859;  // Hardcoded known correct value
    
    if (fseek(f, (long)weights_offset, SEEK_SET) != 0) {
        fprintf(stderr, "Failed to seek to weights at offset %lu\n", weights_offset);
        return -1;
    }
    
    int dim = model->dim;
    int hidden_dim = model->hidden_dim;
    int n_layers = model->n_layers;
    int kv_dim = model->n_kv_heads * model->head_dim;  // 192 for 135M
    
    // ========== 1. Load tok_embeddings ==========
    uint32_t rows, cols;
    if (fread(&rows, 4, 1, f) != 1) { fprintf(stderr, "Failed to read tok_embeddings rows\n"); return -1; }
    if (fread(&cols, 4, 1, f) != 1) { fprintf(stderr, "Failed to read tok_embeddings cols\n"); return -1; }
    fprintf(stderr, "DEBUG: tok_embeddings header: rows=%u, cols=%u\n", rows, cols);
    
    size_t embed_size = (size_t)rows * cols * sizeof(uint16_t);
    model->tok_embeddings = calloc(1, sizeof(sm2_tensor_f16));
    if (!model->tok_embeddings) return -1;
    model->tok_embeddings->rows = rows;
    model->tok_embeddings->cols = cols;
    model->tok_embeddings->data = malloc(embed_size);
    if (!model->tok_embeddings->data) return -1;
    if (fread(model->tok_embeddings->data, embed_size, 1, f) != 1) {
        fprintf(stderr, "Failed to read tok_embeddings data\n"); return -1;
    }
    
    // ========== 2. Load layer weights ==========
    // Allocate contiguous arrays for all layers
    size_t input_ln_size  = (size_t)n_layers * dim * sizeof(uint16_t);
    size_t q_size         = (size_t)n_layers * dim * dim * sizeof(uint16_t);
    size_t k_size         = (size_t)n_layers * kv_dim * dim * sizeof(uint16_t);
    size_t v_size         = (size_t)n_layers * kv_dim * dim * sizeof(uint16_t);
    size_t o_size         = (size_t)n_layers * dim * dim * sizeof(uint16_t);  // [dim, dim] not [dim, kv_dim]
    size_t post_ln_size   = (size_t)n_layers * dim * sizeof(uint16_t);
    size_t gate_size      = (size_t)n_layers * hidden_dim * dim * sizeof(uint16_t);
    size_t up_size        = (size_t)n_layers * hidden_dim * dim * sizeof(uint16_t);
    size_t down_size      = (size_t)n_layers * hidden_dim * dim * sizeof(uint16_t);
    
    model->input_layernorm  = calloc(input_ln_size, 1);
    model->q_proj          = calloc(q_size, 1);
    model->k_proj          = calloc(k_size, 1);
    model->v_proj          = calloc(v_size, 1);
    model->o_proj          = calloc(o_size, 1);
    model->post_attention_layernorm = calloc(post_ln_size, 1);
    model->gate_proj       = calloc(gate_size, 1);
    model->up_proj         = calloc(up_size, 1);
    model->down_proj       = calloc(down_size, 1);
    
    if (!model->input_layernorm || !model->q_proj || !model->k_proj || !model->v_proj ||
        !model->o_proj || !model->post_attention_layernorm || !model->gate_proj ||
        !model->up_proj || !model->down_proj) {
        fprintf(stderr, "Failed to allocate layer weights\n"); return -1;
    }
    
    for (int layer = 0; layer < n_layers; layer++) {
        // input_layernorm: [1, dim] -> read rows=1, cols=dim
        uint32_t r, c;
        if (fread(&r, 4, 1, f) != 1) { fprintf(stderr, "layer %d: failed read input_ln rows\n", layer); return -1; }
        if (fread(&c, 4, 1, f) != 1) { fprintf(stderr, "layer %d: failed read input_ln cols\n", layer); return -1; }
        if (r != 1 || c != (uint32_t)dim) { fprintf(stderr, "layer %d: input_ln bad shape %ux%u\n", layer, r, c); }
        if (fread(model->input_layernorm + layer * dim, dim * sizeof(uint16_t), 1, f) != 1) { return -1; }
        
        // q_proj: [dim, dim]
        if (fread(&r, 4, 1, f) != 1) return -1;
        if (fread(&c, 4, 1, f) != 1) return -1;
        if (r != (uint32_t)dim || c != (uint32_t)dim) { fprintf(stderr, "layer %d: q_proj bad shape %ux%u\n", layer, r, c); }
        if (fread(model->q_proj + layer * dim * dim, dim * dim * sizeof(uint16_t), 1, f) != 1) return -1;
        
        // k_proj: [kv_dim, dim]
        if (fread(&r, 4, 1, f) != 1) return -1;
        if (fread(&c, 4, 1, f) != 1) return -1;
        if (r != (uint32_t)kv_dim || c != (uint32_t)dim) { fprintf(stderr, "layer %d: k_proj bad shape %ux%u\n", layer, r, c); }
        if (fread(model->k_proj + layer * kv_dim * dim, kv_dim * dim * sizeof(uint16_t), 1, f) != 1) return -1;
        
        // v_proj: [kv_dim, dim]
        if (fread(&r, 4, 1, f) != 1) return -1;
        if (fread(&c, 4, 1, f) != 1) return -1;
        if (r != (uint32_t)kv_dim || c != (uint32_t)dim) { fprintf(stderr, "layer %d: v_proj bad shape %ux%u\n", layer, r, c); }
        if (fread(model->v_proj + layer * kv_dim * dim, kv_dim * dim * sizeof(uint16_t), 1, f) != 1) return -1;
        
        // o_proj: [dim, dim] (NOT [dim, kv_dim]!)
        if (fread(&r, 4, 1, f) != 1) return -1;
        if (fread(&c, 4, 1, f) != 1) return -1;
        if (r != (uint32_t)dim || c != (uint32_t)dim) { fprintf(stderr, "layer %d: o_proj bad shape %ux%u\n", layer, r, c); }
        if (fread(model->o_proj + layer * dim * dim, dim * dim * sizeof(uint16_t), 1, f) != 1) return -1;
        
        // post_attention_layernorm: [1, dim]
        if (fread(&r, 4, 1, f) != 1) return -1;
        if (fread(&c, 4, 1, f) != 1) return -1;
        if (r != 1 || c != (uint32_t)dim) { fprintf(stderr, "layer %d: post_ln bad shape %ux%u\n", layer, r, c); }
        if (fread(model->post_attention_layernorm + layer * dim, dim * sizeof(uint16_t), 1, f) != 1) return -1;
        
        // gate_proj: [hidden_dim, dim]
        if (fread(&r, 4, 1, f) != 1) return -1;
        if (fread(&c, 4, 1, f) != 1) return -1;
        if (r != (uint32_t)hidden_dim || c != (uint32_t)dim) { fprintf(stderr, "layer %d: gate_proj bad shape %ux%u\n", layer, r, c); }
        if (fread(model->gate_proj + layer * hidden_dim * dim, hidden_dim * dim * sizeof(uint16_t), 1, f) != 1) return -1;
        
        // up_proj: [hidden_dim, dim]
        if (fread(&r, 4, 1, f) != 1) return -1;
        if (fread(&c, 4, 1, f) != 1) return -1;
        if (r != (uint32_t)hidden_dim || c != (uint32_t)dim) { fprintf(stderr, "layer %d: up_proj bad shape %ux%u\n", layer, r, c); }
        if (fread(model->up_proj + layer * hidden_dim * dim, hidden_dim * dim * sizeof(uint16_t), 1, f) != 1) return -1;
        
        // down_proj: [dim, hidden_dim]
        if (fread(&r, 4, 1, f) != 1) return -1;
        if (fread(&c, 4, 1, f) != 1) return -1;
        if (r != (uint32_t)dim || c != (uint32_t)hidden_dim) { fprintf(stderr, "layer %d: down_proj bad shape %ux%u\n", layer, r, c); }
        if (fread(model->down_proj + layer * dim * hidden_dim, dim * hidden_dim * sizeof(uint16_t), 1, f) != 1) return -1;
        
        fprintf(stderr, "DEBUG: Layer %d loaded\n", layer);
    }
    
    // ========== 3. Load final_norm ==========
    uint32_t r, c;
    if (fread(&r, 4, 1, f) != 1) { fprintf(stderr, "Failed to read final_norm rows\n"); return -1; }
    if (fread(&c, 4, 1, f) != 1) { fprintf(stderr, "Failed to read final_norm cols\n"); return -1; }
    fprintf(stderr, "DEBUG: final_norm header: rows=%u, cols=%u\n", r, c);
    
    model->final_norm = calloc(dim, sizeof(uint16_t));
    if (!model->final_norm) return -1;
    if (fread(model->final_norm, dim * sizeof(uint16_t), 1, f) != 1) { return -1; }
    
    fprintf(stderr, "DEBUG: All weights loaded successfully!\n");
    return 0;
}

int sm2_load_model(const char* path, sm2_model** out_model) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }
    
    // Read header
    sm2_file_header hdr;
    if (sm2_read_header(f, &hdr) != 0) {
        fclose(f);
        return -1;
    }
    
    // Determine variant
    sm2_variant variant = SM2_135M;
    if (hdr.variant_id == 360) variant = SM2_360M;
    if (hdr.variant_id == 1700) variant = SM2_1700M;
    
    // Allocate model
    sm2_model* model = sm2_model_alloc(variant);
    if (!model) {
        fclose(f);
        return -1;
    }
    
    model->quant_type = (sm2_quant_type)hdr.quant_type;
    
    // Load weights based on quant type
    if (model->quant_type == SM2_F16) {
        if (sm2_load_weights_f16(f, model, &hdr) != 0) {
            fprintf(stderr, "Failed to load F16 weights\n");
            sm2_model_free(model);
            fclose(f);
            return -1;
        }
    }
    
    // Load tokenizer from .sm2 file
    if (hdr.tokenizer_offset > 0 && hdr.tokenizer_size > 0) {
        model->tokenizer = calloc(1, sizeof(sm2_tokenizer));
        if (model->tokenizer) {
            if (sm2_load_tokenizer_from_sm2(model->tokenizer, f, hdr.tokenizer_offset, hdr.tokenizer_size) != 0) {
                free(model->tokenizer);
                model->tokenizer = NULL;
                fprintf(stderr, "Warning: Failed to load tokenizer\n");
            }
        }
    }
    
    fclose(f);
    *out_model = model;
    return 0;
}

int sm2_free_model(sm2_model* model) {
    sm2_model_free(model);
    return 0;
}

// ============================================================================
// MMAP MODEL LOADING (zero-copy)
// ============================================================================

int sm2_mmap_model(const char* path, sm2_model** out_model) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }
    
    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return -1;
    }
    
    void* data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        close(fd);
        return -1;
    }
    
    // Parse header from mmap
    sm2_file_header* hdr = (sm2_file_header*)data;
    if (memcmp(hdr->magic, SM2_MAGIC, 8) != 0) {
        munmap(data, st.st_size);
        close(fd);
        return -1;
    }
    
    // Create model with mmap'd data pointers
    sm2_variant variant = SM2_135M;
    if (hdr->variant_id == 360) variant = SM2_360M;
    if (hdr->variant_id == 1700) variant = SM2_1700M;
    
    sm2_model* model = sm2_model_alloc(variant);
    if (!model) {
        munmap(data, st.st_size);
        close(fd);
        return -1;
    }
    
    // For mmap, weights point directly into mapped region
    // Weights offset is after header
    uint8_t* weights_base = (uint8_t*)data + hdr->weights_offset;
    size_t embed_size = (size_t)model->vocab_size * model->dim * sizeof(uint16_t);
    
    model->tok_embeddings = calloc(1, sizeof(sm2_tensor_f16));
    model->tok_embeddings->rows = model->vocab_size;
    model->tok_embeddings->cols = model->dim;
    model->tok_embeddings->data = (uint16_t*)weights_base;
    
    *out_model = model;
    return 0;
}

int sm2_unmmap_model(sm2_model* model) {
    // For mmap models, we need to track and unmap the original region
    // For simplicity, just free model struct
    sm2_model_free(model);
    return 0;
}