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
    
    // Free layer weights
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
    if (model->tok_embeddings) free(model->tok_embeddings);
    
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
    // For now, just allocate and read - simplified
    size_t embed_size = (size_t)model->vocab_size * model->dim * sizeof(uint16_t);
    model->tok_embeddings = calloc(1, sizeof(sm2_tensor_f16));
    if (!model->tok_embeddings) return -1;
    
    model->tok_embeddings->rows = model->vocab_size;
    model->tok_embeddings->cols = model->dim;
    model->tok_embeddings->data = malloc(embed_size);
    if (!model->tok_embeddings->data) return -1;
    
    // Read embedding table
    if (fread(model->tok_embeddings->data, embed_size, 1, f) != 1) {
        return -1;
    }
    
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
    
    // Skip to tensor index
    if (fseek(f, (long)hdr.tensor_index_offset, SEEK_SET) != 0) {
        fprintf(stderr, "Failed to seek to tensor index\n");
        sm2_model_free(model);
        fclose(f);
        return -1;
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