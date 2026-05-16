// sm2dl_batch_decode.c - Batch decode support for concurrent requests

#include <stdlib.h>
#include <math.h>
#include "smollm2.h"

// ============================================================================
// BATCH DECODE - Process multiple sequences simultaneously
//
// Continuous batching: new requests are added to the batch each step,
// finished requests are removed. Maximizes GPU utilization by keeping
// the compute busy with variable-length sequences.
// ============================================================================

// Forward declaration
typedef struct sm2_batch sm2_batch;

typedef struct {
    int seq_id;
    sm2_context* ctx;
    int n_tokens;       // tokens generated so far
    int target_len;    // max tokens to generate
    sm2_req_state state;
} sm2_batch_item;

struct sm2_batch {
    sm2_batch_item* items;
    int capacity;
    int n_active;
};

// Initialize batch processor
int sm2_batch_init(sm2_batch** out_batch, int max_concurrent) {
    sm2_batch* batch = calloc(1, sizeof(sm2_batch));
    if (!batch) return -1;
    
    batch->items = calloc(max_concurrent, sizeof(sm2_batch_item));
    if (!batch->items) {
        free(batch);
        return -1;
    }
    
    batch->capacity = max_concurrent;
    batch->n_active = 0;
    
    *out_batch = batch;
    return 0;
}

void sm2_batch_free(sm2_batch* batch) {
    if (!batch) return;
    for (int i = 0; i < batch->n_active; i++) {
        // Don't free ctx here - caller owns them
    }
    free(batch->items);
    free(batch);
}

// Add request to batch
int sm2_batch_add(sm2_batch* batch, int seq_id, sm2_context* ctx, int target_len) {
    if (batch->n_active >= batch->capacity) return -1; // Batch full
    
    sm2_batch_item* item = &batch->items[batch->n_active];
    item->seq_id = seq_id;
    item->ctx = ctx;
    item->n_tokens = 0;
    item->target_len = target_len;
    item->state = SM2_REQ_DECODE;
    
    batch->n_active++;
    return 0;
}

// Remove finished request from batch
void sm2_batch_remove(sm2_batch* batch, int idx) {
    if (idx < 0 || idx >= batch->n_active) return;
    
    // Swap with last and decrement
    batch->items[idx] = batch->items[batch->n_active - 1];
    batch->n_active--;
}

// Process one decode step for entire batch
int sm2_batch_decode_step(sm2_batch* batch) {
    int n_done = 0;
    
    for (int i = 0; i < batch->n_active; i++) {
        sm2_batch_item* item = &batch->items[i];
        
        if (item->state != SM2_REQ_DECODE) continue;
        
        // Decode one token
        int token;
        int ok = sm2_decode_next(item->ctx, &token);
        
        if (ok != 0) {
            item->state = SM2_REQ_CANCELLED;
            n_done++;
            continue;
        }
        
        item->n_tokens++;
        
        // Check if done
        if (token == 2 || item->n_tokens >= item->target_len) {
            item->state = SM2_REQ_DONE;
            n_done++;
        }
    }
    
    return n_done;
}

// Get done items
int sm2_batch_get_done(sm2_batch* batch, int* out_seq_ids, int max) {
    int n = 0;
    for (int i = 0; i < batch->n_active && n < max; i++) {
        if (batch->items[i].state == SM2_REQ_DONE || 
            batch->items[i].state == SM2_REQ_CANCELLED) {
            out_seq_ids[n++] = batch->items[i].seq_id;
        }
    }
    return n;
}