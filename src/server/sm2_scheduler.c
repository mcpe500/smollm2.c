// sm2_scheduler.c - Request scheduler with continuous batching

#include <stdlib.h>
#include <math.h>
#include "smollm2.h"
#include <string.h>

typedef struct {
    int id;
    int client_fd;
    sm2_context* ctx;
    int n_tokens;
    int max_tokens;
    sm2_req_state state;
    int eos;
} schedule_item;

struct sm2_scheduler {
    schedule_item* items;
    int capacity;
    int n_active;
    int next_id;
};

int sm2_scheduler_init(struct sm2_scheduler** out, int capacity) {
    struct sm2_scheduler* s = calloc(1, sizeof(struct sm2_scheduler));
    if (!s) return -1;
    
    s->items = calloc(capacity, sizeof(schedule_item));
    if (!s->items) {
        free(s);
        return -1;
    }
    
    s->capacity = capacity;
    s->n_active = 0;
    s->next_id = 1;
    
    *out = s;
    return 0;
}

void sm2_scheduler_free(struct sm2_scheduler* s) {
    if (!s) return;
    for (int i = 0; i < s->n_active; i++) {
        if (s->items[i].ctx) {
            sm2_free_context(s->items[i].ctx);
        }
    }
    free(s->items);
    free(s);
}

int sm2_scheduler_add(struct sm2_scheduler* s, int client_fd, sm2_context* ctx, int max_tokens) {
    if (s->n_active >= s->capacity) return -1;
    
    schedule_item* item = &s->items[s->n_active];
    item->id = s->next_id++;
    item->client_fd = client_fd;
    item->ctx = ctx;
    item->n_tokens = 0;
    item->max_tokens = max_tokens;
    item->state = SM2_REQ_DECODE;
    item->eos = 0;
    
    s->n_active++;
    return item->id;
}

void sm2_scheduler_remove(struct sm2_scheduler* s, int id) {
    for (int i = 0; i < s->n_active; i++) {
        if (s->items[i].id == id) {
            if (s->items[i].ctx) {
                sm2_free_context(s->items[i].ctx);
            }
            s->items[i] = s->items[s->n_active - 1];
            s->n_active--;
            return;
        }
    }
}

// Process one step for all active requests
int sm2_scheduler_step(struct sm2_scheduler* s) {
    int n_done = 0;
    
    for (int i = 0; i < s->n_active; i++) {
        schedule_item* item = &s->items[i];
        
        if (item->state != SM2_REQ_DECODE) continue;
        if (item->eos) continue;
        
        // Decode one token
        int token;
        int ok = sm2_decode_next(item->ctx, &token);
        
        if (ok != 0) {
            item->state = SM2_REQ_CANCELLED;
            n_done++;
            continue;
        }
        
        item->n_tokens++;
        
        if (token == 2) { // EOS
            item->eos = 1;
            item->state = SM2_REQ_DONE;
            n_done++;
        } else if (item->n_tokens >= item->max_tokens) {
            item->state = SM2_REQ_DONE;
            n_done++;
        }
    }
    
    return n_done;
}

int sm2_scheduler_get_done(struct sm2_scheduler* s, int* out_ids, int max) {
    int n = 0;
    for (int i = 0; i < s->n_active && n < max; i++) {
        if (s->items[i].state == SM2_REQ_DONE || 
            s->items[i].state == SM2_REQ_CANCELLED) {
            out_ids[n++] = s->items[i].id;
        }
    }
    return n;
}