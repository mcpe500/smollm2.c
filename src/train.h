// train.h — LoRA/QLoRA/FullFT trainer (phase 2)
#ifndef TRAIN_H
#define TRAIN_H

#include "forward.h"
#include "data.h"
#include "hw_probe.h"

typedef enum { TRAIN_LORA = 0, TRAIN_QLORA, TRAIN_FULLFT } train_mode;

typedef struct {
    train_mode mode;
    int   lora_rank;        /* 0 for full FT */
    int   lora_alpha;       /* default = 2*rank */
    int   seq_max;
    int   batch;
    float lr;
    int   epochs;
    int   checkpoint_every;
    int   max_steps;        /* 0 = unlimited */
    int   seed;
    long  simulate_mem_kb;  /* 0 = real, else override for tests */
} train_params;

typedef struct train_state train_state;

train_state* train_create(forward_ctx* f, const train_params* p);
void         train_free  (train_state* t);

/* One step on a packed sample. Returns CE loss. */
float train_step(train_state* t, const int* tokens, int n_tokens);

int  train_save(const train_state* t, const char* path);
int  train_load(train_state* t, const char* path);

/* Main loop: iterate dataset, save adapters, watchdog. */
int  train_run(train_state* t, const char* packed_path,
               const train_params* p, const char* out_dir);

/* Merge LoRA adapter into base GGUF (lm_head only for phase 2). */
int  train_merge(const char* base_gguf, const char* adapter_path,
                 const char* out_gguf);

#endif // TRAIN_H