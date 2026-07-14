// data.h — dataset adapter untuk studio phase 1
#ifndef DATA_H
#define DATA_H

#include "tokenizer.h"

typedef enum { FMT_RAW = 0, FMT_INSTRUCT, FMT_SHAREGPT, FMT_AUTO } data_fmt;

typedef struct {
    int  n_tokens;
    long offset;
} sample_idx;

typedef struct {
    int         n_samples;
    long        total_tokens;
    sample_idx* index;
    char*       packed_path;
    data_fmt    detected_fmt;
} dataset;

const char* data_fmt_name(data_fmt f);

data_fmt data_detect(const char* in_path);

int  data_build(const char* in_path, const char* out_path,
                data_fmt fmt, const tokenizer* tok);
int  data_inspect(const char* packed_path);
void data_free(dataset* d);

#endif // DATA_H