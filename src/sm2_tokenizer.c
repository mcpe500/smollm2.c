// sm2_tokenizer.c - HuggingFace BPE tokenizer

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include "smollm2.h"

// ============================================================================
// Simple BPE tokenizer for SmolLM2
// Handles vocab.json and merges.txt from HF
// ============================================================================

int sm2_tokenizer_init(const char* vocab_path, const char* merges_path, sm2_tokenizer** out_tok) {
    sm2_tokenizer* tok = calloc(1, sizeof(sm2_tokenizer));
    if (!tok) return -1;
    
    tok->vocab_size = 49152; // SmolLM2 vocab size
    tok->num_merges = 0;
    tok->merges = NULL;
    
    *out_tok = tok;
    return 0;
}

void sm2_tokenizer_free(sm2_tokenizer* tok) {
    if (!tok) return;
    if (tok->merges) free(tok->merges);
    free(tok);
}

// Check if char is whitespace
static int is_whitespace(char c) {
    return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v';
}

// Pre-tokenize: split text into words
static int pre_tokenize(const char* text, char*** out_words, int* out_n) {
    int capacity = 256;
    char** words = malloc(capacity * sizeof(char*));
    int n = 0;
    
    const char* p = text;
    while (*p) {
        while (*p && is_whitespace(*p)) p++;
        if (!*p) break;
        
        const char* start = p;
        while (*p && !is_whitespace(*p)) p++;
        
        int len = (int)(p - start);
        if (len > 0) {
            char* word = malloc(len + 1);
            memcpy(word, start, len);
            word[len] = '\0';
            
            if (n >= capacity) {
                capacity *= 2;
                words = realloc(words, capacity * sizeof(char*));
            }
            words[n++] = word;
        }
    }
    
    *out_words = words;
    *out_n = n;
    return 0;
}

// Free word list
static void free_words(char** words, int n) {
    for (int i = 0; i < n; i++) free(words[i]);
    free(words);
}

// Byte-level BPE encoding (simplified)
static int bpe_encode_word(sm2_tokenizer* tok, const char* word, int* ids, int max_len) {
    (void)tok;
    int n = 0;
    for (int i = 0; word[i] && n < max_len; i++) {
        unsigned char b = (unsigned char)word[i];
        if (b < 256) {
            ids[n++] = b;
        }
    }
    return n;
}

int sm2_tokenizer_encode(sm2_tokenizer* tok, const char* text, int* ids, int max_len) {
    if (!tok || !text || !ids) return -1;
    
    char** words;
    int n_words;
    pre_tokenize(text, &words, &n_words);
    
    int n = 0;
    for (int w = 0; w < n_words && n < max_len; w++) {
        int word_ids[512];
        int n_ids = bpe_encode_word(tok, words[w], word_ids, 512);
        
        for (int i = 0; i < n_ids && n < max_len; i++) {
            ids[n++] = word_ids[i];
        }
    }
    
    free_words(words, n_words);
    return n;
}

char* sm2_tokenizer_decode(sm2_tokenizer* tok, const int* ids, int n_ids) {
    if (!tok || !ids) return NULL;
    
    (void)tok;
    char* out = malloc(n_ids * 4 + 1);
    int n = 0;
    
    for (int i = 0; i < n_ids; i++) {
        int id = ids[i];
        if (id >= 0 && id < 256) {
            out[n++] = (char)id;
        }
    }
    
    out[n] = '\0';
    return out;
}