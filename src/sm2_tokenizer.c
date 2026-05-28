// sm2_tokenizer.c - Full BPE tokenizer implementation for SmolLM2
// Loads tokenizer from .sm2 file format and implements proper BPE encoding/decoding

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <sys/stat.h>
#include "smollm2.h"

// ============================================================================
// TOKENIZER LOADING FROM .SM2 FILE
// ============================================================================

int sm2_load_tokenizer_from_sm2(sm2_tokenizer* tok, FILE* f, uint64_t offset, uint64_t size) {
    // Seek to tokenizer section
    if (fseek(f, (long)offset, SEEK_SET) != 0) {
        fprintf(stderr, "Failed to seek to tokenizer at offset %lu\n", offset);
        return -1;
    }

    // Allocate temporary buffer for tokenizer data
    uint8_t* data = malloc(size);
    if (!data) {
        fprintf(stderr, "Failed to allocate %lu bytes for tokenizer\n", size);
        return -1;
    }

    if (fread(data, size, 1, f) != 1) {
        fprintf(stderr, "Failed to read tokenizer data\n");
        free(data);
        return -1;
    }

    tok->vocab_size = 49152; // SmolLM2
    tok->tokens = calloc(tok->vocab_size, sizeof(char*));
    tok->token_to_id = calloc(tok->vocab_size, sizeof(int));
    tok->scores = NULL;  // Not stored in .sm2 format

    if (!tok->tokens || !tok->token_to_id) {
        fprintf(stderr, "Failed to allocate tokenizer arrays\n");
        free(data);
        return -1;
    }

    // Initialize byte_to_token mapping to identity mapping (byte -> byte)
    // This will be updated when we find single-byte tokens in the vocab
    for (int i = 0; i < 256; i++) {
        tok->byte_to_token[i] = i;
    }

    // Parse tokenizer binary format:
    // [49152 x (4-byte length + token_bytes)] followed by [4-byte merges_count] + [merges]
    uint32_t pos = 0;

    // Read token strings
    for (int i = 0; i < tok->vocab_size && pos < size; i++) {
        if (pos + 4 > size) break;
        uint32_t len = *(uint32_t*)(data + pos);
        pos += 4;

        if (len > 0 && pos + len <= size) {
            char* token = malloc(len + 1);
            memcpy(token, data + pos, len);
            token[len] = '\0';
            tok->tokens[i] = token;
            tok->token_to_id[i] = i; // id == index for this format

            // Build byte_to_token mapping for single-byte tokens
            // This maps raw byte values (0-255) to actual vocab token IDs
            if (len == 1) {
                unsigned char byte_val = (unsigned char)token[0];
                tok->byte_to_token[byte_val] = i;
            }

            // Special case: Ġ (U+0120 = 0xC4 0xA0) represents space in SmolLM2 BPE
            // This is the word-start marker used when a word begins with a space
            if (len == 2 && (unsigned char)token[0] == 0xC4 && (unsigned char)token[1] == 0xA0) {
                // Map space (byte 32) to Ġ token
                tok->byte_to_token[32] = i;
            }

            // Special case: Ċ (U+010A = 0xC4 0x8A) represents newline in SmolLM2 BPE
            if (len == 2 && (unsigned char)token[0] == 0xC4 && (unsigned char)token[1] == 0x8A) {
                // Map newline (byte 10) to Ċ token
                tok->byte_to_token[10] = i;
            }

            pos += len;
        } else {
            tok->tokens[i] = NULL;
            tok->token_to_id[i] = -1;
        }
    }

    // Read merges count and merges
    if (pos + 4 <= size) {
        uint32_t num_merges = *(uint32_t*)(data + pos);
        pos += 4;

        tok->num_merges = num_merges;
        tok->merges = calloc(num_merges, sizeof(char*));

        for (uint32_t i = 0; i < num_merges && pos < size; i++) {
            if (pos + 4 > size) break;
            uint32_t merge_len = *(uint32_t*)(data + pos);
            pos += 4;

            if (merge_len > 0 && pos + merge_len <= size) {
                char* merge = malloc(merge_len + 1);
                memcpy(merge, data + pos, merge_len);
                merge[merge_len] = '\0';
                tok->merges[i] = merge;
                pos += merge_len;

                // Normalize merge: remove spaces used as separators between tokens
                // tokenizer.json stores merges as "token1 token2", we need "token1token2"
                {
                    char* src = merge;
                    char* dst = merge;
                    while (*src) {
                        if (*src != ' ') {
                            *dst++ = *src;
                        }
                        src++;
                    }
                    *dst = '\0';
                }
            }
        }
    }

    free(data);
    return 0;
}

// ============================================================================
// TOKENIZER INIT / FREE
// ============================================================================

int sm2_tokenizer_init(const char* vocab_path, const char* merges_path, sm2_tokenizer** out_tok) {
    sm2_tokenizer* tok = calloc(1, sizeof(sm2_tokenizer));
    if (!tok) return -1;

    tok->vocab_size = 49152; // SmolLM2 vocab size
    tok->num_merges = 0;
    tok->merges = NULL;
    tok->tokens = NULL;
    tok->token_to_id = NULL;
    tok->scores = NULL;

    *out_tok = tok;
    return 0;
}

void sm2_tokenizer_free(sm2_tokenizer* tok) {
    if (!tok) return;

    if (tok->tokens) {
        for (int i = 0; i < tok->vocab_size; i++) {
            if (tok->tokens[i]) free(tok->tokens[i]);
        }
        free(tok->tokens);
    }
    if (tok->token_to_id) free(tok->token_to_id);
    if (tok->scores) free(tok->scores);
    if (tok->merges) {
        for (int i = 0; i < tok->num_merges; i++) {
            if (tok->merges[i]) free(tok->merges[i]);
        }
        free(tok->merges);
    }
    if (tok->vocab_data) free(tok->vocab_data);
    free(tok);
}

// ============================================================================
// HELPERS
// ============================================================================

// Check if char is whitespace
static int is_whitespace(char c) {
    return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v';
}

// Find token ID for a given UTF-8 string in the tokens array
// Binary search optimization: try most common token lengths first
static int find_token_id_by_string(sm2_tokenizer* tok, const char* s, int s_len) {
    if (!tok->tokens) return -1;

    // Linear search (could be optimized with hash/Trie for speed)
    for (int i = 0; i < tok->vocab_size; i++) {
        if (tok->tokens[i]) {
            int tok_len = strlen(tok->tokens[i]);
            if (tok_len == s_len && memcmp(tok->tokens[i], s, s_len) == 0) {
                return i;
            }
        }
    }
    return -1;
}

// ============================================================================
// BPE ENCODE WORD - uses token strings stored in merges
// ============================================================================

static int bpe_encode_word(sm2_tokenizer* tok, const char* word, int* ids, int max_len) {
    if (!word || !ids || max_len <= 0) return 0;

    int word_len = strlen(word);
    if (word_len == 0) return 0;

    // If no merges or tokens loaded, fall back to byte-level
    if (!tok->merges || tok->num_merges == 0 || !tok->tokens) {
        int n = 0;
        for (int i = 0; word[i] && n < max_len; i++) {
            unsigned char b = (unsigned char)word[i];
            ids[n++] = b;
        }
        return n;
    }

    // Use token strings as the base "alphabet" - each character becomes a seg_str
    // seg_strs holds the actual token strings (UTF-8 encoded)
    int max_segs = word_len * 2;

    // segments: array of token IDs for final output
    int* segments = calloc(max_segs, sizeof(int));
    // seg_strs: array of string pointers for the segmentation state
    char** seg_strs = calloc(max_segs, sizeof(char*));
    int n_segs = 0;

    // Step 1: Initialize with each character as a separate segment
    // For ASCII text, each character is a single byte = single token
    for (int i = 0; i < word_len; i++) {
        unsigned char byte_val = (unsigned char)word[i];

        // Find the token ID for this single character
        int token_id = find_token_id_by_string(tok, (const char*)&byte_val, 1);
        if (token_id < 0) token_id = byte_val; // fallback

        segments[n_segs] = token_id;

        seg_strs[n_segs] = malloc(2);
        seg_strs[n_segs][0] = (char)byte_val;
        seg_strs[n_segs][1] = '\0';
        n_segs++;
    }

    // Step 2: Iteratively apply BPE merges
    // For each adjacent pair, check if the concatenated string is in the merges table
    // The merge that appears at the lowest index (most frequent) gets applied first
    int iterations = 0;
    int max_iterations = n_segs * 4;

    while (iterations < max_iterations && n_segs > 1) {
        iterations++;

        int best_rank = -1;
        int best_idx = -1;

        // Find the best pair (lowest rank in merges table)
        for (int i = 0; i < n_segs - 1; i++) {
            int len1 = strlen(seg_strs[i]);
            int len2 = strlen(seg_strs[i + 1]);
            int pair_len = len1 + len2;

            // Build the pair string
            char* pair = malloc(pair_len + 1);
            memcpy(pair, seg_strs[i], len1);
            memcpy(pair + len1, seg_strs[i + 1], len2);
            pair[pair_len] = '\0';

            // Search for this pair in merges
            int rank = -1;
            for (int m = 0; m < tok->num_merges; m++) {
                if (tok->merges[m] && strcmp(tok->merges[m], pair) == 0) {
                    rank = m;
                    break;
                }
            }

            free(pair);

            if (rank >= 0 && (best_rank < 0 || rank < best_rank)) {
                best_rank = rank;
                best_idx = i;
            }
        }

        if (best_idx < 0) {
            break; // No more merges possible
        }

        // Build the merged string
        int merged_len = strlen(seg_strs[best_idx]) + strlen(seg_strs[best_idx + 1]);
        char* merged = malloc(merged_len + 1);
        strcpy(merged, seg_strs[best_idx]);
        strcat(merged, seg_strs[best_idx + 1]);

        // Find the token ID for the merged string
        int merged_id = find_token_id_by_string(tok, merged, merged_len);

        if (merged_id < 0) {
            // Merged string not in vocab - no more merges possible
            free(merged);
            break;
        }

        // Apply merge: replace seg[best_idx] with merged_id, remove seg[best_idx+1]
        segments[best_idx] = merged_id;
        free(seg_strs[best_idx]);
        seg_strs[best_idx] = merged; // Keep the malloced string

        // Shift remaining segments
        for (int i = best_idx + 1; i < n_segs - 1; i++) {
            segments[i] = segments[i + 1];
            seg_strs[i] = seg_strs[i + 1];
        }
        seg_strs[n_segs - 1] = NULL;
        n_segs--;

        if (n_segs == 1) break;
    }

    // Step 3: Copy final segments to output
    int n = 0;
    for (int i = 0; i < n_segs && n < max_len; i++) {
        ids[n++] = segments[i];
    }

    // Cleanup
    for (int i = 0; i < n_segs; i++) {
        free(seg_strs[i]);
    }
    free(segments);
    free(seg_strs);

    return n;
}

// ============================================================================
// PRETOKENIZE: split text into word pieces with proper space handling
// For BPE tokenizers, spaces at word starts are preserved as special tokens
// ============================================================================

// Token type for pretokenization
typedef enum {
    TOKEN_PIECE_WORD,      // Regular word text
    TOKEN_PIECE_SPACE,     // Whitespace (will become Ġ prefix)
} token_piece_type;

typedef struct {
    token_piece_type type;
    const char* text;
    int len;
} token_piece;

static int pre_tokenize(const char* text, token_piece* pieces, int max_pieces) {
    int n = 0;
    const char* p = text;

    while (*p && n < max_pieces - 1) {
        // Check if we're at whitespace
        if (is_whitespace(*p)) {
            // Each space becomes a separate piece - don't include trailing spaces
            pieces[n].type = TOKEN_PIECE_SPACE;
            pieces[n].text = p;
            pieces[n].len = 1;
            p++;
            n++;
        } else {
            // Collect word
            const char* start = p;
            while (*p && !is_whitespace(*p)) p++;
            pieces[n].type = TOKEN_PIECE_WORD;
            pieces[n].text = start;
            pieces[n].len = (int)(p - start);
            n++;
        }
    }

    return n;
}

// ============================================================================
// TOKENIZER ENCODE / DECODE
// ============================================================================

int sm2_tokenizer_encode(sm2_tokenizer* tok, const char* text, int* ids, int max_len) {
    if (!tok || !text || !ids || max_len <= 0) return -1;

    // Pretokenize into words and spaces
    token_piece pieces[512];
    int n_pieces = pre_tokenize(text, pieces, 512);

    int n = 0;
    for (int w = 0; w < n_pieces && n < max_len; w++) {
        if (pieces[w].type == TOKEN_PIECE_SPACE) {
            // Check what kind of whitespace it is
            char ws_char = pieces[w].text[0];
            if (ws_char == '\n') {
                // Newline becomes the Ċ token (token 198)
                // byte_to_token[10] should map to Ċ after our fix
                ids[n++] = tok->byte_to_token[10];  // newline byte -> Ċ token
            } else if (ws_char == ' ') {
                // Space becomes the Ġ token (token ID that starts with space in HF vocab)
                // In our tokenizer, this maps byte 32 to the Ġ token
                ids[n++] = tok->byte_to_token[32];  // space byte -> Ġ token
            } else {
                // Other whitespace (tab, etc.) - use byte fallback
                ids[n++] = (unsigned char)ws_char;
            }
        } else {
            // Word - BPE encode it
            int word_ids[512];
            int n_ids = bpe_encode_word(tok, pieces[w].text, word_ids, 512);

            for (int i = 0; i < n_ids && n < max_len; i++) {
                ids[n++] = word_ids[i];
            }
        }
    }

    return n;
}

char* sm2_tokenizer_decode(sm2_tokenizer* tok, const int* ids, int n_ids) {
    if (!tok || !ids || n_ids <= 0) return NULL;

    // Allocate output buffer (max 6 bytes per token for UTF-8 + null)
    // Max UTF-8 char is 4 bytes, but Ġ is 2 bytes so max is 5 bytes for Ġ + 3 bytes
    size_t out_capacity = (size_t)n_ids * 8 + 1;
    char* out = malloc(out_capacity);
    if (!out) return NULL;

    size_t pos = 0;
    for (int i = 0; i < n_ids && pos < out_capacity - 1; i++) {
        int id = ids[i];

        // Handle special tokens
        if (id == 0) {
            break; // <|endoftext|> - stop
        } else if (id == 1) {
            continue; // <|im_start|> - skip
        } else if (id == 2) {
            break; // <|im_end|> - end of message
        } else if (id >= 0 && id < tok->vocab_size) {
            // Regular token - copy UTF-8 bytes directly
            if (tok->tokens && tok->tokens[id]) {
                const char* token_str = tok->tokens[id];
                size_t tok_len = strlen(token_str);

                // Copy the UTF-8 bytes (could be 1-4 bytes for multi-byte chars)
                for (size_t j = 0; j < tok_len && pos < out_capacity - 1; j++) {
                    out[pos++] = token_str[j];
                }
            } else if (id < 256) {
                // Fallback: print as ASCII if printable
                if (id >= 32 && id < 127) {
                    out[pos++] = (char)id;
                }
            }
        }
    }

    out[pos] = '\0';
    return out;
}

// Convert byte value to token ID using tokenizer's byte mapping
int sm2_tokenizer_byte_to_token(sm2_tokenizer* tok, unsigned char byte_val) {
    if (!tok || byte_val >= 256) return byte_val;
    return tok->byte_to_token[byte_val];
}
