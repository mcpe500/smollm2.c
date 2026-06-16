// tokenizer.c — GPT-2 BPE tokenizer reading vocab/merges from GGUF

#include "tokenizer.h"
#include "gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// ============================================================================
// byte_to_unicode (GPT-2 standard)
//
// Maps raw byte 0..255 to a unicode codepoint. Printable ASCII/Latin-1 bytes
// map to themselves; non-printable bytes map to U+0100, U+0101, ... in order.
// Space (0x20) → U+0120 ('Ġ'). Used so every byte has a printable string form.
// ============================================================================

static int  g_b2u_byte_to_cp[256];           // byte → codepoint
static int  g_b2u_unicode_to_byte[256 * 2];  // codepoint-256 → byte (or -1)
static int  g_b2u_initialized = 0;

static void b2u_init(void) {
    if (g_b2u_initialized) return;

    // Step 1: collect printable bytes in the order GPT-2 specifies.
    int printable[256];
    int n_printable = 0;
    for (int b = '!'; b <= '~'; b++) printable[n_printable++] = b;
    for (int b = 0xA1; b <= 0xAC; b++) printable[n_printable++] = b;
    for (int b = 0xAE; b <= 0xFF; b++) printable[n_printable++] = b;

    // Step 2: assign codepoints. Printable → byte value. Non-printable → 256+n.
    char is_printable[256] = {0};
    for (int i = 0; i < n_printable; i++) is_printable[printable[i]] = 1;

    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (is_printable[b]) {
            g_b2u_byte_to_cp[b] = b;
        } else {
            g_b2u_byte_to_cp[b] = 256 + n;
            n++;
        }
    }

    // Step 3: build reverse map (codepoint → byte).
    for (int i = 0; i < 512; i++) g_b2u_unicode_to_byte[i] = -1;
    for (int b = 0; b < 256; b++) {
        g_b2u_unicode_to_byte[g_b2u_byte_to_cp[b]] = b;
    }

    g_b2u_initialized = 1;
}

// Encode a unicode codepoint as UTF-8 bytes into dst. Returns # bytes.
static int cp_to_utf8(int cp, char* dst) {
    if (cp < 0x80) {
        dst[0] = (char)cp;
        return 1;
    } else if (cp < 0x800) {
        dst[0] = (char)(0xC0 | (cp >> 6));
        dst[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    } else if (cp < 0x10000) {
        dst[0] = (char)(0xE0 | (cp >> 12));
        dst[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        dst[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    } else {
        dst[0] = (char)(0xF0 | (cp >> 18));
        dst[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        dst[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        dst[3] = (char)(0x80 | (cp & 0x3F));
        return 4;
    }
}

// ============================================================================
// String hash table (open addressing, FNV-1a).
//
// Keys are arbitrary-length byte strings. Values are ints (token_id or rank).
// ============================================================================

typedef struct {
    char*  key;     // NULL = empty slot
    size_t key_len;
    int    value;
} sh_entry;

typedef struct {
    sh_entry* entries;
    size_t    mask;     // capacity - 1
    size_t    capacity;
    size_t    count;
} sh_table;

static uint64_t fnv1a(const void* data, size_t n) {
    const uint8_t* p = (const uint8_t*)data;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

static void sh_init(sh_table* t, size_t initial_capacity) {
    // Round up to power of 2.
    size_t cap = 16;
    while (cap < initial_capacity) cap <<= 1;
    t->capacity = cap;
    t->mask = cap - 1;
    t->count = 0;
    t->entries = calloc(cap, sizeof(sh_entry));
}

static void sh_free(sh_table* t) {
    for (size_t i = 0; i < t->capacity; i++) free(t->entries[i].key);
    free(t->entries);
    memset(t, 0, sizeof(*t));
}

static void sh_put(sh_table* t, const void* key, size_t klen, int value);

static void sh_grow(sh_table* t) {
    sh_entry* old = t->entries;
    size_t old_cap = t->capacity;
    t->capacity <<= 1;
    t->mask = t->capacity - 1;
    t->count = 0;
    t->entries = calloc(t->capacity, sizeof(sh_entry));
    for (size_t i = 0; i < old_cap; i++) {
        if (old[i].key) sh_put(t, old[i].key, old[i].key_len, old[i].value);
        free(old[i].key);
    }
    free(old);
}

static void sh_put(sh_table* t, const void* key, size_t klen, int value) {
    if ((t->count + 1) * 2 > t->capacity) sh_grow(t);
    uint64_t h = fnv1a(key, klen) & t->mask;
    while (t->entries[h].key) {
        if (t->entries[h].key_len == klen &&
            memcmp(t->entries[h].key, key, klen) == 0) {
            t->entries[h].value = value;  // overwrite
            return;
        }
        h = (h + 1) & t->mask;
    }
    t->entries[h].key = malloc(klen);
    memcpy(t->entries[h].key, key, klen);
    t->entries[h].key_len = klen;
    t->entries[h].value = value;
    t->count++;
}

// Returns 1 if found (writes *value), 0 if not found.
static int sh_get(const sh_table* t, const void* key, size_t klen, int* value) {
    uint64_t h = fnv1a(key, klen) & t->mask;
    while (t->entries[h].key) {
        if (t->entries[h].key_len == klen &&
            memcmp(t->entries[h].key, key, klen) == 0) {
            *value = t->entries[h].value;
            return 1;
        }
        h = (h + 1) & t->mask;
    }
    return 0;
}

// ============================================================================
// Tokenizer state
// ============================================================================

struct tokenizer {
    int            n_vocab;
    char**         inv_vocab;       // token_id → UTF-8 string (owned)
    int*           token_type;      // token_id → type

    sh_table       vocab;           // UTF-8 string → token_id
    sh_table       merges;          // "a b" → rank
    sh_table       specials;        // special-token UTF-8 string → token_id
};

// ============================================================================
// Load from GGUF
// ============================================================================

int tokenizer_load(tokenizer** out, const gguf_ctx* g) {
    b2u_init();

    gguf_vtype et;
    uint64_t n;
    const void* raw_tokens = gguf_kv_arr(g, "tokenizer.ggml.tokens", &et, &n);
    if (!raw_tokens || et != GGUF_V_STRING || n == 0) {
        fprintf(stderr, "tokenizer: tokens array missing\n");
        return -1;
    }
    char** tokens = (char**)raw_tokens;

    uint64_t n_types = 0;
    const void* raw_types = gguf_kv_arr(g, "tokenizer.ggml.token_type", &et, &n_types);
    int* types = NULL;
    if (raw_types && (et == GGUF_V_INT32 || et == GGUF_V_UINT32) && n_types == n) {
        types = (int*)raw_types;
    }

    tokenizer* t = calloc(1, sizeof(tokenizer));
    t->n_vocab = (int)n;
    t->inv_vocab = calloc(t->n_vocab, sizeof(char*));
    t->token_type = calloc(t->n_vocab, sizeof(int));
    sh_init(&t->vocab, t->n_vocab * 2);
    sh_init(&t->specials, 64);

    for (int i = 0; i < t->n_vocab; i++) {
        const char* s = tokens[i];
        size_t slen = strlen(s);
        char* copy = malloc(slen + 1);
        memcpy(copy, s, slen + 1);
        t->inv_vocab[i] = copy;
        t->token_type[i] = types ? types[i] : 1;
        sh_put(&t->vocab, s, slen, i);
        if (t->token_type[i] == 3 || t->token_type[i] == 4) {
            // Control or user-defined → special.
            sh_put(&t->specials, s, slen, i);
        }
    }

    // Merges: array of strings like "Ġh" or "Ġh e" etc. Key is the literal
    // string; rank is its index. BPE algorithm looks up rank of a candidate
    // pair string to decide order.
    uint64_t n_merges = 0;
    const void* raw_merges = gguf_kv_arr(g, "tokenizer.ggml.merges", &et, &n_merges);
    if (raw_merges && et == GGUF_V_STRING) {
        char** merges = (char**)raw_merges;
        sh_init(&t->merges, n_merges * 2);
        for (uint64_t i = 0; i < n_merges; i++) {
            sh_put(&t->merges, merges[i], strlen(merges[i]), (int)i);
        }
    } else {
        sh_init(&t->merges, 64);
    }

    *out = t;
    return 0;
}

void tokenizer_free(tokenizer* t) {
    if (!t) return;
    for (int i = 0; i < t->n_vocab; i++) free(t->inv_vocab[i]);
    free(t->inv_vocab);
    free(t->token_type);
    sh_free(&t->vocab);
    sh_free(&t->merges);
    sh_free(&t->specials);
    free(t);
}

int tokenizer_vocab_size(const tokenizer* t) { return t->n_vocab; }

int tokenizer_lookup(const tokenizer* t, const char* token_text) {
    int id;
    if (sh_get(&t->vocab, token_text, strlen(token_text), &id)) return id;
    return -1;
}

// ============================================================================
// Pre-tokenization (GPT-2 regex approximation, ASCII-focused)
//
// Pattern: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+
//
// We approximate by classifying each byte into one of {space, letter, digit,
// other} and grouping runs. For UTF-8 multibyte chars we treat continuation
// bytes as part of the same class as the lead byte (so accented letters stay
// in the "letter" class, etc.).
// ============================================================================

typedef enum { CL_SPACE, CL_LETTER, CL_DIGIT, CL_OTHER } char_class;

static char_class classify(int b, int prev_was_space) {
    (void)prev_was_space;
    if (b == ' ' || b == '\t' || b == '\n' || b == '\r' || b == '\v' || b == '\f')
        return CL_SPACE;
    if ((b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')) return CL_LETTER;
    if (b >= '0' && b <= '9') return CL_DIGIT;
    // Treat UTF-8 lead bytes (>=0xC0) as letters for grouping purposes.
    if (b >= 0x80) return CL_LETTER;
    return CL_OTHER;
}

// Emit a chunk: [start, end) bytes of input. Caller converts to BPE.
typedef void (*emit_fn)(const char* text, int start, int end, void* user);

static void pretokenize(const char* text, size_t len, emit_fn emit, void* user) {
    size_t i = 0;
    while (i < len) {
        // Apostrophe contractions: 's 't 're 've 'm 'll 'd
        if (text[i] == '\'' && i + 1 < len) {
            char c1 = text[i + 1];
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                emit(text, (int)i, (int)i + 2, user);
                i += 2;
                continue;
            }
            if ((c1 == 'r' && i + 2 < len && text[i + 2] == 'e') ||
                (c1 == 'v' && i + 2 < len && text[i + 2] == 'e') ||
                (c1 == 'l' && i + 2 < len && text[i + 2] == 'l')) {
                emit(text, (int)i, (int)i + 3, user);
                i += 3;
                continue;
            }
        }

        // Optional single leading space.
        size_t j = i;
        int has_lead_space = 0;
        if (text[j] == ' ') {
            has_lead_space = 1;
            j++;
            if (j >= len) {
                // Trailing single space at end → emit as whitespace chunk.
                emit(text, (int)i, (int)len, user);
                i = len;
                break;
            }
        }
        char_class cls = classify((unsigned char)text[j], has_lead_space);
        if (cls == CL_SPACE) {
            // Run of whitespace.
            size_t k = j;
            while (k < len) {
                char_class c2 = classify((unsigned char)text[k], 0);
                if (c2 != CL_SPACE) break;
                k++;
            }
            emit(text, (int)i, (int)k, user);
            i = k;
            continue;
        }
        // Consume run of same class (continuation bytes fold in).
        size_t k = j;
        while (k < len) {
            unsigned char bk = (unsigned char)text[k];
            if (bk >= 0x80 && bk < 0xC0) { k++; continue; } // UTF-8 cont byte
            char_class c2 = classify(bk, 0);
            if (c2 != cls || c2 == CL_SPACE) break;
            k++;
            while (k < len && (unsigned char)text[k] >= 0x80 &&
                   (unsigned char)text[k] < 0xC0) k++;
        }
        emit(text, (int)i, (int)k, user);
        i = k;
    }
}

// ============================================================================
// BPE merge on a chunk
// ============================================================================

// Fragment buffer: each fragment is a heap-allocated UTF-8 string.
// We need scratch space; allocate per encode call.
typedef struct {
    char**  items;
    int*    lengths;
    int     n;
    int     cap;
} frag_buf;

static void fb_init(frag_buf* fb, int cap) {
    fb->items = malloc(cap * sizeof(char*));
    fb->lengths = malloc(cap * sizeof(int));
    fb->n = 0;
    fb->cap = cap;
}

static void fb_clear(frag_buf* fb) {
    for (int i = 0; i < fb->n; i++) free(fb->items[i]);
    fb->n = 0;
}

static void fb_free(frag_buf* fb) {
    fb_clear(fb);
    free(fb->items);
    free(fb->lengths);
}

static void fb_push(frag_buf* fb, const char* data, int len) {
    if (fb->n >= fb->cap) {
        fb->cap *= 2;
        fb->items = realloc(fb->items, fb->cap * sizeof(char*));
        fb->lengths = realloc(fb->lengths, fb->cap * sizeof(int));
    }
    char* copy = malloc(len);
    memcpy(copy, data, len);
    fb->items[fb->n] = copy;
    fb->lengths[fb->n] = len;
    fb->n++;
}

// BPE: repeatedly merge the lowest-rank adjacent pair.
static void bpe_merge(const tokenizer* t, frag_buf* fb) {
    if (fb->n < 2) return;

    // Scratch for building pair keys.
    char keybuf[512];

    while (fb->n > 1) {
        int best_rank = INT32_MAX;
        int best_i = -1;
        for (int i = 0; i + 1 < fb->n; i++) {
            int la = fb->lengths[i];
            int lb = fb->lengths[i + 1];
            if (la + 1 + lb > (int)sizeof(keybuf)) continue;
            memcpy(keybuf, fb->items[i], la);
            keybuf[la] = ' ';
            memcpy(keybuf + la + 1, fb->items[i + 1], lb);
            int r;
            if (sh_get(&t->merges, keybuf, la + 1 + lb, &r)) {
                if (r < best_rank) {
                    best_rank = r;
                    best_i = i;
                }
            }
        }
        if (best_i < 0) break;

        // Merge pair at best_i: replace items[best_i] with concatenated,
        // remove items[best_i+1].
        int la = fb->lengths[best_i];
        int lb = fb->lengths[best_i + 1];
        char* merged = malloc(la + lb);
        memcpy(merged, fb->items[best_i], la);
        memcpy(merged + la, fb->items[best_i + 1], lb);
        free(fb->items[best_i]);
        fb->items[best_i] = merged;
        fb->lengths[best_i] = la + lb;
        free(fb->items[best_i + 1]);
        // Shift down.
        for (int j = best_i + 1; j + 1 < fb->n; j++) {
            fb->items[j] = fb->items[j + 1];
            fb->lengths[j] = fb->lengths[j + 1];
        }
        fb->n--;
    }
}

// ============================================================================
// Encode entry point
// ============================================================================

typedef struct {
    const tokenizer* t;
    int* out;
    int  cap;
    int  n;
    int  truncated;
    frag_buf fb;
} encode_state;

static void emit_chunk(const char* text, int start, int end, void* user) {
    encode_state* s = (encode_state*)user;
    if (s->truncated) return;

    fb_clear(&s->fb);

    // Seed fragments: one per input byte, mapped through byte_to_unicode
    // and encoded as UTF-8.
    char utf8buf[8];
    for (int i = start; i < end; i++) {
        int b = (unsigned char)text[i];
        int cp = g_b2u_byte_to_cp[b];
        int n = cp_to_utf8(cp, utf8buf);
        fb_push(&s->fb, utf8buf, n);
    }

    bpe_merge(s->t, &s->fb);

    // Look up each fragment in vocab.
    for (int i = 0; i < s->fb.n; i++) {
        if (s->n >= s->cap) { s->truncated = 1; return; }
        int id;
        if (sh_get(&s->t->vocab, s->fb.items[i], s->fb.lengths[i], &id)) {
            s->out[s->n++] = id;
        } else {
            // Unknown — fall back to byte token if present.
            // For SmolLM2 we expect every byte to be in vocab as a single
            // codepoint fragment, so this should rarely fire.
            int unk = tokenizer_lookup(s->t, "<|unk|>");
            if (unk < 0) unk = 0;
            s->out[s->n++] = unk;
        }
    }
}

// Split input on known special tokens, emitting literal segments.
static int emit_special_or_chunk(const tokenizer* t, const char* text,
                                 int* out, int max_out, int* n_out) {
    // For each special token in our table, scan input; this is O(n_specials *
    // text_len). Specials are typically <100, text typically <8k → fine.
    encode_state s;
    s.t = t;
    s.out = out;
    s.cap = max_out;
    s.n = 0;
    s.truncated = 0;
    fb_init(&s.fb, 64);

    size_t text_len = strlen(text);
    size_t i = 0;
    while (i < text_len) {
        // Try to match a special token at position i.
        int matched_id = -1;
        int matched_len = 0;
        for (size_t h = 0; h < t->specials.capacity; h++) {
            if (!t->specials.entries[h].key) continue;
            int klen = (int)t->specials.entries[h].key_len;
            if ((int)(text_len - i) < klen) continue;
            if (memcmp(text + i, t->specials.entries[h].key, klen) == 0) {
                // Prefer the longest match.
                if (klen > matched_len) {
                    matched_len = klen;
                    matched_id = t->specials.entries[h].value;
                }
            }
        }
        if (matched_id >= 0) {
            if (s.n >= s.cap) { s.truncated = 1; break; }
            s.out[s.n++] = matched_id;
            i += matched_len;
            continue;
        }
        // No special here — scan forward to next special or end.
        size_t j = i + 1;
        while (j < text_len) {
            int hit = 0;
            for (size_t h = 0; h < t->specials.capacity; h++) {
                if (!t->specials.entries[h].key) continue;
                int klen = (int)t->specials.entries[h].key_len;
                if ((int)(text_len - j) < klen) continue;
                if (memcmp(text + j, t->specials.entries[h].key, klen) == 0) {
                    hit = 1; break;
                }
            }
            if (hit) break;
            j++;
        }
        // BPE-encode [i, j).
        pretokenize(text + i, j - i, emit_chunk, &s);
        i = j;
    }

    fb_free(&s.fb);
    *n_out = s.n;
    return s.truncated ? -1 : 0;
}

int tokenizer_encode(const tokenizer* t, const char* text,
                     int* out, int max_out) {
    int n_out = 0;
    emit_special_or_chunk(t, text, out, max_out, &n_out);
    return n_out;
}

// ============================================================================
// Decode
// ============================================================================

int tokenizer_decode(const tokenizer* t, int token_id,
                     char* buf, int max_buf) {
    if (token_id < 0 || token_id >= t->n_vocab) return -1;
    const char* s = t->inv_vocab[token_id];

    // Parse UTF-8 chars from s, reverse-map each codepoint to a byte.
    int n = 0;
    const unsigned char* p = (const unsigned char*)s;
    while (*p) {
        int cp;
        int adv;
        if (*p < 0x80) { cp = *p; adv = 1; }
        else if ((*p & 0xE0) == 0xC0) {
            cp = (*p & 0x1F) << 6 | (p[1] & 0x3F);
            adv = 2;
        } else if ((*p & 0xF0) == 0xE0) {
            cp = (*p & 0x0F) << 12 | (p[1] & 0x3F) << 6 | (p[2] & 0x3F);
            adv = 3;
        } else {
            cp = (*p & 0x07) << 18 | (p[1] & 0x3F) << 12 |
                 (p[2] & 0x3F) << 6 | (p[3] & 0x3F);
            adv = 4;
        }
        p += adv;
        if (cp >= 256 && cp < 768) {
            int b = g_b2u_unicode_to_byte[cp];
            if (b < 0) continue;
            if (n + 1 > max_buf) return -1;
            buf[n++] = (char)b;
        } else if (cp < 256) {
            // Printable maps to itself.
            if (n + 1 > max_buf) return -1;
            buf[n++] = (char)cp;
        } else {
            // Out of range — emit '?' as fallback.
            if (n + 1 > max_buf) return -1;
            buf[n++] = '?';
        }
    }
    return n;
}
