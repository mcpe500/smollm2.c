#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "include/smollm2.h"

int main() {
    sm2_model* model;
    if (sm2_load_model("smollm2-135m-v2.sm2", &model) != 0) return 1;
    
    // Look for tokens that could represent bytes
    // In GPT-2 style BPE, single-byte tokens are stored
    printf("Looking for single-byte tokens:\n");
    for (int i = 0; i < model->vocab_size; i++) {
        if (model->tokenizer && model->tokenizer->tokens && model->tokenizer->tokens[i]) {
            char* t = model->tokenizer->tokens[i];
            int len = strlen(t);
            // Single character tokens (bytes)
            if (len == 1) {
                unsigned char c = (unsigned char)t[0];
                if (c < 32 || c > 126) {
                    printf("  token %d: single byte 0x%02x ('%c')\n", i, c, c < 32 ? '.' : c);
                }
            }
            // Two-byte tokens that could be UTF-8 representations
            else if (len == 2 && (unsigned char)t[0] == 0xC4 && (unsigned char)t[1] == 0x80) {
                printf("  token %d: represents byte 0 (NULL)\n", i);
            }
        }
    }
    
    // Check what byte_to_token says for key bytes
    printf("\nbyte_to_token mapping for key bytes:\n");
    unsigned char test_bytes[] = {0, 1, 2, 3, 9, 10, 13, 32, 97}; // NULL, SOH, STX, ETX, TAB, LF, CR, SPACE, 'a'
    for (int i = 0; i < 9; i++) {
        int tok = model->tokenizer->byte_to_token[test_bytes[i]];
        printf("  byte 0x%02x (%3d): -> token %d", test_bytes[i], test_bytes[i], tok);
        if (model->tokenizer->tokens && tok >= 0 && tok < model->vocab_size && model->tokenizer->tokens[tok]) {
            printf(" = \"%s\"", model->tokenizer->tokens[tok]);
        }
        printf("\n");
    }
    
    sm2_free_model(model);
    return 0;
}
