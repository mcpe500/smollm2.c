#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "smollm2.h"

int main() {
    sm2_model* model;
    if (sm2_load_model("./smollm2-135m.sm2", &model) != 0) return 1;
    
    sm2_tokenizer* tok = model->tokenizer;
    
    // Test encoding
    char* words[] = {"hello", "world", "the", "hello world", "a"};
    int n_words = 5;
    
    for (int w = 0; w < n_words; w++) {
        int ids[64];
        int n = sm2_tokenizer_encode(tok, words[w], ids, 64);
        printf("'%s' -> %d tokens: ", words[w], n);
        for (int i = 0; i < n; i++) printf("%d ", ids[i]);
        
        // Decode back
        char* decoded = sm2_tokenizer_decode(tok, ids, n);
        printf(" -> '%s'\n", decoded ? decoded : "(null)");
        if (decoded) free(decoded);
    }
    
    sm2_free_model(model);
    return 0;
}
