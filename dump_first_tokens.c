#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "include/smollm2.h"

int main() {
    FILE* f = fopen("smollm2-135m-v2.sm2", "rb");
    if (!f) return 1;
    
    // Skip header
    fseek(f, 256, SEEK_SET);
    
    // Read vocab
    uint32_t pos = 0;
    uint8_t data[256];
    uint32_t size = 256;
    fread(data, size, 1, f);
    
    printf("First 20 tokens from .sm2 file:\n");
    for (int i = 0; i < 20; i++) {
        if (pos + 4 > size) break;
        uint32_t len = *(uint32_t*)(data + pos);
        pos += 4;
        
        if (len > 0 && pos + len <= size) {
            printf("  token %2d: len=%2d bytes=\"", i, len);
            for (uint32_t j = 0; j < len && j < 30; j++) {
                printf("\\x%02x", data[pos + j]);
            }
            if (len > 30) printf("...");
            printf("\" str=\"");
            for (uint32_t j = 0; j < len && j < 30; j++) {
                char c = data[pos + j];
                printf("%c", c >= 32 && c < 127 ? c : '.');
            }
            printf("\"\n");
            pos += len;
        }
    }
    
    fclose(f);
    return 0;
}
