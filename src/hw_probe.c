// hw_probe.c — /proc/meminfo reader + heuristic for training caps
#include "hw_probe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void hw_probe(hw_caps* c) {
    memset(c, 0, sizeof(*c));
    FILE* f = fopen("/proc/meminfo", "r");
    if (!f) {
        c->mem_total_kb = 1;
        c->mem_avail_kb = 1;
        return;
    }
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        long kb = 0;
        char key[64];
        if (sscanf(line, "%63s %ld kB", key, &kb) == 2) {
            if (strcmp(key, "MemTotal:") == 0) c->mem_total_kb = kb;
            else if (strcmp(key, "MemAvailable:") == 0) c->mem_avail_kb = kb;
        }
    }
    fclose(f);
    if (c->mem_avail_kb <= 0) c->mem_avail_kb = c->mem_total_kb / 2;

    long mb = c->mem_avail_kb / 1024;
    c->max_seq_advised    = (mb > 1500) ? 512 : (mb > 800 ? 256 : 128);
    c->max_batch_advised  = (mb > 1500) ? 4 : (mb > 800 ? 2 : 1);
    c->fullft_allowed     = (c->mem_avail_kb >= 2.5L * 1024 * 1024) ? 1 : 0;
    c->qlora_recommended  = (mb < 1500) ? 1 : 0;
}

void hw_print(const hw_caps* c) {
    printf("hw: mem_total=%ld MB mem_avail=%ld MB\n",
           c->mem_total_kb / 1024, c->mem_avail_kb / 1024);
    printf("hw: max_seq_advised=%d max_batch_advised=%d\n",
           c->max_seq_advised, c->max_batch_advised);
    printf("hw: fullft_allowed=%d qlora_recommended=%d\n",
           c->fullft_allowed, c->qlora_recommended);
}