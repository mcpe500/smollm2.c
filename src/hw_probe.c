// hw_probe.c — /proc/meminfo + /proc/cpuinfo + getauxval reader for HW-aware training.
#include "hw_probe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <sys/auxv.h>
#ifndef HWCAP_ASIMD
#define HWCAP_ASIMD    (1 << 1)
#endif
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP  (1 << 20)
#endif
#ifndef HWCAP_SVE
#define HWCAP_SVE      (1 << 22)
#endif
#ifndef HWCAP_ATOMICS
#define HWCAP_ATOMICS  (1 << 8)
#endif
#endif

static void probe_mem(hw_caps* c) {
    c->mem_total_kb = 0;
    c->mem_avail_kb = 0;
    FILE* f = fopen("/proc/meminfo", "r");
    if (f) {
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
    }
    if (c->mem_avail_kb <= 0) c->mem_avail_kb = c->mem_total_kb / 2;

    /* Test override: SMOLLIM2_SIM_MEM_KB env var. */
    const char* sim = getenv("SMOLLIM2_SIM_MEM_KB");
    if (sim && *sim) {
        long v = atol(sim);
        if (v > 0) c->mem_avail_kb = v;
    }

    c->mem_total_mb = c->mem_total_kb / 1024;
    c->mem_available_mb = c->mem_avail_kb / 1024;

    /* VmRSS from /proc/self/status. */
    c->vm_rss_mb = 0;
    FILE* sf = fopen("/proc/self/status", "r");
    if (sf) {
        char line[256];
        while (fgets(line, sizeof(line), sf)) {
            long kb = 0;
            if (sscanf(line, "VmRSS: %ld kB", &kb) == 1) {
                c->vm_rss_mb = kb / 1024;
                break;
            }
        }
        fclose(sf);
    }
}

static void probe_cpu(hw_caps* c) {
    c->cpu_cores = 1;
    c->cpu_neon = 0;
    c->cpu_dotprod = 0;
    c->cpu_sve = 0;
    c->cpu_atomics = 0;
    c->cpu_model[0] = '\0';
    c->cpu_freq_max_mhz = 0;

#if defined(__aarch64__) || defined(__ARM_NEON)
    unsigned long h = getauxval(AT_HWCAP);
    if (h) {
        c->cpu_neon    = (h & HWCAP_ASIMD) ? 1 : 0;
        c->cpu_dotprod = (h & HWCAP_ASIMDDP) ? 1 : 0;
        c->cpu_sve     = (h & HWCAP_SVE) ? 1 : 0;
        c->cpu_atomics = (h & HWCAP_ATOMICS) ? 1 : 0;
    }
#endif

#ifdef _SC_NPROCESSORS_ONLN
    long ncpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpus > 0) c->cpu_cores = (int)ncpus;
#endif

    int model_set = 0;
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[512];
        while (fgets(line, sizeof(line), f)) {
            if (!model_set &&
                (strncmp(line, "model name", 10) == 0 ||
                 strncmp(line, "Hardware", 8) == 0)) {
                char* colon = strchr(line, ':');
                if (colon) {
                    colon++;
                    while (*colon == ' ') colon++;
                    char* nl = strchr(colon, '\n');
                    if (nl) *nl = '\0';
                    strncpy(c->cpu_model, colon, sizeof(c->cpu_model) - 1);
                    c->cpu_model[sizeof(c->cpu_model) - 1] = '\0';
                    model_set = 1;
                }
            }
            /* Fallback if getauxval unavailable. */
            if (strncmp(line, "Features", 8) == 0) {
                if (!c->cpu_neon    && strstr(line, "asimd"))    c->cpu_neon = 1;
                if (!c->cpu_dotprod && strstr(line, "asimddp"))  c->cpu_dotprod = 1;
                if (!c->cpu_sve     && strstr(line, "sve"))      c->cpu_sve = 1;
                if (!c->cpu_atomics && strstr(line, "atomics"))  c->cpu_atomics = 1;
            }
        }
        fclose(f);
    }

    FILE* cf = fopen("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", "r");
    if (cf) {
        int khz = 0;
        if (fscanf(cf, "%d", &khz) == 1 && khz > 0)
            c->cpu_freq_max_mhz = khz / 1000;
        fclose(cf);
    }
}

void hw_advice(hw_caps* c) {
    long mb = c->mem_available_mb;

    c->max_seq_advised   = (mb >= 3000) ? 1024 :
                           (mb >= 2000) ? 512  :
                           (mb >= 1000) ? 256  : 128;
    c->max_batch_advised = 1;  /* single-threaded training; gradient accumulation later */
    c->max_rank_advised  = (mb >= 2500) ? 32 :
                           (mb >= 1500) ? 16 : 8;

    c->lora_min_mb   = MEM_START_LORA_MB;
    c->qlora_min_mb  = MEM_START_QLORA_MB;
    c->fullft_min_mb = MEM_START_FULLFT_MB;

    c->fullft_allowed    = (mb >= MEM_START_FULLFT_MB) ? 1 : 0;
    c->qlora_recommended = (!c->fullft_allowed && mb >= MEM_START_QLORA_MB) ? 1 : 0;
}

void hw_probe(hw_caps* c) {
    memset(c, 0, sizeof(*c));
    probe_mem(c);
    probe_cpu(c);
    hw_advice(c);
}

void hw_print(const hw_caps* c) {
    printf("hw: cpu=%s cores=%d neon=%d dotprod=%d sve=%d atomics=%d freq_max=%d MHz\n",
           c->cpu_model[0] ? c->cpu_model : "(unknown)",
           c->cpu_cores, c->cpu_neon, c->cpu_dotprod, c->cpu_sve, c->cpu_atomics,
           c->cpu_freq_max_mhz);
    printf("hw: mem_total=%ld MB mem_avail=%ld MB vm_rss=%ld MB\n",
           c->mem_total_mb, c->mem_available_mb, c->vm_rss_mb);
    printf("hw: advisory max_seq=%d max_batch=%d max_rank=%d\n",
           c->max_seq_advised, c->max_batch_advised, c->max_rank_advised);
    printf("hw: gates lora_min=%d qlora_min=%d fullft_min=%d MB emergency=%d MB\n",
           c->lora_min_mb, c->qlora_min_mb, c->fullft_min_mb, MEM_EMERGENCY_MB);
    printf("hw: fullft_allowed=%d qlora_recommended=%d\n",
           c->fullft_allowed, c->qlora_recommended);
}

char* hw_json(const hw_caps* c) {
    size_t cap = 1024;
    char* s = (char*)malloc(cap);
    if (!s) return NULL;
    /* Escape cpu_model — it's from /proc/cpuinfo and could contain quotes
     * in theory. Simple escape: replace " with \. */
    char escaped[256];
    const char* src = c->cpu_model[0] ? c->cpu_model : "unknown";
    size_t i, j = 0;
    for (i = 0; src[i] && j < sizeof(escaped) - 2; i++) {
        if (src[i] == '"' || src[i] == '\\') escaped[j++] = '\\';
        if ((unsigned char)src[i] < 0x20) continue;
        escaped[j++] = src[i];
    }
    escaped[j] = '\0';

    snprintf(s, cap,
        "{"
        "\"cpu_model\":\"%s\","
        "\"cpu_cores\":%d,"
        "\"cpu_neon\":%d,"
        "\"cpu_dotprod\":%d,"
        "\"cpu_sve\":%d,"
        "\"cpu_atomics\":%d,"
        "\"cpu_freq_max_mhz\":%d,"
        "\"mem_total_mb\":%ld,"
        "\"mem_available_mb\":%ld,"
        "\"vm_rss_mb\":%ld,"
        "\"max_seq_advised\":%d,"
        "\"max_batch_advised\":%d,"
        "\"max_rank_advised\":%d,"
        "\"fullft_allowed\":%d,"
        "\"qlora_recommended\":%d,"
        "\"lora_min_mb\":%d,"
        "\"qlora_min_mb\":%d,"
        "\"fullft_min_mb\":%d,"
        "\"emergency_min_mb\":%d"
        "}\n",
        escaped, c->cpu_cores, c->cpu_neon, c->cpu_dotprod,
        c->cpu_sve, c->cpu_atomics, c->cpu_freq_max_mhz,
        c->mem_total_mb, c->mem_available_mb, c->vm_rss_mb,
        c->max_seq_advised, c->max_batch_advised, c->max_rank_advised,
        c->fullft_allowed, c->qlora_recommended,
        c->lora_min_mb, c->qlora_min_mb, c->fullft_min_mb,
        MEM_EMERGENCY_MB);
    return s;
}

char* hw_suggest(const hw_caps* c) {
    char mode[16];
    int rank = 0, seq = 0;
    const char* targets = "(none)";
    long mb = c->mem_available_mb;

    if (c->fullft_allowed) {
        snprintf(mode, sizeof(mode), "fullft");
        seq = c->max_seq_advised;
        targets = "(all weights)";
    } else if (mb >= MEM_START_LORA_MB) {
        snprintf(mode, sizeof(mode), "lora");
        rank = c->max_rank_advised;
        seq = c->max_seq_advised;
        targets = (mb >= 2500) ? "all" : "q,v";
    } else if (mb >= MEM_START_QLORA_MB) {
        snprintf(mode, sizeof(mode), "qlora");
        rank = c->max_rank_advised;
        seq = c->max_seq_advised;
        targets = "q,v";
    } else {
        snprintf(mode, sizeof(mode), "(none)");
        targets = "(insufficient RAM)";
    }

    char* s = (char*)malloc(512);
    if (!s) return NULL;
    snprintf(s, 512,
        "Given mem_available=%ld MB, vm_rss=%ld MB, cpu_cores=%d:\n"
        "  recommended: mode=%s rank=%d seq=%d targets=%s\n"
        "  gates (MB): lora_min=%d qlora_min=%d fullft_min=%d emergency=%d\n"
        "  fullft_allowed=%d qlora_recommended=%d\n",
        mb, c->vm_rss_mb, c->cpu_cores, mode, rank, seq, targets,
        c->lora_min_mb, c->qlora_min_mb, c->fullft_min_mb, MEM_EMERGENCY_MB,
        c->fullft_allowed, c->qlora_recommended);
    return s;
}
