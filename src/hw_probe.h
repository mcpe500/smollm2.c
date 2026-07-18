// hw_probe.h — hardware capabilities for studio training and inference.
// Phase A0: extended for HW-aware auto-config + WebUI JSON output.
#ifndef HW_PROBE_H
#define HW_PROBE_H

#include <stddef.h>

/* Memory thresholds in MB. Used by training gates and watchdog. */
#define MEM_START_LORA_MB    800
#define MEM_START_QLORA_MB   900
#define MEM_START_FULLFT_MB  2560
#define MEM_EMERGENCY_MB     150

typedef struct {
    /* Memory in kB (backwards compat with existing callers). */
    long mem_total_kb;
    long mem_avail_kb;
    /* New: same in MB (convenience) + VmRSS. */
    long mem_total_mb;
    long mem_available_mb;
    long vm_rss_mb;

    /* CPU. */
    int  cpu_cores;
    int  cpu_neon;       /* ARM NEON / asimd */
    int  cpu_dotprod;    /* ARM asimddp */
    int  cpu_sve;
    int  cpu_atomics;
    char cpu_model[128];
    int  cpu_freq_max_mhz;

    /* Derived advisory (filled by hw_advice). */
    int  max_seq_advised;
    int  max_batch_advised;
    int  max_rank_advised;
    int  fullft_allowed;
    int  qlora_recommended;
    int  lora_min_mb;
    int  qlora_min_mb;
    int  fullft_min_mb;
} hw_caps;

/* Probe hardware (mem + cpu), then compute derived advisory.
 * Reads SMOLLIM2_SIM_MEM_KB env var if set (for deterministic tests). */
void hw_probe(hw_caps* c);

/* Recompute advisory fields from mem + cpu. Called internally by hw_probe
 * but exposed for callers that tweak mem_avail_kb after probe. */
void hw_advice(hw_caps* c);

/* Human-readable multi-line print (backwards compat). */
void hw_print(const hw_caps* c);

/* JSON output. Caller frees the returned string. */
char* hw_json(const hw_caps* c);

/* Suggestion string. Caller frees. */
char* hw_suggest(const hw_caps* c);

#endif // HW_PROBE_H
