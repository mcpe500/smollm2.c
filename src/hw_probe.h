// hw_probe.h — hardware capabilities for studio training
#ifndef HW_PROBE_H
#define HW_PROBE_H

typedef struct {
    long mem_total_kb;
    long mem_avail_kb;
    int  max_seq_advised;
    int  max_batch_advised;
    int  fullft_allowed;
    int  qlora_recommended;
} hw_caps;

void hw_probe(hw_caps* c);
void hw_print(const hw_caps* c);

#endif // HW_PROBE_H