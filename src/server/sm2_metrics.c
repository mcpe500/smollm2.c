// sm2_metrics.c - Prometheus metrics

#include <stdio.h>
#include <time.h>

typedef struct {
    uint64_t total_requests;
    uint64_t total_tokens;
    uint64_t total_prefill_tokens;
    uint64_t total_decode_tokens;
    double total_time_ms;
    int active_requests;
} sm2_metrics;

static sm2_metrics g_metrics = {0};

void sm2_metrics_inc_request() {
    g_metrics.total_requests++;
}

void sm2_metrics_add_tokens(int prefill, int decode) {
    g_metrics.total_prefill_tokens += prefill;
    g_metrics.total_decode_tokens += decode;
    g_metrics.total_tokens += prefill + decode;
}

void sm2_metrics_inc_active() {
    g_metrics.active_requests++;
}

void sm2_metrics_dec_active() {
    if (g_metrics.active_requests > 0) g_metrics.active_requests--;
}

void sm2_metrics_add_time(double ms) {
    g_metrics.total_time_ms += ms;
}

int sm2_metrics_prometheus(char* out, size_t max_len) {
    int n = 0;
    
    n += snprintf(out + n, max_len - n,
        "# HELP smollm2_requests_total Total chat completions requests\n"
        "# TYPE smollm2_requests_total counter\n"
        "smollm2_requests_total %lu\n\n",
        g_metrics.total_requests);
    
    n += snprintf(out + n, max_len - n,
        "# HELP smollm2_tokens_total Total tokens processed\n"
        "# TYPE smollm2_tokens_total counter\n"
        "smollm2_tokens_total %lu\n"
        "smollm2_tokens_prefill %lu\n"
        "smollm2_tokens_decode %lu\n\n",
        g_metrics.total_tokens,
        g_metrics.total_prefill_tokens,
        g_metrics.total_decode_tokens);
    
    double avg_latency = g_metrics.total_requests > 0 ?
        g_metrics.total_time_ms / g_metrics.total_requests : 0;
    
    n += snprintf(out + n, max_len - n,
        "# HELP smollm2_avg_latency_ms Average latency per request\n"
        "# TYPE smollm2_avg_latency_ms gauge\n"
        "smollm2_avg_latency_ms %.2f\n\n",
        avg_latency);
    
    n += snprintf(out + n, max_len - n,
        "# HELP smollm2_active_requests Current active requests\n"
        "# TYPE smollm2_active_requests gauge\n"
        "smollm2_active_requests %d\n",
        g_metrics.active_requests);
    
    return n;
}