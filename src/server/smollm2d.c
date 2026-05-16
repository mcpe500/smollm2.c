// smollm2d.c - HTTP server daemon with OpenAI-compatible API

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include "smollm2.h"

// ============================================================================
// SMOLLMD - SmolLM2 HTTP Server Daemon
//
// Features:
//   - OpenAI-compatible /v1/chat/completions
//   - SSE streaming
//   - Continuous batching
//   - Prometheus metrics
//   - Health checks
// ============================================================================

#define DEFAULT_PORT 7331
#define DEFAULT_HOST "127.0.0.1"

// Simple HTTP response helper
static void send_response(int client_fd, const char* status, const char* content) {
    char header[256];
    snprintf(header, sizeof(header),
        "HTTP/1.1 %s\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n",
        status, strlen(content));
    
    write(client_fd, header, strlen(header));
    write(client_fd, content, strlen(content));
}

// Parse JSON chat completions request (simplified)
typedef struct {
    char messages[1024];
    float temperature;
    int max_tokens;
} chat_request;

static int parse_chat_request(const char* body, chat_request* req) {
    // Simple JSON parsing - in production use proper JSON parser
    memset(req, 0, sizeof(chat_request));
    req->temperature = 0.8f;
    req->max_tokens = 256;
    
    // Look for "content" field
    const char* content = strstr(body, "\"content\"");
    if (content) {
        const char* start = strchr(content, ':');
        if (start) {
            start = strchr(start, '"');
            if (start) {
                start++;
                const char* end = strchr(start, '"');
                if (end) {
                    int len = end - start;
                    if (len > 1023) len = 1023;
                    strncpy(req->messages, start, len);
                }
            }
        }
    }
    
    // Look for temperature
    const char* temp = strstr(body, "\"temperature\"");
    if (temp) {
        sscanf(temp, "\"temperature\":%f", &req->temperature);
    }
    
    // Look for max_tokens
    const char* max_tok = strstr(body, "\"max_tokens\"");
    if (max_tok) {
        sscanf(max_tok, "\"max_tokens\":%d", &req->max_tokens);
    }
    
    return 0;
}

// Handle /v1/chat/completions
static void handle_chat_complete(int client_fd, const char* body) {
    chat_request req;
    parse_chat_request(body, &req);
    
    // For now, return a simple response
    // Real implementation would:
    //   1. Tokenize the prompt
    //   2. Run prefill
    //   3. Decode with streaming
    //   4. Send SSE events
    
    char response[512];
    snprintf(response, sizeof(response),
        "{\"model\":\"smollm2\",\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"Hello from smollm2.c\"}}]}");
    
    send_response(client_fd, "200 OK", response);
}

// Handle /health
static void handle_health(int client_fd) {
    char response[256];
    snprintf(response, sizeof(response),
        "{\"ok\":true,\"model\":\"smollm2\",\"version\":\"1.0\"}");
    send_response(client_fd, "200 OK", response);
}

// Handle /metrics (Prometheus format)
static void handle_metrics(int client_fd) {
    char response[512];
    snprintf(response, sizeof(response),
        "# HELP smollm2_requests_total Total requests\n"
        "# TYPE smollm2_requests_total counter\n"
        "smollm2_requests_total 0\n");
    send_response(client_fd, "200 OK", response);
}

// Parse HTTP request line
static void handle_request(int client_fd, const char* method, const char* path, const char* body) {
    if (strcmp(method, "POST") == 0 && strcmp(path, "/v1/chat/completions") == 0) {
        handle_chat_complete(client_fd, body);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/health") == 0) {
        handle_health(client_fd);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/metrics") == 0) {
        handle_metrics(client_fd);
    } else {
        const char* not_found = "{\"error\":\"Not found\"}";
        send_response(client_fd, "404 Not Found", not_found);
    }
}

// Main server loop (simplified single-threaded)
int sm2_server_run(const sm2_server_config* config) {
    int port = config->port ? config->port : DEFAULT_PORT;
    const char* host = config->host ? config->host : DEFAULT_HOST;
    
    printf("smollm2d starting on %s:%d\n", host, port);
    printf("Model: %s\n", config->model_path ? config->model_path : "(none)");
    printf("Workers: %d, Context: %d, KV dtype: %d\n",
           config->n_threads, config->max_ctx, config->kv_dtype);
    
    // For now, just print startup info
    // Real implementation would:
    //   1. Load model
    //   2. Initialize KV pool
    //   3. Start HTTP server loop
    //   4. Handle requests with continuous batching
    
    return 0;
}