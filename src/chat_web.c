// chat_web.c - Web server mode for SmolLM2
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <time.h>
#include "chat_history.h"
#include "smollm2.h"

#define DEFAULT_PORT 7331
#define DEFAULT_HOST "127.0.0.1"
#define BUFFER_SIZE 65536
#define WEB_ROOT "web"

// Time measurement
static double time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// Send HTTP response
static void http_response(int fd, int status, const char* content_type,
                         const char* body, size_t body_len) {
    char header[1024];
    int header_len = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "\r\n",
        status,
        status == 200 ? "OK" : status == 404 ? "Not Found" : "Error",
        content_type,
        body_len);

    write(fd, header, header_len);
    if (body && body_len > 0) {
        write(fd, body, body_len);
    }
}

// Send SSE event
static void sse_send(int fd, const char* event, const char* data) {
    char buf[4096];
    int len = snprintf(buf, sizeof(buf), "event: %s\ndata: %s\n\n", event, data);
    write(fd, buf, len);
}

static void sse_send_token(int fd, const char* token_str, int token_id) {
    char buf[4096];
    int len = snprintf(buf, sizeof(buf),
        "event: token\ndata: {\"token\":%d,\"text\":\"", token_id);

    // Escape JSON
    char* dst = buf + len;
    for (const char* src = token_str; *src && dst - buf < 3800; src++) {
        unsigned char c = (unsigned char)*src;
        if (c == '"') {
            *dst++ = '\\';
            *dst++ = '"';
        } else if (c == '\\') {
            *dst++ = '\\';
            *dst++ = '\\';
        } else if (c == 0xC4 && (unsigned char)src[1] == 0xA0) {
            *dst++ = ' ';
            src++;
        } else if (c == 0xC4 && (unsigned char)src[1] == 0x8A) {
            *dst++ = '\n';
            src++;
        } else if (c >= 32 && c < 127) {
            *dst++ = *src;
        } else {
            dst += snprintf(dst, 10, "\\x%02X", c);
        }
    }
    len = dst - buf;
    len += snprintf(buf + len, sizeof(buf) - len, "\"}\n\n");
    write(fd, buf, len);
}

// Parse HTTP request
static int parse_request(const char* buf, char* method, char* path, char* body, int* body_len) {
    // Method
    const char* p = buf;
    const char* e = strchr(p, ' ');
    if (!e) return -1;
    int method_len = e - p;
    if (method_len > 15) method_len = 15;
    strncpy(method, p, method_len);
    method[method_len] = '\0';

    // Path
    p = e + 1;
    e = strchr(p, ' ');
    if (!e) return -1;
    int path_len = e - p;
    if (path_len > 255) path_len = 255;
    strncpy(path, p, path_len);
    path[path_len] = '\0';

    // Body (look for \r\n\r\n)
    const char* header_end = strstr(buf, "\r\n\r\n");
    if (header_end) {
        *body_len = 0;
        const char* body_start = header_end + 4;
        const char* next_line = strstr(body_start, "\r\n");
        if (next_line) {
            // Get Content-Length
            const char* cl = strstr(buf, "Content-Length: ");
            if (cl && cl < header_end) {
                *body_len = atoi(cl + 16);
                if (*body_len > BUFFER_SIZE - (body_start - buf) - 1) {
                    *body_len = BUFFER_SIZE - (body_start - buf) - 1;
                }
            }
            memcpy(body, body_start, *body_len);
            body[*body_len] = '\0';
        }
    }

    return 0;
}

// Extract JSON value for key
static int json_extract_str(const char* json, const char* key, char* out, int out_size) {
    char search[64];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char* p = strstr(json, search);
    if (!p) return -1;

    p = strchr(p, ':');
    if (!p) return -1;
    p++;

    // Skip whitespace and quotes
    while (*p == ' ' || *p == '\t' || *p == '"') p++;

    int len = 0;
    while (*p && *p != '"' && *p != ',' && *p != '}' && len < out_size - 1) {
        if (*p == '\\' && p[1]) {
            p++;
            if (*p == 'n') out[len++] = '\n';
            else if (*p == 't') out[len++] = '\t';
            else if (*p == '"') out[len++] = '"';
            else if (*p == '\\') out[len++] = '\\';
            else out[len++] = *p;
        } else {
            out[len++] = *p;
        }
        p++;
    }
    out[len] = '\0';

    return len;
}

// Handle chat API request
static int handle_chat(int fd, const char* body, int body_len,
                       sm2_model* model, sm2_tokenizer* tok,
                       const cli_args* args, int streaming) {
    // Extract message from JSON
    char message[4096] = {0};

    // Try "messages" array format first (OpenAI compatible)
    const char* content_p = strstr(body, "\"content\"");
    if (content_p) {
        content_p = strchr(content_p, ':');
        if (content_p) {
            content_p++;
            while (*content_p == ' ' || *content_p == '\t') content_p++;
            if (*content_p == '"') content_p++;

            int i = 0;
            while (*content_p && *content_p != '"' && i < (int)sizeof(message) - 1) {
                if (*content_p == '\\' && content_p[1]) {
                    content_p++;
                    if (*content_p == 'n') message[i++] = '\n';
                    else if (*content_p == 't') message[i++] = '\t';
                    else if (*content_p == '"') message[i++] = '"';
                    else message[i++] = *content_p;
                } else {
                    message[i++] = *content_p;
                }
                content_p++;
            }
            message[i] = '\0';
        }
    }

    // Also check for "prompt" key (simple fallback)
    if (!message[0]) {
        json_extract_str(body, "prompt", message, sizeof(message));
    }

    if (!message[0]) {
        char err[256];
        snprintf(err, sizeof(err), "{\"error\":\"No message found\"}");
        http_response(fd, 400, "application/json", err, strlen(err));
        return -1;
    }

    // Tokenize
    char full_prompt[8192];
    snprintf(full_prompt, sizeof(full_prompt),
        "<|im_start|>system\nGive short answers. Say only the number for math. Examples: 2+2=4, 5*5=25.<|im_end|>\n"
        "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
        message);

    int tokens[4096];
    int n_tokens = sm2_tokenizer_encode(tok, full_prompt, tokens, 4096);

    if (n_tokens <= 0) {
        char err[256];
        snprintf(err, sizeof(err), "{\"error\":\"Tokenization failed\"}");
        http_response(fd, 400, "application/json", err, strlen(err));
        return -1;
    }

    // Create context
    sm2_context* ctx;
    if (sm2_create_context(model, &ctx) != 0) {
        char err[256];
        snprintf(err, sizeof(err), "{\"error\":\"Failed to create context\"}");
        http_response(fd, 500, "application/json", err, strlen(err));
        return -1;
    }

    ctx->params.temperature = args->temperature;
    ctx->params.top_p = args->top_p;
    ctx->params.top_k = args->top_k;
    ctx->params.max_context = 8192;
    ctx->params.max_output = 512;
    ctx->params.repetition_penalty = args->repetition_penalty;
    ctx->params.penalty_window = 32;

    if (sm2_prefill(ctx, tokens, n_tokens) != 0) {
        sm2_free_context(ctx);
        char err[256];
        snprintf(err, sizeof(err), "{\"error\":\"Prefill failed\"}");
        http_response(fd, 500, "application/json", err, strlen(err));
        return -1;
    }

    if (streaming) {
        // SSE streaming
        char sse_header[512];
        int header_len = snprintf(sse_header, sizeof(sse_header),
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Connection: close\r\n"
            "\r\n");
        write(fd, sse_header, header_len);

        sse_send(fd, "start", "{}");

        char response[8192];
        int resp_len = 0;
        double t0 = time_ms();

        while (resp_len < (int)sizeof(response) - 1) {
            int token;
            if (sm2_decode_next(ctx, &token) != 0) break;

            if (token < 3) break; // EOS

            char* decoded = sm2_tokenizer_decode(tok, &token, 1);
            if (decoded) {
                sse_send_token(fd, decoded, token);

                // Accumulate for final response
                for (const char* p = decoded; *p && resp_len < (int)sizeof(response) - 1; p++) {
                    unsigned char c = (unsigned char)*p;
                    if (c == 0xC4 && (unsigned char)p[1] == 0xA0) {
                        response[resp_len++] = ' ';
                        p++;
                    } else if (c == 0xC4 && (unsigned char)p[1] == 0x8A) {
                        response[resp_len++] = '\n';
                        p++;
                    } else {
                        response[resp_len++] = *p;
                    }
                }
                free(decoded);
            }
        }

        response[resp_len] = '\0';
        sm2_free_context(ctx);

        double dt = time_ms() - t0;
        char done[512];
        snprintf(done, sizeof(done),
            "{\"tokens\":%d,\"time_ms\":%.1f,\"text\":\"%s\"}",
            resp_len, dt, response);
        sse_send(fd, "done", done);

    } else {
        // Non-streaming - collect all tokens then return JSON
        char response[8192];
        int resp_len = 0;
        double t0 = time_ms();

        while (resp_len < (int)sizeof(response) - 1) {
            int token;
            if (sm2_decode_next(ctx, &token) != 0) break;

            if (token < 3) break; // EOS

            char* decoded = sm2_tokenizer_decode(tok, &token, 1);
            if (decoded) {
                for (const char* p = decoded; *p && resp_len < (int)sizeof(response) - 1; p++) {
                    unsigned char c = (unsigned char)*p;
                    if (c == 0xC4 && (unsigned char)p[1] == 0xA0) {
                        response[resp_len++] = ' ';
                        p++;
                    } else if (c == 0xC4 && (unsigned char)p[1] == 0x8A) {
                        response[resp_len++] = '\n';
                        p++;
                    } else {
                        response[resp_len++] = *p;
                    }
                }
                free(decoded);
            }
        }

        response[resp_len] = '\0';
        sm2_free_context(ctx);

        double dt = time_ms() - t0;

        // Build JSON response
        char json[16384];
        int json_len = snprintf(json, sizeof(json),
            "{\"model\":\"smollm2\",\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"");
        json_len--; // Don't include null terminator

        // Escape and append response
        for (int i = 0; response[i] && json_len < (int)sizeof(json) - 3; i++) {
            char c = response[i];
            if (c == '"') {
                json[json_len++] = '\\';
                json[json_len++] = '"';
            } else if (c == '\\') {
                json[json_len++] = '\\';
                json[json_len++] = '\\';
            } else if (c == '\n') {
                json[json_len++] = '\\';
                json[json_len++] = 'n';
            } else {
                json[json_len++] = c;
            }
        }

        int remaining = snprintf(json + json_len, sizeof(json) - json_len,
            "\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d},\"latency_ms\":%.1f}",
            n_tokens, resp_len, n_tokens + resp_len, dt);

        http_response(fd, 200, "application/json", json, json_len + remaining);
    }

    return 0;
}

// Serve static file
static void serve_file(int fd, const char* path) {
    char full_path[512];
    snprintf(full_path, sizeof(full_path), "%s/%s", WEB_ROOT, path);

    // Security: prevent directory traversal
    if (strstr(path, "..") != NULL) {
        http_response(fd, 403, "text/plain", "Forbidden", 9);
        return;
    }

    FILE* f = fopen(full_path, "rb");
    if (!f) {
        http_response(fd, 404, "text/plain", "Not Found", 9);
        return;
    }

    // Get file size
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    // Determine content type
    const char* content_type = "application/octet-stream";
    if (strstr(path, ".html")) content_type = "text/html";
    else if (strstr(path, ".css")) content_type = "text/css";
    else if (strstr(path, ".js")) content_type = "application/javascript";
    else if (strstr(path, ".json")) content_type = "application/json";

    // Read file
    char* buffer = malloc(size);
    if (!buffer) {
        fclose(f);
        http_response(fd, 500, "text/plain", "Error", 5);
        return;
    }

    fread(buffer, size, 1, f);
    fclose(f);

    http_response(fd, 200, content_type, buffer, size);
    free(buffer);
}

// Main server loop
int run_chat_web(sm2_model* model, const cli_args* args) {
    int port = args->web_port > 0 ? args->web_port : DEFAULT_PORT;
    const char* host = args->web_host ? args->web_host : DEFAULT_HOST;

    printf("Starting SmolLM2 WebUI at http://%s:%d\n", host, port);

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket");
        return -1;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(server_fd);
        return -1;
    }

    if (listen(server_fd, 10) < 0) {
        perror("listen");
        close(server_fd);
        return -1;
    }

    printf("WebUI ready! Press Ctrl+C to stop.\n");

    while (1) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);

        if (client_fd < 0) {
            perror("accept");
            continue;
        }

        // Read request
        char buffer[BUFFER_SIZE];
        int n = read(client_fd, buffer, sizeof(buffer) - 1);

        if (n > 0) {
            buffer[n] = '\0';

            char method[16], path[256], body[BUFFER_SIZE];
            int body_len = 0;

            if (parse_request(buffer, method, path, body, &body_len) == 0) {
                printf("[%s] %s\n", method, path);

                if (strcmp(method, "GET") == 0 && strcmp(path, "/") == 0) {
                    serve_file(client_fd, "index.html");
                } else if (strcmp(method, "GET") == 0 && strncmp(path, "/static/", 8) == 0) {
                    serve_file(client_fd, path + 1);
                } else if (strcmp(method, "POST") == 0 && strcmp(path, "/chat/stream") == 0) {
                    handle_chat(client_fd, body, body_len, model, model->tokenizer, args, 1);
                } else if (strcmp(method, "POST") == 0 && strcmp(path, "/chat") == 0) {
                    handle_chat(client_fd, body, body_len, model, model->tokenizer, args, 0);
                } else if (strcmp(method, "GET") == 0 && strcmp(path, "/health") == 0) {
                    http_response(client_fd, 200, "application/json", "{\"status\":\"ok\"}", 15);
                } else {
                    http_response(client_fd, 404, "application/json", "{\"error\":\"Not found\"}", 19);
                }
            }
        }

        close(client_fd);
    }

    close(server_fd);
    return 0;
}