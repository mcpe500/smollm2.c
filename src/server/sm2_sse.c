// sm2_sse.c - Server-Sent Events for streaming

#include <stdio.h>
#include <string.h>

// Send SSE data event
int sse_send(int fd, const char* event, const char* data) {
    char buf[1024];
    int len = snprintf(buf, sizeof(buf), "event: %s\ndata: %s\n\n", event, data);
    
    if (write(fd, buf, len) < 0) return -1;
    return 0;
}

// Send SSE token
int sse_send_token(int fd, const char* token, int token_id) {
    char data[256];
    snprintf(data, sizeof(data), "{\"token\":\"%s\",\"id\":%d}", token, token_id);
    return sse_send(fd, "message", data);
}

// Send SSE done
int sse_send_done(int fd) {
    return sse_send(fd, "done", "");
}

// Send SSE error
int sse_send_error(int fd, const char* error) {
    char data[256];
    snprintf(data, sizeof(data), "{\"error\":\"%s\"}", error);
    return sse_send(fd, "error", data);
}