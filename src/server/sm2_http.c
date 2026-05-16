// sm2_http.c - HTTP parsing utilities

#include <stdio.h>
#include <string.h>

typedef struct {
    char method[16];
    char path[256];
    char body[4096];
} http_request;

int http_parse_request(const char* data, http_request* req) {
    memset(req, 0, sizeof(http_request));
    
    // Find headers/body separator
    const char* sep = strstr(data, "\r\n\r\n");
    if (!sep) return -1;
    
    // Parse request line
    const char* line_start = data;
    const char* line_end = strstr(data, "\r\n");
    if (!line_end) return -1;
    
    int line_len = line_end - line_start;
    if (line_len > sizeof(req->method) + sizeof(req->path) + 1) return -1;
    
    char line[512];
    strncpy(line, line_start, line_len);
    line[line_len] = '\0';
    
    // Parse method and path
    char* space1 = strchr(line, ' ');
    if (!space1) return -1;
    
    int method_len = space1 - line;
    strncpy(req->method, line, method_len);
    req->method[method_len] = '\0';
    
    char* space2 = strchr(space1 + 1, ' ');
    if (!space2) return -1;
    
    int path_len = space2 - (space1 + 1);
    strncpy(req->path, space1 + 1, path_len);
    req->path[path_len] = '\0';
    
    // Extract body
    int body_len = strlen(sep + 4);
    if (body_len > sizeof(req->body) - 1) body_len = sizeof(req->body) - 1;
    strncpy(req->body, sep + 4, body_len);
    req->body[body_len] = '\0';
    
    return 0;
}

int http_response(int fd, int status, const char* content_type, const char* body, size_t body_len) {
    char header[256];
    int header_len = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n",
        status,
        status == 200 ? "OK" : "ERROR",
        content_type,
        body_len);
    
    if (write(fd, header, header_len) < 0) return -1;
    if (write(fd, body, body_len) < 0) return -1;
    
    return 0;
}