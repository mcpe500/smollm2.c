// web.c — minimal HTTP WebUI for smollm2
//
// Two endpoints:
//   GET  /          → embedded HTML page
//   POST /generate  → SSE stream of token chunks, terminated with [DONE]
//
// Pure POSIX sockets, single-threaded, blocks per request.

#include "web.h"
#include "gguf.h"
#include "tokenizer.h"
#include "forward.h"
#include "sampling.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define MAX_REQ     65536
#define IM_END_TOKEN 2

// ---------------------------------------------------------------------------
// Embedded HTML/CSS/JS as a single C string literal
// ---------------------------------------------------------------------------
static const char* kHTML =
"<!DOCTYPE html>\n"
"<html lang=\"en\"><head><meta charset=\"utf-8\">"
"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
"<title>smollm2</title>"
"<style>"
"*{box-sizing:border-box;margin:0;padding:0}"
"body{font-family:system-ui,-apple-system,sans-serif;background:#0f1117;"
"color:#e6e6e6;height:100vh;display:flex;flex-direction:column}"
"header{padding:12px 16px;background:#161922;border-bottom:1px solid #252836;"
"font-weight:600;font-size:14px;color:#9ca3af}"
"header b{color:#60a5fa}"
"#chat{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px}"
".msg{padding:10px 14px;border-radius:8px;max-width:80%;white-space:pre-wrap;"
"word-wrap:break-word;line-height:1.5;font-size:14px}"
".user{background:#1e3a5f;align-self:flex-end}"
".bot{background:#1f2937;align-self:flex-start;border:1px solid #2d3748}"
".bot *{font-family:inherit}"
"#input-area{padding:12px 16px;background:#161922;border-top:1px solid #252836;"
"display:flex;gap:8px}"
"#prompt{flex:1;padding:10px 14px;background:#1f2937;border:1px solid #2d3748;"
"border-radius:8px;color:#e6e6e6;font-size:14px;font-family:inherit;resize:none;"
"min-height:44px;max-height:140px}"
"#prompt:focus{outline:none;border-color:#60a5fa}"
"button{padding:10px 20px;background:#3b82f6;border:0;border-radius:8px;"
"color:white;font-weight:600;font-size:14px;cursor:pointer}"
"button:disabled{background:#4b5563;cursor:not-allowed}"
".status{font-size:12px;color:#9ca3af;padding:4px 8px}"
"</style></head><body>"
"<header><b>smollm2</b> · 135M · pure-C inference</header>"
"<div id=\"chat\"><div class=\"msg bot\">Hi! Ask me anything.</div></div>"
"<div id=\"input-area\">"
"<textarea id=\"prompt\" rows=\"1\" placeholder=\"Send a message...\" "
"autofocus></textarea>"
"<button id=\"send\">Send</button>"
"</div>"
"<div class=\"status\" id=\"status\">Ready.</div>"
"<script>"
"const chat=document.getElementById('chat');"
"const ta=document.getElementById('prompt');"
"const btn=document.getElementById('send');"
"const st=document.getElementById('status');"
"ta.addEventListener('input',()=>{"
"ta.style.height='auto';"
"ta.style.height=Math.min(ta.scrollHeight,140)+'px';"
"});"
"ta.addEventListener('keydown',e=>{"
"if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}"
"});"
"btn.addEventListener('click',send);"
"function addMsg(cls,text){const d=document.createElement('div');"
"d.className='msg '+cls;d.textContent=text;chat.appendChild(d);"
"chat.scrollTop=chat.scrollHeight;return d;}"
"async function send(){const p=ta.value.trim();if(!p)return;"
"addMsg('user',p);ta.value='';ta.style.height='auto';"
"const bot=addMsg('bot','');btn.disabled=true;st.textContent='Generating...';"
"try{"
"const r=await fetch('/generate',{method:'POST',"
"headers:{'Content-Type':'application/json'},"
"body:JSON.stringify({prompt:p,n:200})});"
"if(!r.ok){st.textContent='Error: '+r.status;return;}"
"const reader=r.body.getReader();"
"const dec=new TextDecoder();let buf='';let text='';"
"while(true){const{done,value}=await reader.read();"
"if(done)break;buf+=dec.decode(value,{stream:true});"
"let idx;"
"while((idx=buf.indexOf('\\n'))>=0){"
"const line=buf.slice(0,idx);buf=buf.slice(idx+1);"
"if(!line.startsWith('data: '))continue;"
"const payload=line.slice(6);"
"if(payload==='[DONE]'){st.textContent='Done.';continue;}"
"try{const j=JSON.parse(payload);if(j.token){text+=j.token;"
"bot.textContent=text;chat.scrollTop=chat.scrollHeight;}}"
"catch(e){}}"
"}"
"}catch(e){st.textContent='Network error.';}"
"finally{btn.disabled=false;ta.focus();"
"if(bot.textContent===''){bot.textContent='(no response)';}"
"}"
"}"
"</script></body></html>";

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------
static const char* json_find_key(const char* body, const char* key) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char* p = strstr(body, pattern);
    if (!p) return NULL;
    p += strlen(pattern);
    while (*p && (*p == ' ' || *p == ':' || *p == '\t')) p++;
    return p;
}

static int json_extract_string(const char* body, const char* key,
                               char* out, int out_max) {
    const char* p = json_find_key(body, key);
    if (!p || *p != '"') return -1;
    p++;
    int n = 0;
    while (*p && *p != '"' && n < out_max - 1) {
        if (*p == '\\' && p[1]) {
            char esc = p[1];
            if (esc == 'n') out[n++] = '\n';
            else if (esc == 't') out[n++] = '\t';
            else if (esc == 'r') out[n++] = '\r';
            else if (esc == '"') out[n++] = '"';
            else if (esc == '\\') out[n++] = '\\';
            else if (esc == '/') out[n++] = '/';
            else out[n++] = esc;
            p += 2;
        } else {
            out[n++] = *p++;
        }
    }
    out[n] = '\0';
    return n;
}

static int json_extract_int(const char* body, const char* key, int def) {
    const char* p = json_find_key(body, key);
    if (!p) return def;
    return atoi(p);
}

// Escape string for JSON output (handles quotes, backslashes, control chars)
static void json_escape(const char* in, int in_len, char* out, int out_max) {
    int n = 0;
    for (int i = 0; i < in_len && n < out_max - 8; i++) {
        unsigned char c = (unsigned char)in[i];
        if (c == '"') { out[n++]='\\'; out[n++]='"'; }
        else if (c == '\\') { out[n++]='\\'; out[n++]='\\'; }
        else if (c == '\n') { out[n++]='\\'; out[n++]='n'; }
        else if (c == '\r') { out[n++]='\\'; out[n++]='r'; }
        else if (c == '\t') { out[n++]='\\'; out[n++]='t'; }
        else if (c < 32) { n += snprintf(out+n, out_max-n, "\\u%04x", c); }
        else out[n++] = (char)c;
    }
    out[n] = '\0';
}

// ---------------------------------------------------------------------------
// HTTP send helpers
// ---------------------------------------------------------------------------
static void send_all(int fd, const char* buf, size_t len) {
    size_t off = 0;
    while (off < len) {
        ssize_t w = send(fd, buf + off, len - off, MSG_NOSIGNAL);
        if (w <= 0) {
            if (errno == EINTR) continue;
            break;
        }
        off += (size_t)w;
    }
}

static void send_chunk(int fd, const char* data, size_t len) {
    char header[32];
    int hlen = snprintf(header, sizeof(header), "%zx\r\n", len);
    send_all(fd, header, (size_t)hlen);
    send_all(fd, data, len);
    send_all(fd, "\r\n", 2);
}

// ---------------------------------------------------------------------------
// Request handlers
// ---------------------------------------------------------------------------
static void handle_home(int fd) {
    char header[256];
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/html; charset=utf-8\r\n"
        "Transfer-Encoding: chunked\r\n"
        "Connection: close\r\n\r\n");
    send_all(fd, header, (size_t)hlen);
    size_t html_len = strlen(kHTML);
    // Send HTML in chunks of 4096 to keep within chunked-encoding limits
    size_t off = 0;
    while (off < html_len) {
        size_t chunk = html_len - off;
        if (chunk > 4096) chunk = 4096;
        send_chunk(fd, kHTML + off, chunk);
        off += chunk;
    }
    send_chunk(fd, "", 0);  // end-of-body
}

static void handle_generate(int fd, const char* body,
                            const char* model_path,
                            const sample_params* sp) {
    char prompt[2048];
    int plen = json_extract_string(body, "prompt", prompt, sizeof(prompt));
    int max_new = json_extract_int(body, "n", 200);
    if (plen <= 0) {
        const char* err = "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n";
        send_all(fd, err, strlen(err));
        return;
    }

    // Send SSE headers
    const char* header =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n"
        "Transfer-Encoding: chunked\r\n\r\n";
    send_all(fd, header, strlen(header));

    // Load model fresh for each request (simple but slow)
    // For better perf, load once at startup and mutex-protect — out of scope here.
    gguf_ctx ctx;
    if (gguf_load(model_path, &ctx) < 0) {
        send_chunk(fd, "data: {\"error\":\"model load failed\"}\n\n",
                   strlen("data: {\"error\":\"model load failed\"}\n\n"));
        send_chunk(fd, "data: [DONE]\n\n", strlen("data: [DONE]\n\n"));
        send_chunk(fd, "", 0);
        return;
    }
    tokenizer* tok = NULL;
    if (tokenizer_load(&tok, &ctx) < 0) {
        gguf_free(&ctx);
        send_chunk(fd, "", 0);
        return;
    }
    forward_ctx* fwd = NULL;
    if (forward_load(&fwd, &ctx, 2048) < 0) {
        tokenizer_free(tok); gguf_free(&ctx);
        send_chunk(fd, "", 0);
        return;
    }

    // Build chat template
    char tmpl[4096];
    snprintf(tmpl, sizeof(tmpl),
        "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", prompt);
    int prompt_ids[1024];
    int prompt_len = tokenizer_encode(tok, tmpl, prompt_ids, 1024);
    if (prompt_len <= 0) {
        forward_free(fwd); tokenizer_free(tok); gguf_free(&ctx);
        send_chunk(fd, "", 0);
        return;
    }

    int vocab = forward_vocab_size(fwd);
    float* logits = malloc((size_t)vocab * sizeof(float));
    int*   gen    = malloc((size_t)max_new * sizeof(int));
    if (!logits || !gen) {
        free(logits); free(gen);
        forward_free(fwd); tokenizer_free(tok); gguf_free(&ctx);
        send_chunk(fd, "", 0);
        return;
    }

    if (forward_prefill(fwd, prompt_ids, prompt_len, logits) < 0) {
        free(logits); free(gen);
        forward_free(fwd); tokenizer_free(tok); gguf_free(&ctx);
        send_chunk(fd, "", 0);
        return;
    }

    int pos = prompt_len;
    int gen_n = 0;
    char dec_buf[512];
    char json_buf[1024];

    while (gen_n < max_new) {
        int next = sample_token(logits, vocab, sp, gen, gen_n);
        if (next == IM_END_TOKEN) break;
        gen[gen_n++] = next;

        int bytes = tokenizer_decode(tok, next, dec_buf, sizeof(dec_buf));
        if (bytes > 0) {
            json_escape(dec_buf, bytes, json_buf, sizeof(json_buf));
            char sse[2048];
            int slen = snprintf(sse, sizeof(sse),
                "data: {\"token\":\"%s\"}\n\n", json_buf);
            send_chunk(fd, sse, (size_t)slen);
        }

        if (pos < 2047) {
            if (forward_decode(fwd, next, pos, logits) < 0) break;
            pos++;
        } else break;
    }

    send_chunk(fd, "data: [DONE]\n\n", strlen("data: [DONE]\n\n"));
    send_chunk(fd, "", 0);

    free(logits); free(gen);
    forward_free(fwd); tokenizer_free(tok); gguf_free(&ctx);
}

// ---------------------------------------------------------------------------
// Parse HTTP request: method, path, body
// ---------------------------------------------------------------------------
static int parse_request(const char* req, int req_len __attribute__((unused)),
                         char* method, int method_max,
                         char* path,   int path_max,
                         char** body_out) {
    const char* end = strstr(req, "\r\n\r\n");
    if (!end) return -1;
    *body_out = (char*)(end + 4);

    const char* sp = strchr(req, ' ');
    if (!sp) return -1;
    int mlen = (int)(sp - req);
    if (mlen >= method_max) mlen = method_max - 1;
    memcpy(method, req, mlen);
    method[mlen] = '\0';

    const char* path_start = sp + 1;
    const char* path_end = strchr(path_start, ' ');
    if (!path_end) return -1;
    int plen = (int)(path_end - path_start);
    if (plen >= path_max) plen = path_max - 1;
    memcpy(path, path_start, plen);
    path[plen] = '\0';
    return 0;
}

// ---------------------------------------------------------------------------
// Main server loop
// ---------------------------------------------------------------------------
static volatile int g_stop = 0;

static void on_signal(int sig) {
    (void)sig;
    g_stop = 1;
}

int web_run(const char* model_path, int port, const sample_params* sp) {
    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) { perror("socket"); return -1; }

    int yes = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((uint16_t)port);

    if (bind(srv, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind"); close(srv); return -1;
    }
    if (listen(srv, 8) < 0) {
        perror("listen"); close(srv); return -1;
    }

    fprintf(stderr, "smollm2 web: http://localhost:%d/  (Ctrl+C to stop)\n", port);
    fflush(stderr);

    while (!g_stop) {
        struct sockaddr_in cli;
        socklen_t clen = sizeof(cli);
        int cli_fd = accept(srv, (struct sockaddr*)&cli, &clen);
        if (cli_fd < 0) {
            if (errno == EINTR) continue;
            if (g_stop) break;
            perror("accept");
            continue;
        }

        char req[MAX_REQ];
        ssize_t n = recv(cli_fd, req, sizeof(req) - 1, 0);
        if (n <= 0) { close(cli_fd); continue; }
        req[n] = '\0';

        char method[16], path[256];
        char* body = NULL;
        if (parse_request(req, (int)n, method, sizeof(method),
                          path, sizeof(path), &body) < 0) {
            close(cli_fd);
            continue;
        }

        if (strcmp(method, "GET") == 0 && strcmp(path, "/") == 0) {
            handle_home(cli_fd);
        } else if (strcmp(method, "POST") == 0 &&
                   strcmp(path, "/generate") == 0) {
            handle_generate(cli_fd, body ? body : "", model_path, sp);
        } else {
            const char* nf = "HTTP/1.1 404 Not Found\r\n"
                             "Content-Length: 0\r\nConnection: close\r\n\r\n";
            send_all(cli_fd, nf, strlen(nf));
        }

        close(cli_fd);
    }

    close(srv);
    fprintf(stderr, "\nsmollm2 web: stopped.\n");
    return 0;
}
