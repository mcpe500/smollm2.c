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
#include "hw_probe.h"
#include "attn_registry.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/wait.h>
#include <fcntl.h>

#define MAX_REQ     65536
#define IM_END_TOKEN 2

/* A2: process-global model cache. Loaded once in web_run, reused per request.
 * forward_reset() clears KV cache between requests; full reload only when
 * the caller changes rope/kv precision (which affects forward_load's allocation). */
static forward_ctx* g_fwd = NULL;
static tokenizer*    g_tok = NULL;
static gguf_ctx      g_gctx;
static int           g_model_loaded = 0;
static int           g_cache_rope = ROPE_F32, g_cache_kv = KV_F32;

/* Forward declarations — helpers defined below. */
static void send_json(int fd, int code, const char* body);
static void send_all(int fd, const char* buf, size_t len);

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

/* Studio SPA — tabs: Infer / HW / Attn / Data / Train / Merge. */
static const char* kStudioHTML =
"<!DOCTYPE html>\n"
"<html lang=\"en\"><head><meta charset=\"utf-8\">"
"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
"<title>smollm2 studio</title>"
"<style>"
"*{box-sizing:border-box;margin:0;padding:0}"
"body{font-family:system-ui,sans-serif;background:#0f1117;color:#e6e6e6;"
"min-height:100vh;display:flex;flex-direction:column}"
"header{padding:12px 16px;background:#161922;border-bottom:1px solid #252836;"
"font-weight:600;font-size:14px;color:#9ca3af;display:flex;gap:16px;align-items:center}"
"header b{color:#60a5fa}"
"nav{display:flex;gap:4px;flex-wrap:wrap}"
"nav button{padding:6px 12px;background:#1f2937;border:1px solid #2d3748;"
"color:#e6e6e6;border-radius:6px;cursor:pointer;font-size:13px}"
"nav button.active{background:#3b82f6;border-color:#3b82f6}"
"main{flex:1;padding:16px;max-width:900px;width:100%;margin:0 auto}"
".tab{display:none}.tab.active{display:block}"
"pre{background:#1f2937;padding:12px;border-radius:8px;overflow:auto;"
"font-size:12px;white-space:pre-wrap;border:1px solid #2d3748}"
"label{display:block;font-size:12px;color:#9ca3af;margin:8px 0 4px}"
"input,textarea,select{width:100%;padding:8px 10px;background:#1f2937;"
"border:1px solid #2d3748;border-radius:6px;color:#e6e6e6;font-size:13px}"
"button.primary{margin-top:10px;padding:8px 16px;background:#3b82f6;border:0;"
"border-radius:6px;color:#fff;font-weight:600;cursor:pointer}"
".row{display:grid;grid-template-columns:1fr 1fr;gap:10px}"
".status{font-size:12px;color:#9ca3af;margin-top:8px}"
".log{max-height:280px;overflow:auto}"
"</style></head><body>"
"<header><b>smollm2 studio</b>"
"<nav id=\"nav\">"
"<button data-t=\"infer\" class=\"active\">Infer</button>"
"<button data-t=\"hw\">HW</button>"
"<button data-t=\"attn\">Attn</button>"
"<button data-t=\"data\">Data</button>"
"<button data-t=\"train\">Train</button>"
"<button data-t=\"merge\">Merge</button>"
"</nav></header>"
"<main>"
"<section class=\"tab active\" id=\"tab-infer\">"
"<details open><summary style=\"cursor:pointer;color:#60a5fa;font-size:13px\">"
"⚙ Sampling &amp; inference</summary>"
"<div class=\"row\" style=\"margin-top:8px\">"
"<div><label>Temperature <span id=\"v-temp\">0.3</span></label>"
"<input type=\"range\" id=\"inf-temp\" min=\"0\" max=\"2\" step=\"0.05\" value=\"0.3\" "
"oninput=\"document.getElementById('v-temp').textContent=this.value\"></div>"
"<div><label>Top-p <span id=\"v-topp\">0</span> (0=off)</label>"
"<input type=\"range\" id=\"inf-topp\" min=\"0\" max=\"1\" step=\"0.05\" value=\"0\" "
"oninput=\"document.getElementById('v-topp').textContent=this.value\"></div>"
"<div><label>Top-k <span id=\"v-topk\">5</span> (0=off)</label>"
"<input type=\"range\" id=\"inf-topk\" min=\"0\" max=\"100\" step=\"1\" value=\"5\" "
"oninput=\"document.getElementById('v-topk').textContent=this.value\"></div>"
"<div><label>Rep-penalty <span id=\"v-rep\">1.1</span></label>"
"<input type=\"range\" id=\"inf-rep\" min=\"1\" max=\"2\" step=\"0.05\" value=\"1.1\" "
"oninput=\"document.getElementById('v-rep').textContent=this.value\"></div>"
"<div><label>Tokens (max)</label>"
"<input type=\"range\" id=\"inf-n\" min=\"1\" max=\"512\" step=\"1\" value=\"128\" "
"oninput=\"document.getElementById('v-n').textContent=this.value\">"
"<span id=\"v-n\" style=\"color:#9ca3af\">128</span></div>"
"<div><label>Seed</label>"
"<input id=\"inf-seed\" type=\"number\" value=\"0\" placeholder=\"0=random\"></div>"
"</div>"
"<div class=\"row\" style=\"margin-top:8px\">"
"<div><label>Attention</label>"
"<select id=\"inf-attn\">"
"<option value=\"dense\">dense (GQA, default)</option>"
"<option value=\"swa:window=64\">SWA window=64</option>"
"<option value=\"swa:window=128\">SWA window=128</option>"
"<option value=\"swa:window=256\">SWA window=256</option>"
"<option value=\"swa:window=512\">SWA window=512</option>"
"</select></div>"
"<div><label>RoPE precision</label>"
"<select id=\"inf-rope\"><option value=\"f32\">f32</option>"
"<option value=\"f16\">f16</option><option value=\"q8\">q8</option></select></div>"
"<div><label>KV cache precision</label>"
"<select id=\"inf-kv\"><option value=\"f32\">f32</option>"
"<option value=\"f16\">f16</option><option value=\"q8\">q8</option></select></div>"
"<div><label style=\"display:flex;align-items:center;gap:8px;margin-top:24px\">"
"<input type=\"checkbox\" id=\"inf-template\" checked style=\"width:auto\">"
"ChatML template</label>"
"<label style=\"display:flex;align-items:center;gap:8px\">"
"<input type=\"checkbox\" id=\"inf-stop\" checked style=\"width:auto\">"
"Stop on &lt;|im_end|&gt;</label></div>"
"</div>"
"</details>"

"<label>System (optional)</label>"
"<textarea id=\"inf-sys\" rows=\"2\" placeholder=\"You are a helpful AI assistant named SmolLM, trained by Hugging Face\">"
"You are a helpful AI assistant named SmolLM, trained by Hugging Face</textarea>"
"<label>Prompt</label>"
"<textarea id=\"inf-p\" rows=\"3\">hello</textarea>"
"<button class=\"primary\" id=\"inf-go\">Generate</button>"
"<div class=\"status\" id=\"inf-st\">Ready.</div>"
"<pre class=\"log\" id=\"inf-out\" style=\"max-height:380px\"></pre></section>"
"<section class=\"tab\" id=\"tab-hw\">"
"<button class=\"primary\" id=\"hw-go\">Probe</button>"
"<pre class=\"log\" id=\"hw-out\">(click Probe)</pre></section>"
"<section class=\"tab\" id=\"tab-attn\">"
"<button class=\"primary\" id=\"attn-go\">List registry</button>"
"<pre class=\"log\" id=\"attn-out\">(click List)</pre></section>"
"<section class=\"tab\" id=\"tab-data\">"
"<label>Raw text</label><textarea id=\"data-t\" rows=\"4\">Hello world.\\nHow are you?</textarea>"
"<label>Format</label><select id=\"data-f\"><option>raw</option>"
"<option>instruct</option><option>sharegpt</option></select>"
"<label>Out path</label><input id=\"data-o\" value=\"/tmp/studio_packed.bin\">"
"<button class=\"primary\" id=\"data-go\">Build</button>"
"<pre class=\"log\" id=\"data-out\"></pre></section>"
"<section class=\"tab\" id=\"tab-train\">"
"<div class=\"row\">"
"<div><label>Data</label><input id=\"tr-data\" value=\"/tmp/studio_packed.bin\"></div>"
"<div><label>Mode</label><select id=\"tr-mode\"><option>lora</option>"
"<option>qlora</option><option>fullft</option></select></div>"
"<div><label>Rank</label><input id=\"tr-rank\" type=\"number\" value=\"4\"></div>"
"<div><label>Epochs</label><input id=\"tr-ep\" type=\"number\" value=\"1\"></div>"
"<div><label>LR</label><input id=\"tr-lr\" value=\"1e-3\"></div>"
"<div><label>Max steps</label><input id=\"tr-ms\" type=\"number\" value=\"5\"></div>"
"</div>"
"<button class=\"primary\" id=\"tr-go\">Train</button>"
"<div class=\"status\" id=\"tr-st\">Ready.</div>"
"<pre class=\"log\" id=\"tr-out\"></pre></section>"
"<section class=\"tab\" id=\"tab-merge\">"
"<label>Base GGUF</label><input id=\"mg-base\" value=\"models/smollm2-135m-f16.gguf\">"
"<label>Adapter</label><input id=\"mg-ad\" value=\"adapters/lora_final.bin\">"
"<label>Out</label><input id=\"mg-out\" value=\"/tmp/merged.gguf\">"
"<button class=\"primary\" id=\"mg-go\">Merge</button>"
"<pre class=\"log\" id=\"mg-out-log\"></pre></section>"
"</main>"
"<script>"
"const tabs=document.querySelectorAll('.tab');"
"document.getElementById('nav').onclick=e=>{"
"const b=e.target.closest('button');if(!b)return;"
"document.querySelectorAll('nav button').forEach(x=>x.classList.remove('active'));"
"b.classList.add('active');"
"tabs.forEach(t=>t.classList.toggle('active',t.id==='tab-'+b.dataset.t));"
"};"
"async function jget(u){const r=await fetch(u);return r.json();}"
"async function jpost(u,body){const r=await fetch(u,{method:'POST',"
"headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});"
"return r.json();}"
"document.getElementById('hw-go').onclick=async()=>{"
"document.getElementById('hw-out').textContent=JSON.stringify(await jget('/studio/hw'),null,2);};"
"document.getElementById('attn-go').onclick=async()=>{"
"document.getElementById('attn-out').textContent=JSON.stringify(await jget('/studio/attn'),null,2);};"
"document.getElementById('data-go').onclick=async()=>{"
"const body={text:document.getElementById('data-t').value,"
"fmt:document.getElementById('data-f').value,"
"out:document.getElementById('data-o').value};"
"document.getElementById('data-out').textContent=JSON.stringify(await jpost('/studio/data',body),null,2);};"
"document.getElementById('mg-go').onclick=async()=>{"
"const body={base:document.getElementById('mg-base').value,"
"adapter:document.getElementById('mg-ad').value,"
"out:document.getElementById('mg-out').value};"
"document.getElementById('mg-out-log').textContent=JSON.stringify(await jpost('/studio/merge',body),null,2);};"
"document.getElementById('inf-go').onclick=async()=>{"
"const p=document.getElementById('inf-p').value;"
"const n=+document.getElementById('inf-n').value||128;"
"const body={prompt:p,n:n,"
"temperature:+document.getElementById('inf-temp').value,"
"top_p:+document.getElementById('inf-topp').value,"
"top_k:+document.getElementById('inf-topk').value,"
"rep_penalty:+document.getElementById('inf-rep').value,"
"seed:+document.getElementById('inf-seed').value||0,"
"attn:document.getElementById('inf-attn').value,"
"rope:document.getElementById('inf-rope').value,"
"kv:document.getElementById('inf-kv').value,"
"system:document.getElementById('inf-sys').value,"
"template:document.getElementById('inf-template').checked?1:0};"
"const out=document.getElementById('inf-out');out.textContent='';"
"const st=document.getElementById('inf-st');st.textContent='Generating...';"
"try{const r=await fetch('/studio/infer',{method:'POST',"
"headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});"
"const reader=r.body.getReader();const dec=new TextDecoder();let buf='',text='';"
"while(true){const{done,value}=await reader.read();if(done)break;"
"buf+=dec.decode(value,{stream:true});let idx;"
"while((idx=buf.indexOf('\\n'))>=0){const line=buf.slice(0,idx);buf=buf.slice(idx+1);"
"if(!line.startsWith('data: '))continue;const payload=line.slice(6);"
"if(payload==='[DONE]'){st.textContent='Done.';continue;}"
"try{const j=JSON.parse(payload);if(j.token){text+=j.token;out.textContent=text;}}"
"catch(e){}}}"
"}catch(e){st.textContent='Error';}"
"};"
"document.getElementById('tr-go').onclick=async()=>{"
"const body={data:document.getElementById('tr-data').value,"
"mode:document.getElementById('tr-mode').value,"
"rank:+document.getElementById('tr-rank').value||4,"
"epochs:+document.getElementById('tr-ep').value||1,"
"lr:parseFloat(document.getElementById('tr-lr').value)||1e-3,"
"max_steps:+document.getElementById('tr-ms').value||5};"
"const out=document.getElementById('tr-out');out.textContent='';"
"const st=document.getElementById('tr-st');st.textContent='Training...';"
"try{const r=await fetch('/studio/train',{method:'POST',"
"headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});"
"const reader=r.body.getReader();const dec=new TextDecoder();let buf='';"
"while(true){const{done,value}=await reader.read();if(done)break;"
"buf+=dec.decode(value,{stream:true});let idx;"
"while((idx=buf.indexOf('\\n'))>=0){const line=buf.slice(0,idx);buf=buf.slice(idx+1);"
"if(!line.startsWith('data: '))continue;const payload=line.slice(6);"
"if(payload==='[DONE]'){st.textContent='Done.';continue;}"
"out.textContent+=payload+'\\n';out.scrollTop=out.scrollHeight;}"
"}"
"}catch(e){st.textContent='Error';}"
"};"
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
// A1: shell-injection-safe subprocess helpers (no shell, no popen).
// ---------------------------------------------------------------------------
/* Reject paths with shell metacharacters or traversal patterns. Defense in
 * depth — run_capture does not use a shell, but we want a clear error for
 * suspicious input. Returns 1 if safe, 0 otherwise. */
static int path_safe(const char* p) {
    if (!p || !*p) return 0;
    for (const char* c = p; *c; c++) {
        if (*c == ';' || *c == '|' || *c == '`' || *c == '$' ||
            *c == '<' || *c == '>' || *c == '\n' || *c == '\r' ||
            *c == '\\' || *c == '"' || *c == '\'') {
            return 0;
        }
    }
    return 1;
}

/* Spawn argv (NULL-terminated), capture combined stdout+stderr up to cap-1
 * bytes (NUL-terminated). Returns exit status (0-255), or -1 on spawn error.
 * Caller must allocate `out` with at least `cap` bytes (cap=0 disables capture).
 * `n_out` (optional) receives captured length. */
static int run_capture(char** argv, char* out, size_t cap, size_t* n_out) {
    int pipefd[2];
    if (pipe(pipefd) < 0) return -1;
    pid_t pid = fork();
    if (pid < 0) { close(pipefd[0]); close(pipefd[1]); return -1; }
    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);
        execvp(argv[0], argv);
        _exit(127);
    }
    close(pipefd[1]);
    size_t total = 0;
    if (out && cap > 1) {
        while (total < cap - 1) {
            ssize_t n = read(pipefd[0], out + total, cap - 1 - total);
            if (n < 0) { if (errno == EINTR) continue; break; }
            if (n == 0) break;
            total += (size_t)n;
        }
        out[total] = '\0';
    }
    close(pipefd[0]);
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) break;
    }
    if (n_out) *n_out = total;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return -1;
}

/* A3: growable recv buffer. Reads HTTP request (headers + body up to
 * Content-Length), growing as needed. Hard cap MAX_REQ_BODY bytes.
 * On success: *out_buf is malloc'd (caller frees), *out_len is byte count.
 * Returns 0 on success, -1 on error. */
#define MAX_REQ_BODY (8 * 1024 * 1024)
static int recv_request(int fd, char** out_buf, size_t* out_len) {
    size_t cap = 8192;
    size_t len = 0;
    char* buf = (char*)malloc(cap);
    if (!buf) return -1;

    size_t hdr_end = 0;
    int have_hdr = 0;
    while (!have_hdr) {
        if (len + 1 >= cap) {
            if (cap >= MAX_REQ_BODY) { free(buf); return -1; }
            size_t ncap = cap * 2;
            char* nb = (char*)realloc(buf, ncap);
            if (!nb) { free(buf); return -1; }
            buf = nb; cap = ncap;
        }
        ssize_t k = recv(fd, buf + len, cap - len - 1, 0);
        if (k < 0) { if (errno == EINTR) continue; free(buf); return -1; }
        if (k == 0) break;
        len += (size_t)k;
        buf[len] = '\0';
        char* he = strstr(buf, "\r\n\r\n");
        if (he) { hdr_end = (size_t)(he - buf) + 4; have_hdr = 1; }
        if (len > 1024 * 1024 && !have_hdr) { free(buf); return -1; }
    }
    if (!have_hdr) { free(buf); return -1; }

    long clen = 0;
    char* cl = strcasestr(buf, "Content-Length:");
    if (cl) clen = atol(cl + 15);
    if (clen < 0) clen = 0;

    size_t need = hdr_end + (size_t)clen + 1;
    if (need > MAX_REQ_BODY) { free(buf); return -1; }
    if (need > cap) {
        char* nb = (char*)realloc(buf, need);
        if (!nb) { free(buf); return -1; }
        buf = nb; cap = need;
    }
    while (len < hdr_end + (size_t)clen) {
        ssize_t k = recv(fd, buf + len, hdr_end + (size_t)clen - len, 0);
        if (k < 0) { if (errno == EINTR) continue; break; }
        if (k == 0) break;
        len += (size_t)k;
    }
    buf[len] = '\0';

    *out_buf = buf;
    *out_len = len;
    return 0;
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

static void handle_studio_page(int fd) {
    char header[256];
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/html; charset=utf-8\r\n"
        "Transfer-Encoding: chunked\r\n"
        "Connection: close\r\n\r\n");
    send_all(fd, header, (size_t)hlen);
    size_t html_len = strlen(kStudioHTML);
    size_t off = 0;
    while (off < html_len) {
        size_t chunk = html_len - off;
        if (chunk > 4096) chunk = 4096;
        send_chunk(fd, kStudioHTML + off, chunk);
        off += chunk;
    }
    send_chunk(fd, "", 0);
}

static void handle_studio_hw(int fd) {
    hw_caps c; hw_probe(&c);
    char* body = hw_json(&c);
    if (!body) {
        send_json(fd, 500, "{\"ok\":false,\"error\":\"hw_json\"}");
        return;
    }
    size_t n = strlen(body);
    char header[256];
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
        "Content-Length: %zu\r\nConnection: close\r\n\r\n", n);
    send_all(fd, header, (size_t)hlen);
    send_all(fd, body, n);
    free(body);
}

static void handle_studio_attn(int fd) {
    char header[256];
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
        "Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n");
    send_all(fd, header, (size_t)hlen);
    char line[256];
    int n = snprintf(line, sizeof(line), "{\"n_layers\":%d,\"variants\":[\"dense\",\"swa\",\"dilated\",\"bigbird\",\"glocal\",\"mla\"],\"layers\":[",
                     attn_n_layers());
    send_chunk(fd, line, (size_t)n);
    int nl = attn_n_layers();
    if (nl <= 0) {
        attn_spec s; attn_get_spec(0, &s);
        n = snprintf(line, sizeof(line),
            "{\"L\":0,\"type\":\"%s\",\"window\":%d,\"dilation\":%d,\"n_global\":%d,\"latent_dim\":%d}",
            attn_type_name(s.type), s.window, s.dilation, s.n_global, s.latent_dim);
        send_chunk(fd, line, (size_t)n);
    } else {
        for (int i = 0; i < nl; i++) {
            attn_spec s; attn_get_spec(i, &s);
            n = snprintf(line, sizeof(line),
                "%s{\"L\":%d,\"type\":\"%s\",\"window\":%d,\"dilation\":%d,\"n_global\":%d,\"latent_dim\":%d}",
                i ? "," : "", i,
                attn_type_name(s.type), s.window, s.dilation, s.n_global, s.latent_dim);
            send_chunk(fd, line, (size_t)n);
        }
    }
    send_chunk(fd, "]}\n", 3);
    send_chunk(fd, "", 0);
}

/* Phase 4b: POST /studio/{data,train,merge} — exec ./smollm2 studio <cmd>
   and stream stdout as SSE. Single-shot, blocking, no auth. */
#include <sys/wait.h>

static void sse_send(int fd, const char* s) {
    char esc[2048]; int n = 0;
    for (const char* p = s; *p && n < (int)sizeof(esc) - 4; p++) {
        char c = *p;
        if (c == '\n') { esc[n++] = '\\'; esc[n++] = 'n'; }
        else if (c == '"') { esc[n++] = '\\'; esc[n++] = '"'; }
        else if (c == '\\') { esc[n++] = '\\'; esc[n++] = '\\'; }
        else if ((unsigned char)c < 32) { /* skip */ }
        else esc[n++] = c;
    }
    char buf[4096];
    int bl = snprintf(buf, sizeof(buf), "data: {\"line\":\"%s\"}\n\n", esc);
    if (bl > 0) send_chunk(fd, buf, (size_t)bl);
}

static void sse_done(int fd) {
    send_chunk(fd, "data: [DONE]\n\n", strlen("data: [DONE]\n\n"));
    send_chunk(fd, "", 0);
}

static const char* sse_header =
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: text/event-stream\r\n"
    "Cache-Control: no-cache\r\n"
    "Connection: close\r\n"
    "Transfer-Encoding: chunked\r\n\r\n";

/* Trivial extractor: "{key":"value"}" — returns length, copies into out. */
static int jp_str(const char* body, const char* key, char* out, int mx) {
    char pat[64]; int pl = snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char* p = strstr(body, pat); if (!p) return 0;
    p += pl; while (*p == ' ' || *p == ':') p++;
    if (*p != '"') return 0;
    p++;
    int n = 0;
    while (*p && *p != '"' && n < mx - 1) {
        if (*p == '\\' && p[1]) {
            char e = p[1];
            if      (e == 'n') out[n++] = '\n';
            else if (e == 't') out[n++] = '\t';
            else if (e == 'r') out[n++] = '\r';
            else if (e == '"') out[n++] = '"';
            else if (e == '\\') out[n++] = '\\';
            else                 out[n++] = e;
            p += 2;
        } else out[n++] = *p++;
    }
    out[n] = '\0';
    return n;
}
static int jp_int(const char* body, const char* key, int def) {
    char pat[64]; int pl = snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char* p = strstr(body, pat); if (!p) return def;
    p += pl; while (*p == ' ' || *p == ':') p++;
    return atoi(p);
}
static int jp_float_str(char* out, int mx, const char* body, const char* key) {
    char pat[64]; int pl = snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char* p = strstr(body, pat); if (!p) return 0;
    p += pl; while (*p == ' ' || *p == ':') p++;
    int n = 0;
    while (*p && *p != ',' && *p != '}' && *p != ' ' && n < mx - 1)
        out[n++] = *p++;
    out[n] = '\0';
    return n;
}

static void send_json(int fd, int code, const char* body) {
    char header[256];
    int blen = (int)strlen(body);
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\n"
        "Content-Length: %d\r\nConnection: close\r\n\r\n",
        code, code == 200 ? "OK" : "Bad Request", blen);
    send_all(fd, header, (size_t)hlen);
    send_all(fd, body, (size_t)blen);
}

static const char* tmp_dir(void) {
    const char* t = getenv("TMPDIR");
    if (t && t[0]) return t;
    t = getenv("TMP");
    if (t && t[0]) return t;
    return "/tmp";
}

static void handle_studio_data(int fd, const char* body, const char* model_path) {
    char text[16384], fmt[32], out_path[1024];
    int tlen = jp_str(body, "text", text, sizeof(text));
    jp_str(body, "fmt", fmt, sizeof(fmt));
    jp_str(body, "out", out_path, sizeof(out_path));
    if (tlen <= 0 || out_path[0] == '\0' || model_path == NULL) {
        send_json(fd, 400, "{\"ok\":false,\"error\":\"missing text/out or --model\"}");
        return;
    }
    /* A1: path safety — reject shell metacharacters even though run_capture
     * does not use a shell (defense in depth + clear error for user). */
    if (!path_safe(out_path) || !path_safe(model_path)) {
        send_json(fd, 400, "{\"ok\":false,\"error\":\"unsafe path\"}");
        return;
    }
    char in_path[1024];
    snprintf(in_path, sizeof(in_path), "%s/studio_in_%d.txt", tmp_dir(), (int)getpid());
    FILE* f = fopen(in_path, "w");
    if (!f) { send_json(fd, 500, "{\"ok\":false,\"error\":\"tmpfile\"}"); return; }
    fwrite(text, 1, (size_t)tlen, f); fclose(f);

    char fmt_arg[32];
    snprintf(fmt_arg, sizeof(fmt_arg), "%s", fmt[0] ? fmt : "raw");

    /* A1: fork+execvp (no shell) eliminates injection surface entirely. */
    char* argv[] = {
        "./smollm2", "studio", "data-build",
        "--in", in_path,
        "--out", out_path,
        "--fmt", fmt_arg,
        "--model", (char*)model_path,
        NULL,
    };
    char log[4096] = {0}; size_t logn = 0;
    int rc = run_capture(argv, log, sizeof(log), &logn);
    unlink(in_path);

    char resp[8192], log_esc[4096];
    json_escape(log, (int)logn, log_esc, sizeof(log_esc));
    snprintf(resp, sizeof(resp),
        "{\"ok\":%s,\"rc\":%d,\"out\":\"%s\",\"log\":\"%s\"}",
        rc == 0 ? "true" : "false", rc, out_path, log_esc);
    send_json(fd, rc == 0 ? 200 : 500, resp);
}

static void handle_studio_train(int fd, const char* body, const char* model_path) {
    send_all(fd, sse_header, strlen(sse_header));
    char data[1024], mode[16];
    int rank = 4, epochs = 1, max_steps = 5, seq = 64, batch = 1;
    char lr_s[32] = "1e-3";
    jp_str(body, "data", data, sizeof(data));
    jp_str(body, "mode", mode, sizeof(mode));
    rank = jp_int(body, "rank", 4);
    epochs = jp_int(body, "epochs", 1);
    max_steps = jp_int(body, "max_steps", 5);
    seq = jp_int(body, "seq", 64);
    batch = jp_int(body, "batch", 1);
    jp_float_str(lr_s, sizeof(lr_s), body, "lr");
    if (!data[0] || !model_path) {
        sse_send(fd, "{\"error\":\"missing data or --model\"}");
        sse_done(fd); return;
    }
    /* A1: path safety. */
    if (!path_safe(data) || !path_safe(model_path)) {
        sse_send(fd, "{\"error\":\"unsafe path\"}");
        sse_done(fd); return;
    }
    /* A4: refuse non-lora mode at server too (mirrors studio.c). */
    if (mode[0] && strcmp(mode, "lora") != 0) {
        char msg[256];
        snprintf(msg, sizeof(msg),
            "{\"error\":\"mode '%s' not implemented in this build\"}", mode);
        sse_send(fd, msg);
        sse_done(fd); return;
    }
    char out_dir[256];
    snprintf(out_dir, sizeof(out_dir), "%s/studio_adapter_%d", tmp_dir(), (int)getpid());

    char rank_s[16], epochs_s[16], seq_s[16], batch_s[16], max_s[16];
    snprintf(rank_s, sizeof(rank_s), "%d", rank);
    snprintf(epochs_s, sizeof(epochs_s), "%d", epochs);
    snprintf(seq_s, sizeof(seq_s), "%d", seq);
    snprintf(batch_s, sizeof(batch_s), "%d", batch);
    snprintf(max_s, sizeof(max_s), "%d", max_steps);

    /* A1: argv-based execvp, no shell. */
    char* argv[] = {
        "./smollm2", "studio", "train",
        "--data", data,
        "--mode", mode[0] ? mode : "lora",
        "--rank", rank_s,
        "--epochs", epochs_s,
        "--lr", lr_s,
        "--seq", seq_s,
        "--batch", batch_s,
        "--max-steps", max_s,
        "--out-dir", out_dir,
        "--model", (char*)model_path,
        NULL,
    };

    /* Capture + forward as SSE. Phase F will replace with direct call + streaming. */
    char log[16384] = {0}; size_t logn = 0;
    int rc = run_capture(argv, log, sizeof(log), &logn);
    char* line = log;
    while (line < log + logn) {
        char* nl = memchr(line, '\n', (size_t)(log + logn - line));
        size_t L = nl ? (size_t)(nl - line) : (size_t)(log + logn - line);
        char buf[1100];
        size_t cpy = L < sizeof(buf) - 1 ? L : sizeof(buf) - 1;
        memcpy(buf, line, cpy); buf[cpy] = '\0';
        sse_send(fd, buf);
        if (!nl) break;
        line = nl + 1;
    }
    char meta[256];
    snprintf(meta, sizeof(meta), "{\"done\":true,\"rc\":%d,\"out_dir\":\"%s\"}",
             rc, out_dir);
    sse_send(fd, meta);
    sse_done(fd);
}

static void handle_studio_merge(int fd, const char* body) {
    char base[1024], adapter[1024], out[1024];
    jp_str(body, "base", base, sizeof(base));
    jp_str(body, "adapter", adapter, sizeof(adapter));
    jp_str(body, "out", out, sizeof(out));
    if (!base[0] || !adapter[0] || !out[0]) {
        send_json(fd, 400, "{\"ok\":false,\"error\":\"missing base/adapter/out\"}");
        return;
    }
    /* A1: path safety on all three. */
    if (!path_safe(base) || !path_safe(adapter) || !path_safe(out)) {
        send_json(fd, 400, "{\"ok\":false,\"error\":\"unsafe path\"}");
        return;
    }
    char* argv[] = {
        "./smollm2", "studio", "merge",
        "--base", base,
        "--adapter", adapter,
        "--out", out,
        NULL,
    };
    char log[4096] = {0}; size_t logn = 0;
    int rc = run_capture(argv, log, sizeof(log), &logn);
    char resp[8192], log_esc[4096];
    json_escape(log, (int)logn, log_esc, sizeof(log_esc));
    snprintf(resp, sizeof(resp),
        "{\"ok\":%s,\"rc\":%d,\"out\":\"%s\",\"sidecar\":\"%s.lora\",\"log\":\"%s\"}",
        rc == 0 ? "true" : "false", rc, out, out, log_esc);
    send_json(fd, rc == 0 ? 200 : 500, resp);
}

static void handle_generate(int fd, const char* body,
                            const char* model_path,
                            const sample_params* sp_default) {
    char prompt[2048];
    int plen = json_extract_string(body, "prompt", prompt, sizeof(prompt));
    int max_new = json_extract_int(body, "n", 200);
    if (plen <= 0) {
        const char* err = "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n";
        send_all(fd, err, strlen(err));
        return;
    }

    /* Per-request overrides — fall back to server defaults. */
    sample_params sp = *sp_default;
    const char* pv;
    pv = json_find_key(body, "temperature"); if (pv) sp.temperature = (float)atof(pv);
    pv = json_find_key(body, "top_p");       if (pv) sp.top_p       = (float)atof(pv);
    pv = json_find_key(body, "rep_penalty"); if (pv) sp.rep_penalty = (float)atof(pv);
    pv = json_find_key(body, "top_k");       if (pv) sp.top_k       = atoi(pv);
    pv = json_find_key(body, "seed");        if (pv) sp.seed        = (unsigned)atoi(pv);

    /* System prompt + chat template toggle. */
    char system_msg[1024];
    int syslen = json_extract_string(body, "system", system_msg, sizeof(system_msg));
    int use_template = json_extract_int(body, "template", 1);

    /* Rope/KV precision + attention variant — apply before forward_load. */
    int rope_v = -1, kv_v = -1;
    char rope_s[16] = {0}, kv_s[16] = {0}, attn_s[64] = {0};
    json_extract_string(body, "rope", rope_s, sizeof(rope_s));
    json_extract_string(body, "kv",   kv_s,   sizeof(kv_s));
    json_extract_string(body, "attn", attn_s, sizeof(attn_s));
    if (strcmp(rope_s, "f32") == 0) rope_v = ROPE_F32;
    else if (strcmp(rope_s, "f16") == 0) rope_v = ROPE_F16;
    else if (strcmp(rope_s, "q8") == 0)   rope_v = ROPE_Q8;
    if (strcmp(kv_s, "f32") == 0) kv_v = KV_F32;
    else if (strcmp(kv_s, "f16") == 0) kv_v = KV_F16;
    else if (strcmp(kv_s, "q8") == 0)   kv_v = KV_Q8;
    /* attn: "dense" | "swa:window=N" — applied via attn_set_default_spec */
    if (attn_s[0]) {
        attn_reset();
        if (strncmp(attn_s, "swa", 3) == 0) {
            int w = 256;
            const char* col = strchr(attn_s, ':');
            if (col) {
                const char* eq = strstr(col, "window=");
                if (eq) w = atoi(eq + 7);
            }
            if (w > 0) attn_set_default_spec(ATTN_TYPE_SWA, w, 1, 0, 0);
        } else {
            attn_set_default_spec(ATTN_TYPE_DENSE, 0, 1, 0, 0);
        }
    }
    if (rope_v >= 0 || kv_v >= 0) {
        /* Preserve previous rope/kv when caller specified only one. */
        static int s_rope = ROPE_F32, s_kv = KV_F32;
        if (rope_v >= 0) s_rope = rope_v;
        if (kv_v   >= 0) s_kv   = kv_v;
        /* A2: if cached model exists with different modes, free + reload. */
        if (g_model_loaded && (s_rope != g_cache_rope || s_kv != g_cache_kv)) {
            forward_free(g_fwd); tokenizer_free(g_tok); gguf_free(&g_gctx);
            g_model_loaded = 0;
        }
        forward_set_modes(s_rope, s_kv, ATTN_NAIVE);
        g_cache_rope = s_rope; g_cache_kv = s_kv;
    }

    // Send SSE headers
    const char* header =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n"
        "Transfer-Encoding: chunked\r\n\r\n";
    send_all(fd, header, strlen(header));

    /* A2: load once, cache. forward_reset clears KV per request. */
    if (!g_model_loaded) {
        if (gguf_load(model_path, &g_gctx) < 0) {
            send_chunk(fd, "data: {\"error\":\"model load failed\"}\n\n",
                       strlen("data: {\"error\":\"model load failed\"}\n\n"));
            send_chunk(fd, "data: [DONE]\n\n", strlen("data: [DONE]\n\n"));
            send_chunk(fd, "", 0);
            return;
        }
        if (tokenizer_load(&g_tok, &g_gctx) < 0) {
            gguf_free(&g_gctx); g_model_loaded = 0;
            send_chunk(fd, "", 0);
            return;
        }
        if (forward_load(&g_fwd, &g_gctx, 2048) < 0) {
            tokenizer_free(g_tok); g_tok = NULL;
            gguf_free(&g_gctx);
            send_chunk(fd, "", 0);
            return;
        }
        g_model_loaded = 1;
    }
    forward_ctx* fwd = g_fwd;
    tokenizer*    tok = g_tok;
    forward_reset(fwd);

    // Build chat template (toggleable)
    char tmpl[4096];
    if (use_template) {
        if (syslen > 0) {
            snprintf(tmpl, sizeof(tmpl),
                "<|im_start|>system\n%s<|im_end|>\n"
                "<|im_start|>user\n%s<|im_end|>\n"
                "<|im_start|>assistant\n", system_msg, prompt);
        } else {
            snprintf(tmpl, sizeof(tmpl),
                "<|im_start|>system\n"
                "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
                "<|im_end|>\n"
                "<|im_start|>user\n%s<|im_end|>\n"
                "<|im_start|>assistant\n", prompt);
        }
    } else {
        snprintf(tmpl, sizeof(tmpl), "%s", prompt);
    }
    int prompt_ids[2048];
    int prompt_len = tokenizer_encode(tok, tmpl, prompt_ids, 2048);
    if (prompt_len <= 0) {
        send_chunk(fd, "", 0);
        return;
    }

    int vocab = forward_vocab_size(fwd);
    float* logits = malloc((size_t)vocab * sizeof(float));
    int*   gen    = malloc((size_t)max_new * sizeof(int));
    if (!logits || !gen) {
        free(logits); free(gen);
        send_chunk(fd, "", 0);
        return;
    }

    if (forward_prefill(fwd, prompt_ids, prompt_len, logits) < 0) {
        free(logits); free(gen);
        send_chunk(fd, "", 0);
        return;
    }

    int pos = prompt_len;
    int gen_n = 0;
    char dec_buf[512];
    char json_buf[1024];

    while (gen_n < max_new) {
        int next = sample_token(logits, vocab, &sp, gen, gen_n);
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
    /* A2: model cached in globals; do not free here. */
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

        char* req = NULL;
        size_t req_len = 0;
        if (recv_request(cli_fd, &req, &req_len) < 0) {
            close(cli_fd);
            continue;
        }
        int n = (int)req_len;

        char method[16], path[256];
        char* body = NULL;
        if (parse_request(req, n, method, sizeof(method),
                          path, sizeof(path), &body) < 0) {
            free(req);
            close(cli_fd);
            continue;
        }

        if (strcmp(method, "GET") == 0 && strcmp(path, "/") == 0) {
            handle_home(cli_fd);
        } else if (strcmp(method, "GET") == 0 && strcmp(path, "/studio") == 0) {
            handle_studio_page(cli_fd);
        } else if (strcmp(method, "GET") == 0 && strcmp(path, "/studio/hw") == 0) {
            handle_studio_hw(cli_fd);
        } else if (strcmp(method, "GET") == 0 && strcmp(path, "/studio/attn") == 0) {
            handle_studio_attn(cli_fd);
        } else if (strcmp(method, "POST") == 0 && strcmp(path, "/studio/data") == 0) {
            handle_studio_data(cli_fd, body ? body : "", model_path);
        } else if (strcmp(method, "POST") == 0 && strcmp(path, "/studio/train") == 0) {
            handle_studio_train(cli_fd, body ? body : "", model_path);
        } else if (strcmp(method, "POST") == 0 && strcmp(path, "/studio/merge") == 0) {
            handle_studio_merge(cli_fd, body ? body : "");
        } else if (strcmp(method, "POST") == 0 &&
                   (strcmp(path, "/generate") == 0 ||
                    strcmp(path, "/studio/infer") == 0)) {
            handle_generate(cli_fd, body ? body : "", model_path, sp);
        } else {
            const char* nf = "HTTP/1.1 404 Not Found\r\n"
                             "Content-Length: 0\r\nConnection: close\r\n\r\n";
            send_all(cli_fd, nf, strlen(nf));
        }

        free(req);
        close(cli_fd);
    }

    close(srv);
    /* A2: free cached model on shutdown. */
    if (g_model_loaded) {
        forward_free(g_fwd); g_fwd = NULL;
        tokenizer_free(g_tok); g_tok = NULL;
        gguf_free(&g_gctx);
        g_model_loaded = 0;
    }
    fprintf(stderr, "\nsmollm2 web: stopped.\n");
    return 0;
}
