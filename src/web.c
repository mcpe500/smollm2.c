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
"<label>Prompt</label><textarea id=\"inf-p\" rows=\"3\">hello</textarea>"
"<label>Tokens</label><input id=\"inf-n\" type=\"number\" value=\"64\">"
"<button class=\"primary\" id=\"inf-go\">Generate</button>"
"<div class=\"status\" id=\"inf-st\">Ready.</div>"
"<pre class=\"log\" id=\"inf-out\"></pre></section>"
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
"const n=+document.getElementById('inf-n').value||64;"
"const out=document.getElementById('inf-out');out.textContent='';"
"const st=document.getElementById('inf-st');st.textContent='Generating...';"
"try{const r=await fetch('/studio/infer',{method:'POST',"
"headers:{'Content-Type':'application/json'},"
"body:JSON.stringify({prompt:p,n:n})});"
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
    char buf[512];
    int n = snprintf(buf, sizeof(buf),
        "{\"mem_total_kb\":%ld,\"mem_avail_kb\":%ld,"
        "\"max_seq_advised\":%d,\"max_batch_advised\":%d,"
        "\"fullft_allowed\":%d,\"qlora_recommended\":%d}\n",
        c.mem_total_kb, c.mem_avail_kb,
        c.max_seq_advised, c.max_batch_advised,
        c.fullft_allowed, c.qlora_recommended);
    char header[256];
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
        "Content-Length: %d\r\nConnection: close\r\n\r\n", n);
    send_all(fd, header, (size_t)hlen);
    send_all(fd, buf, (size_t)n);
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
        "<|im_start|>user\n%s<|im_end|>\n"
        "<|im_start|>assistant\n", prompt);
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
        } else if (strcmp(method, "GET") == 0 && strcmp(path, "/studio") == 0) {
            handle_studio_page(cli_fd);
        } else if (strcmp(method, "GET") == 0 && strcmp(path, "/studio/hw") == 0) {
            handle_studio_hw(cli_fd);
        } else if (strcmp(method, "GET") == 0 && strcmp(path, "/studio/attn") == 0) {
            handle_studio_attn(cli_fd);
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
