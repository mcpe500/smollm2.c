// tui.c — full-screen ncurses TUI chat for smollm2
//
// Layout:
//   +-------------------------------+
//   | scrollable chat history       |
//   | ...                           |
//   +-------------------------------+
//   | > user input line             |
//   +-------------------------------+
// Keys: Enter=submit, Ctrl+C/q=quit, PgUp/PgDn=scroll, Ctrl+K=interrupt, F1=help

#include "tui.h"
#include "gguf.h"
#include "tokenizer.h"
#include "forward.h"
#include "sampling.h"

#include <curses.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

#define MAX_HISTORY_LINES 4096
#define MAX_LINE_LEN      1024
#define INPUT_BUF_LEN     512
#define IM_END_TOKEN      2

// ---------------------------------------------------------------------------
// History buffer
// ---------------------------------------------------------------------------
typedef struct {
    char lines[MAX_HISTORY_LINES][MAX_LINE_LEN];
    int  colors[MAX_HISTORY_LINES];   // 1=user, 2=bot, 3=status
    int  count;
    int  scroll;  // first visible line index
} history_t;

static history_t g_hist;

static void hist_add(const char* line, int color) {
    if (g_hist.count >= MAX_HISTORY_LINES) {
        // rotate
        memmove(g_hist.lines, g_hist.lines + 1,
                (MAX_HISTORY_LINES - 1) * MAX_LINE_LEN);
        memmove(g_hist.colors, g_hist.colors + 1,
                (MAX_HISTORY_LINES - 1) * sizeof(int));
        g_hist.count = MAX_HISTORY_LINES - 1;
    }
    strncpy(g_hist.lines[g_hist.count], line, MAX_LINE_LEN - 1);
    g_hist.colors[g_hist.count] = color;
    g_hist.count++;
}

static void hist_append_char(char ch) {
    if (g_hist.count == 0) hist_add("", 2);
    int idx = g_hist.count - 1;
    size_t len = strlen(g_hist.lines[idx]);
    if (len + 1 < MAX_LINE_LEN) {
        g_hist.lines[idx][len] = ch;
        g_hist.lines[idx][len + 1] = '\0';
    }
}

// Append raw bytes (may include newlines, wrapping into new lines)
static void hist_append_bytes(const char* s, int n, int color) {
    for (int i = 0; i < n; i++) {
        if (s[i] == '\n' || s[i] == '\r') {
            hist_add("", color);
        } else {
            if (g_hist.count == 0) hist_add("", color);
            hist_append_char(s[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
static volatile int g_interrupted = 0;

/* Count how many screen rows a history line occupies with word-wrap at cols. */
static int line_visual_rows(const char* line, int cols) {
    if (cols < 1) cols = 1;
    int len = (int)strlen(line);
    if (len == 0) return 1;
    int rows = 0, col = 0;
    for (int i = 0; i < len; ) {
        /* find end of next word */
        int ws = i;
        while (ws < len && line[ws] == ' ') ws++;  /* leading spaces */
        int we = ws;
        while (we < len && line[we] != ' ') we++;  /* word chars */
        int word_len = we - i;  /* spaces + word */
        if (word_len == 0) break;
        if (col == 0) {
            /* always place at least one word per row even if it overflows */
            int placed = (word_len <= cols) ? word_len : cols;
            col += placed;
            if (col >= cols) { rows++; col = 0; }
            i += placed;
        } else if (col + word_len <= cols) {
            col += word_len;
            if (col >= cols) { rows++; col = 0; }
            i = we;
        } else {
            /* doesn't fit on current row, wrap */
            rows++; col = 0;
        }
    }
    if (col > 0) rows++;
    return rows > 0 ? rows : 1;
}

/* Total screen rows consumed by entries [0..count). */
static int count_visual_rows(int count, int cols) {
    int total = 0;
    for (int i = 0; i < count; i++)
        total += line_visual_rows(g_hist.lines[i], cols);
    return total;
}

static void render(WINDOW* hist_win, WINDOW* input_win,
                   const char* input_buf, int history_rows) {
    int cols = getmaxx(hist_win);
    if (cols < 1) cols = 1;

    /* Convert scroll (line index) to visual-row offset and clamp. */
    int total_vrows = count_visual_rows(g_hist.count, cols);
    /* g_hist.scroll is a logical line index; we render from there. */
    int first = g_hist.scroll;
    if (first < 0) first = 0;
    if (first >= g_hist.count && g_hist.count > 0)
        first = g_hist.count - 1;
    g_hist.scroll = first;
    (void)total_vrows;

    werase(hist_win);
    int screen_row = 0;
    for (int idx = first; idx < g_hist.count && screen_row < history_rows; idx++) {
        const char* line = g_hist.lines[idx];
        int len = (int)strlen(line);
        int c = g_hist.colors[idx];
        attr_t attr = (c == 1) ? (COLOR_PAIR(1) | A_BOLD)
                    : (c == 2) ? COLOR_PAIR(2)
                               : COLOR_PAIR(3);
        wattron(hist_win, attr);
        if (len == 0) {
            screen_row++;
        } else {
            int i = 0;
            int col = 0;
            while (i < len && screen_row < history_rows) {
                /* Gather word-wrap chunk: spaces then word */
                int ws = i;
                while (ws < len && line[ws] == ' ') ws++;
                int we = ws;
                while (we < len && line[we] != ' ') we++;
                int word_len = we - i;
                if (word_len == 0) break;

                if (col == 0) {
                    /* start of row: print up to cols chars */
                    int chunk = (word_len <= cols) ? word_len : cols;
                    mvwaddnstr(hist_win, screen_row, col, line + i, chunk);
                    col += chunk;
                    i += chunk;
                    if (col >= cols) { screen_row++; col = 0; }
                } else if (col + word_len <= cols) {
                    mvwaddnstr(hist_win, screen_row, col, line + i, word_len);
                    col += word_len;
                    i = we;
                    if (col >= cols) { screen_row++; col = 0; }
                } else {
                    /* wrap: skip leading spaces at start of new row */
                    screen_row++; col = 0;
                    while (i < len && line[i] == ' ') i++;
                }
            }
            if (col > 0) screen_row++;
        }
        wattroff(hist_win, attr);
    }
    wrefresh(hist_win);

    werase(input_win);
    mvwprintw(input_win, 0, 0, "> %s", input_buf);
    wrefresh(input_win);
}

static void scroll_to_bottom(int history_rows) {
    /* Walk backwards through lines until we've accumulated history_rows
       worth of visual rows, then set scroll to that line index. */
    int cols = 80;  /* conservative fallback; render() uses actual cols */
    int vrows = 0;
    int first = 0;
    for (int i = g_hist.count - 1; i >= 0; i--) {
        int lv = line_visual_rows(g_hist.lines[i], cols);
        if (vrows + lv > history_rows) { first = i + 1; break; }
        vrows += lv;
        if (i == 0) first = 0;
    }
    g_hist.scroll = first;
}

// ---------------------------------------------------------------------------
// Model state (shared across turns, forward_reset on /reset or new session)
// ---------------------------------------------------------------------------
typedef struct {
    gguf_ctx     gguf;
    tokenizer*   tok;
    forward_ctx* fwd;
    float*       logits;
    int          vocab;
    int*         gen_hist;
    int          gen_n;
    int          pos;      // current KV cache position
} model_t;

static int model_init(model_t* m, const char* path) {
    if (gguf_load(path, &m->gguf) < 0) return -1;
    if (tokenizer_load(&m->tok, &m->gguf) < 0) { gguf_free(&m->gguf); return -1; }
    if (forward_load(&m->fwd, &m->gguf, 2048) < 0) {
        tokenizer_free(m->tok); gguf_free(&m->gguf); return -1;
    }
    m->vocab = forward_vocab_size(m->fwd);
    m->logits = malloc((size_t)m->vocab * sizeof(float));
    m->gen_hist = malloc(2048 * sizeof(int));
    m->gen_n = 0;
    m->pos = 0;
    return (!m->logits || !m->gen_hist) ? -1 : 0;
}

static void model_free(model_t* m) {
    free(m->logits);
    free(m->gen_hist);
    forward_free(m->fwd);
    tokenizer_free(m->tok);
    gguf_free(&m->gguf);
}

#define CHAT_BUF_MAX 4096
static char g_chat_tmpl[CHAT_BUF_MAX];

static void build_prompt(const char* user_text) {
    snprintf(g_chat_tmpl, sizeof(g_chat_tmpl),
        "<|im_start|>user\n%s<|im_end|>\n"
        "<|im_start|>assistant\n",
        user_text);
}

// ---------------------------------------------------------------------------
// Generate one response, streaming tokens to hist_win
// ---------------------------------------------------------------------------
static void generate_response(model_t* m, const sample_params* sp,
                               WINDOW* hist_win, WINDOW* input_win,
                               int history_rows) {
    int prompt_ids[1024];
    int prompt_len = tokenizer_encode(m->tok, g_chat_tmpl,
                                      prompt_ids, 1024);
    if (prompt_len <= 0) return;

    forward_reset(m->fwd);
    m->gen_n = 0;
    m->pos = 0;
    g_interrupted = 0;

    if (forward_prefill(m->fwd, prompt_ids, prompt_len, m->logits) < 0) {
        hist_add("[error: prefill failed]", 3);
        return;
    }
    m->pos = prompt_len;

    hist_add("Bot: ", 2);
    char dec_buf[512];

    for (int step = 0; step < 200; step++) {
        if (g_interrupted) {
            hist_add(" [interrupted]", 3);
            break;
        }
        int next = sample_token(m->logits, m->vocab, sp,
                                m->gen_hist, m->gen_n);
        if (next == IM_END_TOKEN) break;
        if (m->gen_n < 2047) m->gen_hist[m->gen_n++] = next;

        int bytes = tokenizer_decode(m->tok, next, dec_buf, sizeof(dec_buf));
        if (bytes > 0)
            hist_append_bytes(dec_buf, bytes, 2);

        scroll_to_bottom(history_rows);
        render(hist_win, input_win, "", history_rows);

        if (m->pos < 2047) {
            if (forward_decode(m->fwd, next, m->pos, m->logits) < 0) break;
            m->pos++;
        } else break;
    }
}

// ---------------------------------------------------------------------------
// Help overlay
// ---------------------------------------------------------------------------
static void show_help(WINDOW* parent) {
    int rows, cols;
    getmaxyx(parent, rows, cols);
    int h = 12, w = 50;
    int y = (rows - h) / 2, x = (cols - w) / 2;
    WINDOW* hw = newwin(h, w, y, x);
    box(hw, 0, 0);
    mvwprintw(hw, 0, 2, " Help ");
    mvwprintw(hw, 2, 3, "Enter      Send message");
    mvwprintw(hw, 3, 3, "Ctrl+C     Quit");
    mvwprintw(hw, 4, 3, "Ctrl+K     Interrupt generation");
    mvwprintw(hw, 5, 3, "PgUp/PgDn  Scroll history");
    mvwprintw(hw, 6, 3, "/reset     Clear conversation");
    mvwprintw(hw, 7, 3, "/quit      Exit");
    mvwprintw(hw, 9, 3, "Press any key to close");
    wrefresh(hw);
    wgetch(hw);
    delwin(hw);
    touchwin(parent);
    wrefresh(parent);
}

// ---------------------------------------------------------------------------
// Main TUI loop
// ---------------------------------------------------------------------------
int tui_run(const char* model_path, sample_params* sp) {
    model_t m;
    memset(&m, 0, sizeof(m));

    /* Load model before starting ncurses so error messages show cleanly */
    if (model_init(&m, model_path) < 0) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return -1;
    }

    initscr();
    start_color();
    use_default_colors();
    cbreak();
    noecho();
    keypad(stdscr, TRUE);
    curs_set(1);

    /* Color pairs: 1=user cyan+bold, 2=bot white, 3=status dim */
    init_pair(1, COLOR_CYAN, -1);
    init_pair(2, COLOR_WHITE, -1);
    init_pair(3, COLOR_YELLOW, -1);

    int rows, cols;
    getmaxyx(stdscr, rows, cols);
    int history_rows = rows - 3;  /* leave 1 for separator, 2 for input */
    if (history_rows < 2) history_rows = 2;

    /* Windows */
    WINDOW* hist_win  = newwin(history_rows, cols, 0, 0);
    WINDOW* sep_win   = newwin(1, cols, history_rows, 0);
    WINDOW* input_win = newwin(2, cols, history_rows + 1, 0);
    scrollok(hist_win, FALSE);

    /* Status bar */
    wbkgd(sep_win, COLOR_PAIR(3) | A_REVERSE);
    mvwprintw(sep_win, 0, 0, " smollm2  F1=help  Ctrl+K=interrupt  /reset  /quit ");
    wrefresh(sep_win);

    hist_add("smollm2:135m ready. Type a message and press Enter.", 3);
    hist_add("", 3);

    char input_buf[INPUT_BUF_LEN];
    memset(input_buf, 0, sizeof(input_buf));
    int input_len = 0;

    render(hist_win, input_win, input_buf, history_rows);

    int quit = 0;
    while (!quit) {
        /* Position cursor at input */
        wmove(input_win, 0, 2 + input_len);
        wrefresh(input_win);

        int ch = wgetch(input_win);

        if (ch == 3) {  /* Ctrl+C */
            if (g_interrupted == 0) g_interrupted = 1;
            else quit = 1;
            continue;
        }

        if (ch == KEY_F(1)) {
            show_help(hist_win);
            render(hist_win, input_win, input_buf, history_rows);
            continue;
        }

        if (ch == KEY_PPAGE) {
            g_hist.scroll -= history_rows / 2;
            if (g_hist.scroll < 0) g_hist.scroll = 0;
            render(hist_win, input_win, input_buf, history_rows);
            continue;
        }
        if (ch == KEY_NPAGE) {
            g_hist.scroll += history_rows / 2;
            if (g_hist.scroll > g_hist.count - history_rows)
                g_hist.scroll = g_hist.count - history_rows;
            if (g_hist.scroll < 0) g_hist.scroll = 0;
            render(hist_win, input_win, input_buf, history_rows);
            continue;
        }

        if (ch == 11) {  /* Ctrl+K: interrupt generation */
            g_interrupted = 1;
            continue;
        }

        if (ch == KEY_BACKSPACE || ch == 127 || ch == 8) {
            if (input_len > 0) {
                input_buf[--input_len] = '\0';
                render(hist_win, input_win, input_buf, history_rows);
            }
            continue;
        }

        if (ch == '\n' || ch == KEY_ENTER) {
            if (input_len == 0) continue;

            /* Commands */
            if (strcmp(input_buf, "/quit") == 0 || strcmp(input_buf, "q") == 0) {
                quit = 1;
                break;
            }
            if (strcmp(input_buf, "/reset") == 0) {
                g_hist.count = 0;
                g_hist.scroll = 0;
                hist_add("Conversation reset.", 3);
                forward_reset(m.fwd);
                m.pos = 0; m.gen_n = 0;
                input_len = 0; input_buf[0] = '\0';
                render(hist_win, input_win, input_buf, history_rows);
                continue;
            }
            if (strncmp(input_buf, "/temp ", 6) == 0) {
                sp->temperature = (float)atof(input_buf + 6);
                char msg[64];
                snprintf(msg, sizeof(msg), "Temperature set to %.2f", sp->temperature);
                hist_add(msg, 3);
                input_len = 0; input_buf[0] = '\0';
                render(hist_win, input_win, input_buf, history_rows);
                continue;
            }
            if (strncmp(input_buf, "/topk ", 6) == 0) {
                sp->top_k = atoi(input_buf + 6);
                char msg[64];
                snprintf(msg, sizeof(msg), "Top-k set to %d", sp->top_k);
                hist_add(msg, 3);
                input_len = 0; input_buf[0] = '\0';
                render(hist_win, input_win, input_buf, history_rows);
                continue;
            }
            if (strncmp(input_buf, "/topp ", 6) == 0) {
                sp->top_p = (float)atof(input_buf + 6);
                char msg[64];
                snprintf(msg, sizeof(msg), "Top-p set to %.2f", sp->top_p);
                hist_add(msg, 3);
                input_len = 0; input_buf[0] = '\0';
                render(hist_win, input_win, input_buf, history_rows);
                continue;
            }
            if (strcmp(input_buf, "/settings") == 0) {
                char msg[128];
                snprintf(msg, sizeof(msg),
                    "temp=%.2f top-p=%.2f top-k=%d rep-penalty=%.2f",
                    sp->temperature, sp->top_p, sp->top_k, sp->rep_penalty);
                hist_add(msg, 3);
                input_len = 0; input_buf[0] = '\0';
                render(hist_win, input_win, input_buf, history_rows);
                continue;
            }

            /* Normal message */
            char user_line[INPUT_BUF_LEN + 6];
            snprintf(user_line, sizeof(user_line), "You: %s", input_buf);
            hist_add(user_line, 1);

            build_prompt(input_buf);
            input_len = 0; input_buf[0] = '\0';

            scroll_to_bottom(history_rows);
            render(hist_win, input_win, input_buf, history_rows);

            generate_response(&m, sp, hist_win, input_win, history_rows);

            scroll_to_bottom(history_rows);
            hist_add("", 3);  /* blank line between turns */
            render(hist_win, input_win, input_buf, history_rows);
            continue;
        }

        /* Regular printable character */
        if (ch >= 32 && ch < 127 && input_len < INPUT_BUF_LEN - 1) {
            input_buf[input_len++] = (char)ch;
            input_buf[input_len] = '\0';
            render(hist_win, input_win, input_buf, history_rows);
        }
    }

    delwin(input_win);
    delwin(sep_win);
    delwin(hist_win);
    endwin();

    model_free(&m);
    return 0;
}
