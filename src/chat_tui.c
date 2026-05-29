// chat_tui.c - Text-based UI chat mode for SmolLM2
// Uses ANSI escape codes (no ncurses dependency)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <termios.h>
#include <signal.h>
#include <time.h>
#include "chat_history.h"
#include "smollm2.h"

// ANSI escape codes
#define ANSI_CLEAR     "\033[2J"
#define ANSI_HOME      "\033[H"
#define ANSI_SHOW_CURSOR "\033[?25h"
#define ANSI_HIDE_CURSOR "\033[?25l"
#define ANSI_BOLD      "\033[1m"
#define ANSI_RESET     "\033[0m"
#define ANSI_CYAN      "\033[36m"
#define ANSI_GREEN     "\033[32m"
#define ANSI_YELLOW    "\033[33m"
#define ANSI_MAGENTA   "\033[35m"
#define ANSI_WHITE     "\033[37m"
#define ANSI_DIM       "\033[2m"

// TUI state
typedef struct {
    chat_history history;
    char input_buffer[2048];
    int input_pos;
    int input_scroll;       // Scroll position in input
    int generating;
    double tokens_per_sec;
    int window_width;
    int window_height;
} tui_state;

static tui_state g_state = {0};
static struct termios g_orig_term;
static int g_raw_mode = 0;
static sm2_model* g_model = NULL;
static sm2_tokenizer* g_tok = NULL;
static cli_args* g_args = NULL;

// Time measurement
static double time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// Print token helper
static void print_token(const char* decoded) {
    if (!decoded) return;
    for (const char* p = decoded; *p; p++) {
        unsigned char c = (unsigned char)*p;
        if (c == 0xC4 && (unsigned char)p[1] == 0xA0) {
            putchar(' ');
            p++;
        } else if (c == 0xC4 && (unsigned char)p[1] == 0x8A) {
            putchar('\n');
            p++;
        } else {
            putchar(*p);
        }
    }
}

// Cleanup handler for Ctrl+C
static void cleanup(int sig) {
    (void)sig;
    if (g_raw_mode) {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &g_orig_term);
        printf("\n%s\n", ANSI_SHOW_CURSOR ANSI_RESET);
    }
    exit(0);
}

// Setup raw terminal mode
static int setup_raw_mode() {
    struct termios raw;
    if (tcgetattr(STDIN_FILENO, &g_orig_term) == -1) return -1;
    raw = g_orig_term;
    raw.c_lflag &= ~(ICANON | ECHO);
    raw.c_cc[VMIN] = 1;
    raw.c_cc[VTIME] = 0;
    if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw) == -1) return -1;
    g_raw_mode = 1;
    return 0;
}

// Restore terminal
static void restore_terminal() {
    if (g_raw_mode) {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &g_orig_term);
        g_raw_mode = 0;
    }
}

// Clear line and move to beginning
static void clear_line() {
    printf("\033[2K\r");
}

// Get terminal size (simple version)
static void update_window_size() {
    g_state.window_width = 80;
    g_state.window_height = 24;
    // Could use ioctl(TIOCGWINSZ) for real size
}

// Render the TUI
static void tui_render() {
    printf(ANSI_CLEAR ANSI_HOME);

    // Header
    printf(ANSI_BOLD " SmolLM2 Chat " ANSI_RESET);
    printf(ANSI_DIM "(Press Ctrl+C to quit)" ANSI_RESET "\n");
    printf("\n");

    // Chat history
    for (int i = 0; i < g_state.history.n_messages; i++) {
        chat_message* msg = &g_state.history.messages[i];

        if (strcmp(msg->role, "user") == 0) {
            printf(ANSI_CYAN "▌ You: " ANSI_RESET);
            // Word wrap
            int col = 7;
            const char* p = msg->content;
            while (*p) {
                if (col >= g_state.window_width - 1) {
                    printf("\n       ");
                    col = 7;
                }
                unsigned char c = (unsigned char)*p;
                if (c == 0xC4 && (unsigned char)p[1] == 0xA0) {
                    putchar(' ');
                    p++; col++;
                } else if (c == 0xC4 && (unsigned char)p[1] == 0x8A) {
                    putchar('\n');
                    printf("       ");
                    col = 7;
                    p++;
                } else {
                    putchar(*p);
                    col++;
                }
                p++;
            }
            printf("\n\n");
        } else if (strcmp(msg->role, "assistant") == 0) {
            printf(ANSI_GREEN "▌ SmolLM: " ANSI_RESET);
            int col = 9;
            const char* p = msg->content;
            while (*p) {
                if (col >= g_state.window_width - 1) {
                    printf("\n         ");
                    col = 9;
                }
                unsigned char c = (unsigned char)*p;
                if (c == 0xC4 && (unsigned char)p[1] == 0xA0) {
                    putchar(' ');
                    p++; col++;
                } else if (c == 0xC4 && (unsigned char)p[1] == 0x8A) {
                    putchar('\n');
                    printf("         ");
                    col = 9;
                    p++;
                } else {
                    putchar(*p);
                    col++;
                }
                p++;
            }
            printf("\n\n");
        } else if (strcmp(msg->role, "system") == 0) {
            printf(ANSI_MAGENTA "▌ System: " ANSI_RESET);
            int col = 9;
            const char* p = msg->content;
            while (*p && col < 50) {
                putchar(*p);
                col++;
                p++;
            }
            if (strlen(msg->content) > 50) printf(ANSI_DIM "..." ANSI_RESET);
            printf("\n\n");
        }
    }

    // Input line
    printf("\n");
    printf(ANSI_YELLOW "▐> " ANSI_RESET);

    // Show input with scroll
    const char* input = g_state.input_buffer;
    int display_len = strlen(input);
    int start = 0;

    if (display_len > g_state.window_width - 4) {
        start = display_len - (g_state.window_width - 4);
    }

    for (int i = start; i < display_len; i++) {
        putchar(input[i]);
    }

    // Cursor
    int cursor_x = 2 + (display_len - start);
    printf("\033[%dG", cursor_x);

    // Status bar
    if (g_state.generating) {
        printf("\n" ANSI_DIM "Generating... %.1f tok/s" ANSI_RESET, g_state.tokens_per_sec);
    } else {
        printf("\n" ANSI_DIM "[Enter] send  [Ctrl+L] clear" ANSI_RESET);
    }

    fflush(stdout);
}

// Generate response asynchronously (simplified: synchronous)
static int generate_response(const char* user_input) {
    chat_history_add_user(&g_state.history, user_input);

    char* prompt = chat_history_build_prompt(&g_state.history, NULL);
    if (!prompt) return -1;

    int tokens[4096];
    int n_tokens = sm2_tokenizer_encode(g_tok, prompt, tokens, 4096);
    free(prompt);

    if (n_tokens <= 0) return -1;

    sm2_context* ctx;
    if (sm2_create_context(g_model, &ctx) != 0) return -1;

    ctx->params.temperature = g_args->temperature;
    ctx->params.top_p = g_args->top_p;
    ctx->params.top_k = g_args->top_k;
    ctx->params.max_context = 8192;
    ctx->params.max_output = 512;
    ctx->params.repetition_penalty = g_args->repetition_penalty;
    ctx->params.penalty_window = 32;

    if (sm2_prefill(ctx, tokens, n_tokens) != 0) {
        sm2_free_context(ctx);
        return -1;
    }

    g_state.generating = 1;
    double t0 = time_ms();

    char response[8192];
    int resp_len = 0;
    int gen_tokens = 0;

    while (resp_len < (int)sizeof(response) - 1) {
        int token;
        if (sm2_decode_next(ctx, &token) != 0) break;

        if (token < 3) break; // EOS

        char* decoded = sm2_tokenizer_decode(g_tok, &token, 1);
        if (decoded) {
            // Print and accumulate
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
        gen_tokens++;

        // Update stats
        double dt = (time_ms() - t0) / 1000.0;
        if (dt > 0.1) {
            g_state.tokens_per_sec = gen_tokens / dt;
        }
    }

    response[resp_len] = '\0';
    sm2_free_context(ctx);

    g_state.generating = 0;
    g_state.tokens_per_sec = gen_tokens / ((time_ms() - t0) / 1000.0);

    chat_history_add_assistant(&g_state.history, response);
    chat_history_trim(&g_state.history, 1800);

    return 0;
}

// Handle escape sequences (arrow keys, etc.)
static int handle_escape(char c) {
    (void)c;
    // Read remaining bytes of escape sequence
    char seq[3] = {0};
    if (read(STDIN_FILENO, &seq[0], 1) <= 0) return 0;
    if (read(STDIN_FILENO, &seq[1], 1) <= 0) return 0;

    if (seq[0] == '[') {
        switch (seq[1]) {
            case 'A': // Up arrow - scroll history
                break;
            case 'B': // Down arrow
                break;
            case 'C': // Right - move cursor
                if (g_state.input_pos < (int)strlen(g_state.input_buffer)) {
                    g_state.input_pos++;
                }
                break;
            case 'D': // Left - move cursor
                if (g_state.input_pos > 0) {
                    g_state.input_pos--;
                }
                break;
        }
    }
    return 0;
}

int run_chat_tui(sm2_model* model, const cli_args* args) {
    g_model = model;
    g_tok = model->tokenizer;
    g_args = (cli_args*)args;

    chat_history_init(&g_state.history);
    chat_history_set_system(&g_state.history, "You are a helpful AI assistant named SmolLM, trained by Hugging Face.");

    g_state.input_buffer[0] = '\0';
    g_state.input_pos = 0;
    g_state.generating = 0;
    g_state.tokens_per_sec = 0;
    update_window_size();

    // Setup
    signal(SIGINT, cleanup);
    if (setup_raw_mode() != 0) {
        fprintf(stderr, "Failed to setup raw terminal mode\n");
        return -1;
    }

    printf(ANSI_HIDE_CURSOR);
    tui_render();

    // Main loop
    while (1) {
        char c;
        int n = read(STDIN_FILENO, &c, 1);
        if (n <= 0) break;

        if (g_state.generating) {
            // Ignore input while generating
            continue;
        }

        if (c == 3) { // Ctrl+C
            break;
        } else if (c == 12) { // Ctrl+L - clear screen and history
            chat_history_clear(&g_state.history);
            g_state.input_buffer[0] = '\0';
            g_state.input_pos = 0;
            tui_render();
        } else if (c == 27) { // Escape
            handle_escape(c);
        } else if (c == 127 || c == 8) { // Backspace
            if (g_state.input_pos > 0) {
                g_state.input_pos--;
                int len = strlen(g_state.input_buffer);
                for (int i = g_state.input_pos; i < len; i++) {
                    g_state.input_buffer[i] = g_state.input_buffer[i + 1];
                }
            }
            tui_render();
        } else if (c == '\n' || c == '\r') { // Enter - submit
            if (strlen(g_state.input_buffer) > 0) {
                // Save input and clear
                char input_copy[2048];
                strncpy(input_copy, g_state.input_buffer, sizeof(input_copy) - 1);
                input_copy[sizeof(input_copy) - 1] = '\0';

                g_state.input_buffer[0] = '\0';
                g_state.input_pos = 0;

                tui_render();

                // Generate response
                generate_response(input_copy);

                tui_render();
            }
        } else if (c >= 32 && c < 127) { // Regular character
            int len = strlen(g_state.input_buffer);
            if (len < (int)sizeof(g_state.input_buffer) - 1) {
                // Insert at cursor position
                for (int i = len; i > g_state.input_pos; i--) {
                    g_state.input_buffer[i] = g_state.input_buffer[i - 1];
                }
                g_state.input_buffer[g_state.input_pos++] = c;
                g_state.input_buffer[len + 1] = '\0';
            }
            tui_render();
        }
    }

    // Cleanup
    restore_terminal();
    printf(ANSI_SHOW_CURSOR ANSI_CLEAR ANSI_HOME);
    printf("\nGoodbye!\n");

    return 0;
}