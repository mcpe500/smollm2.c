// chat_history.c - Chat history management implementation
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "chat_history.h"
#include "smollm2.h"

void chat_history_init(chat_history* hist) {
    if (!hist) return;
    hist->n_messages = 0;
    memset(hist->messages, 0, sizeof(hist->messages));
}

int chat_history_add_user(chat_history* hist, const char* content) {
    if (!hist || !content) return -1;
    if (hist->n_messages >= MAX_MESSAGES) return -1;

    chat_message* msg = &hist->messages[hist->n_messages++];
    strncpy(msg->role, "user", sizeof(msg->role) - 1);
    msg->role[sizeof(msg->role) - 1] = '\0';
    strncpy(msg->content, content, MAX_MESSAGE_LEN - 1);
    msg->content[MAX_MESSAGE_LEN - 1] = '\0';

    return 0;
}

int chat_history_add_assistant(chat_history* hist, const char* content) {
    if (!hist || !content) return -1;
    if (hist->n_messages >= MAX_MESSAGES) return -1;

    chat_message* msg = &hist->messages[hist->n_messages++];
    strncpy(msg->role, "assistant", sizeof(msg->role) - 1);
    msg->role[sizeof(msg->role) - 1] = '\0';
    strncpy(msg->content, content, MAX_MESSAGE_LEN - 1);
    msg->content[MAX_MESSAGE_LEN - 1] = '\0';

    return 0;
}

int chat_history_set_system(chat_history* hist, const char* system_prompt) {
    if (!hist || !system_prompt) return -1;

    // Check if system message already exists
    for (int i = 0; i < hist->n_messages; i++) {
        if (strcmp(hist->messages[i].role, "system") == 0) {
            strncpy(hist->messages[i].content, system_prompt, MAX_MESSAGE_LEN - 1);
            hist->messages[i].content[MAX_MESSAGE_LEN - 1] = '\0';
            return 0;
        }
    }

    // Add new system message at the beginning
    if (hist->n_messages >= MAX_MESSAGES) return -1;

    // Shift existing messages down
    for (int i = hist->n_messages; i > 0; i--) {
        hist->messages[i] = hist->messages[i - 1];
    }

    chat_message* msg = &hist->messages[0];
    strncpy(msg->role, "system", sizeof(msg->role) - 1);
    msg->role[sizeof(msg->role) - 1] = '\0';
    strncpy(msg->content, system_prompt, MAX_MESSAGE_LEN - 1);
    msg->content[MAX_MESSAGE_LEN - 1] = '\0';
    hist->n_messages++;

    return 0;
}

void chat_history_clear(chat_history* hist) {
    if (!hist) return;
    // Keep system message, remove all others
    int sys_count = 0;
    for (int i = 0; i < hist->n_messages; i++) {
        if (strcmp(hist->messages[i].role, "system") == 0) {
            hist->messages[sys_count++] = hist->messages[i];
        }
    }
    hist->n_messages = sys_count;
}

// Build chat template: <|im_start|>role\ncontent<|im_end|>\n...
// Uses special token bytes: 0x80 = <|im_start|>, 0x81 = <|im_end|>, 0x82 = Ċ (newline)
// These map to vocab tokens 1, 2, 198 respectively
char* chat_history_build_prompt(chat_history* hist, const char* new_user_input) {
    if (!hist) return NULL;

    // Calculate required buffer size
    size_t capacity = 8192;
    char* result = malloc(capacity);
    if (!result) return NULL;

    size_t pos = 0;

    // Special tokens as raw bytes
    // 0x80 = <|im_start|> (maps to token 1)
    // 0x81 = <|im_end|> (maps to token 2)
    // 0x82 = Ċ newline (maps to token 198)

    // Add existing messages
    for (int i = 0; i < hist->n_messages; i++) {
        chat_message* msg = &hist->messages[i];

        // Reserve space for: 1 byte + role + 1 byte + content + 1 byte + 1 byte
        size_t needed = 5 + strlen(msg->role) + strlen(msg->content);
        if (pos + needed >= capacity) {
            capacity *= 2;
            char* new_result = realloc(result, capacity);
            if (!new_result) {
                free(result);
                return NULL;
            }
            result = new_result;
        }

        // <|im_start|> (0x80)
        result[pos++] = '\x80';
        // role
        size_t role_len = strlen(msg->role);
        memcpy(result + pos, msg->role, role_len);
        pos += role_len;
        // Ċ newline (0x82)
        result[pos++] = '\x82';
        // content
        size_t content_len = strlen(msg->content);
        memcpy(result + pos, msg->content, content_len);
        pos += content_len;
        // <|im_end|> (0x81)
        result[pos++] = '\x81';
        // Ċ newline (0x82)
        result[pos++] = '\x82';
    }

    // Add new user input if provided
    if (new_user_input && new_user_input[0]) {
        size_t needed = 30 + strlen(new_user_input);
        if (pos + needed >= capacity) {
            capacity += needed + 1024;
            char* new_result = realloc(result, capacity);
            if (!new_result) {
                free(result);
                return NULL;
            }
            result = new_result;
        }

        // <|im_start|>user (0x80 + "user")
        result[pos++] = '\x80';
        memcpy(result + pos, "user", 4);
        pos += 4;
        // Ċ newline (0x82)
        result[pos++] = '\x82';
        // user message
        size_t msg_len = strlen(new_user_input);
        memcpy(result + pos, new_user_input, msg_len);
        pos += msg_len;
        // <|im_end|> (0x81)
        result[pos++] = '\x81';
        // Ċ newline (0x82)
        result[pos++] = '\x82';
        // <|im_start|>assistant (0x80 + "assistant")
        result[pos++] = '\x80';
        memcpy(result + pos, "assistant", 9);
        pos += 9;
        // Ċ newline (0x82)
        result[pos++] = '\x82';
    }

    result[pos] = '\0';
    return result;
}

// Rough estimate: ~1.5 chars per token for typical English
int chat_history_token_count(chat_history* hist) {
    if (!hist) return 0;

    int total_chars = 0;
    for (int i = 0; i < hist->n_messages; i++) {
        total_chars += 20 + strlen(hist->messages[i].role) + strlen(hist->messages[i].content);
    }
    return (total_chars * 2) / 3;  // Conservative estimate
}

void chat_history_trim(chat_history* hist, int max_tokens) {
    if (!hist || max_tokens <= 0) return;

    while (chat_history_token_count(hist) > max_tokens && hist->n_messages > 1) {
        // Remove oldest non-system message
        for (int i = 0; i < hist->n_messages - 1; i++) {
            if (strcmp(hist->messages[i].role, "system") != 0) {
                // Remove this message
                for (int j = i; j < hist->n_messages - 1; j++) {
                    hist->messages[j] = hist->messages[j + 1];
                }
                hist->n_messages--;
                break;
            }
        }
    }
}

// Build chat template directly as tokens
// Uses actual token IDs: <|im_start|>=1, <|im_end|>=2, Ċ (newline)=198
int chat_history_build_prompt_tokens(chat_history* hist, const char* new_user_input,
                                      int* tokens, int max_tokens, sm2_tokenizer* tok) {
    if (!hist || !tokens || max_tokens < 16) return -1;

    int n = 0;

    // Add existing messages
    for (int i = 0; i < hist->n_messages && n < max_tokens - 10; i++) {
        chat_message* msg = &hist->messages[i];

        // <|im_start|> = 1
        tokens[n++] = SM2_TOKEN_IM_START;

        // Encode role
        if (tok && msg->role[0]) {
            int role_tokens[32];
            int n_role = sm2_tokenizer_encode(tok, msg->role, role_tokens, 32);
            for (int j = 0; j < n_role && n < max_tokens; j++) {
                tokens[n++] = role_tokens[j];
            }
        }

        // Ċ newline = 198
        tokens[n++] = SM2_TOKEN_NEWLINE;

        // Encode content
        if (tok && msg->content[0]) {
            int content_tokens[256];
            int n_content = sm2_tokenizer_encode(tok, msg->content, content_tokens,
                                                max_tokens - n - 5);
            for (int j = 0; j < n_content && n < max_tokens - 5; j++) {
                tokens[n++] = content_tokens[j];
            }
        }

        // <|im_end|> = 2
        tokens[n++] = SM2_TOKEN_IM_END;
        // Ċ newline = 198
        tokens[n++] = SM2_TOKEN_NEWLINE;
    }

    // Add new user input if provided
    if (new_user_input && new_user_input[0] && n < max_tokens - 10) {
        // <|im_start|>user
        tokens[n++] = SM2_TOKEN_IM_START;
        if (tok) {
            int role_tokens[32];
            int n_role = sm2_tokenizer_encode(tok, "user", role_tokens, 32);
            for (int j = 0; j < n_role && n < max_tokens; j++) {
                tokens[n++] = role_tokens[j];
            }
        }
        tokens[n++] = SM2_TOKEN_NEWLINE;

        // User message
        if (tok) {
            int msg_tokens[512];
            int n_msg = sm2_tokenizer_encode(tok, new_user_input, msg_tokens,
                                            max_tokens - n - 10);
            for (int j = 0; j < n_msg && n < max_tokens - 10; j++) {
                tokens[n++] = msg_tokens[j];
            }
        }

        // <|im_end|>
        tokens[n++] = SM2_TOKEN_IM_END;
        tokens[n++] = SM2_TOKEN_NEWLINE;
    }

    // Always add <|im_start|>assistant at the end (model needs this to start generating)
    if (n < max_tokens - 10) {
        tokens[n++] = SM2_TOKEN_IM_START;
        if (tok) {
            int role_tokens[32];
            int n_role = sm2_tokenizer_encode(tok, "assistant", role_tokens, 32);
            for (int j = 0; j < n_role && n < max_tokens; j++) {
                tokens[n++] = role_tokens[j];
            }
        }
        tokens[n++] = SM2_TOKEN_NEWLINE;
    }

    return n;
}