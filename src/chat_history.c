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
char* chat_history_build_prompt(chat_history* hist, const char* new_user_input) {
    if (!hist) return NULL;

    // Calculate required buffer size
    size_t capacity = 8192;
    char* result = malloc(capacity);
    if (!result) return NULL;

    size_t pos = 0;
    int add_newline = 0;

    // Add existing messages
    for (int i = 0; i < hist->n_messages; i++) {
        chat_message* msg = &hist->messages[i];

        // Reserve space for: <|im_start|> + role + \n + content + <|im_end|> + \n
        size_t needed = 20 + strlen(msg->role) + strlen(msg->content);
        if (pos + needed >= capacity) {
            capacity *= 2;
            char* new_result = realloc(result, capacity);
            if (!new_result) {
                free(result);
                return NULL;
            }
            result = new_result;
        }

        pos += snprintf(result + pos, capacity - pos,
            "<|im_start|>%s\n%s<|im_end|>\n",
            msg->role, msg->content);
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

        pos += snprintf(result + pos, capacity - pos,
            "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
            new_user_input);
    }

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