// chat_history.h - Chat history management for SmolLM2
#ifndef CHAT_HISTORY_H
#define CHAT_HISTORY_H

#include <stddef.h>

// Forward declaration (actual type in smollm2.h)
typedef struct sm2_tokenizer sm2_tokenizer;

#define MAX_MESSAGES 200
#define MAX_MESSAGE_LEN 4096
#define MAX_TOKENS 2048

typedef struct {
    char role[16];        // "system", "user", "assistant"
    char content[MAX_MESSAGE_LEN];
} chat_message;

typedef struct {
    chat_message messages[MAX_MESSAGES];
    int n_messages;
} chat_history;

// Initialize empty chat history
void chat_history_init(chat_history* hist);

// Add user message to history
int chat_history_add_user(chat_history* hist, const char* content);

// Add assistant response to history
int chat_history_add_assistant(chat_history* hist, const char* content);

// Set system prompt (replaces existing system message if any)
int chat_history_set_system(chat_history* hist, const char* system_prompt);

// Clear all messages except system
void chat_history_clear(chat_history* hist);

// Build chat template string for tokenization
// Returns allocated string, caller must free
char* chat_history_build_prompt(chat_history* hist, const char* new_user_input);

// Get approximate token count for history
int chat_history_token_count(chat_history* hist);

// Trim history to fit within max_tokens
void chat_history_trim(chat_history* hist, int max_tokens);

// Special token IDs for SmolLM2 chat template
// These are the actual vocab token IDs, not string representations
#define SM2_TOKEN_IM_START 1    // <|im_start|>
#define SM2_TOKEN_IM_END  2    // <|im_end|>
#define SM2_TOKEN_NEWLINE 198  // Ċ (newline character)

// Build chat template directly as tokens (more reliable than string encoding)
// This bypasses the tokenizer to avoid string encoding issues with special tokens
// Returns number of tokens written, or -1 on error
int chat_history_build_prompt_tokens(chat_history* hist, const char* new_user_input,
                                     int* tokens, int max_tokens, sm2_tokenizer* tok);

#endif // CHAT_HISTORY_H