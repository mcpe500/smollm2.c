// resolve_model.h — locate SmolLM2 GGUF on disk
//
// Auto-resolves the Ollama-installed blob path when no explicit --model
// is given. Returns heap-allocated path (caller frees) or NULL on miss.
#ifndef RESOLVE_MODEL_H
#define RESOLVE_MODEL_H

char* resolve_ollama_model_path(void);

#endif
