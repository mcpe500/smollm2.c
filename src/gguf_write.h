// gguf_write.h — minimal GGUF copy for studio phase 1
#ifndef GGUF_WRITE_H
#define GGUF_WRITE_H

/* Byte-exact copy. Reader (src/gguf.c) accepts the result. Phase 2 will
   add writer API for tensor modification / LoRA merge. */
int  gguf_copy(const char* src_path, const char* dst_path);

#endif // GGUF_WRITE_H