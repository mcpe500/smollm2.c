// gguf_write.h — minimal GGUF copy + tensor patch for LoRA merge
#ifndef GGUF_WRITE_H
#define GGUF_WRITE_H

#include <stddef.h>
#include <stdint.h>

/* Byte-exact copy. Reader (src/gguf.c) accepts the result. */
int  gguf_copy(const char* src_path, const char* dst_path);

/* Patch a single tensor in src GGUF, writing result to dst_path.
 * new_data must be exactly the tensor's byte size (in its source dtype).
 * Returns 0 on success, -1 on error. */
int  gguf_patch_tensor(const char* src_path, const char* dst_path,
                       const char* tensor_name,
                       const void* new_data, size_t n_bytes);

/* Patch multiple tensors in one pass (avoids repeated header parse + copy).
 * names[i] / data[i] / sizes[i] are parallel arrays of `count` entries. */
int  gguf_patch_tensors(const char* src_path, const char* dst_path,
                        const char* const* names,
                        const void* const* data,
                        const size_t* sizes,
                        int count);

#endif // GGUF_WRITE_H
