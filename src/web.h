// web.h — minimal C HTTP WebUI with SSE streaming

#ifndef WEB_H
#define WEB_H

#include "sampling.h"

// Start HTTP server on `port`. Blocks until Ctrl+C.
// Returns -1 on error (port in use, etc.).
int web_run(const char* model_path, int port, const sample_params* sp);

#endif // WEB_H
