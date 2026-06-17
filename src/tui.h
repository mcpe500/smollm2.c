// tui.h — full-screen ncurses TUI chat

#ifndef TUI_H
#define TUI_H

#include "sampling.h"

// Launch interactive TUI. Blocks until user quits (Ctrl+C or /quit).
// Returns 0 on clean exit, -1 on error.
int tui_run(const char* model_path, sample_params* sp);

#endif // TUI_H
