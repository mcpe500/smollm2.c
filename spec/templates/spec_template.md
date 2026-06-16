# XXX_<task_name>.md

> Replace `XXX` with the next 3-digit number (continue from highest existing in `spec/`).
> Replace `<task_name>` with a short snake_case identifier for the task.

## Prompt

> Verbatim (or close paraphrase) of what the user asked for.

## Goal

> What you understand needs to be done. One paragraph.

## Why

> Motivation for the goal. Why does this matter for the project?

## Codebase Context

> Relevant files, modules, functions. Where does the change live?
> - `path/to/file.c` — what it does, why relevant.
> - `path/to/other.c` — same.

## Logical Change

> What will change at the logic / algorithm level. Plain English, no code yet.

## Code Change

> Concrete file-by-file modifications.
> - `path/to/file.c`: add function X, modify Y.
> - `path/to/header.h`: expose Z.

## Why This Change

> Justification. Why this approach over alternatives?

## Logic / Pseudocode

```
function foo():
    step 1
    step 2
    ...
```

## Test Simulation & Tracing

> If user provided test cases, simulate execution here. Trace state step by step.

## Manual Testing Plan

```bash
# Build
make

# Run specific test
./smollm2 ...

# Expected output
...
```

## Status

- [ ] Spec written
- [ ] Implementation
- [ ] Verified
- [ ] Handoff written
