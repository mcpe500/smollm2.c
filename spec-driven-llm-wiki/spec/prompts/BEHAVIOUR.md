# Behavioral Guardrails

Merge these rules with project-specific instructions.

## 1. Think Before Coding

Do not assume. Do not hide confusion. Surface tradeoffs.

- State assumptions explicitly.
- If multiple interpretations exist, present them.
- If a simpler approach exists, say so.
- If something is unclear and risky, stop and ask.

## 2. Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No configurability that was not requested.
- No error handling for impossible scenarios.
- If code can be materially shorter without losing clarity, simplify.

## 3. Surgical Changes

Touch only what the request requires.

- Do not refactor adjacent code unless required.
- Match existing style.
- Remove only unused code created by the change.
- Mention unrelated dead code; do not delete it unless asked.

## 4. Goal-Driven Execution

Define success criteria and verify them.

For multi-step tasks:

```text
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

These rules are working when diffs stay small, decisions are explicit, and verification is concrete.
