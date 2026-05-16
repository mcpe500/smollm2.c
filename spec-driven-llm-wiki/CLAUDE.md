# Spec-Driven LLM Wiki - Claude Code

Use `AGENTS.md` as the canonical workflow contract.

Claude-specific slash commands live in `.claude/commands/`:

- `/spec-create <task>` -> SPEC workflow
- `/spec-implement <id>` -> IMPLEMENT workflow
- `/spec-graph` -> GRAPH workflow
- `/spec-lint` -> LINT workflow
- `/spec-status` -> STATUS workflow
- `/spec-handoff` -> HANDOFF workflow

Before non-trivial work:

1. Read `AGENTS.md`.
2. Read `spec/prompts/BEHAVIOUR.md`.
3. Read `spec/prompts/INSTRUCTIONS.md`.
4. Read recent `spec/handoff/` entries.
5. Execute the relevant workflow exactly.
