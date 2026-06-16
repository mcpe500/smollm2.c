# INSTRUCTIONS.md — Agent Operating Guide for smollm2.c

## Output Format

**IF NORMAL:**
1. **Rationale:** (1 sentence on why the elements were placed there).
2. **The Code.**

**IF "ULTRATHINK" IS ACTIVE:**
1. **Deep Reasoning Chain:** (Detailed breakdown of the architectural and design decisions).
2. **Edge Case Analysis:** (What could go wrong and how we prevented it).
3. **The Code:** (Optimized, bespoke, production-ready, utilizing existing libraries).

## Mindset

You are a high quality and extreme thinker. Always doubt yourself and see and test since you can make a mistake and you need to fix your mistake. Don't give up or submit or complete before it is fixed. If it is still not good enough then fix it, don't stop, just keep fixing until it's fixed.

## Required Reading

Files you must read before starting a task:
- `@/spec/` — all spec documents (the persistent design record)
- `@/spec/handoff/` — all handoff documents (session-to-session continuity)
- `BEHAVIOUR.md` — coding behavior rules

After the entire task is completed you must:
- Write the spec (in `@/spec/`) following the existing numbering `XXX_<task>.md`.
- Write the handoff (in `@/spec/handoff/`) following the existing numbering `NNNN_<session>.md`.

Spec = the design specs; everything about how the code should behave must be written here.
Handoff = session log; after you finish one session you must write what was done, what state the code is in, and what's next.

## Research Protocol

You are the person who will research the entire codebase to gather all important information. The goal is to answer this task:

<TASK>
{TASK}
</TASK>

Steps:
1. Understand what this project does, what it contains, its structure, and the core of how it works.
2. Understand the meaning of the task given.
3. Validate by searching the entire codebase to ground your understanding in actual code — find where changes should happen and what.
4. Research, understand, and think deeply until you know everything about this codebase and the relationship between the task and the code.

## Spec Document Required Sections

After understanding, write a markdown file at `spec/XXX_<task_name>.md` (XXX = 3-digit number from 000 to 999, continuing from the highest existing). Include:

- **Prompt**: what the user asked (verbatim or close paraphrase).
- **Goal**: what you understand needs to be done.
- **Why**: motivation for the goal.
- **Codebase context**: relevant files, modules, functions.
- **Logical change**: what will change at the logic level.
- **Code change**: concrete file/line modifications.
- **Why this change**: justification.
- **Logic / pseudocode**: for review.
- **Test simulation & tracing**: if user provided test cases, simulate and trace.
- **Manual testing plan**: how to verify end-to-end.

Before writing, check `spec/` for related existing files that could speed up research.

Content must be detailed, correct, and validated.

## Requirements

- **NEVER ASSUME, ALWAYS ASK** for any inconsistencies or unclear requirements.
- Use ultrathink, use sequential-thinking.
- Read `BEHAVIOUR.md` for coding rules (simplicity, surgical changes, goal-driven).
