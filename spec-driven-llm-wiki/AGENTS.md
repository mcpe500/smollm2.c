# Spec-Driven LLM Wiki - Agent Instructions

This directory is a portable spec-driven development memory system. OpenCode, Codex, Kilocode, and compatible agents should treat this file as the canonical operating contract.

## Prompt Layers

1. Read `spec/prompts/BEHAVIOUR.md` for behavioral guardrails.
2. Read `spec/prompts/INSTRUCTIONS.md` for the exact system prompt persona.
3. Apply the spec-driven workflows in this file.

If instructions conflict, workflow procedure in this file wins for artifact maintenance; behavior guardrails still apply to ambiguity, safety, and scope.

## Core Principle

Every non-trivial task becomes durable project memory:

```text
task -> research -> numbered spec -> implementation plan -> wiki updates -> graph updates -> validation -> handoff
```

The repository is self-contained. Specs, wiki pages, graph data, templates, tools, and handoffs live inside `spec-driven-llm-wiki/`.

## Directory Contract

```text
spec/                  numbered specs, handoffs, handovers, prompt source files
wiki/                  modular project memory
graph/                 generated graph JSON and HTML visualizations
tools/                 TypeScript automation
templates/             artifact templates
docs/                  user-facing documentation
.claude/commands/      Claude Code command wrappers
```

## Language Rule

Use the user's prompt language for generated specs and handoffs. If the user mixes languages, preserve that mix when it improves clarity.

## SPEC Workflow

Triggered by: `spec <task>`, `create spec`, or a task description referencing this file.

1. Read this file, `spec/prompts/BEHAVIOUR.md`, and `spec/prompts/INSTRUCTIONS.md`.
2. Detect parent project root from `PROJECT_ROOT` or the parent of this folder.
3. Search and read relevant project files, configs, existing specs, wiki pages, and handoffs.
4. Identify affected components and existing patterns.
5. Allocate the next `NNN.slug.md` number. Never renumber existing specs.
6. Write a spec using `templates/spec-template.md`.
7. Update or create relevant `wiki/components/*.md`, `wiki/decisions/*.md`, or `wiki/patterns/*.md`.
8. Append `wiki/log.md`.
9. Run validation or explain why validation could not run.

Required spec sections:

- `Task/Prompt`
- `Tujuan` or goal section in the user's language
- `Codebase Overview`
- `Logic Changes`
- `Code Changes`
- `Pseudocode`
- `Test Plan`
- `Graph Plan`
- `Review Checklist`

## IMPLEMENT Workflow

Triggered by: `implement <spec-id>`.

1. Read the target spec fully.
2. Read related wiki component pages and graph data if present.
3. Re-read files to be edited.
4. Apply only changes that trace to the spec.
5. Validate with available tests/checks.
6. Update spec status and wiki pages.
7. Rebuild graph when relationships changed.
8. Write handoff if session context changed materially.

## GRAPH Workflow

Triggered by: `build graph`, `rebuild graph`, or `spec-graph`.

Preferred command:

```bash
npm run build-graph -- --no-infer
```

Equivalent package-manager commands may use npm, pnpm, or bun from `tools/`.

Graph build rules:

- Deterministic pass extracts `[[wikilinks]]`.
- LLM inference is optional and off by default.
- Missing LLM config must not fail deterministic graph build.
- Community detection uses library mode when available and native fallback otherwise.
- Component HTML is generated for every component node found in the graph.

## QUERY Workflow

Triggered by: `query: <question>`.

1. Read `wiki/index.md` and `wiki/overview.md`.
2. Read relevant specs and wiki pages.
3. Use graph neighbors when relationship context matters.
4. Answer with `[[wikilink]]` citations where possible.
5. Ask before saving to `wiki/syntheses/`.

## LINT Workflow

Triggered by: `lint`.

Run:

```bash
npm run lint-repo
```

If dependencies are unavailable, perform manual checks:

- Missing required spec sections.
- Broken `[[wikilinks]]`.
- Duplicate spec numbers.
- Graph edges pointing to missing nodes.
- Missing index entries.
- Undocumented tools under `tools/additional/`.

## STATUS Workflow

Triggered by: `status`.

Report:

- Spec counts by status.
- Wiki page counts by type.
- Graph node/edge counts if graph exists.
- Recent `wiki/log.md` entries.
- Open handoff items.

## HANDOFF Workflow

Triggered by: `handoff` or session end.

Write `spec/handoff/NNN.slug.md` using `templates/handoff-template.md`.

Include:

- Session summary.
- Specs created/changed/completed.
- Files modified.
- Graph/wiki updates.
- Validation performed.
- Open questions.
- Next steps.

## Tool Rules

- Core tools live under `tools/src/`.
- User-added tools live under `tools/additional/`.
- Dynamic tools must include a README or top-of-file usage comment.
- `.env` is local and gitignored.
- `.env.example` documents all supported settings.

## Artifact Policy

- Generated graph files may be regenerated.
- Do not hand-edit `graph/graph.json`, `graph/graph.html`, or `graph/components/*.html`.
- Do not commit real secrets.
- Do not auto-renumber existing specs.

## Verification

Preferred checks from `tools/`:

```bash
npm run typecheck
npm run validate-spec -- --all
npm run build-graph -- --no-infer
npm run lint-repo
```

If checks cannot run, state the exact blocker and perform the closest manual validation.
