# Spec-Driven LLM Wiki

A portable coding-agent methodology for spec-driven development, graph memory, and session handoff.

Drop this folder into any project, tag `AGENTS.md`, describe a task, and the agent maintains:

- numbered specs in `spec/`
- project memory in `wiki/`
- interactive graphs in `graph/`
- reusable tools in `tools/`
- session continuity in `spec/handoff/`

## Quick Start

```bash
cd spec-driven-llm-wiki/tools
npm install
npm run validate-spec -- --all
npm run build-graph -- --no-infer
```

Equivalent package managers:

```bash
pnpm install
pnpm run build-graph -- --no-infer

bun install
bun run build-graph -- --no-infer
```

## Usage

Natural language works:

```text
@AGENTS.md tasknya adalah: buat auth JWT dan tulis spec dulu
```

Shorthand triggers:

```text
spec <task>
implement <spec-id>
build graph
lint
status
handoff
query: <question>
```

## Configuration

Copy `.env.example` to `.env`. Core deterministic tools work without LLM keys. LLM inference for semantic graph edges is optional and off by default.

## Current Scope

MVP includes deterministic spec validation and graph generation. Advanced semantic inference and query intelligence are optional follow-up work.
