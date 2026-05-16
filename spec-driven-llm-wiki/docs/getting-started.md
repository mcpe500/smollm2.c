# Getting Started

## Install

Place `spec-driven-llm-wiki/` inside a project.

```bash
cd spec-driven-llm-wiki/tools
npm install
```

Use `pnpm install` or `bun install` if preferred.

## First Checks

```bash
npm run validate-spec -- --all
npm run build-graph -- --no-infer
```

## View Knowledge Graph

```bash
npm run serve-graph
```

Then open **http://localhost:8080/graph.html** in your browser.

> **Note:** Always use `/graph.html`, not `/graph`.
> Edge/Chrome may block graph visualization when opened via `file://`. Use HTTP server for reliable viewing.

## First Agent Task

```text
@AGENTS.md tasknya adalah: [your task]
```

The agent should research the parent project, create a numbered spec, update wiki memory, and rebuild the graph when useful.
