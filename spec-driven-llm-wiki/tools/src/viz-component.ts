#!/usr/bin/env node
/**
 * Generate one component subgraph from graph/graph.json.
 *
 * Usage:
 *   npm run viz-component -- wiki/components/auth-service
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { loadEnv } from "./lib/env";
import { componentSubgraph, renderGraphHtml } from "./lib/html-renderer";
import type { GraphData } from "./lib/types";

if (process.argv.includes("--help") || process.argv.includes("-h")) {
  console.log("Usage: viz-component <component-id>");
  process.exit(0);
}

const componentId = process.argv[2];
if (!componentId) {
  console.error("missing component id");
  process.exit(1);
}

const config = loadEnv();
const graphPath = path.join(config.repoRoot, "graph", "graph.json");
if (!existsSync(graphPath)) {
  console.error("graph/graph.json not found. Run build-graph first.");
  process.exit(1);
}

const graph = JSON.parse(readFileSync(graphPath, "utf8")) as GraphData;
const node = graph.nodes.find((n) => n.id === componentId || n.id.endsWith(componentId));
if (!node) {
  console.error(`component not found: ${componentId}`);
  process.exit(1);
}

const outputDir = path.join(config.repoRoot, "graph", "components");
mkdirSync(outputDir, { recursive: true });
const slug = path.basename(node.id).replace(/[^a-z0-9_-]+/gi, "-").toLowerCase();
const subgraph = componentSubgraph(node.id, graph, 2);
writeFileSync(path.join(outputDir, `${slug}.html`), renderGraphHtml(`${node.label} - Component Graph`, subgraph, "../lib/vis-network.min.js"), "utf8");
console.log(`component graph written: graph/components/${slug}.html`);
