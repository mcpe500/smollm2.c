#!/usr/bin/env node
/**
 * Build graph/graph.json, graph/graph.html, and component graph HTML files.
 *
 * Usage:
 *   npm run build-graph
 *   npm run build-graph -- --no-infer
 */
import { loadEnv } from "./lib/env";
import { buildGraph, writeGraphOutputs } from "./lib/graph-builder";
import { appendWikiLog } from "./lib/wiki-reader";

function hasHelp(): boolean {
  return process.argv.includes("--help") || process.argv.includes("-h");
}

if (hasHelp()) {
  console.log("Usage: build-graph [--no-infer]\nBuild deterministic graph artifacts from spec/ and wiki/.");
  process.exit(0);
}

const config = loadEnv();
if (process.argv.includes("--no-infer")) config.graphInferEnabled = false;

const graph = await buildGraph(config);
writeGraphOutputs(config, graph);
appendWikiLog(config.repoRoot, "graph", "Knowledge graph rebuilt", `${graph.nodes.length} nodes, ${graph.edges.length} edges.`);
console.log(`graph built: ${graph.nodes.length} nodes, ${graph.edges.length} edges`);
