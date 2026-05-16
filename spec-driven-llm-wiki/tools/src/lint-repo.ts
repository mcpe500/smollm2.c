#!/usr/bin/env node
/**
 * Lint specs, wiki links, graph consistency, and additional tool docs.
 */
import { existsSync, readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { loadEnv } from "./lib/env";
import { extractWikilinks, scanMarkdownFiles, validateSpecFile } from "./lib/spec-parser";
import { readKnowledgePages } from "./lib/wiki-reader";
import type { GraphData, LintFinding } from "./lib/types";

const config = loadEnv();
const findings: LintFinding[] = [];

for (const file of scanMarkdownFiles(path.join(config.repoRoot, "spec"), { excludeParts: ["prompts", "handoff", "handover", "docs"] }).filter((f) => /^\d{3}\./.test(path.basename(f)))) {
  const report = validateSpecFile(file, config.repoRoot);
  findings.push(...report.errors, ...report.warnings, ...report.info, ...report.hints);
}

const pages = readKnowledgePages(config.repoRoot);
const ids = new Set(pages.map((page) => page.relativePath.replace(/\.md$/, "")));
const stems = new Set(pages.map((page) => path.basename(page.relativePath, ".md").toLowerCase()));
for (const page of pages) {
  for (const link of extractWikilinks(page.content)) {
    const normalized = link.replace(/\.md$/, "").toLowerCase();
    if (!ids.has(normalized) && !stems.has(path.basename(normalized))) {
      findings.push({ severity: "WARNING", layer: "wiki", file: page.relativePath, message: `Broken wikilink: [[${link}]]` });
    }
  }
}

const graphPath = path.join(config.repoRoot, "graph", "graph.json");
if (existsSync(graphPath)) {
  const graph = JSON.parse(readFileSync(graphPath, "utf8")) as GraphData;
  const nodeIds = new Set(graph.nodes.map((node) => node.id));
  for (const edge of graph.edges) {
    if (!nodeIds.has(edge.from) || !nodeIds.has(edge.to)) {
      findings.push({ severity: "ERROR", layer: "graph", message: `Dangling edge: ${edge.id}` });
    }
  }
} else {
  findings.push({ severity: "INFO", layer: "graph", message: "graph/graph.json does not exist yet" });
}

if (existsSync(config.toolsAdditionalDir)) {
  for (const entry of readdirSync(config.toolsAdditionalDir, { withFileTypes: true })) {
    if (entry.isFile() && /\.(ts|js)$/.test(entry.name)) {
      const tool = path.join(config.toolsAdditionalDir, entry.name);
      const readme = path.join(config.toolsAdditionalDir, `${path.basename(entry.name, path.extname(entry.name))}.md`);
      const content = readFileSync(tool, "utf8");
      if (!existsSync(readme) && !content.trimStart().startsWith("/**")) {
        findings.push({ severity: "WARNING", layer: "tool", file: path.relative(config.repoRoot, tool), message: "Additional tool lacks README or docstring" });
      }
    }
  }
}

for (const item of findings) {
  console.log(`${item.severity}: ${item.layer}${item.file ? ` ${item.file}` : ""} - ${item.message}`);
}
const hasErrors = findings.some((item) => item.severity === "ERROR");
console.log(hasErrors ? "lint failed" : "lint passed");
process.exit(hasErrors ? 1 : 0);
