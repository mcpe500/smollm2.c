import { copyFileSync, mkdirSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { detectCommunities } from "./community";
import { extractWikilinks, pageId, pageStem, preview, readKnowledgePages } from "./wiki-reader";
import { componentSubgraph, renderGraphHtml } from "./html-renderer";
import type { EnvConfig, GraphData, GraphEdge, GraphNode, NodeType } from "./types";

const TYPE_COLORS: Record<NodeType, string> = {
  component: "#2196F3",
  spec: "#4CAF50",
  decision: "#9C27B0",
  pattern: "#FF9800",
  concept: "#009688",
  entity: "#E91E63",
  synthesis: "#64748B",
  unknown: "#9E9E9E"
};

const EDGE_COLORS = {
  EXTRACTED: "#94A3B8",
  INFERRED: "#FF5722",
  AMBIGUOUS: "#BDBDBD"
} as const;

function inferType(relativePath: string, frontmatterType?: string): NodeType {
  if (frontmatterType && ["component", "spec", "decision", "pattern", "concept", "entity", "synthesis"].includes(frontmatterType)) {
    return frontmatterType as NodeType;
  }
  if (relativePath.startsWith("spec/")) return "spec";
  if (relativePath.startsWith("wiki/components/")) return "component";
  if (relativePath.startsWith("wiki/decisions/")) return "decision";
  if (relativePath.startsWith("wiki/patterns/")) return "pattern";
  if (relativePath.startsWith("wiki/syntheses/")) return "synthesis";
  return "unknown";
}

function edgeId(from: string, to: string, type: string): string {
  return `${from}->${to}:${type}`;
}

function copyVisNetworkBundle(graphDir: string): void {
  const libDir = path.join(graphDir, "lib");
  mkdirSync(libDir, { recursive: true });
  try {
    const require = createRequire(import.meta.url);
    const src = require.resolve("vis-network/standalone/umd/vis-network.min.js");
    copyFileSync(src, path.join(libDir, "vis-network.min.js"));
  } catch {
    throw new Error(
      "vis-network is not installed. Run npm install in tools/, then rerun npm run build-graph."
    );
  }
}

export async function buildGraph(config: EnvConfig): Promise<GraphData> {
  const pages = readKnowledgePages(config.repoRoot);
  const nodes: GraphNode[] = pages.map((page) => {
    const type = inferType(page.relativePath, page.frontmatter.type);
    return {
      id: pageId(page),
      label: page.frontmatter.title || pageStem(page),
      type,
      status: page.frontmatter.status,
      color: TYPE_COLORS[type],
      path: page.relativePath,
      preview: preview(page.body),
      markdown: page.content,
      value: 1,
      last_updated: page.frontmatter.last_updated
    };
  });

  const byStem = new Map<string, string>();
  const byId = new Map<string, string>();
  for (const node of nodes) {
    byStem.set(path.basename(node.id).toLowerCase(), node.id);
    byId.set(node.id.toLowerCase(), node.id);
  }

  const edges: GraphEdge[] = [];
  const seen = new Set<string>();
  for (const page of pages) {
    const from = pageId(page);
    for (const link of extractWikilinks(page.content)) {
      const normalized = link.replace(/\.md$/, "").toLowerCase();
      const to = byId.get(normalized) || byStem.get(path.basename(normalized));
      if (!to || to === from) continue;
      const id = edgeId(from, to, "EXTRACTED");
      if (seen.has(id)) continue;
      seen.add(id);
      edges.push({ id, from, to, type: "EXTRACTED", color: EDGE_COLORS.EXTRACTED, confidence: 1, title: "explicit wikilink", label: "" });
    }
  }

  const communities = await detectCommunities(nodes, edges, config.communityDetection);
  const degree = new Map<string, number>();
  for (const edge of edges) {
    degree.set(edge.from, (degree.get(edge.from) || 0) + 1);
    degree.set(edge.to, (degree.get(edge.to) || 0) + 1);
  }
  for (const node of nodes) {
    node.group = communities.get(node.id) ?? -1;
    node.value = (degree.get(node.id) || 0) + 1;
  }

  const communitySet = new Set([...communities.values()]);
  const orphanCount = nodes.filter((node) => (degree.get(node.id) || 0) === 0).length;
  return {
    nodes,
    edges,
    communities: Object.fromEntries(communities),
    built: new Date().toISOString().slice(0, 10),
    version: "0.1.0",
    stats: {
      node_count: nodes.length,
      edge_count: edges.length,
      community_count: communitySet.size,
      orphan_count: orphanCount,
      extracted_edge_count: edges.filter((e) => e.type === "EXTRACTED").length,
      inferred_edge_count: edges.filter((e) => e.type === "INFERRED").length,
      ambiguous_edge_count: edges.filter((e) => e.type === "AMBIGUOUS").length
    }
  };
}

export function writeGraphOutputs(config: EnvConfig, graph: GraphData): void {
  const graphDir = path.join(config.repoRoot, "graph");
  const componentDir = path.join(graphDir, "components");
  mkdirSync(componentDir, { recursive: true });
  copyVisNetworkBundle(graphDir);
  writeFileSync(path.join(graphDir, "graph.json"), JSON.stringify(graph, null, 2), "utf8");
  writeFileSync(path.join(graphDir, "graph.html"), renderGraphHtml("Spec-Driven LLM Wiki Graph", graph, "./lib/vis-network.min.js"), "utf8");

  for (const node of graph.nodes.filter((n) => n.type === "component")) {
    const slug = path.basename(node.id).replace(/[^a-z0-9_-]+/gi, "-").toLowerCase();
    const subgraph = componentSubgraph(node.id, graph, 2);
    writeFileSync(path.join(componentDir, `${slug}.html`), renderGraphHtml(`${node.label} - Component Graph`, subgraph, "../lib/vis-network.min.js"), "utf8");
  }
}
