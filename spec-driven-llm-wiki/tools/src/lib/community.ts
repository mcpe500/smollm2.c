import type { GraphEdge, GraphNode } from "./types";

function neighbors(nodeId: string, edges: GraphEdge[]): Set<string> {
  const out = new Set<string>();
  for (const edge of edges) {
    if (edge.from === nodeId) out.add(edge.to);
    if (edge.to === nodeId) out.add(edge.from);
  }
  return out;
}

function nativeCommunities(nodes: GraphNode[], edges: GraphEdge[]): Map<string, number> {
  const seen = new Set<string>();
  const result = new Map<string, number>();
  let group = 0;
  for (const node of nodes) {
    if (seen.has(node.id)) continue;
    const queue = [node.id];
    seen.add(node.id);
    while (queue.length) {
      const current = queue.shift()!;
      result.set(current, group);
      for (const next of neighbors(current, edges)) {
        if (!seen.has(next)) {
          seen.add(next);
          queue.push(next);
        }
      }
    }
    group += 1;
  }
  return result;
}

async function libraryCommunities(nodes: GraphNode[], edges: GraphEdge[]): Promise<Map<string, number> | null> {
  try {
    const moduleName = "jlouvain";
    const mod: any = await import(moduleName);
    const factory = mod.default || mod.jLouvain || mod;
    if (typeof factory !== "function") return null;
    const algo = factory().nodes(nodes.map((n) => n.id)).edges(edges.map((e) => ({ source: e.from, target: e.to, weight: 1 })));
    const raw = algo();
    const out = new Map<string, number>();
    for (const [id, value] of Object.entries(raw || {})) out.set(id, Number(value));
    return out.size ? out : null;
  } catch {
    return null;
  }
}

export async function detectCommunities(nodes: GraphNode[], edges: GraphEdge[], strategy: "library" | "native" | "both"): Promise<Map<string, number>> {
  if (strategy !== "native") {
    const fromLibrary = await libraryCommunities(nodes, edges);
    if (fromLibrary) return fromLibrary;
    if (strategy === "library") return nativeCommunities(nodes, edges);
  }
  return nativeCommunities(nodes, edges);
}
