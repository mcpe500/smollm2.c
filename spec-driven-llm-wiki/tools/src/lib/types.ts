export type NodeType =
  | "component"
  | "spec"
  | "decision"
  | "pattern"
  | "concept"
  | "entity"
  | "synthesis"
  | "unknown";

export type EdgeType =
  | "EXTRACTED"
  | "INFERRED"
  | "AMBIGUOUS"
  | "IMPLEMENTS"
  | "AFFECTS"
  | "DEPENDS_ON"
  | "USES_PATTERN"
  | "DECIDED_BY";

export interface GraphNode {
  id: string;
  label: string;
  type: NodeType;
  status?: string;
  group?: number;
  color: string;
  path: string;
  preview: string;
  markdown: string;
  value: number;
  last_updated?: string;
}

export interface GraphEdge {
  id: string;
  from: string;
  to: string;
  type: EdgeType;
  color: string;
  confidence: number;
  title?: string;
  label?: string;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  communities: Record<string, number>;
  built: string;
  version: string;
  stats: GraphStats;
}

export interface GraphStats {
  node_count: number;
  edge_count: number;
  community_count: number;
  orphan_count: number;
  extracted_edge_count: number;
  inferred_edge_count: number;
  ambiguous_edge_count: number;
}

export interface EnvConfig {
  toolRoot: string;
  repoRoot: string;
  projectRoot: string;
  llmApiKey?: string;
  llmModel: string;
  llmBaseUrl?: string;
  llmModelFast: string;
  graphInferEnabled: boolean;
  graphCacheEnabled: boolean;
  communityDetection: "library" | "native" | "both";
  toolsAdditionalDir: string;
}

export interface LintFinding {
  severity: "ERROR" | "WARNING" | "INFO" | "HINT";
  layer: "spec" | "wiki" | "graph" | "tool" | "config";
  file?: string;
  message: string;
  suggestion?: string;
}

export interface LintReport {
  valid: boolean;
  errors: LintFinding[];
  warnings: LintFinding[];
  info: LintFinding[];
  hints: LintFinding[];
}

export interface MarkdownPage {
  filePath: string;
  relativePath: string;
  content: string;
  frontmatter: Record<string, string>;
  body: string;
}
