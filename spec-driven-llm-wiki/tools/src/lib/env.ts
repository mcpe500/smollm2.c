import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { EnvConfig } from "./types";

function parseEnvFile(filePath: string): Record<string, string> {
  if (!existsSync(filePath)) return {};
  const out: Record<string, string> = {};
  const raw = readFileSync(filePath, "utf8");
  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const idx = trimmed.indexOf("=");
    if (idx < 0) continue;
    const key = trimmed.slice(0, idx).trim();
    const value = trimmed.slice(idx + 1).trim().replace(/^["']|["']$/g, "");
    out[key] = value;
  }
  return out;
}

function bool(value: string | undefined, fallback: boolean): boolean {
  if (value == null || value === "") return fallback;
  return ["1", "true", "yes", "on"].includes(value.toLowerCase());
}

function community(value: string | undefined): EnvConfig["communityDetection"] {
  if (value === "library" || value === "native" || value === "both") return value;
  return "both";
}

export function getRepoRoot(): string {
  const thisFile = fileURLToPath(import.meta.url);
  return path.resolve(path.dirname(thisFile), "..", "..", "..");
}

export function getToolRoot(): string {
  return path.join(getRepoRoot(), "tools");
}

export function loadEnv(): EnvConfig {
  const repoRoot = getRepoRoot();
  const toolRoot = getToolRoot();
  const fileEnv = parseEnvFile(path.join(repoRoot, ".env"));
  const merged = { ...fileEnv, ...process.env } as Record<string, string | undefined>;
  const projectRoot = path.resolve(repoRoot, merged.PROJECT_ROOT || "..");

  return {
    toolRoot,
    repoRoot,
    projectRoot,
    llmApiKey: merged.LLM_API_KEY || undefined,
    llmModel: merged.LLM_MODEL || "claude-sonnet-4-20250514",
    llmBaseUrl: merged.LLM_BASE_URL || undefined,
    llmModelFast: merged.LLM_MODEL_FAST || "claude-3-5-haiku-latest",
    graphInferEnabled: bool(merged.GRAPH_INFER_ENABLED, false),
    graphCacheEnabled: bool(merged.GRAPH_CACHE_ENABLED, true),
    communityDetection: community(merged.COMMUNITY_DETECTION),
    toolsAdditionalDir: path.resolve(repoRoot, merged.TOOLS_ADDITIONAL_DIR || "tools/additional")
  };
}
