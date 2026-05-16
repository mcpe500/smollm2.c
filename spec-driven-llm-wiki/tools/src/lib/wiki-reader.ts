import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { extractWikilinks, readMarkdownPage, scanMarkdownFiles } from "./spec-parser";
import type { MarkdownPage } from "./types";

export function readKnowledgePages(repoRoot: string): MarkdownPage[] {
  const files = [
    path.join(repoRoot, "full-spec.md"),
    ...scanMarkdownFiles(path.join(repoRoot, "wiki")),
    ...scanMarkdownFiles(path.join(repoRoot, "spec"), { excludeParts: ["prompts", "handoff", "handover"] })
  ].filter((file, idx, arr) => existsSync(file) && arr.indexOf(file) === idx);

  return files.map((file) => readMarkdownPage(file, repoRoot));
}

export function pageId(page: MarkdownPage): string {
  return page.relativePath.replace(/\.md$/, "");
}

export function pageStem(page: MarkdownPage): string {
  return path.basename(page.relativePath, ".md");
}

export function preview(body: string): string {
  return body
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .slice(0, 3)
    .join(" ")
    .slice(0, 240);
}

export { extractWikilinks };

export function appendWikiLog(repoRoot: string, operation: string, title: string, details = ""): void {
  const logPath = path.join(repoRoot, "wiki", "log.md");
  mkdirSync(path.dirname(logPath), { recursive: true });
  if (!existsSync(logPath)) writeFileSync(logPath, "# Wiki Log\n\n", "utf8");
  const date = new Date().toISOString().slice(0, 10);
  const body = `\n## [${date}] ${operation} | ${title}\n\n${details.trim()}\n`;
  appendFileSync(logPath, body, "utf8");
}

export function readTextIfExists(filePath: string): string {
  return existsSync(filePath) ? readFileSync(filePath, "utf8") : "";
}
