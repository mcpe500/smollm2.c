import { existsSync, readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import type { LintFinding, LintReport, MarkdownPage } from "./types";

export function parseFrontmatter(content: string): { frontmatter: Record<string, string>; body: string } {
  const match = content.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
  if (!match) return { frontmatter: {}, body: content };
  const frontmatter: Record<string, string> = {};
  for (const line of match[1]?.split(/\r?\n/) || []) {
    const idx = line.indexOf(":");
    if (idx < 0) continue;
    const key = line.slice(0, idx).trim();
    const value = line.slice(idx + 1).trim().replace(/^["']|["']$/g, "");
    frontmatter[key] = value;
  }
  return { frontmatter, body: content.slice(match[0].length) };
}

export function readMarkdownPage(filePath: string, root: string): MarkdownPage {
  const content = readFileSync(filePath, "utf8");
  const parsed = parseFrontmatter(content);
  return {
    filePath,
    relativePath: path.relative(root, filePath).replace(/\\/g, "/"),
    content,
    frontmatter: parsed.frontmatter,
    body: parsed.body
  };
}

export function extractSections(content: string): string[] {
  return [...content.matchAll(/^#{2,6}\s+(.+)$/gm)].map((m) => (m[1] || "").trim());
}

export function extractWikilinks(content: string): string[] {
  return [...new Set([...content.matchAll(/\[\[([^\]]+)\]\]/g)].map((m) => (m[1] || "").trim()))].filter(Boolean);
}

export function scanMarkdownFiles(root: string, options: { excludeParts?: string[] } = {}): string[] {
  if (!existsSync(root)) return [];
  const exclude = options.excludeParts || [];
  const out: string[] = [];
  const walk = (dir: string) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      const rel = path.relative(root, full).replace(/\\/g, "/");
      if (exclude.some((part) => rel === part || rel.startsWith(`${part}/`))) continue;
      if (entry.isDirectory()) walk(full);
      else if (entry.isFile() && entry.name.endsWith(".md")) out.push(full);
    }
  };
  walk(root);
  return out.sort();
}

function requiredSectionFound(sections: string[], pattern: RegExp): boolean {
  return sections.some((section) => pattern.test(section));
}

export function validateSpecFile(filePath: string, repoRoot: string): LintReport {
  const errors: LintFinding[] = [];
  const warnings: LintFinding[] = [];
  const info: LintFinding[] = [];
  const hints: LintFinding[] = [];
  const page = readMarkdownPage(filePath, repoRoot);
  const sections = extractSections(page.content);
  const fm = page.frontmatter;

  for (const field of ["title", "status", "type"]) {
    if (!fm[field]) errors.push({ severity: "ERROR", layer: "spec", file: page.relativePath, message: `Missing frontmatter field: ${field}` });
  }

  if (fm.status && !["DRAFT", "IN-PROGRESS", "COMPLETED", "CANCELLED"].includes(fm.status)) {
    errors.push({ severity: "ERROR", layer: "spec", file: page.relativePath, message: `Invalid status: ${fm.status}` });
  }

  const required = [
    [/Task\/Prompt|Prompt/i, "Task/Prompt"],
    [/Tujuan|Goal/i, "Tujuan/Goal"],
    [/Codebase Overview/i, "Codebase Overview"],
    [/Logic Changes|Perubahan/i, "Logic Changes/Perubahan"],
    [/Pseudocode/i, "Pseudocode"],
    [/Test Plan|Manual Testing/i, "Test Plan"]
  ] as const;

  for (const [pattern, label] of required) {
    if (!requiredSectionFound(sections, pattern)) {
      errors.push({ severity: "ERROR", layer: "spec", file: page.relativePath, message: `Missing required section: ${label}` });
    }
  }

  if (!/```[\s\S]*?```/.test(page.content)) {
    warnings.push({ severity: "WARNING", layer: "spec", file: page.relativePath, message: "No fenced code block found for pseudocode or tests" });
  }

  if (!/^\d{3}\./.test(path.basename(filePath))) {
    hints.push({ severity: "HINT", layer: "spec", file: page.relativePath, message: "Spec filename does not start with NNN." });
  }

  return { valid: errors.length === 0, errors, warnings, info, hints };
}

export function mergeReports(reports: LintReport[]): LintReport {
  const errors = reports.flatMap((r) => r.errors);
  const warnings = reports.flatMap((r) => r.warnings);
  const info = reports.flatMap((r) => r.info);
  const hints = reports.flatMap((r) => r.hints);
  return { valid: errors.length === 0, errors, warnings, info, hints };
}
