#!/usr/bin/env node
/**
 * Validate one spec or all numbered specs.
 *
 * Usage:
 *   npm run validate-spec -- --all
 *   npm run validate-spec -- ../spec/001.example.md
 */
import { existsSync } from "node:fs";
import path from "node:path";
import { loadEnv } from "./lib/env";
import { mergeReports, scanMarkdownFiles, validateSpecFile } from "./lib/spec-parser";
import type { LintFinding, LintReport } from "./lib/types";

function printReport(report: LintReport): void {
  const groups: Array<[string, LintFinding[]]> = [
    ["ERROR", report.errors],
    ["WARNING", report.warnings],
    ["INFO", report.info],
    ["HINT", report.hints]
  ];
  for (const [label, findings] of groups) {
    for (const item of findings) {
      console.log(`${label}: ${item.file || ""} ${item.message}${item.suggestion ? ` (${item.suggestion})` : ""}`);
    }
  }
  console.log(report.valid ? "valid" : "invalid");
}

if (process.argv.includes("--help") || process.argv.includes("-h")) {
  console.log("Usage: validate-spec --all | <spec-path>");
  process.exit(0);
}

const config = loadEnv();
const all = process.argv.includes("--all");
const target = process.argv.find((arg, idx) => idx > 1 && !arg.startsWith("-"));
const specRoot = path.join(config.repoRoot, "spec");
const files = all
  ? scanMarkdownFiles(specRoot, { excludeParts: ["prompts", "handoff", "handover", "docs"] }).filter((file) => /^\d{3}\./.test(path.basename(file)))
  : target
    ? [path.resolve(process.cwd(), target)]
    : [];

if (!files.length) {
  console.error("no spec files selected");
  process.exit(1);
}

const numberCounts = new Map<string, number>();
for (const file of files) {
  const num = path.basename(file).match(/^(\d{3})\./)?.[1];
  if (num) numberCounts.set(num, (numberCounts.get(num) || 0) + 1);
}

const reports = files.map((file) => {
  if (!existsSync(file)) {
    return { valid: false, errors: [{ severity: "ERROR", layer: "spec", file, message: "Spec file not found" }], warnings: [], info: [], hints: [] } satisfies LintReport;
  }
  return validateSpecFile(file, config.repoRoot);
});

const merged = mergeReports(reports);
for (const [num, count] of numberCounts) {
  if (count > 1) merged.errors.push({ severity: "ERROR", layer: "spec", message: `Duplicate spec number: ${num}` });
}
merged.valid = merged.errors.length === 0;
printReport(merged);
process.exit(merged.valid ? 0 : 1);
