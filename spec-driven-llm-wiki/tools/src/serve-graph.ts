#!/usr/bin/env node
import { createReadStream, existsSync, readFileSync, statSync } from "node:fs";
import { createServer, type ServerResponse } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..", "..");
const graphDir = path.join(repoRoot, "graph");
const defaultPort = Number(process.env.GRAPH_PORT || 8080);
const maxPort = defaultPort + 10;

const contentTypes: Record<string, string> = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".txt": "text/plain; charset=utf-8"
};

function send(res: ServerResponse, status: number, body: string): void {
  res.writeHead(status, { "content-type": "text/plain; charset=utf-8" });
  res.end(body);
}

function safeFilePath(urlPath: string): string | undefined {
  const pathOnly = decodeURIComponent(urlPath.split("?")[0] || "/");
  const relative = pathOnly === "/" || pathOnly === "/graph" ? "graph.html" : pathOnly.replace(/^\/+/, "");
  const resolved = path.resolve(graphDir, relative);
  const graphRoot = path.resolve(graphDir);
  if (resolved !== graphRoot && !resolved.startsWith(graphRoot + path.sep)) return undefined;
  return resolved;
}

function readGraphStats(): string {
  const graphJson = path.join(graphDir, "graph.json");
  if (!existsSync(graphJson)) return "graph.json missing";
  try {
    const graph = JSON.parse(readFileSync(graphJson, "utf8")) as {
      stats?: { node_count?: number; edge_count?: number; community_count?: number };
      built?: string;
    };
    const stats = graph.stats || {};
    return `${stats.node_count ?? 0} nodes, ${stats.edge_count ?? 0} edges, ${stats.community_count ?? 0} groups, built ${graph.built ?? "unknown"}`;
  } catch (error) {
    return `graph.json unreadable: ${error instanceof Error ? error.message : String(error)}`;
  }
}

function createGraphServer() {
  return createServer((req, res) => {
    if (!req.url) return send(res, 400, "Bad request");
    const urlPath = req.url.split("?")[0] || "/";
    if (urlPath === "/favicon.ico") {
      res.writeHead(204, { "cache-control": "no-store" });
      res.end();
      return;
    }

    const filePath = safeFilePath(req.url);
    if (!filePath) return send(res, 403, "Forbidden");

    if (urlPath === "/" || urlPath === "/graph") {
      res.writeHead(302, { location: "/graph.html" });
      res.end();
      return;
    }

    if (!existsSync(filePath) || !statSync(filePath).isFile()) {
      send(res, 404, `Not found: ${path.relative(graphDir, filePath)}`);
      return;
    }

    const ext = path.extname(filePath).toLowerCase();
    res.writeHead(200, {
      "cache-control": "no-store",
      "content-type": contentTypes[ext] || "application/octet-stream"
    });
    createReadStream(filePath).pipe(res);
  });
}

async function listenOnAvailablePort(port: number): Promise<number> {
  if (port > maxPort) {
    throw new Error(`No available port found from ${defaultPort} to ${maxPort}.`);
  }

  const server = createGraphServer();
  return new Promise((resolve, reject) => {
    server.once("error", (error: NodeJS.ErrnoException) => {
      if (error.code === "EADDRINUSE") {
        console.warn(`Port ${port} is busy; trying ${port + 1}.`);
        listenOnAvailablePort(port + 1).then(resolve, reject);
        return;
      }
      reject(error);
    });

    server.listen(port, () => {
      resolve(port);
    });
  });
}

if (!existsSync(path.join(graphDir, "graph.html"))) {
  console.error(`Missing graph.html in ${graphDir}`);
  console.error("Run: npm run build-graph -- --no-infer");
  process.exit(1);
}

const port = await listenOnAvailablePort(defaultPort);

console.log(`Serving graph directory: ${graphDir}`);
console.log(`Graph data: ${readGraphStats()}`);
console.log(`Open: http://localhost:${port}/graph.html`);
if (port !== defaultPort) {
  console.warn(`Port ${defaultPort} is already in use. Use the printed URL above, not the stale server.`);
}
