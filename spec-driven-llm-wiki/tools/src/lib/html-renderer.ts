import type { GraphData, GraphEdge, GraphNode } from "./types";

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

const style = `
body{margin:0;background:#111827;color:#e5e7eb;font-family:Georgia,'Times New Roman',serif;overflow:hidden}
#graph{width:100vw;height:100vh}
#controls{position:fixed;left:14px;top:14px;z-index:5;width:280px;background:rgba(17,24,39,.88);border:1px solid rgba(255,255,255,.12);border-radius:18px;padding:16px;backdrop-filter:blur(12px)}
#controls h1{font-size:18px;margin:0 0 12px;line-height:1.1}
#controls input[type=text],#search{width:100%;box-sizing:border-box;margin:8px 0 12px;padding:8px 10px;border-radius:10px;border:1px solid rgba(255,255,255,.18);background:#030712;color:#f9fafb}
#controls label{display:block;font-size:13px;margin:7px 0;color:#d1d5db}
#controls p{font-size:12px;line-height:1.45;color:#9ca3af}
#stats{position:fixed;right:14px;top:14px;z-index:5;background:rgba(17,24,39,.88);border:1px solid rgba(255,255,255,.12);border-radius:14px;padding:10px 12px;font-size:12px;color:#d1d5db}
#drawer{position:fixed;right:0;top:0;width:min(560px,100vw);height:100vh;z-index:10;display:none;flex-direction:column;background:rgba(3,7,18,.96);border-left:1px solid rgba(255,255,255,.12);box-shadow:-22px 0 50px rgba(0,0,0,.4)}
#drawer.open{display:flex}
#close{align-self:flex-end;margin:14px;background:transparent;color:#f9fafb;border:1px solid rgba(255,255,255,.2);border-radius:999px;padding:7px 12px;cursor:pointer}
#drawer-title{font-size:24px;margin:0 22px 6px}
#drawer-meta{margin:0 22px 14px;color:#9ca3af;font-size:13px}
#drawer-content{white-space:pre-wrap;overflow:auto;margin:0;padding:18px 22px 32px;font:13px/1.6 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;color:#d1d5db}
@media(max-width:800px){#controls{width:calc(100vw - 60px);max-height:34vh;overflow:auto}#stats{top:auto;bottom:14px}}
`;

const script = `
const nodes = new vis.DataSet(GRAPH_DATA.nodes.map(n => ({...n,title:n.preview || n.label})));
const edges = new vis.DataSet(GRAPH_DATA.edges);
const nodeMap = new Map(GRAPH_DATA.nodes.map(n => [n.id,n]));
const container = document.getElementById('graph');
const network = new vis.Network(container,{nodes,edges},{
  nodes:{shape:'dot',font:{color:'#f9fafb',strokeWidth:4,strokeColor:'#111827'},borderWidth:2,scaling:{min:8,max:36}},
  edges:{arrows:{to:{enabled:true,scaleFactor:.35}},smooth:{type:'continuous'},color:{inherit:false}},
  physics:{stabilization:{enabled:true,iterations:220},barnesHut:{gravitationalConstant:-5200,springLength:180,springConstant:.025,damping:.18}},
  interaction:{hover:true,tooltipDelay:120,hideEdgesOnDrag:true,hideEdgesOnZoom:true}
});
const search = document.getElementById('search');
const stats = document.getElementById('stats');
const drawer = document.getElementById('drawer');
const closeBtn = document.getElementById('close');
const checks = {
  EXTRACTED: document.getElementById('show-extracted'),
  INFERRED: document.getElementById('show-inferred'),
  AMBIGUOUS: document.getElementById('show-ambiguous')
};
function updateGraphHealth(){
  const canvas = document.querySelector('#graph canvas');
  window.__GRAPH_HEALTH__ = {
    nodes: GRAPH_DATA.nodes.length,
    edges: GRAPH_DATA.edges.length,
    visibleNodes: nodes.get({filter:n=>n.hidden !== true}).length,
    internalVisibleNodes: network.body.nodeIndices.length,
    canvasCount: document.querySelectorAll('#graph canvas').length,
    canvasWidth: canvas ? canvas.width : 0,
    canvasHeight: canvas ? canvas.height : 0,
    visNetworkSource: Array.from(document.scripts).map(s=>s.src || 'inline').find(src=>src.includes('vis-network')) || ''
  };
}
function esc(s){return String(s||'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function apply(){
  const q = search.value.toLowerCase().trim();
  const allowed = new Set(Object.entries(checks).filter(([,el])=>el.checked).map(([k])=>k));
  nodes.update(GRAPH_DATA.nodes.map(n=>({id:n.id,hidden:q ? !n.label.toLowerCase().includes(q) : false})));
  edges.update(GRAPH_DATA.edges.map(e=>({id:e.id,hidden:!allowed.has(e.type)})));
  stats.textContent = GRAPH_DATA.nodes.length + ' nodes · ' + GRAPH_DATA.edges.length + ' edges · ' + GRAPH_DATA.stats.community_count + ' groups';
  updateGraphHealth();
}
function openNode(id){
  const n = nodeMap.get(id);
  if(!n)return;
  drawer.classList.add('open');
  document.getElementById('drawer-title').textContent = n.label;
  document.getElementById('drawer-meta').textContent = n.type + ' · ' + n.path + (Number.isInteger(n.group)?' · group '+n.group:'');
  document.getElementById('drawer-content').innerHTML = esc(n.markdown || n.preview || '');
}
network.on('click', params => { if(params.nodes.length) openNode(params.nodes[0]); });
closeBtn.onclick = () => drawer.classList.remove('open');
search.oninput = apply;
Object.values(checks).forEach(el => el.onchange = apply);
apply();
network.once('stabilized', () => { network.fit({animation:false}); updateGraphHealth(); });
setTimeout(() => { network.fit({animation:false}); updateGraphHealth(); }, 500);
`;

export function renderGraphHtml(title: string, data: GraphData, visNetworkSrc: string): string {
  const safeTitle = escapeHtml(title);
  const graphJson = JSON.stringify(data).replace(/<\/script>/g, "<\\/script>");
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${safeTitle}</title>
<script src="${visNetworkSrc}"></script>
<style>${style}</style>
</head>
<body>
<div id="controls">
<h1>${safeTitle}</h1>
<input id="search" type="text" placeholder="Search nodes..." aria-label="Search nodes">
<label><input id="show-extracted" type="checkbox" checked> Extracted</label>
<label><input id="show-inferred" type="checkbox" checked> Inferred</label>
<label><input id="show-ambiguous" type="checkbox"> Ambiguous</label>
<p>Click a node to inspect source markdown. Generated locally from specs and wiki pages.</p>
</div>
<div id="stats"></div>
<div id="graph"></div>
<aside id="drawer"><button id="close">Close</button><h2 id="drawer-title"></h2><p id="drawer-meta"></p><pre id="drawer-content"></pre></aside>
<script>const GRAPH_DATA=${graphJson};${script}</script>
</body>
</html>`;
}

export function componentSubgraph(componentId: string, data: GraphData, depth = 2): GraphData {
  const keep = new Set<string>([componentId]);
  for (let i = 0; i < depth; i += 1) {
    for (const edge of data.edges) {
      if (keep.has(edge.from)) keep.add(edge.to);
      if (keep.has(edge.to)) keep.add(edge.from);
    }
  }
  const nodes = data.nodes.filter((node) => keep.has(node.id));
  const nodeIds = new Set(nodes.map((node) => node.id));
  const edges = data.edges.filter((edge) => nodeIds.has(edge.from) && nodeIds.has(edge.to));
  const communities = Object.fromEntries(Object.entries(data.communities).filter(([id]) => nodeIds.has(id)));
  return { ...data, nodes, edges, communities, stats: { ...data.stats, node_count: nodes.length, edge_count: edges.length } };
}
