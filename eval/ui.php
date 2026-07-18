<?php
// ── helpers ──────────────────────────────────────────────────────────────────
function load_records(string $path): array {
    $records = [];
    foreach (file($path, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES) as $line) {
        $r = json_decode($line, true);
        if ($r) $records[] = $r;
    }
    return $records;
}
function record_stats(array $records): array {
    $coherent = 0; $failed = []; $tps_list = [];
    foreach ($records as $r) {
        $s = $r['smollm2'];
        if ($s['coherent']) $coherent++;
        else $failed[] = $r;
        if (($s['tok_per_sec'] ?? 0) > 0) $tps_list[] = $s['tok_per_sec'];
    }
    $total = count($records);
    return [
        'total'    => $total,
        'coherent' => $coherent,
        'failed'   => $failed,
        'avg_tps'  => $tps_list ? round(array_sum($tps_list)/count($tps_list),1) : 0,
        'pct'      => $total > 0 ? round(100*$coherent/$total) : 0,
    ];
}

// ── routing ───────────────────────────────────────────────────────────────────
$results_dir = __DIR__ . '/results';
$files = glob($results_dir . '/*.jsonl');
if (!$files) die("No results found.");
usort($files, fn($a,$b) => filemtime($b) - filemtime($a));

$selected = isset($_GET['f']) ? basename($_GET['f']) : basename($files[0]);
$path = $results_dir . '/' . $selected;
if (!file_exists($path)) { $path = $files[0]; $selected = basename($files[0]); }

$mode = 'dashboard';
if (isset($_GET['detail']))  $mode = 'detail';
elseif (isset($_GET['compare'])) $mode = 'compare';

$records = load_records($path);
$st = record_stats($records);
$total = $st['total']; $coherent = $st['coherent']; $failed = $st['failed'];
$avg_tps = $st['avg_tps']; $pct = $st['pct']; $fail_count = count($failed);
$running = (time() - filemtime($path)) < 90;

// ── detail mode data ──────────────────────────────────────────────────────────
$detail_rec = null; $detail_idx = -1;
if ($mode === 'detail') {
    $want = (int)$_GET['detail'];
    foreach ($records as $idx => $r) {
        if ((int)$r['i'] === $want) { $detail_rec = $r; $detail_idx = $idx; break; }
    }
    if (!$detail_rec) $mode = 'dashboard';
}

// ── compare mode data ─────────────────────────────────────────────────────────
$cmp_records_b = []; $cmp_stats_b = null; $cmp_selected_b = '';
if ($mode === 'compare') {
    $cmp_selected_b = basename($_GET['vs'] ?? '');
    $path_b = $results_dir . '/' . $cmp_selected_b;
    if ($cmp_selected_b && file_exists($path_b)) {
        $cmp_records_b = load_records($path_b);
        $cmp_stats_b   = record_stats($cmp_records_b);
    } else {
        $mode = 'dashboard';
    }
}
?><!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>smollm2.c eval<?php if($mode==='detail'&&$detail_rec): ?> — #<?=(int)$detail_rec['i']?><?php elseif($mode==='compare'): ?> — compare<?php endif; ?></title>
<?php if ($running && $mode==='dashboard'): ?><meta http-equiv="refresh" content="12"><?php endif; ?>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --bg:      #080c10;
  --surface: #0d1117;
  --card:    #111820;
  --border:  #1e2a38;
  --border2: #243040;
  --text:    #cdd9e5;
  --muted:   #768390;
  --blue:    #58a6ff;
  --green:   #3fb950;
  --red:     #f85149;
  --amber:   #e3b341;
  --glow-g:  0 0 18px #3fb95044;
  --glow-r:  0 0 18px #f8514944;
  --glow-b:  0 0 18px #58a6ff33;
  --radius:  10px;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;font-size:14px;line-height:1.5}
a{color:var(--blue);text-decoration:none}
/* layout */
.shell{max-width:1200px;margin:0 auto;padding:28px 20px}
header{display:flex;align-items:center;justify-content:space-between;margin-bottom:28px;gap:12px;flex-wrap:wrap}
.logo{display:flex;align-items:center;gap:10px}
.logo-icon{width:32px;height:32px;border-radius:8px;background:linear-gradient(135deg,#1e3a5f,#0d5c2e);display:flex;align-items:center;justify-content:center;font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:500;color:var(--blue);border:1px solid var(--border2)}
.logo-text{font-size:16px;font-weight:600;color:var(--text);letter-spacing:-.2px}
.logo-sub{font-size:12px;color:var(--muted);font-family:'JetBrains Mono',monospace;margin-top:1px}
.file-select select{background:var(--card);border:1px solid var(--border2);color:var(--text);padding:6px 12px;border-radius:7px;font-size:12px;font-family:'JetBrains Mono',monospace;cursor:pointer;outline:none}
.file-select select:focus{border-color:var(--blue)}
/* status pill */
.status-pill{display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:500}
.pill-running{background:#e3b34115;border:1px solid #e3b34130;color:var(--amber)}
.pill-done{background:#3fb95015;border:1px solid #3fb95030;color:var(--green)}
.pulse{width:7px;height:7px;border-radius:50%;background:var(--amber);animation:pulse 1.4s infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.8)}}
/* metric cards */
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin-bottom:24px}
.card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:18px 20px;position:relative;overflow:hidden;transition:border-color .2s}
.card::before{content:'';position:absolute;inset:0;background:linear-gradient(135deg,rgba(255,255,255,.02) 0%,transparent 60%);pointer-events:none}
.card:hover{border-color:var(--border2)}
.card-label{font-size:11px;font-weight:500;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-bottom:10px}
.card-val{font-size:30px;font-weight:700;letter-spacing:-1px;line-height:1;font-family:'JetBrains Mono',monospace}
.card-sub{font-size:11px;color:var(--muted);margin-top:5px}
.val-blue{color:var(--blue)}
.val-green{color:var(--green);text-shadow:var(--glow-g)}
.val-red{color:var(--red);text-shadow:var(--glow-r)}
.val-amber{color:var(--amber)}
/* breadcrumb */
.breadcrumb{display:flex;align-items:center;gap:8px;margin-bottom:20px;font-size:12px;color:var(--muted)}
.breadcrumb a{color:var(--muted)}
.breadcrumb a:hover{color:var(--blue)}
.breadcrumb-sep{color:var(--border2)}
.breadcrumb-cur{color:var(--text)}
/* detail stats */
.detail-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin-bottom:20px}
.dstat{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px 14px}
.dstat-label{font-size:10px;font-weight:500;color:var(--muted);text-transform:uppercase;letter-spacing:.7px;margin-bottom:6px}
.dstat-val{font-size:20px;font-weight:700;font-family:'JetBrains Mono',monospace;letter-spacing:-.5px}
/* output box */
.output-box{background:#0a1018;border:1px solid var(--border);border-radius:var(--radius);padding:16px 18px;font-family:'JetBrains Mono',monospace;font-size:12.5px;line-height:1.65;color:var(--text);max-height:60vh;overflow-y:auto;white-space:pre-wrap;word-break:break-word;margin-bottom:16px}
.output-box::-webkit-scrollbar{width:6px}
.output-box::-webkit-scrollbar-track{background:transparent}
.output-box::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px}
/* side-by-side */
.side-by-side{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
@media(max-width:700px){.side-by-side{grid-template-columns:1fr}}
.side-label{font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.7px;margin-bottom:8px;display:flex;align-items:center;gap:6px}
/* nav prev/next */
.nav-pn{display:flex;align-items:center;gap:10px;margin-top:16px}
.btn{display:inline-flex;align-items:center;gap:6px;padding:7px 14px;border-radius:7px;font-size:12px;font-weight:500;cursor:pointer;border:1px solid var(--border2);background:var(--card);color:var(--text);text-decoration:none;transition:border-color .15s,background .15s}
.btn:hover{border-color:var(--blue);background:#111d2e}
.btn-ghost{background:transparent;border-color:transparent;color:var(--muted)}
.btn-ghost:hover{background:var(--card);border-color:var(--border);color:var(--text)}
/* compare */
.cmp-form{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:20px;padding:14px 16px;background:var(--card);border:1px solid var(--border);border-radius:var(--radius)}
.cmp-form label{font-size:12px;color:var(--muted);font-weight:500}
.cmp-form select{background:var(--surface);border:1px solid var(--border2);color:var(--text);padding:5px 10px;border-radius:6px;font-size:12px;font-family:'JetBrains Mono',monospace;outline:none}
.cmp-form select:focus{border-color:var(--blue)}
.cmp-form button{padding:6px 14px;border-radius:6px;border:1px solid var(--blue);background:#0d2040;color:var(--blue);font-size:12px;font-weight:500;cursor:pointer}
.cmp-form button:hover{background:#1a3a5c}
.cmp-summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:20px}
.badge-reg{background:#f8514918;border:1px solid #f8514930;color:var(--red)}
.badge-imp{background:#3fb95018;border:1px solid #3fb95030;color:var(--green)}
.badge-unch{background:#58a6ff10;border:1px solid #58a6ff20;color:var(--blue)}
td.cmp-ok{color:var(--green)}
td.cmp-fail{color:var(--red)}
td.cmp-reg{background:#f851491a}
td.cmp-imp{background:#3fb9501a}
/* progress */
.progress-wrap{margin-bottom:24px}
.progress-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.progress-title{font-size:12px;font-weight:500;color:var(--muted);text-transform:uppercase;letter-spacing:.8px}
.progress-pct{font-size:12px;font-family:'JetBrains Mono',monospace;color:var(--text)}
.track{background:var(--card);border:1px solid var(--border);border-radius:20px;height:10px;overflow:hidden}
.fill{height:100%;border-radius:20px;background:linear-gradient(90deg,#1a7f37,#3fb950);box-shadow:0 0 12px #3fb95066;transition:width .6s cubic-bezier(.4,0,.2,1)}
/* sections */
.section{margin-bottom:24px}
.section-header{display:flex;align-items:center;gap:10px;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid var(--border)}
.section-title{font-size:13px;font-weight:600;color:var(--text)}
.badge{display:inline-flex;align-items:center;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600}
.badge-red{background:#f8514918;border:1px solid #f8514930;color:var(--red)}
.badge-green{background:#3fb95018;border:1px solid #3fb95030;color:var(--green)}
/* table */
.table-wrap{border:1px solid var(--border);border-radius:var(--radius);overflow:hidden}
table{width:100%;border-collapse:collapse;font-size:13px}
th{background:#0a1018;color:var(--muted);padding:9px 14px;text-align:left;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.6px;border-bottom:1px solid var(--border)}
th.r{text-align:right}
td{padding:9px 14px;border-bottom:1px solid var(--border);vertical-align:middle;transition:background .1s}
tr:last-child td{border-bottom:none}
tbody tr:hover td{background:#0d1520}
.mono{font-family:'JetBrains Mono',monospace;font-size:12px}
.r{text-align:right}
.prompt-cell{max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--text)}
.snippet-cell{max-width:380px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted);font-size:12px}
.ok-badge{display:inline-flex;align-items:center;gap:4px;padding:2px 9px;border-radius:12px;font-size:11px;font-weight:600;background:#3fb95015;border:1px solid #3fb95028;color:var(--green)}
.fail-badge{display:inline-flex;align-items:center;gap:4px;padding:2px 9px;border-radius:12px;font-size:11px;font-weight:600;background:#f8514915;border:1px solid #f8514928;color:var(--red)}
.dot{width:6px;height:6px;border-radius:50%}
.dot-g{background:var(--green)}
.dot-r{background:var(--red)}
.rep-bar{display:inline-block;height:4px;border-radius:2px;background:var(--border2);vertical-align:middle;position:relative;width:60px}
.rep-fill{position:absolute;left:0;top:0;height:100%;border-radius:2px}
.rep-low{background:var(--green)}
.rep-mid{background:var(--amber)}
.rep-high{background:var(--red)}
.num{color:var(--muted);font-family:'JetBrains Mono',monospace;font-size:12px}
</style>
</head>
<body>
<div class="shell">

<?php
// ════════════════════════════════════════════════════════════════════════════
// MODE: DETAIL
// ════════════════════════════════════════════════════════════════════════════
if ($mode === 'detail' && $detail_rec):
    $s = $detail_rec['smollm2'];
    $has_ollama = isset($detail_rec['ollama']) && is_array($detail_rec['ollama']);
    $o = $has_ollama ? $detail_rec['ollama'] : null;
    // find prev/next by index in $records
    $prev_i = null; $next_i = null;
    if ($detail_idx > 0) $prev_i = $records[$detail_idx-1]['i'];
    if ($detail_idx < count($records)-1) $next_i = $records[$detail_idx+1]['i'];
?>
<header>
  <div class="logo">
    <div class="logo-icon">sm</div>
    <div>
      <div class="logo-text">smollm2.c eval</div>
      <div class="logo-sub">prompt #<?=(int)$detail_rec['i']?> detail</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
    <a class="btn btn-ghost" href="?f=<?=urlencode($selected)?>">← Dashboard</a>
    <?php if($prev_i!==null): ?><a class="btn btn-ghost" href="?f=<?=urlencode($selected)?>&detail=<?=$prev_i?>">← #<?=$prev_i?></a><?php endif; ?>
    <?php if($next_i!==null): ?><a class="btn btn-ghost" href="?f=<?=urlencode($selected)?>&detail=<?=$next_i?>">#<?=$next_i?> →</a><?php endif; ?>
  </div>
</header>

<div class="breadcrumb">
  <a href="?f=<?=urlencode($selected)?>">Dashboard</a>
  <span class="breadcrumb-sep">/</span>
  <span class="breadcrumb-cur">Prompt #<?=(int)$detail_rec['i']?> &nbsp;·&nbsp; <?=htmlspecialchars(mb_substr($detail_rec['prompt'],0,80))?><?=mb_strlen($detail_rec['prompt'])>80?'…':''?></span>
</div>

<div class="section" style="margin-bottom:20px">
  <div class="section-header">
    <span class="section-title">Prompt</span>
    <?php if($s['coherent']): ?><span class="badge badge-green">OK</span><?php else: ?><span class="badge badge-red">FAIL</span><?php endif; ?>
  </div>
  <div class="output-box" style="max-height:140px"><?=htmlspecialchars($detail_rec['prompt'])?></div>
</div>

<div class="detail-stats">
  <div class="dstat"><div class="dstat-label">tok/s</div><div class="dstat-val val-blue"><?=round($s['tok_per_sec']??0,1)?></div></div>
  <div class="dstat"><div class="dstat-label">words</div><div class="dstat-val"><?=intval($s['word_count']??0)?></div></div>
  <div class="dstat"><div class="dstat-label">chars</div><div class="dstat-val"><?=intval($s['char_count']??0)?></div></div>
  <div class="dstat"><div class="dstat-label">printable</div><div class="dstat-val <?=($s['printable_ratio']??1)>=0.95?'':'val-red'?>"><?=round($s['printable_ratio']??0,4)?></div></div>
  <div class="dstat"><div class="dstat-label">repetition</div><div class="dstat-val <?=($s['repetition_4gram']??0)<0.1?'val-green':(($s['repetition_4gram']??0)<0.2?'val-amber':'val-red')?>"><?=$s['repetition_4gram']??0?></div></div>
</div>

<?php if($has_ollama): ?>
<div class="side-by-side">
  <div>
    <div class="side-label"><span class="dot dot-g"></span>smollm2.c</div>
    <div class="output-box"><?=htmlspecialchars($s['text']??'')?></div>
  </div>
  <div>
    <div class="side-label"><span class="dot" style="background:var(--blue)"></span>ollama (reference)</div>
    <div class="output-box"><?=htmlspecialchars($o['text']??'')?></div>
  </div>
</div>
<?php else: ?>
<div class="section">
  <div class="section-header">
    <span class="section-title">Output</span>
    <span class="badge badge-green"><?=intval($s['word_count']??0)?> words</span>
  </div>
  <div class="output-box"><?=htmlspecialchars($s['text']??'')?></div>
</div>
<?php endif; ?>

<div class="nav-pn">
  <?php if($prev_i!==null): ?><a class="btn" href="?f=<?=urlencode($selected)?>&detail=<?=$prev_i?>">← Prev #<?=$prev_i?></a><?php endif; ?>
  <?php if($next_i!==null): ?><a class="btn" href="?f=<?=urlencode($selected)?>&detail=<?=$next_i?>">Next #<?=$next_i?> →</a><?php endif; ?>
</div>

<?php
// ════════════════════════════════════════════════════════════════════════════
// MODE: COMPARE
// ════════════════════════════════════════════════════════════════════════════
elseif ($mode === 'compare'):
    // build lookup by prompt index for B
    $b_by_i = [];
    foreach ($cmp_records_b as $r) $b_by_i[(int)$r['i']] = $r;
    $reg=0; $imp=0; $unch_ok=0; $unch_fail=0; $only_a=0; $only_b=0;
    foreach ($records as $r) {
        $i = (int)$r['i'];
        $ak = $r['smollm2']['coherent'] ?? false;
        if (!isset($b_by_i[$i])) { $only_a++; continue; }
        $bk = $b_by_i[$i]['smollm2']['coherent'] ?? false;
        if ($ak && !$bk) $reg++;
        elseif (!$ak && $bk) $imp++;
        elseif ($ak && $bk) $unch_ok++;
        else $unch_fail++;
    }
    foreach ($cmp_records_b as $r) if (!isset($records[(int)$r['i']])) { /* counted via lookup */ }
    // count only_b: prompts in B not in A
    $a_by_i = [];
    foreach ($records as $r) $a_by_i[(int)$r['i']] = true;
    foreach ($cmp_records_b as $r) if (!isset($a_by_i[(int)$r['i']])) $only_b++;
?>
<header>
  <div class="logo">
    <div class="logo-icon">sm</div>
    <div>
      <div class="logo-text">smollm2.c eval</div>
      <div class="logo-sub">compare runs</div>
    </div>
  </div>
  <a class="btn btn-ghost" href="?f=<?=urlencode($selected)?>">← Dashboard</a>
</header>

<form class="cmp-form" method="get">
  <input type="hidden" name="compare" value="1">
  <label>A:</label>
  <select name="f"><?php foreach(array_reverse($files) as $f): $bn=basename($f); ?>
    <option value="<?=htmlspecialchars($bn)?>" <?=$bn===$selected?'selected':''?>><?=htmlspecialchars($bn)?></option>
  <?php endforeach; ?></select>
  <label>vs B:</label>
  <select name="vs"><?php foreach(array_reverse($files) as $f): $bn=basename($f); ?>
    <option value="<?=htmlspecialchars($bn)?>" <?=$bn===$cmp_selected_b?'selected':''?>><?=htmlspecialchars($bn)?></option>
  <?php endforeach; ?></select>
  <button type="submit">Compare</button>
</form>

<div class="cmp-summary">
  <div class="card"><div class="card-label">Regressions</div><div class="card-val val-red"><?=$reg?></div><div class="card-sub">A ok → B fail</div></div>
  <div class="card"><div class="card-label">Improvements</div><div class="card-val val-green"><?=$imp?></div><div class="card-sub">A fail → B ok</div></div>
  <div class="card"><div class="card-label">Both OK</div><div class="card-val val-green"><?=$unch_ok?></div><div class="card-sub">unchanged</div></div>
  <div class="card"><div class="card-label">Both FAIL</div><div class="card-val val-red"><?=$unch_fail?></div><div class="card-sub">unchanged</div></div>
  <div class="card"><div class="card-label">Only in A</div><div class="card-val val-amber"><?=$only_a?></div><div class="card-sub">missing in B</div></div>
  <div class="card"><div class="card-label">Only in B</div><div class="card-val val-amber"><?=$only_b?></div><div class="card-sub">missing in A</div></div>
</div>

<div class="section">
  <div class="section-header">
    <span class="section-title">Per-prompt diff</span>
    <span class="badge badge-green"><?=count($records)?> prompts</span>
  </div>
  <div class="table-wrap">
  <table>
  <tr><th>#</th><th>Prompt</th><th>A (<?=$selected?>)</th><th>B (<?=$cmp_selected_b?>)</th><th class="r">Δ</th></tr>
  <?php foreach($records as $r):
      $i = (int)$r['i']; $ak = $r['smollm2']['coherent'] ?? false;
      $bk = isset($b_by_i[$i]) ? ($b_by_i[$i]['smollm2']['coherent'] ?? false) : null;
      $rowcls=''; $delta='';
      if ($bk===null) { $delta='<span class=\"num\">—</span>'; }
      elseif ($ak && !$bk) { $rowcls='cmp-reg'; $delta='<span class=\"fail-badge\">REGRESS</span>'; }
      elseif (!$ak && $bk) { $rowcls='cmp-imp'; $delta='<span class=\"ok-badge\">IMPROVE</span>'; }
      else { $delta = ($ak&&$bk)?'<span class="num">ok</span>':'<span class="num">fail</span>'; }
  ?>
  <tr class="<?=$rowcls?>">
    <td class="num"><?=$i?></td>
    <td class="prompt-cell"><a href="?f=<?=urlencode($selected)?>&detail=<?=$i?>"><?=htmlspecialchars(mb_substr($r['prompt'],0,55))?></a></td>
    <td><?php if($ak): ?><span class="ok-badge"><span class="dot dot-g"></span>OK</span><?php else: ?><span class="fail-badge"><span class="dot dot-r"></span>FAIL</span><?php endif; ?></td>
    <td><?php if($bk===null): ?><span class="num">—</span><?php elseif($bk): ?><span class="ok-badge"><span class="dot dot-g"></span>OK</span><?php else: ?><span class="fail-badge"><span class="dot dot-r"></span>FAIL</span><?php endif; ?></td>
    <td class="r"><?=$delta?></td>
  </tr>
  <?php endforeach; ?>
  </table>
  </div>
</div>

<?php
// ════════════════════════════════════════════════════════════════════════════
// MODE: DASHBOARD (default)
// ════════════════════════════════════════════════════════════════════════════
else:
?>
<header>
  <div class="logo">
    <div class="logo-icon">sm</div>
    <div>
      <div class="logo-text">smollm2.c eval</div>
      <div class="logo-sub">inference quality benchmark</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
    <?php if ($running): ?>
      <span class="status-pill pill-running"><span class="pulse"></span>Running</span>
    <?php else: ?>
      <span class="status-pill pill-done">● Complete</span>
    <?php endif; ?>
    <a class="btn btn-ghost" href="?compare=1&f=<?=urlencode($selected)?>&vs=<?=urlencode(basename($files[0]))?>">Compare runs</a>
    <form class="file-select" method="get">
      <select name="f" onchange="this.form.submit()">
        <?php foreach(array_reverse($files) as $f): $bn=basename($f); ?>
        <option value="<?=htmlspecialchars($bn)?>" <?=$bn===$selected?'selected':''?>><?=htmlspecialchars($bn)?></option>
        <?php endforeach; ?>
      </select>
    </form>
  </div>
</header>

<div class="metrics">
  <div class="card">
    <div class="card-label">Prompts run</div>
    <div class="card-val val-blue"><?=$total?></div>
    <div class="card-sub">of 219 total</div>
  </div>
  <div class="card">
    <div class="card-label">Coherent</div>
    <div class="card-val val-green"><?=$coherent?></div>
    <div class="card-sub"><?=$pct?>% pass rate</div>
  </div>
  <div class="card">
    <div class="card-label">Failed</div>
    <div class="card-val <?=$fail_count>0?'val-red':'val-green'?>"><?=$fail_count?></div>
    <div class="card-sub"><?=$fail_count>0?'needs investigation':'all good'?></div>
  </div>
  <div class="card">
    <div class="card-label">Avg tok/s</div>
    <div class="card-val val-blue"><?=$avg_tps?></div>
    <div class="card-sub">greedy, temp=0</div>
  </div>
</div>

<div class="progress-wrap">
  <div class="progress-header">
    <span class="progress-title">Pass rate</span>
    <span class="progress-pct"><?=$coherent?> / <?=$total?> &nbsp;·&nbsp; <?=$pct?>%</span>
  </div>
  <div class="track"><div class="fill" style="width:<?=$pct?>%"></div></div>
</div>

<?php if ($failed): ?>
<div class="section">
  <div class="section-header">
    <span class="section-title">Failures</span>
    <span class="badge badge-red"><?=$fail_count?></span>
  </div>
  <div class="table-wrap">
  <table>
  <tr><th>#</th><th>Prompt</th><th class="r">pr</th><th class="r">rep</th><th>Output</th></tr>
  <?php foreach($failed as $r): $s=$r['smollm2']; ?>
  <tr style="cursor:pointer" onclick="location.href='?f=<?=urlencode($selected)?>&detail=<?=$r['i']?>'">
    <td class="num"><?=$r['i']?></td>
    <td class="prompt-cell"><?=htmlspecialchars(mb_substr($r['prompt'],0,60))?></td>
    <td class="num r"><?=$s['printable_ratio']?></td>
    <td class="num r"><?=$s['repetition_4gram']?></td>
    <td class="snippet-cell"><?=htmlspecialchars(mb_substr(str_replace("\n"," ",$s['text']),0,100))?></td>
  </tr>
  <?php endforeach; ?>
  </table>
  </div>
</div>
<?php endif; ?>

<?php if ($records): ?>
<div class="section">
  <div class="section-header">
    <span class="section-title">All results</span>
    <span class="badge badge-green"><?=$total?> prompts</span>
  </div>
  <div class="table-wrap">
  <table>
  <tr><th>#</th><th>Prompt</th><th>Status</th><th class="r">tok/s</th><th class="r">words</th><th class="r">repetition</th></tr>
  <?php foreach($records as $r): $s=$r['smollm2']; $rep=$s['repetition_4gram']; $repw=min(100,round($rep*200));
    $rpc=$rep<0.1?'rep-low':($rep<0.2?'rep-mid':'rep-high'); ?>
  <tr style="cursor:pointer" onclick="location.href='?f=<?=urlencode($selected)?>&detail=<?=$r['i']?>'">
    <td class="num"><?=$r['i']?></td>
    <td class="prompt-cell"><?=htmlspecialchars(mb_substr($r['prompt'],0,55))?></td>
    <td><?php if($s['coherent']): ?><span class="ok-badge"><span class="dot dot-g"></span>OK</span><?php else: ?><span class="fail-badge"><span class="dot dot-r"></span>FAIL</span><?php endif; ?></td>
    <td class="num r"><?=round($s['tok_per_sec']??0,1)?></td>
    <td class="num r"><?=$s['word_count']?></td>
    <td class="r"><span class="rep-bar"><span class="rep-fill <?=$rpc?>" style="width:<?=$repw?>%"></span></span> <span class="num"><?=$rep?></span></td>
  </tr>
  <?php endforeach; ?>
  </table>
  </div>
</div>
<?php endif; ?>

<?php endif; // end mode switch ?>

</div>
</body>
</html>
