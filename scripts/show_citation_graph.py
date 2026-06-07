# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Citation network visualizer for zotero-local-rag.
Uses sigma.js (WebGL) + graphology for scalable rendering of 1000–10000+ nodes.

Usage:
  uv run scripts/show_citation_graph.py               # デフォルト（上位20件）
  uv run scripts/show_citation_graph.py --top 100     # 上位100件
  uv run scripts/show_citation_graph.py --item KEY    # 1アイテムに絞る
  uv run scripts/show_citation_graph.py --no-refs     # 参照先を非表示
  uv run scripts/show_citation_graph.py --no-open     # ブラウザを自動で開かない

Node colors:
  Blue   = Zotero アイテム（自分のライブラリ）
  Amber  = 外部論文（被引用: 自分の文献を引用している論文）
  Green  = 外部論文（参照先: 自分の文献が引用している論文）

Edge colors:
  Red   = 外部論文 → Zotero アイテム（被引用）
  Green = Zotero アイテム → 参照先論文（参照）
"""
import argparse
import math
import os
import sqlite3
import sys
import webbrowser
from pathlib import Path

PROJECT_ROOT   = Path(__file__).resolve().parents[1]
DB_PATH        = os.environ.get("RELATIONS_DB_PATH", str(PROJECT_ROOT / "data" / "relations.db"))
CHROMA_DB      = os.environ.get("CHROMA_DIR", str(PROJECT_ROOT / "data" / "chroma")) + "/chroma.sqlite3"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "citation_graph.html"


# ── ChromaDB title / author lookup ───────────────────────────────────────────

def get_item_meta(item_keys: list[str]) -> dict[str, dict]:
    """ChromaDB SQLite から itemKey → {title, creators} のマップを返す。"""
    if not os.path.exists(CHROMA_DB):
        return {}
    try:
        conn = sqlite3.connect(CHROMA_DB, timeout=5)
        conn.row_factory = sqlite3.Row
        placeholders = ",".join("?" * len(item_keys))
        rows = conn.execute(f"""
            SELECT
                ikey.string_value AS item_key,
                MAX(CASE WHEN em.key = 'title'    THEN em.string_value END) AS title,
                MAX(CASE WHEN em.key = 'creators' THEN em.string_value END) AS creators
            FROM embedding_metadata ikey
            JOIN embedding_metadata em ON em.id = ikey.id
            WHERE ikey.key = 'itemKey'
              AND ikey.string_value IN ({placeholders})
            GROUP BY ikey.string_value
        """, item_keys).fetchall()
        conn.close()
        return {
            r["item_key"]: {"title": r["title"] or "", "creators": r["creators"] or ""}
            for r in rows
        }
    except Exception as e:
        print(f"  (ChromaDB metadata lookup failed: {e})", file=sys.stderr)
        return {}


# ── DB helpers ───────────────────────────────────────────────────────────────

def get_top_items(conn: sqlite3.Connection, limit: int) -> list[dict]:
    """被引用数の多い順に Zotero アイテムを返す。"""
    rows = conn.execute("""
        SELECT
            gc.cited_item_key                       AS item_key,
            COUNT(DISTINCT gc.citing_paper_id)      AS citer_count,
            COUNT(*)                                AS context_count,
            ics.s2_status
        FROM global_citations gc
        LEFT JOIN item_citation_status ics ON ics.item_key = gc.cited_item_key
        GROUP BY gc.cited_item_key
        ORDER BY citer_count DESC
        LIMIT ?
    """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_item_row(conn: sqlite3.Connection, item_key: str) -> dict | None:
    """1アイテムの統計を返す。"""
    row = conn.execute("""
        SELECT
            gc.cited_item_key                       AS item_key,
            COUNT(DISTINCT gc.citing_paper_id)      AS citer_count,
            COUNT(*)                                AS context_count,
            ics.s2_status
        FROM global_citations gc
        LEFT JOIN item_citation_status ics ON ics.item_key = gc.cited_item_key
        WHERE gc.cited_item_key = ?
        GROUP BY gc.cited_item_key
    """, (item_key,)).fetchone()
    return dict(row) if row else None


def get_citers(conn: sqlite3.Connection, item_keys: list[str], per_item: int) -> list[dict]:
    """各アイテムについて、被引用数の多い外部論文を返す（per_item 件まで）。"""
    placeholders = ",".join("?" * len(item_keys))
    rows = conn.execute(f"""
        SELECT
            cited_item_key,
            citing_paper_id,
            citing_title,
            citing_year,
            citing_citation_count,
            COUNT(*) AS context_count
        FROM global_citations
        WHERE cited_item_key IN ({placeholders})
        GROUP BY cited_item_key, citing_paper_id
        ORDER BY cited_item_key, citing_citation_count DESC, context_count DESC
    """, item_keys).fetchall()

    seen: dict[str, int] = {}
    result = []
    for row in rows:
        key = row[0]
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= per_item:
            result.append(dict(row))
    return result


def get_refs(conn: sqlite3.Connection, item_keys: list[str], per_item: int) -> list[dict]:
    """各アイテムの参照先論文を返す（per_item 件まで）。"""
    placeholders = ",".join("?" * len(item_keys))
    rows = conn.execute(f"""
        SELECT
            citing_item_key,
            cited_paper_id,
            cited_title,
            cited_year,
            cited_citation_count,
            COUNT(*) AS context_count
        FROM global_references
        WHERE citing_item_key IN ({placeholders})
          AND cited_paper_id IS NOT NULL
        GROUP BY citing_item_key, cited_paper_id
        ORDER BY citing_item_key, cited_citation_count DESC, context_count DESC
    """, item_keys).fetchall()

    seen: dict[str, int] = {}
    result = []
    for row in rows:
        key = row[0]
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= per_item:
            result.append(dict(row))
    return result


def get_chunk_expand_data(
    conn: sqlite3.Connection,
    item_keys: list[str],
    rendered_papers: dict[str, str],
) -> dict:
    """ダブルクリック展開用のチャンクレベル引用データを返す。

    Returns:
        { item_key: { "in_edges":  [[paper_node_id, chunk_id, width], ...],
                      "out_edges": [[chunk_id, paper_node_id, width], ...] } }
    """
    if not item_keys or not rendered_papers:
        return {}
    placeholders = ",".join("?" * len(item_keys))

    citer_rows = conn.execute(f"""
        SELECT cited_item_key, cited_chunk_id, citing_paper_id, COUNT(*) AS ctx
        FROM global_citations
        WHERE cited_item_key IN ({placeholders})
        GROUP BY cited_item_key, cited_chunk_id, citing_paper_id
    """, item_keys).fetchall()

    ref_rows = conn.execute(f"""
        SELECT citing_item_key, citing_chunk_id, cited_paper_id, COUNT(*) AS ctx
        FROM global_references
        WHERE citing_item_key IN ({placeholders})
          AND cited_paper_id IS NOT NULL
        GROUP BY citing_item_key, citing_chunk_id, cited_paper_id
    """, item_keys).fetchall()

    result: dict = {k: {"in_edges": [], "out_edges": []} for k in item_keys}

    for r in citer_rows:
        nid = rendered_papers.get(r["citing_paper_id"])
        if nid:
            w = round(max(0.8, min(4.0, r["ctx"] * 0.5)), 2)
            result[r["cited_item_key"]]["in_edges"].append([nid, r["cited_chunk_id"], w])

    for r in ref_rows:
        nid = rendered_papers.get(r["cited_paper_id"])
        if nid:
            w = round(max(0.8, min(3.0, r["ctx"] * 0.4)), 2)
            result[r["citing_item_key"]]["out_edges"].append([r["citing_chunk_id"], nid, w])

    return result


# ── node / tooltip helpers ────────────────────────────────────────────────────

_SIZE_MIN = 5
_SIZE_MAX = 28

def _node_size(count: int | None) -> float:
    """被引用数をノードサイズに変換する（sigma の pixel 単位）。
    log10 スケール：count=10→8, count=100→11, count=1000→14.5, count=10000→18, count=100000→21.5
    """
    if not count or count <= 0:
        return _SIZE_MIN
    return min(_SIZE_MAX, _SIZE_MIN + (_SIZE_MAX - _SIZE_MIN) * math.log10(count + 1) / 6.0)


def _short(title: str | None, max_len: int = 35) -> str:
    if not title:
        return "(no title)"
    return title[:max_len] + "…" if len(title) > max_len else title


def _fmt_creators(raw: str, max_len: int = 55) -> str:
    """ChromaDB の creators フィールドを短い著者リスト文字列に変換する。"""
    if not raw:
        return ""
    try:
        import json
        data = json.loads(raw)
        names: list[str] = []
        for c in data:
            last  = c.get("lastName", "")
            first = c.get("firstName", "")
            name  = c.get("name", "")
            if last and first:
                names.append(f"{last}, {first[0]}.")
            elif last:
                names.append(last)
            elif name:
                names.append(name)
        result = "; ".join(names)
    except (ValueError, TypeError, AttributeError):
        result = raw

    if len(result) > max_len:
        result = result[:max_len].rsplit(";", 1)[0] + "; …"
    return result


def _tooltip_wrap(text: str, width: int = 38) -> str:
    """単語境界でテキストを折り返す。"""
    if len(text) <= width:
        return text
    words = text.split()
    lines: list[str] = []
    line: list[str] = []
    cur = 0
    for w in words:
        need = len(w) + (1 if line else 0)
        if cur + need > width and line:
            lines.append(" ".join(line))
            line = [w]
            cur = len(w)
        else:
            line.append(w)
            cur += need
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)


def _tooltip(title: str, extra: list[tuple[str, str]]) -> str:
    """ホバー時ツールチップをプレーンテキストで生成する。"""
    wrapped = _tooltip_wrap(title, 38)
    sep = "─" * 22
    lines = [wrapped, sep]
    for k, v in extra:
        lines.append(f"{k}: {_tooltip_wrap(v, 36 - len(k) - 2)}")
    return "\n".join(lines)


# ── sigma.js HTML builder ─────────────────────────────────────────────────────

def _build_sigma_html(
    graph_json: str,
    expand_json: str,
    n_items: int,
    n_nodes: int,
    n_edges: int,
    n_citer: int,
    n_ref: int,
) -> str:
    """sigma.js (WebGL) ベースの完全な HTML を生成する。
    f-string 衝突を避けるため CSS / JS 部分は通常文字列、変数挿入部分のみ f-string。
    """

    # ── CSS (plain string – no Python vars) ──────────────────────────────────
    css = """<style>
  *, *::before, *::after { box-sizing: border-box; }
  body {
    margin: 0; overflow: hidden;
    font-family: 'Inter', system-ui, sans-serif;
    background: #080f1e;
  }
  #sigma-container {
    position: fixed; inset: 0;
    background: radial-gradient(ellipse at 35% 45%, #0f1e3d 0%, #080f1e 70%);
  }

  /* ── Layout progress badge (shown while D3 runs) ── */
  #layout-badge {
    display: none;
    position: fixed; bottom: 16px; right: 16px; z-index: 1000;
    background: rgba(10,16,32,0.85);
    backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 8px; padding: 7px 14px;
    font-size: 11.5px; color: #93c5fd;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    cursor: default;
  }
  #layout-badge span { color: #3b82f6; font-weight: 600; margin-left: 4px; }

  /* ── Legend card ── */
  #rag-legend {
    position: fixed; top: 16px; left: 16px; z-index: 1000;
    background: rgba(10,16,32,0.88);
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 18px 20px 14px;
    color: #e2e8f0; font-size: 13px; min-width: 210px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.06);
  }
  #rag-legend h3 { margin: 0 0 14px; font-size: 15px; font-weight: 600; color: #f8fafc; }
  .rl-row { display: flex; align-items: center; gap: 9px; margin-bottom: 7px; }
  .rl-dot  { width: 11px; height: 11px; border-radius: 50%; flex-shrink: 0; box-shadow: 0 0 6px currentColor; }
  .rl-label { font-size: 12.5px; color: #cbd5e1; }
  .rl-count { margin-left: auto; font-size: 11px; color: #64748b; font-variant-numeric: tabular-nums; }
  .rl-edge-row { display: flex; align-items: center; gap: 9px; margin-bottom: 6px; }
  .rl-line  { width: 22px; height: 2px; border-radius: 1px; flex-shrink: 0; }
  .rl-dashed { background: repeating-linear-gradient(to right,#34d399 0,#34d399 4px,transparent 4px,transparent 7px); height: 2px; }
  .rl-divider { border: none; border-top: 1px solid rgba(255,255,255,0.07); margin: 12px 0; }
  .rl-stats { font-size: 11.5px; color: #64748b; line-height: 1.9; }
  .rl-stats span { color: #94a3b8; }
  .rl-size-scale { display: flex; align-items: center; gap: 5px; margin: 4px 0 10px; }
  .rl-size-scale .dot { background: #475569; border-radius: 50%; flex-shrink: 0; }
  .rl-size-scale .sl { font-size: 10px; color: #475569; }
  .rl-hint { font-size: 10.5px; color: #334155; line-height: 1.7; margin-top: 2px; }

  /* ── Title badge ── */
  #rag-title {
    position: fixed; top: 16px; right: 16px; z-index: 1000;
    background: rgba(10,16,32,0.75);
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; padding: 10px 16px;
    color: #94a3b8; font-size: 11px; text-align: right;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
  }
  #rag-title strong { display: block; font-size: 14px; color: #e2e8f0; font-weight: 600; margin-bottom: 2px; }

  /* ── Tooltip ── */
  #rag-tooltip {
    position: fixed; display: none; z-index: 2000;
    background: rgba(10,16,32,0.95);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px; padding: 10px 14px;
    font-family: 'Inter', monospace; font-size: 12.5px; line-height: 1.65;
    color: #e2e8f0; max-width: 360px;
    white-space: pre-wrap; word-break: break-word;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    pointer-events: none;
  }
</style>"""

    # ── Legend HTML (f-string – uses Python vars) ─────────────────────────────
    legend = f"""<div id="rag-legend">
  <h3>Citation Network</h3>
  <div class="rl-row">
    <span class="rl-dot" style="background:#2563eb;color:#2563eb"></span>
    <span class="rl-label">Zotero アイテム</span>
    <span class="rl-count">{n_items}</span>
  </div>
  <div class="rl-row">
    <span class="rl-dot" style="background:#d97706;color:#d97706"></span>
    <span class="rl-label">被引用元（外部論文）</span>
    <span class="rl-count">{n_citer}</span>
  </div>
  <div class="rl-row">
    <span class="rl-dot" style="background:#059669;color:#059669"></span>
    <span class="rl-label">参照先（外部論文）</span>
    <span class="rl-count">{n_ref}</span>
  </div>
  <div style="margin:10px 0 6px;font-size:11px;color:#475569">エッジ</div>
  <div class="rl-edge-row">
    <div class="rl-line" style="background:#f87171"></div>
    <span class="rl-label" style="font-size:11.5px">引用（外部 → Zotero）</span>
  </div>
  <div class="rl-edge-row">
    <div class="rl-dashed rl-line"></div>
    <span class="rl-label" style="font-size:11.5px">参照（Zotero → 外部）</span>
  </div>
  <div style="margin:10px 0 6px;font-size:11px;color:#475569">ノードサイズ ＝ 被引用数</div>
  <div class="rl-size-scale">
    <div class="dot" style="width:5px;height:5px"></div>
    <span class="sl">少</span>
    <div class="dot" style="width:9px;height:9px"></div>
    <div class="dot" style="width:14px;height:14px"></div>
    <div class="dot" style="width:20px;height:20px"></div>
    <span class="sl">多</span>
  </div>
  <hr class="rl-divider">
  <div class="rl-stats">
    <span>{n_nodes}</span> nodes &nbsp;·&nbsp; <span>{n_edges}</span> edges
  </div>
  <div class="rl-hint">
    スクロールでズーム · ドラッグで移動<br>
    ホバーで詳細 · 青ノードをダブルクリックでチャンク展開
  </div>
</div>"""

    # ── Embedded data (f-string) ──────────────────────────────────────────────
    data_script = f"""<script>
const GRAPH_DATA = {graph_json};
const EXPAND_DATA = {expand_json};
</script>"""

    # ── sigma.js logic (plain string – no Python vars, no brace escaping) ─────
    logic_script = """<script>
(function () {
'use strict';

/* ── 0. Layout geometry – computed first so sections 2 & 3 can use it.
        bboxR:    half-size of the fixed normalisation bounding box.
                  Must contain the D3 equilibrium spread (empirically ≈ 70√N).
                  Set bigger than expected spread to avoid clipping outliers.
        initMaxR: outer radius of the phyllotaxis seed spiral (≈ 65 % of bbox).
                  Nodes start spread out so the animation is visible from frame 1.
*/
var _N0   = GRAPH_DATA.nodes.length;
// bboxR is larger now to accommodate the collision-based cluster radii.
// With forceCollide, clusters expand until nodes physically don't overlap,
// which for 15 citers around a hub requires cluster radius ≈ 300-500 graph units.
var bboxR = _N0 < 50   ? 1800 :
            _N0 < 150  ? 3000 :
            _N0 < 400  ? 4500 :
            _N0 < 1000 ? 7500 : 12000;
var initMaxR = bboxR * 0.60;   // initial spread = 60 % of bbox → visible in viewport

/* ── 1. Build graphology graph ─────────────────────────────── */
const graph = new graphology.DirectedGraph();
GRAPH_DATA.nodes.forEach(function (n) { graph.addNode(n.id, n); });
GRAPH_DATA.edges.forEach(function (e) {
  try { graph.addEdgeWithKey(e.id, e.source, e.target, e); } catch (_) {}
});

/* ── 2. Spread-out phyllotaxis seed positions ───────────────
        Nodes spiral from 0.3×initMaxR (inner) to initMaxR (outer) so they
        fill ~65 % of the bbox from frame 1.  D3 then reorganises them into
        force-directed clusters – producing the vis.js-style spring-settle
        animation that is clearly visible at the default camera zoom. */
var _ni = 0, _N = graph.order || 1;
graph.forEachNode(function (node) {
  var r     = initMaxR * (0.3 + 0.7 * _ni / Math.max(1, _N - 1));
  var theta = _ni * 2.39996;   // golden angle
  graph.setNodeAttribute(node, 'x', r * Math.cos(theta));
  graph.setNodeAttribute(node, 'y', r * Math.sin(theta));
  _ni++;
});

/* ── 3. Create sigma renderer ───────────────────────────── */
const SigmaClass = (typeof Sigma === 'function') ? Sigma
                 : (Sigma && typeof Sigma.Sigma === 'function') ? Sigma.Sigma
                 : null;
if (!SigmaClass) { console.error('[RAG] sigma.js not loaded'); return; }

// Custom hover renderer: dark background + white text so it's readable on the
// dark canvas. Sigma's default uses a white background (#fff) which makes the
// light labelColor (#e2e8f0) nearly invisible.
function customDrawHover(context, data, settings) {
  var size   = settings.labelSize   || 12;
  var font   = settings.labelFont   || 'sans-serif';
  var weight = settings.labelWeight || '500';
  // Node ring (white outline → node color, matching sigma default)
  context.beginPath();
  context.arc(data.x, data.y, data.size + 3, 0, Math.PI * 2);
  context.closePath();
  context.fillStyle = 'rgba(255,255,255,0.25)';
  context.fill();
  context.beginPath();
  context.arc(data.x, data.y, data.size + 1, 0, Math.PI * 2);
  context.closePath();
  context.fillStyle = data.color;
  context.fill();
  if (!data.label) return;
  context.font = weight + ' ' + size + 'px ' + font;
  var tw  = context.measureText(data.label).width;
  var pad = 5, br = 5;
  var bx  = data.x + data.size + 5;
  var by  = data.y - size / 2 - pad;
  var bw  = tw + pad * 2;
  var bh  = size + pad * 2;
  context.shadowOffsetX = 0; context.shadowOffsetY = 2;
  context.shadowBlur = 8; context.shadowColor = 'rgba(0,0,0,0.5)';
  context.fillStyle = '#ffffff';
  context.beginPath();
  if (context.roundRect) { context.roundRect(bx, by, bw, bh, br); }
  else { context.rect(bx, by, bw, bh); }
  context.fill();
  context.shadowBlur = 0;
  context.fillStyle = '#111827';
  context.fillText(data.label, bx + pad, data.y + size / 3);
}

const container = document.getElementById('sigma-container');
const renderer  = new SigmaClass(graph, container, {
  renderEdgeLabels: false,
  defaultEdgeType:  'arrow',
  labelFont:        'Inter, system-ui, sans-serif',
  labelSize:        12,
  labelWeight:      '500',
  labelThreshold:   7,
  labelColor:       { color: '#e2e8f0' },
  defaultNodeColor: '#2563eb',
  defaultEdgeColor: '#f87171',
  minCameraRatio:   0.01,
  maxCameraRatio:   20,
  hoverRenderer:    customDrawHover,
});
const camera = renderer.getCamera();

// Fix sigma's normalisation to the pre-computed bbox so D3 movements
// produce stable framed-coordinate changes (visible animation).
// bboxR is computed at the top based on graph size.
renderer.setCustomBBox({ x: [-bboxR, bboxR], y: [-bboxR, bboxR] });

/* ── 4. Tooltip ─────────────────────────────────────────── */
var hoveredNode = null, mouseX = 0, mouseY = 0;
var tooltip = document.getElementById('rag-tooltip');

container.addEventListener('mousemove', function (e) {
  mouseX = e.clientX; mouseY = e.clientY;
  if (tooltip.style.display !== 'none') {
    var tw = tooltip.offsetWidth || 360, th = tooltip.offsetHeight || 80;
    tooltip.style.left = Math.min(mouseX + 15, window.innerWidth  - tw - 8) + 'px';
    tooltip.style.top  = Math.min(mouseY + 10, window.innerHeight - th - 8) + 'px';
  }
});
renderer.on('enterNode', function (ev) {
  hoveredNode = ev.node;
  var tip = (graph.getNodeAttributes(ev.node).tooltip ||
             graph.getNodeAttribute(ev.node, 'label') || '');
  if (tip) {
    tooltip.textContent = tip;
    tooltip.style.display = 'block';
    tooltip.style.left = (mouseX + 15) + 'px';
    tooltip.style.top  = (mouseY + 10) + 'px';
  }
});
renderer.on('leaveNode', function () {
  hoveredNode = null;
  tooltip.style.display = 'none';
});

/* ── 5. Smooth zoom ─────────────────────────────────────── */
var zoomVel = 0, zoomId = null;
container.addEventListener('wheel', function (e) {
  e.preventDefault(); e.stopPropagation();
  zoomVel = Math.max(-0.06, Math.min(0.06, zoomVel + (e.deltaY < 0 ? 0.008 : -0.008)));
  if (!zoomId) (function az() {
    if (Math.abs(zoomVel) < 0.001) { zoomId = null; return; }
    camera.setState({ ratio: Math.max(0.01, Math.min(20, camera.getState().ratio * (1 - zoomVel))) });
    zoomVel *= 0.72;
    zoomId = requestAnimationFrame(az);
  })();
}, { capture: true, passive: false });

/* ── 6. Pan inertia ─────────────────────────────────────── */
var isDragging = false, velCamX = 0, velCamY = 0, inertiaId = null;
var prevCamX = 0, prevCamY = 0;
var mc = renderer.getMouseCaptor();

mc.on('mousedown', function () {
  cancelAnimationFrame(inertiaId); inertiaId = null;
  isDragging = !hoveredNode;
  if (isDragging) {
    velCamX = velCamY = 0;
    var s = camera.getState(); prevCamX = s.x; prevCamY = s.y;
  }
});
mc.on('mousemove', function () {
  if (!isDragging) return;
  var s = camera.getState(), a = 0.45;
  velCamX = velCamX * (1-a) + (s.x - prevCamX) * a;
  velCamY = velCamY * (1-a) + (s.y - prevCamY) * a;
  prevCamX = s.x; prevCamY = s.y;
});
mc.on('mouseup', function () {
  if (!isDragging) return;
  isDragging = false;
  if (Math.abs(velCamX) + Math.abs(velCamY) < 0.00005) return;
  (function glide() {
    velCamX *= 0.93; velCamY *= 0.93;
    if (Math.abs(velCamX) + Math.abs(velCamY) < 0.00003) { inertiaId = null; return; }
    var s = camera.getState();
    camera.setState({ x: s.x + velCamX, y: s.y + velCamY });
    inertiaId = requestAnimationFrame(glide);
  })();
});

/* ── 6b. Node drag ──────────────────────────────────────── */
var draggedNode = null;

renderer.on('downNode', function (ev) {
  draggedNode = ev.node;
  isDragging  = false;  // prevent canvas pan while dragging a node
  cancelAnimationFrame(inertiaId); inertiaId = null;
});

// Use window-level listener so drag keeps working even when mouse leaves container
window.addEventListener('mousemove', function (e) {
  if (!draggedNode) return;
  var rect = container.getBoundingClientRect();
  var gpos = renderer.viewportToGraph({ x: e.clientX - rect.left, y: e.clientY - rect.top });
  graph.setNodeAttribute(draggedNode, 'x', gpos.x);
  graph.setNodeAttribute(draggedNode, 'y', gpos.y);
  // Keep D3 simulation in sync so it resumes from new dragged position
  var d3n = d3nodeById[draggedNode];
  if (d3n) { d3n.x = gpos.x; d3n.y = gpos.y; d3n.vx = 0; d3n.vy = 0; }
  renderer.refresh();
});

window.addEventListener('mouseup', function () {
  draggedNode = null;
});

/* ── 7. D3-force layout ─────────────────────────────────── */
// Build D3 node/link arrays from the (now seeded) graphology graph
var d3nodes = [], d3nodeById = {};
graph.forEachNode(function (node, attrs) {
  var n = { id: node, x: attrs.x, y: attrs.y };
  d3nodes.push(n); d3nodeById[node] = n;
});
var d3links = [];
graph.forEachEdge(function (e, attrs, src, tgt) {
  d3links.push({ source: src, target: tgt });
});

var nNodes    = d3nodes.length;
// bboxR was computed in section 0; D3 params scale to match the graph size.
// Charge handles long-range repulsion (pushing clusters apart).
// forceCollide handles short-range: it prevents nodes from visually overlapping
// by enforcing minimum distances proportional to sigma's CSS-pixel node sizes.
var chargeStr = nNodes < 100  ? -1200 :
                nNodes < 500  ? -800 :
                nNodes < 2000 ? -500 : -280;
var linkDist  = nNodes < 100  ?  300 :
                nNodes < 500  ?  260 :
                nNodes < 2000 ?  170 : 110;

// Scale factor: graph-coord units per 1 CSS pixel for collision radii.
// Fixed at 8 so collision radii don't grow with bboxR (which would cause a
// feedback loop: larger bbox → larger gup → larger collide radius → nodes
// spread further → bbox needs to be even larger).
var gup = 8;

var simulation = d3.forceSimulation(d3nodes)
  .force('charge', d3.forceManyBody().strength(chargeStr).theta(0.8))
  // link: use D3's degree-aware default strength (1/min(src_degree, tgt_degree))
  // so hubs (many edges) aren't over-pulled by all their leaves at once
  .force('link',   d3.forceLink(d3links).id(function (d) { return d.id; })
                     .distance(linkDist))
  // collide: prevent visual overlap – radius = sigma visual radius in graph units
  .force('collide', d3.forceCollide()
    .radius(function (d) {
      var sz = 5;
      try { sz = graph.getNodeAttribute(d.id, 'size') || 5; } catch (_) {}
      return sz * gup * 1.6;    // node CSS-pixel radius → graph units + 60 % padding
    })
    .strength(0.85)
    .iterations(3))            // extra passes per tick to resolve dense clusters faster
  .force('x', d3.forceX(0).strength(0.05))
  .force('y', d3.forceY(0).strength(0.05))
  .velocityDecay(0.45)
  .alphaDecay(0.012)
  .stop();

console.log('[RAG] bboxR=' + bboxR + ' gup=' + gup.toFixed(2) + ' nNodes=' + nNodes);

var badge    = document.getElementById('layout-badge');
var badgePct = document.getElementById('layout-pct');
badge.style.display = 'block';
var layoutDone = false;

function finishLayout() {
  if (layoutDone) return;
  layoutDone = true;
  renderer.refresh();
  badge.style.display = 'none';
  console.log('[RAG] layout complete, alpha =', simulation.alpha().toFixed(4));
}

document.getElementById('layout-skip').addEventListener('click', function () {
  finishLayout();
});

function layoutStep() {
  if (layoutDone) return;

  // 1 tick/frame for small graphs → slow, clearly visible animation (~7 s at 60 fps)
  // More ticks for large graphs to avoid layout taking minutes
  var tpf = nNodes < 200 ? 1 : nNodes < 1000 ? 2 : nNodes < 3000 ? 4 : 8;
  for (var i = 0; i < tpf; i++) simulation.tick();

  // Update graphology so sigma's extent/normalization tracks current positions
  d3nodes.forEach(function (n) {
    if (isFinite(n.x) && isFinite(n.y)) {
      graph.setNodeAttribute(n.id, 'x', n.x);
      graph.setNodeAttribute(n.id, 'y', n.y);
    }
  });
  renderer.refresh();

  var pct = Math.round((1 - simulation.alpha()) * 100);
  badgePct.textContent = pct + '%';

  if (simulation.alpha() <= simulation.alphaMin()) {
    finishLayout();
    return;
  }
  requestAnimationFrame(layoutStep);
}
requestAnimationFrame(layoutStep);

/* ── 8. Chunk drill-down ─────────────────────────────────── */
var expandedItems = new Set();
var expandedState = new Map();

function chunkLabel(id) {
  var p = id.split(':'); if (p.length < 2) return '?';
  var loc = p[1], para = p[2] || '', part = p[3] || '';
  var paraNum = para.startsWith('para') ? parseInt(para.slice(4)) + 1 : '';
  var partNum = part.startsWith('part') ? parseInt(part.slice(4)) : 0;
  var base;
  if      (loc[0] === 'p') base = 'p.'  + loc.slice(1);
  else if (loc[0] === 'c') base = 'ch.' + loc.slice(1);
  else if (loc[0] === 'h') base = 'HTML';
  else if (loc[0] === 'n') base = 'Note';
  else                     base = loc;
  var suffix     = paraNum > 1 ? '-' + paraNum       : '';
  var partSuffix = partNum > 0 ? '/' + (partNum + 1) : '';
  return base + suffix + partSuffix;
}

function chunkTooltip(id) {
  var p = id.split(':'), loc = p[1]||'', para = p[2]||'', part = p[3]||'';
  var ls = loc;
  if      (loc[0]==='p') ls = 'Page '    + loc.slice(1);
  else if (loc[0]==='c') ls = 'Chapter ' + loc.slice(1);
  else if (loc[0]==='h') ls = 'HTML snapshot';
  else if (loc[0]==='n') ls = 'Note';
  var paraNum = para.startsWith('para') ? parseInt(para.slice(4))+1 : para;
  var partNum = part.startsWith('part') ? parseInt(part.slice(4)) : 0;
  var partStr = partNum > 0 ? ' (part ' + (partNum+1) + ')' : '';
  return 'Chunk\\n──────────────────────\\nLocation: ' + ls
       + '\\nParagraph: ' + paraNum + partStr;
}

function computeRadius(pos) {
  var total = 0, cnt = 0;
  graph.forEachNode(function (n, a) {
    if (n.startsWith('chunk:')) return;
    var dx = a.x - pos.x, dy = a.y - pos.y;
    total += Math.sqrt(dx*dx + dy*dy); cnt++;
  });
  var avg = cnt > 0 ? total / cnt : 100;
  return Math.max(avg * 0.12, 30);
}

function expandItem(nodeId, itemKey, data) {
  var attrs = graph.getNodeAttributes(nodeId);
  var pos   = { x: attrs.x, y: attrs.y };
  var r     = computeRadius(pos);
  var chunkSet = new Set();
  data.in_edges.forEach(function(e)  { chunkSet.add(e[1]); });
  data.out_edges.forEach(function(e) { chunkSet.add(e[0]); });
  var chunkIds = Array.from(chunkSet);

  var hiddenEdges = [];
  graph.forEachEdge(nodeId, function (edge) {
    if (!graph.getEdgeAttribute(edge, '_structural')) {
      graph.setEdgeAttribute(edge, 'hidden', true);
      hiddenEdges.push(edge);
    }
  });

  var origColor = attrs.color;
  graph.setNodeAttribute(nodeId, 'color', '#1d4ed8');

  var step = (2*Math.PI) / Math.max(chunkIds.length, 1);
  var addedNodes = [], addedEdges = [];

  chunkIds.forEach(function (cid, i) {
    var angle = step*i - Math.PI/2;
    var cnid  = 'chunk:' + cid;
    if (graph.hasNode(cnid)) return;
    graph.addNode(cnid, {
      x: pos.x + r*Math.cos(angle), y: pos.y + r*Math.sin(angle),
      size: 7, color: '#1e3a8a',
      label: chunkLabel(cid), tooltip: chunkTooltip(cid), group: 'chunk',
    });
    addedNodes.push(cnid);
    var seid = graph.addEdge(nodeId, cnid, {
      size: 0.5, color: 'rgba(37,99,235,0.12)', type: 'line', _structural: true,
    });
    addedEdges.push(seid);
  });

  data.in_edges.forEach(function (e) {
    if (!graph.hasNode(e[0]) || !graph.hasNode('chunk:'+e[1])) return;
    try { addedEdges.push(graph.addEdge(e[0], 'chunk:'+e[1], { size:e[2], color:'#f87171', type:'arrow' })); } catch(_){}
  });
  data.out_edges.forEach(function (e) {
    if (!graph.hasNode('chunk:'+e[0]) || !graph.hasNode(e[1])) return;
    try { addedEdges.push(graph.addEdge('chunk:'+e[0], e[1], { size:e[2], color:'#34d399', type:'arrow' })); } catch(_){}
  });

  expandedItems.add(itemKey);
  expandedState.set(itemKey, { addedNodes:addedNodes, addedEdges:addedEdges,
                               hiddenEdges:hiddenEdges, origColor:origColor });

  // ── Add chunk nodes to D3 simulation and reheat so they animate into place ──
  var newD3Nodes = [];
  addedNodes.forEach(function(cnid) {
    if (!d3nodeById[cnid] && graph.hasNode(cnid)) {
      var attrs = graph.getNodeAttributes(cnid);
      var dn = { id: cnid, x: attrs.x, y: attrs.y };
      d3nodes.push(dn); d3nodeById[cnid] = dn;
      newD3Nodes.push(dn);
    }
  });
  if (newD3Nodes.length > 0) {
    // Rebuild full link list from current graph edges
    d3links = [];
    graph.forEachEdge(function(edge, attrs, src, tgt) {
      if (graph.hasNode(src) && graph.hasNode(tgt)) {
        d3links.push({ source: src, target: tgt });
      }
    });
    simulation.nodes(d3nodes);
    simulation.force('link').links(d3links);
    simulation.alpha(0.5);  // reheat – DO NOT call restart(), we use manual RAF
    layoutDone = false;
    badge.style.display = 'block';
    requestAnimationFrame(layoutStep);
  }
}

function collapseItem(nodeId, itemKey) {
  var st = expandedState.get(itemKey); if (!st) return;
  st.addedEdges.forEach(function(e)  { if(graph.hasEdge(e)) graph.dropEdge(e); });
  st.addedNodes.forEach(function(n)  {
    if (d3nodeById[n]) { d3nodes.splice(d3nodes.indexOf(d3nodeById[n]), 1); delete d3nodeById[n]; }
    if (graph.hasNode(n)) graph.dropNode(n);
  });
  st.hiddenEdges.forEach(function(e) { if(graph.hasEdge(e)) graph.setEdgeAttribute(e,'hidden',false); });
  graph.setNodeAttribute(nodeId, 'color', st.origColor);
  expandedItems.delete(itemKey); expandedState.delete(itemKey);
  // Rebuild D3 links after node removal and do a short reheat
  d3links = [];
  graph.forEachEdge(function(edge, attrs, src, tgt) {
    if (graph.hasNode(src) && graph.hasNode(tgt)) {
      d3links.push({ source: src, target: tgt });
    }
  });
  simulation.nodes(d3nodes);
  simulation.force('link').links(d3links);
  simulation.alpha(0.3);
  layoutDone = false;
  badge.style.display = 'block';
  requestAnimationFrame(layoutStep);
}

renderer.on('doubleClickNode', function (ev) {
  ev.event.original.preventDefault(); ev.event.original.stopPropagation();
  var node = ev.node;
  if (!String(node).startsWith('item:')) return;
  var itemKey = String(node).slice(5);
  var data = EXPAND_DATA[itemKey];
  if (!data || (data.in_edges.length===0 && data.out_edges.length===0)) return;
  if (expandedItems.has(itemKey)) { collapseItem(node, itemKey); }
  else { try { expandItem(node, itemKey, data); } catch(err){ console.error('[RAG]', err); } }
});

console.log('[RAG] sigma ready – nodes:', graph.order, 'edges:', graph.size,
            '· D3-force layout starting…');

})();
</script>"""

    # ── Assemble full HTML ────────────────────────────────────────────────────
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Citation Network – Zotero Local RAG</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
  <!-- WebGL graph renderer stack -->
  <script src="https://cdn.jsdelivr.net/npm/graphology@0.25.4/dist/graphology.umd.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/sigma@2.4.0/build/sigma.min.js"></script>
  <!-- D3-force for graph layout (includes all required sub-modules) -->
  <script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
  {css}
</head>
<body>
  <!-- sigma.js renders into this div -->
  <div id="sigma-container"></div>

  <!-- Layout progress badge (bottom-right corner, hidden after layout done) -->
  <div id="layout-badge">
    レイアウト計算中<span id="layout-pct">0%</span>
    <button id="layout-skip" style="margin-left:10px;background:none;border:1px solid rgba(59,130,246,0.4);border-radius:4px;color:#93c5fd;font-size:10px;padding:2px 7px;cursor:pointer;">スキップ</button>
  </div>

  {legend}

  <div id="rag-title">
    <strong>Zotero Local RAG</strong>
    Citation Network
  </div>

  <!-- Tooltip (positioned by JS) -->
  <div id="rag-tooltip"></div>

  {data_script}
  {logic_script}
</body>
</html>"""


# ── build_html: assemble node/edge data → call sigma HTML builder ─────────────

def build_html(
    items: list[dict],
    citers: list[dict],
    refs: list[dict],
    output_path: Path,
    item_meta: dict[str, dict] | None = None,
    chunk_data: dict | None = None,
) -> None:
    import json as _json

    meta       = item_meta or {}
    chunk_data = chunk_data or {}

    C_ZOTERO = "#2563eb"
    C_CITER  = "#d97706"
    C_REF    = "#059669"
    C_UNK    = "#475569"

    nodes: list[dict] = []
    edges: list[dict] = []
    added_papers: set[str] = set()
    edge_counter = 0

    def _eid() -> str:
        nonlocal edge_counter
        edge_counter += 1
        return f"e{edge_counter}"

    # ── Zotero item nodes ──────────────────────────────────────────────────
    for d in items:
        key      = d["item_key"]
        count    = d["citer_count"]
        status   = d.get("s2_status") or "unknown"
        m        = meta.get(key, {})
        full     = m.get("title", "") or ""
        creators = _fmt_creators(m.get("creators", "") or "")
        color    = C_ZOTERO if status in ("mapped", "s2_done") else C_UNK

        extra = [("Key", key)]
        if creators:
            extra.append(("Authors", creators))
        extra += [
            ("Cited by", f"{count:,} papers"),
            ("Contexts", f"{d['context_count']:,}"),
            ("Status",   status),
        ]

        nodes.append({
            "id":      f"item:{key}",
            "label":   _short(full, 28) if full else key,
            "size":    _node_size(count),
            "color":   color,
            "tooltip": _tooltip(full or key, extra),
            "group":   "zotero",
        })

    # ── external citing-paper nodes + edges ───────────────────────────────
    for d in citers:
        pid   = d["citing_paper_id"]
        title = d["citing_title"] or pid
        year  = str(d["citing_year"] or "")
        cc    = d["citing_citation_count"] or 0
        nid   = f"paper:{pid}"

        if nid not in added_papers:
            added_papers.add(nid)
            nodes.append({
                "id":      nid,
                "label":   _short(title, 24),
                "size":    _node_size(cc),
                "color":   C_CITER,
                "tooltip": _tooltip(title, [
                    ("Year",      year or "不明"),
                    ("Citations", f"{cc:,}" if cc else "不明"),
                ]),
                "group": "external",
            })

        edges.append({
            "id":     _eid(),
            "source": nid,
            "target": f"item:{d['cited_item_key']}",
            "size":   max(0.5, min(4.0, d["context_count"] * 0.4)),
            "color":  "#f87171",
            "type":   "arrow",
        })

    # ── reference-paper nodes + edges ─────────────────────────────────────
    n_citer = len(added_papers)
    ref_set: set[str] = set()

    for d in refs:
        pid   = d["cited_paper_id"]
        title = d["cited_title"] or pid
        year  = str(d["cited_year"] or "")
        cc    = d["cited_citation_count"] or 0
        nid   = f"paper:{pid}" if f"paper:{pid}" in added_papers else f"ref:{pid}"

        if nid not in added_papers:
            added_papers.add(nid)
            ref_set.add(nid)
            nodes.append({
                "id":      nid,
                "label":   _short(title, 24),
                "size":    _node_size(cc),
                "color":   C_REF,
                "tooltip": _tooltip(f"[Ref] {title}", [
                    ("Year",      year or "不明"),
                    ("Citations", f"{cc:,}" if cc else "不明"),
                ]),
                "group": "reference",
            })

        edges.append({
            "id":     _eid(),
            "source": f"item:{d['citing_item_key']}",
            "target": nid,
            "size":   max(0.5, min(3.0, d["context_count"] * 0.3)),
            "color":  "#34d399",
            "type":   "line",      # dashed look via color; arrows added by sigma default
        })

    n_nodes = len(nodes)
    n_edges = len(edges)
    n_ref   = len(ref_set)

    graph_json  = _json.dumps({"nodes": nodes, "edges": edges},
                              ensure_ascii=False, separators=(",", ":"))
    expand_json = _json.dumps(chunk_data, ensure_ascii=False, separators=(",", ":"))

    html = _build_sigma_html(
        graph_json, expand_json,
        n_items=len(items),
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_citer=n_citer,
        n_ref=n_ref,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    print(f"\n✅ Graph saved: {output_path}")
    print(f"   {n_nodes} nodes  |  {n_edges} edges")
    print(f"   Blue={len(items)} Zotero items  Amber={n_citer} citers  Green={n_ref} refs")


# ── local HTTP server ─────────────────────────────────────────────────────────

def _serve(output_path: Path, port: int = 8765) -> None:
    """file:// の制限を回避するため簡易 HTTP サーバーで HTML を配信する。"""
    import http.server
    import socketserver
    import threading

    directory = str(output_path.parent)

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

        def end_headers(self):
            # Prevent browser caching so the user always sees the latest HTML
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            super().end_headers()

        def log_message(self, fmt, *args):
            pass

    for p in range(port, port + 10):
        try:
            httpd = socketserver.TCPServer(("", p), _Handler)
            httpd.allow_reuse_address = True
            break
        except OSError:
            continue
    else:
        print("⚠ HTTP サーバーを起動できませんでした。手動で開いてください：")
        print(f"   {output_path}")
        return

    url = f"http://localhost:{p}/{output_path.name}"
    threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    print(f"  → {url}")
    print("  グラフを確認したら Ctrl+C で終了")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        httpd.server_close()


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize citation network as interactive WebGL graph (sigma.js).")
    parser.add_argument("--top",     type=int, default=20,
                        help="表示する上位アイテム数 (default: 20)")
    parser.add_argument("--citers",  type=int, default=15,
                        help="1アイテムあたりの被引用論文数 (default: 15)")
    parser.add_argument("--refs",    type=int, default=10,
                        help="1アイテムあたりの参照先論文数 (default: 10)")
    parser.add_argument("--item",    type=str, default=None,
                        help="特定のZoteroアイテムキーに絞る")
    parser.add_argument("--no-refs", action="store_true",
                        help="参照先（緑ノード）を非表示")
    parser.add_argument("--no-open", action="store_true",
                        help="ブラウザを自動で開かない")
    parser.add_argument("--output",  type=str, default=str(DEFAULT_OUTPUT),
                        help="出力先HTMLパス")
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        print(f"Error: DB not found: {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # ── fetch data ────────────────────────────────────────────────────────
    if args.item:
        item_row = get_item_row(conn, args.item)
        if not item_row:
            print(f"Item '{args.item}' not found in citation DB.", file=sys.stderr)
            sys.exit(1)
        items = [item_row]
    else:
        items = get_top_items(conn, args.top)

    if not items:
        print("No citation data found. Run build_citation_network first.")
        sys.exit(0)

    item_keys = [d["item_key"] for d in items]
    citers    = get_citers(conn, item_keys, args.citers)
    refs      = [] if args.no_refs else get_refs(conn, item_keys, args.refs)

    # ChromaDB からタイトル・著者を取得
    item_meta = get_item_meta(item_keys)
    titled    = sum(1 for k in item_keys if item_meta.get(k, {}).get("title"))
    auth      = sum(1 for k in item_keys if item_meta.get(k, {}).get("creators"))

    # チャンク展開用データを取得
    citer_pids: set[str] = {d["citing_paper_id"] for d in citers}
    rendered_papers: dict[str, str] = {}
    for d in citers:
        rendered_papers[d["citing_paper_id"]] = f"paper:{d['citing_paper_id']}"
    for d in refs:
        pid = d["cited_paper_id"]
        rendered_papers[pid] = f"paper:{pid}" if pid in citer_pids else f"ref:{pid}"
    chunk_data = get_chunk_expand_data(conn, item_keys, rendered_papers)

    conn.close()

    unique_citers = len(set(d["citing_paper_id"] for d in citers))
    unique_refs   = len(set(d["cited_paper_id"]  for d in refs))
    print(f"Building graph: {len(items)} Zotero items "
          f"({titled} with titles, {auth} with authors), "
          f"{unique_citers} unique citers, {unique_refs} unique refs…")

    build_html(items, citers, refs, output_path,
               item_meta=item_meta, chunk_data=chunk_data)

    if not args.no_open:
        _serve(output_path)


if __name__ == "__main__":
    main()
