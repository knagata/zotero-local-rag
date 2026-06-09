# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "networkx",
#   "fa2-modified",
# ]
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
import random
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
    """ChromaDB SQLite から itemKey → {title, creators, year} のマップを返す。"""
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
                MAX(CASE WHEN em.key = 'creators' THEN em.string_value END) AS creators,
                MAX(CASE WHEN em.key = 'year'     THEN em.string_value END) AS year
            FROM embedding_metadata ikey
            JOIN embedding_metadata em ON em.id = ikey.id
            WHERE ikey.key = 'itemKey'
              AND ikey.string_value IN ({placeholders})
            GROUP BY ikey.string_value
        """, item_keys).fetchall()
        conn.close()
        return {
            r["item_key"]: {
                "title":    r["title"]    or "",
                "creators": r["creators"] or "",
                "year":     r["year"]     or "",
            }
            for r in rows
        }
    except Exception as e:
        print(f"  (ChromaDB metadata lookup failed: {e})", file=sys.stderr)
        return {}


def get_item_ref_counts(conn: sqlite3.Connection, item_keys: list[str]) -> dict[str, int]:
    """各 Zotero アイテムが持つ参照先論文数（global_references）を返す。"""
    if not item_keys:
        return {}
    placeholders = ",".join("?" * len(item_keys))
    rows = conn.execute(f"""
        SELECT citing_item_key, COUNT(DISTINCT cited_paper_id) AS ref_count
        FROM global_references
        WHERE citing_item_key IN ({placeholders})
        GROUP BY citing_item_key
    """, item_keys).fetchall()
    return {r[0]: r[1] for r in rows}


# ── DB helpers ───────────────────────────────────────────────────────────────

def get_top_items(conn: sqlite3.Connection, limit: int) -> list[dict]:
    """被引用数の多い順に Zotero アイテムを返す。
    global_citations・global_references のどちらかにデータがあれば含む。
    limit=0 の場合は全件。
    """
    base_sql = """
        SELECT
            k.item_key,
            COALESCE(cit.citer_count, 0)    AS citer_count,
            COALESCE(cit.context_count, 0)  AS context_count,
            ics.s2_status,
            ics.s2_paper_id,
            ics.s2_year,
            ics.s2_citation_count,
            ics.doi,
            ics.isbn
        FROM (
            SELECT cited_item_key  AS item_key FROM global_citations
            UNION
            SELECT citing_item_key AS item_key FROM global_references
        ) k
        LEFT JOIN (
            SELECT
                cited_item_key                      AS item_key,
                COUNT(DISTINCT citing_paper_id)     AS citer_count,
                COUNT(*)                            AS context_count
            FROM global_citations
            GROUP BY cited_item_key
        ) cit ON cit.item_key = k.item_key
        LEFT JOIN item_citation_status ics ON ics.item_key = k.item_key
        ORDER BY citer_count DESC
    """
    if limit and limit < 9999:
        rows = conn.execute(base_sql + "LIMIT ?", (limit,)).fetchall()
    else:
        rows = conn.execute(base_sql).fetchall()
    return [dict(r) for r in rows]


def get_item_row(conn: sqlite3.Connection, item_key: str) -> dict | None:
    """1アイテムの統計を返す。citations / references どちらかにあれば返す。"""
    row = conn.execute("""
        SELECT
            ? AS item_key,
            COALESCE(cit.citer_count, 0)   AS citer_count,
            COALESCE(cit.context_count, 0) AS context_count,
            ics.s2_status,
            ics.s2_paper_id,
            ics.s2_year,
            ics.s2_citation_count,
            ics.doi,
            ics.isbn
        FROM (SELECT 1)
        LEFT JOIN (
            SELECT
                cited_item_key                      AS item_key,
                COUNT(DISTINCT citing_paper_id)     AS citer_count,
                COUNT(*)                            AS context_count
            FROM global_citations
            WHERE cited_item_key = ?
            GROUP BY cited_item_key
        ) cit ON 1=1
        LEFT JOIN item_citation_status ics ON ics.item_key = ?
        WHERE cit.item_key IS NOT NULL
           OR EXISTS (SELECT 1 FROM global_references WHERE citing_item_key = ?)
    """, (item_key, item_key, item_key, item_key)).fetchone()
    return dict(row) if row else None


def get_citers(conn: sqlite3.Connection, item_keys: list[str], per_item: int,
               min_cc: int = 0) -> list[dict]:
    """各アイテムについて、被引用数の多い外部論文を返す（per_item 件まで）。
    min_cc: 外部論文自身の被引用数の下限（0=フィルタなし）。
    """
    placeholders = ",".join("?" * len(item_keys))
    cc_filter = f"AND (citing_citation_count IS NOT NULL AND citing_citation_count >= {min_cc})" if min_cc > 0 else ""
    rows = conn.execute(f"""
        SELECT
            cited_item_key,
            citing_paper_id,
            citing_title,
            citing_year,
            citing_citation_count,
            MAX(citing_doi) AS citing_doi,
            COUNT(*) AS context_count
        FROM global_citations
        WHERE cited_item_key IN ({placeholders})
          AND citing_paper_id IS NOT NULL
          {cc_filter}
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


def get_refs(conn: sqlite3.Connection, item_keys: list[str], per_item: int,
             min_cc: int = 0) -> list[dict]:
    """各アイテムの参照先論文を返す（per_item 件まで）。
    min_cc: 参照先論文自身の被引用数の下限（0=フィルタなし）。
    """
    placeholders = ",".join("?" * len(item_keys))
    cc_filter = f"AND (cited_citation_count IS NOT NULL AND cited_citation_count >= {min_cc})" if min_cc > 0 else ""
    rows = conn.execute(f"""
        SELECT
            citing_item_key,
            cited_paper_id,
            cited_title,
            cited_year,
            cited_citation_count,
            MAX(cited_doi) AS cited_doi,
            COUNT(*) AS context_count
        FROM global_references
        WHERE citing_item_key IN ({placeholders})
          AND cited_paper_id IS NOT NULL
          {cc_filter}
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


# ── server-side FA2 layout ───────────────────────────────────────────────────

def compute_layout(
    item_keys: list[str],
    citer_rows: list[dict],
    ref_rows: list[dict],
) -> dict[str, tuple[float, float]]:
    """
    三段階ForceAtlas2でレイアウトを計算する。
    Phase 1a: Zotero + Ref のみ, 強い斥力 → Zoteroノードを十分に離し、Refを近くに配置
    Phase 1b: Zotero + Citer のみ, 弱い斥力 → CiterをZoteroの周囲に配置
    Phase 2:  全ノード, 弱い斥力 → 微調整
    Returns: { node_id: (x, y) }
    """
    try:
        import networkx as nx
        from fa2_modified import ForceAtlas2
    except ImportError as _e:
        print(f"  [layout] ImportError: {_e}", file=__import__('sys').stderr)
        return {}  # フォールバック: JS側のランダム配置を使用

    import random as _random, math as _math
    _rng = _random.Random(42)
    _stderr = __import__('sys').stderr

    def _fa2(scalingRatio, gravity=1.0):
        return ForceAtlas2(
            outboundAttractionDistribution=True,
            barnesHutOptimize=True,
            barnesHutTheta=1.2,
            scalingRatio=scalingRatio,
            gravity=gravity,
            verbose=False,
        )

    def _run_until_convergence(fa2, G, pos_init, *, batch=50, max_iter=3000,
                               tol_frac=0.0002, consec=3, label=""):
        """
        batchイテレーション毎に max_disp / span を計算し、
        consec 回連続で tol_frac を下回ったら収束と判断する。
        強い斥力で振動しやすい場合は tol_frac を大きく設定すること。
        max_iter を超えたら強制終了。
        """
        pos = pos_init if pos_init else None
        total = 0
        consec_ok = 0
        for _ in range(0, max_iter, batch):
            new_pos = fa2.forceatlas2_networkx_layout(G, pos=pos, iterations=batch)
            total += batch
            if pos is None:
                pos = new_pos
                continue
            disps = [
                _math.sqrt((new_pos[n][0]-pos[n][0])**2 + (new_pos[n][1]-pos[n][1])**2)
                for n in G.nodes()
            ]
            max_disp = max(disps) if disps else 0.0
            all_x = [new_pos[n][0] for n in G.nodes()]
            all_y = [new_pos[n][1] for n in G.nodes()]
            span = max(max(all_x)-min(all_x), max(all_y)-min(all_y), 1e-9)
            rel = max_disp / span
            pos = new_pos
            print(f"    {label} iter={total:4d}  max_disp/span={rel:.5f}  consec={consec_ok}", file=_stderr)
            if rel < tol_frac:
                consec_ok += 1
                if consec_ok >= consec:
                    print(f"    {label} → 収束 (iter={total})", file=_stderr)
                    break
            else:
                consec_ok = 0
        else:
            print(f"    {label} → max_iter={max_iter} 到達", file=_stderr)
        return pos

    # 最終的なノードID体系を先に決定（ref vs paper のプレフィックス）
    citer_paper_ids: set[str] = set()
    for r in citer_rows:
        pid = r.get("citing_paper_id")
        if pid:
            citer_paper_ids.add(f"paper:{pid}")

    def _ref_nid(pid: str) -> str:
        return f"paper:{pid}" if f"paper:{pid}" in citer_paper_ids else f"ref:{pid}"

    # ── Phase 1a: Zotero + Ref ────────────────────────────────────────────
    G1a = nx.Graph()
    for k in item_keys:
        G1a.add_node(f"item:{k}")
    for r in ref_rows:
        pid = r.get("cited_paper_id")
        if not pid:
            continue
        G1a.add_edge(f"item:{r['citing_item_key']}", _ref_nid(pid))

    print(f"  [layout] phase1a (Zotero+Ref): {G1a.number_of_nodes()} nodes …", file=_stderr)
    pos1a = _run_until_convergence(
        _fa2(scalingRatio=10.0, gravity=1.0), G1a, None,
        batch=50, max_iter=3000, tol_frac=0.001, consec=3, label="1a"
    )

    # ── Phase 1b: Zotero + Citer（Zoteroの初期配置はPhase1aの結果を使用）────
    G1b = nx.Graph()
    G1b.add_nodes_from(f"item:{k}" for k in item_keys)
    for r in citer_rows:
        pid = r.get("citing_paper_id")
        if pid:
            G1b.add_edge(f"paper:{pid}", f"item:{r['cited_item_key']}")

    # Zoteroはphase1aの位置を初期値、Citerは対応するZoteroの近くにランダム配置
    zotero_pos_1a = {nid: pos1a[nid] for nid in pos1a if nid.startswith("item:")}
    citer_anchor: dict[str, str] = {}  # citer_nid → closest Zotero item nid
    for r in citer_rows:
        pid = r.get("citing_paper_id")
        if pid:
            citer_anchor[f"paper:{pid}"] = f"item:{r['cited_item_key']}"
    pos1b_init: dict[str, tuple[float, float]] = {}
    for nid in G1b.nodes():
        if nid in zotero_pos_1a:
            pos1b_init[nid] = zotero_pos_1a[nid]
        else:
            base = zotero_pos_1a.get(citer_anchor.get(nid, ""), (0.0, 0.0))
            pos1b_init[nid] = (base[0] + _rng.uniform(-0.5, 0.5),
                               base[1] + _rng.uniform(-0.5, 0.5))

    print(f"  [layout] phase1b (Zotero+Citer): {G1b.number_of_nodes()} nodes …", file=_stderr)
    pos1b = _run_until_convergence(
        _fa2(scalingRatio=3.0, gravity=1.0), G1b, pos1b_init,
        batch=50, max_iter=2000, tol_frac=0.0002, label="1b"
    )

    # ── Phase 2: 全ノード統合 ─────────────────────────────────────────────
    G2 = nx.Graph()
    G2.add_nodes_from(G1a.nodes())
    G2.add_nodes_from(G1b.nodes())
    for u, v in G1a.edges():
        G2.add_edge(u, v)
    for u, v in G1b.edges():
        G2.add_edge(u, v)

    # 初期配置: Zotero・Citer=Phase1b座標系, Ref=接続先ZoteroのPhase1b座標の近くに配置
    # （Phase1aとPhase1bは別スケールなので, RefもPhase1b座標系に揃える）
    ref_to_zotero: dict[str, str] = {}
    for r in ref_rows:
        pid = r.get("cited_paper_id")
        if not pid:
            continue
        ref_to_zotero.setdefault(_ref_nid(pid), f"item:{r['citing_item_key']}")

    cx1b = sum(v[0] for v in pos1b.values()) / max(len(pos1b), 1)
    cy1b = sum(v[1] for v in pos1b.values()) / max(len(pos1b), 1)
    pos2_init: dict[str, tuple[float, float]] = {}
    for nid in G2.nodes():
        if nid in pos1b:
            pos2_init[nid] = pos1b[nid]
        elif nid in ref_to_zotero:
            # 接続先ZoteroのPhase1b位置の近くにランダム配置
            z_pos = pos1b.get(ref_to_zotero[nid], (cx1b, cy1b))
            pos2_init[nid] = (z_pos[0] + _rng.uniform(-0.5, 0.5),
                              z_pos[1] + _rng.uniform(-0.5, 0.5))
        else:
            pos2_init[nid] = (cx1b + _rng.uniform(-0.5, 0.5),
                              cy1b + _rng.uniform(-0.5, 0.5))

    print(f"  [layout] phase2 (all): {G2.number_of_nodes()} nodes …", file=_stderr)
    pos = _run_until_convergence(
        _fa2(scalingRatio=2.0, gravity=1.5), G2, pos2_init,
        batch=50, max_iter=2000, tol_frac=0.0002, label="2 "
    )

    # 座標を graph-units にスケーリング（広めに取る）
    xs = [v[0] for v in pos.values()]
    ys = [v[1] for v in pos.values()]
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1)
    scale = 24000 / span  # 広めにスケーリング

    return {nid: ((x - cx) * scale, (y - cy) * scale) for nid, (x, y) in pos.items()}


# ── node / tooltip helpers ────────────────────────────────────────────────────

_SIZE_MIN = 1
_SIZE_MAX = 8
_SIZE_REF = 10000  # このCC値で最大サイズになる

def _node_size(count: int | None) -> float:
    """被引用数をノードサイズに変換する（sigma の pixel 単位）。
    べき乗スケール（指数0.3）: count=0→2, count=100→5.5, count=1000→9, count=10000→16
    log よりも低 CC と高 CC の視覚差が大きい。
    """
    if not count or count <= 0:
        return _SIZE_MIN
    ratio = min(1.0, count / _SIZE_REF)
    return _SIZE_MIN + (_SIZE_MAX - _SIZE_MIN) * (ratio ** 0.3)


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
    palette: dict,
    css_root: str,
    js_theme: str,
) -> str:
    """sigma.js (WebGL) ベースの完全な HTML を生成する。
    f-string 衝突を避けるため CSS / JS 部分は通常文字列、変数挿入部分のみ f-string。
    """

    # ── CSS (plain string – no Python vars) ──────────────────────────────────
    # CSS uses var(--xxx) throughout; _css_root (generated from PALETTE) is
    # prepended so all values come from one place.
    _css_body = """  *, *::before, *::after { box-sizing: border-box; }
  body {
    margin: 0; overflow: hidden;
    font-family: 'Inter', system-ui, sans-serif;
    background: var(--surface);
  }
  :root { --sb-width: 340px; }
  #sigma-container {
    position: fixed; top: 0; left: 0; bottom: 0; right: var(--sb-width);
    background: var(--surface); transition: right 0.25s ease;
  }
  body.sb-collapsed #sigma-container { right: 0; }

  /* ── Sidebar ── */
  #sidebar {
    position: fixed; right: 0; top: 0; bottom: 0; width: var(--sb-width);
    background: var(--surface-container);
    border-left: 1px solid var(--outline-variant);
    display: flex; flex-direction: column; z-index: 500; overflow: hidden;
    transition: width 0.25s ease, opacity 0.2s ease;
  }
  body.sb-collapsed #sidebar { width: 0; opacity: 0; pointer-events: none; }
  #sb-toggle {
    position: fixed; right: var(--sb-width); top: 50%; transform: translateY(-50%);
    z-index: 600; width: 18px; height: 48px;
    background: var(--surface-container-high);
    border: 1px solid var(--outline-variant); border-right: none;
    border-radius: 6px 0 0 6px; color: var(--on-surface-variant);
    cursor: pointer; font-size: 14px; padding: 0;
    display: flex; align-items: center; justify-content: center;
    transition: right 0.25s ease;
  }
  body.sb-collapsed #sb-toggle { right: 0; }
  #sb-toggle:hover { color: var(--on-surface); }
  #sb-header {
    display: flex; align-items: center; gap: 6px;
    padding: 10px 12px 6px; flex-shrink: 0;
    font-size: 13px; font-weight: 600; color: var(--on-surface);
    border-bottom: 1px solid var(--outline-variant);
  }
  #sb-count { font-weight: 400; font-size: 11px; color: var(--text-dis); }
  #sb-search {
    display: block; width: 100%; box-sizing: border-box;
    background: var(--surface-container-high);
    border: 1px solid var(--outline-variant); border-radius: 4px;
    color: var(--on-surface); font-size: 12px;
    padding: 5px 8px; outline: none;
  }
  #sb-search:focus { border-color: var(--node-zotero); }
  #sb-list-wrap { flex: 1 1 0; overflow-y: auto; min-height: 0; }
  #sb-list { width: 100%; border-collapse: collapse; font-size: 11.5px; }
  #sb-list thead th {
    position: sticky; top: 0;
    background: var(--surface-container-high);
    border-bottom: 1px solid var(--outline-variant);
    padding: 5px 6px; text-align: left;
    color: var(--on-surface-variant); font-weight: 500;
    cursor: pointer; user-select: none; white-space: nowrap;
  }
  #sb-list thead th:hover { color: var(--on-surface); }
  #sb-list thead th.sort-asc::after  { content: ' ↑'; }
  #sb-list thead th.sort-desc::after { content: ' ↓'; }
  #sb-list tbody tr { cursor: pointer; border-bottom: 1px solid var(--outline-variant); }
  #sb-list tbody tr:hover td { background: var(--surface-container-high); }
  #sb-list tbody tr.sb-active td {
    background: color-mix(in srgb, var(--node-zotero) 18%, transparent);
  }
  #sb-list td { padding: 5px 6px; color: var(--on-surface); vertical-align: top; }
  #sb-list td.sb-num {
    color: var(--on-surface-variant); text-align: right;
    width: 40px; white-space: nowrap;
  }
  #sb-detail {
    flex-shrink: 0; border-top: 1px solid var(--outline-variant);
    padding: 12px 14px; overflow-y: auto; max-height: 260px;
    font-size: 12px; display: none;
  }
  #sb-detail.sb-active { display: block; }
  #sb-detail-title {
    font-size: 13px; font-weight: 600; color: var(--on-surface);
    line-height: 1.4; margin-bottom: 10px;
  }
  .sb-kv { display: flex; gap: 6px; margin-bottom: 3px; line-height: 1.5; }
  .sb-k  { color: var(--text-dis); flex-shrink: 0; min-width: 64px; }
  .sb-v  { color: var(--on-surface-variant); word-break: break-all; }
  .sb-v a { color: var(--node-zotero); text-decoration: none; }
  .sb-v a:hover { text-decoration: underline; }

  /* ── Layout progress badge  (surface dp1) ── */
  #layout-badge {
    display: none;
    position: fixed; bottom: 16px; right: calc(var(--sb-width) + 16px); z-index: 1000;
    transition: right 0.25s ease;
  }
  body.sb-collapsed #layout-badge { right: 16px; }
  #layout-badge {
    background: var(--surface-container-low);
    border: 1px solid var(--outline-variant);
    border-radius: 4px; padding: 6px 14px;
    font-size: 11.5px; color: var(--on-surface-variant);
    box-shadow: 0 2px 8px rgba(0,0,0,0.5); cursor: default;
  }
  #layout-badge span { color: var(--node-zotero); font-weight: 500; margin-left: 4px; }

  /* ── Legend card  (surface dp1) ── */
  #rag-legend {
    position: fixed; top: 16px; left: 16px; z-index: 1000;
    background: var(--surface-container-low);
    border: 1px solid var(--outline-variant);
    border-radius: 4px; padding: 18px 20px 14px;
    color: var(--on-surface); font-size: 13px; min-width: 210px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.5);
  }
  #rag-legend h3 { margin: 0 0 14px; font-size: 14px; font-weight: 500;
                   color: var(--on-surface); letter-spacing: .01em; }
  .rl-row  { display: flex; align-items: center; gap: 10px; margin-bottom: 7px; }
  .rl-dot  { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  .rl-label { font-size: 12px; color: var(--on-surface-variant); }
  .rl-count { margin-left: auto; font-size: 11px; color: var(--text-dis);
              font-variant-numeric: tabular-nums; }
  .rl-edge-row { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
  .rl-line  { width: 22px; height: 2px; border-radius: 1px; flex-shrink: 0; position: relative; }
  /* 矢印付き線：右端に三角 */
  .rl-line::after {
    content: ''; position: absolute; right: -1px; top: 50%;
    transform: translateY(-50%);
    border-left: 6px solid currentColor;
    border-top: 3px solid transparent;
    border-bottom: 3px solid transparent;
    color: inherit;
  }
  .rl-divider { border: none; border-top: 1px solid var(--outline-variant); margin: 12px 0; }
  .rl-stats { font-size: 11px; color: var(--text-dis); line-height: 2.0; }
  .rl-stats span { color: var(--on-surface-variant); }
  .rl-size-scale { display: flex; align-items: center; gap: 5px; margin: 4px 0 10px; }
  .rl-size-scale .dot { background: var(--outline-variant); border-radius: 50%; flex-shrink: 0; }
  .rl-size-scale .sl  { font-size: 10px; color: var(--text-dis); }
  .rl-hint { font-size: 10.5px; color: var(--text-dis); line-height: 1.7; margin-top: 2px; }
  .rl-filter { margin-top: 10px; }
  .rl-filter label { display: block; font-size: 10.5px; color: var(--text-dis); margin-bottom: 3px; }
  .rl-filter-row { display: flex; align-items: center; gap: 6px; margin-bottom: 6px; }
  .rl-filter-row .rl-dot { flex: none; width: 10px; height: 10px; }
  .rl-filter-row .rl-filter-lbl { font-size: 10.5px; color: var(--on-surface-variant); flex: 1; }
  .rl-filter-row input[type=number] {
    width: 54px; background: var(--surface-container-high);
    border: 1px solid var(--outline-variant); border-radius: 4px;
    color: var(--on-surface); font-size: 11px; padding: 3px 6px;
    outline: none;
  }
  .rl-filter-row input[type=number]:focus { border-color: var(--node-zotero); }

  /* ── Title badge  (surface dp1) ── */
  #rag-title {
    position: fixed; top: 16px; right: calc(var(--sb-width) + 16px); z-index: 1000;
    background: var(--surface-container-low);
    border: 1px solid var(--outline-variant);
    border-radius: 4px; padding: 10px 16px;
    color: var(--on-surface-variant); font-size: 11px; text-align: right;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    transition: right 0.25s ease;
  }
  body.sb-collapsed #rag-title { right: 16px; }
  #rag-title strong { display: block; font-size: 14px; font-weight: 500;
                      color: var(--on-surface); margin-bottom: 2px; }

  /* ── Tooltip  (surface dp4) ── */
  #rag-tooltip {
    position: fixed; display: none; z-index: 2000;
    background: var(--surface-container-high);
    border: 1px solid var(--outline-variant);
    border-radius: 4px; padding: 10px 14px;
    font-family: 'Inter', monospace; font-size: 12px; line-height: 1.7;
    color: var(--on-surface); max-width: 360px;
    white-space: pre-wrap; word-break: break-word;
    box-shadow: 0 8px 16px rgba(0,0,0,0.6); pointer-events: none;
  }"""
    css = "<style>\n" + css_root + "\n\n" + _css_body + "\n</style>"

    # ── Legend HTML (f-string – uses Python vars) ─────────────────────────────
    legend = f"""<div id="rag-legend">
  <h3>Citation Network</h3>
  <div class="rl-row">
    <span class="rl-dot" style="background:{palette['nodeZotero']}"></span>
    <span class="rl-label">Zotero アイテム</span>
    <span class="rl-count">{n_items}</span>
  </div>
  <div class="rl-row">
    <span class="rl-dot" style="background:{palette['nodeExternal']}"></span>
    <span class="rl-label">外部論文（被引用元 / 参照先）</span>
    <span class="rl-count" id="stat-citer"><span id="stat-citer-vis">{n_citer}</span>({n_citer})</span>
    <span style="font-size:10px;color:var(--text-dis)"> / </span>
    <span class="rl-count" id="stat-ref"><span id="stat-ref-vis">{n_ref}</span>({n_ref})</span>
  </div>
  <div style="margin:8px 0 4px;font-size:11px;color:var(--text-dis)">ノード選択時の色分け</div>
  <div class="rl-row" style="margin-bottom:2px">
    <span class="rl-dot" style="background:{palette['nodeCiter']}"></span>
    <span class="rl-label" style="font-size:11px">被引用元（選択ノードを引用）</span>
  </div>
  <div class="rl-row" style="margin-bottom:8px">
    <span class="rl-dot" style="background:{palette['nodeRef']}"></span>
    <span class="rl-label" style="font-size:11px">参照先（選択ノードが引用）</span>
  </div>
  <div style="margin:4px 0 6px;font-size:11px;color:var(--text-dis)">エッジ（太さ＝関連強度）</div>
  <div class="rl-edge-row">
    <div class="rl-line" style="background:{palette['edgeDefault']};color:{palette['edgeDefault']}"></div>
    <span class="rl-label" style="font-size:11.5px">通常時</span>
  </div>
  <div class="rl-edge-row">
    <div class="rl-line" style="background:{palette['edgeCitation']};color:{palette['edgeCitation']}"></div>
    <span class="rl-label" style="font-size:11.5px">選択時：引用（外部 → Zotero）</span>
  </div>
  <div class="rl-edge-row">
    <div class="rl-line" style="background:{palette['edgeReference']};color:{palette['edgeReference']}"></div>
    <span class="rl-label" style="font-size:11.5px">選択時：参照（Zotero → 外部）</span>
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
  <div class="rl-filter">
    <label>最低被引用数フィルタ（変更後 Enter）</label>
    <div class="rl-filter-row">
      <span class="rl-dot" style="background:{palette['nodeExternal']}"></span>
      <span class="rl-filter-lbl">被引用元 ≥</span>
      <input type="number" id="filter-citer-cc" min="0" value="50" title="被引用元ノードの最低被引用数">
    </div>
    <div class="rl-filter-row">
      <span class="rl-dot" style="background:{palette['nodeExternal']}"></span>
      <span class="rl-filter-lbl">参照先 ≥</span>
      <input type="number" id="filter-ref-cc" min="0" value="50" title="参照先ノードの最低被引用数">
    </div>
  </div>
  <hr class="rl-divider">
  <div class="rl-stats">
    <span id="stat-nodes">{n_nodes}</span> nodes &nbsp;·&nbsp; <span>{n_edges}</span> edges
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
    logic_script = "<script>\n(function () {\n'use strict';\n\n" + js_theme + """

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

/* ── 2. Initial positions ────────────────────────────────────
   If the Python backend embedded pre-computed FA2 positions (x/y on each node),
   use them directly — no browser-side physics needed.
   Otherwise fall back to phyllotaxis seeding for the D3 simulation. */
var _hasPrecomputedLayout = GRAPH_DATA.nodes.length > 0 &&
    GRAPH_DATA.nodes[0].x != null && GRAPH_DATA.nodes[0].x !== undefined;

if (_hasPrecomputedLayout) {
  // Positions already embedded — just copy them into graphology attributes.
  // bboxR is set from actual coordinate spread so sigma's normalisation fits.
  var _allX = [], _allY = [];
  graph.forEachNode(function (id, attrs) {
    if (attrs.x != null) { _allX.push(attrs.x); _allY.push(attrs.y); }
  });
  if (_allX.length) {
    var _span = Math.max(
      Math.max.apply(null, _allX) - Math.min.apply(null, _allX),
      Math.max.apply(null, _allY) - Math.min.apply(null, _allY)
    );
    bboxR = Math.ceil(_span * 0.65);  // bbox slightly larger than spread
  }
  console.log('[RAG] using pre-computed layout, bboxR=' + bboxR);
} else {
  // Fallback: phyllotaxis for Zotero items, anchored spread for external nodes.
  var _zoteroIds = [], _externalIds = [];
  graph.forEachNode(function (id, attrs) {
    if (attrs.group === 'zotero') _zoteroIds.push(id);
    else _externalIds.push(id);
  });
  var _NZ = _zoteroIds.length || 1;
  var _zoteroPos = {};
  _zoteroIds.forEach(function (id, i) {
    var r = initMaxR * (0.25 + 0.55 * i / Math.max(1, _NZ - 1));
    var theta = i * 2.39996;
    var x = r * Math.cos(theta), y = r * Math.sin(theta);
    graph.setNodeAttribute(id, 'x', x);
    graph.setNodeAttribute(id, 'y', y);
    _zoteroPos[id] = { x: x, y: y };
  });
  var _anchor = {};
  graph.forEachEdge(function (e, attrs, src, tgt) {
    if (!_anchor[src] && _zoteroPos[tgt]) _anchor[src] = tgt;
    if (!_anchor[tgt] && _zoteroPos[src]) _anchor[tgt] = src;
  });
  var _spreadR = initMaxR * 0.18;
  _externalIds.forEach(function (id) {
    var anch = _anchor[id];
    var base = anch ? _zoteroPos[anch] : { x: 0, y: 0 };
    var ang = Math.random() * 2 * Math.PI;
    var d = _spreadR * (0.4 + Math.random() * 0.8);
    graph.setNodeAttribute(id, 'x', base.x + d * Math.cos(ang));
    graph.setNodeAttribute(id, 'y', base.y + d * Math.sin(ang));
  });
  console.log('[RAG] using fallback phyllotaxis layout');
}

/* ── 3. Create sigma renderer ───────────────────────────── */
const SigmaClass = (typeof Sigma === 'function') ? Sigma
                 : (Sigma && typeof Sigma.Sigma === 'function') ? Sigma.Sigma
                 : null;
if (!SigmaClass) { console.error('[RAG] sigma.js not loaded'); return; }

// ── CC filter thresholds (updated by legend inputs) ──────────────────────────
var filterCiterCC = 50;  // min citation count for citer nodes (group='external')
var filterRefCC   = 50;  // min citation count for ref nodes   (group='reference')

// Click-selection state – declared here so nodeReducer / edgeReducer can close
// over them.  Hovering only shows a label pill; clicking toggles the focus
// (selected node + 1st-degree neighbours highlighted, everything else dimmed).
var selectedNode           = null;
var selectedNeighbors      = new Set();  // all direct neighbours of selectedNode
var selectedChunkNodes     = new Set();  // chunk-type direct neighbours
var selectedChunkNeighbors = new Set();  // nodes connected through those chunks
// 選択ノードからみた関係方向（選択時のハイライト色分け用）
var selectedCiterNodes     = new Set();  // selected ← other (other が選択ノードを引用)
var selectedRefNodes       = new Set();  // selected → other (選択ノードが other を参照)

// ── 選択トランジション ───────────────────────────────────────────────────────
// _selectionT: 0 = 選択直後（未適用）→ 1 = 完全適用
// nodeReducer でこの値を使い、非選択ノードの色を背景色へ補間する。
var _selectionT    = 0;
var _selectionAnimId = null;
var _SELECTION_DUR = 380;  // ms
var _BG_COLOR      = THEME.surface || '#141218';

// ── 起動フェードイン ─────────────────────────────────────────────────────────
// _fadeInT: 0 → 1 (500ms) で nodeReducer/edgeReducer の色を背景色から補間する。
// renderer 作成後に _fadeInRefresh に renderer.refresh を差し込む。
var _fadeInT = 0;
var _fadeInRefresh = null;
setTimeout(function() {
  var _FADE_DUR = 380;
  var _start = Date.now();
  function _step() {
    var raw = Math.min(1, (Date.now() - _start) / _FADE_DUR);
    _fadeInT = raw * raw * (3 - 2 * raw); // smoothstep
    if (_fadeInRefresh) _fadeInRefresh();
    if (raw < 1) { requestAnimationFrame(_step); }
    else { _fadeInT = 1; }
  }
  requestAnimationFrame(_step);
}, 500);

function _hexLerp(a, b, t) {
  // '#rrggbb' 同士を補間
  var ar = parseInt(a.slice(1,3),16), ag = parseInt(a.slice(3,5),16), ab = parseInt(a.slice(5,7),16);
  var br = parseInt(b.slice(1,3),16), bg = parseInt(b.slice(3,5),16), bb = parseInt(b.slice(5,7),16);
  var r = (ar + (br-ar)*t|0).toString(16).padStart(2,'0');
  var g = (ag + (bg-ag)*t|0).toString(16).padStart(2,'0');
  var bl = (ab + (bb-ab)*t|0).toString(16).padStart(2,'0');
  return '#' + r + g + bl;
}

function _startSelectionAnim() {
  if (_selectionAnimId) cancelAnimationFrame(_selectionAnimId);
  _selectionT = 0;
  var start = Date.now();
  function step() {
    var raw = Math.min(1, (Date.now() - start) / _SELECTION_DUR);
    // quadraticOut easing
    _selectionT = 1 - (1 - raw) * (1 - raw);
    renderer.refresh();
    if (raw < 1) { _selectionAnimId = requestAnimationFrame(step); }
    else { _selectionAnimId = null; }
  }
  _selectionAnimId = requestAnimationFrame(step);
}

function recomputeSelection(node) {
  selectedNode           = node;
  selectedNeighbors      = new Set(graph.neighbors(node));
  selectedChunkNodes     = new Set();
  selectedChunkNeighbors = new Set();
  selectedCiterNodes     = new Set();
  selectedRefNodes       = new Set();

  // 辺の向きからciter/refを分類:
  //   other → selected : other は selected を引用している = citer
  //   selected → other : selected は other を参照している = ref
  graph.edges(node).forEach(function(edge) {
    var src = graph.source(edge), tgt = graph.target(edge);
    if (tgt === node) {
      selectedCiterNodes.add(src);
    } else {
      selectedRefNodes.add(tgt);
    }
  });

  selectedNeighbors.forEach(function(n) {
    try {
      if (graph.getNodeAttribute(n, 'group') === 'chunk') {
        selectedChunkNodes.add(n);
        graph.neighbors(n).forEach(function(nn) {
          if (nn !== node) selectedChunkNeighbors.add(nn);
        });
      }
    } catch(_) {}
  });
  _startSelectionAnim();
}

function clearSelection() {
  if (_selectionAnimId) { cancelAnimationFrame(_selectionAnimId); _selectionAnimId = null; }
  // _selectionT を 1→0 方向に戻しながら選択状態を解除する。
  // nodeReducer は selectedNode===null なら dimming を一切行わないので、
  // _selectionT を 0 へ向けて下げながら refresh すると非選択ノードが徐々に復帰する。
  var startT = _selectionT;
  var start  = Date.now();
  selectedNode           = null;
  selectedNeighbors      = new Set();
  selectedChunkNodes     = new Set();
  selectedChunkNeighbors = new Set();
  selectedCiterNodes     = new Set();
  selectedRefNodes       = new Set();
  // すでに完全解除済みなら即終了
  if (startT <= 0) { _selectionT = 0; return; }
  function step() {
    var raw = Math.min(1, (Date.now() - start) / _SELECTION_DUR);
    var eased = 1 - (1 - raw) * (1 - raw);
    _selectionT = startT * (1 - eased);
    renderer.refresh();
    if (raw < 1) { _selectionAnimId = requestAnimationFrame(step); }
    else { _selectionT = 0; _selectionAnimId = null; }
  }
  _selectionAnimId = requestAnimationFrame(step);
}

// ── Custom hover renderer ─────────────────────────────────────────────────────
// Subtle ring + white pill label on hover.  No dimming here – that is done on
// click via nodeReducer / edgeReducer.
function customDrawHover(context, data, settings) {
  var size   = settings.labelSize   || 12;
  var font   = settings.labelFont   || 'sans-serif';
  var weight = settings.labelWeight || '500';
  var nr     = data.size;

  // Soft halo
  context.beginPath();
  context.arc(data.x, data.y, nr + 9, 0, Math.PI * 2);
  context.closePath();
  context.fillStyle = 'rgba(255,255,255,0.07)';
  context.fill();

  // White border ring
  context.beginPath();
  context.arc(data.x, data.y, nr + 3, 0, Math.PI * 2);
  context.closePath();
  context.fillStyle = 'rgba(255,255,255,0.85)';
  context.fill();

  // Node colour fill (use original colour from graph attributes, not reducer)
  var origColor = graph.getNodeAttribute(data.key || '', 'color') || data.color;
  context.beginPath();
  context.arc(data.x, data.y, nr + 1, 0, Math.PI * 2);
  context.closePath();
  context.fillStyle = origColor;
  context.fill();

  if (!data.label) return;

  context.font = weight + ' ' + size + 'px ' + font;
  var tw  = context.measureText(data.label).width;
  var pad = 6, br = 6;
  var bx  = data.x + nr + 8;
  var by  = data.y - size / 2 - pad;
  var bw  = tw + pad * 2;
  var bh  = size + pad * 2;

  // White pill with shadow
  context.shadowOffsetX = 0; context.shadowOffsetY = 2;
  context.shadowBlur    = 10; context.shadowColor = 'rgba(0,0,0,0.4)';
  context.fillStyle = '#ffffff';
  context.beginPath();
  if (context.roundRect) { context.roundRect(bx, by, bw, bh, br); }
  else { context.rect(bx, by, bw, bh); }
  context.fill();
  context.shadowBlur = 0;

  context.fillStyle = '#0f172a';
  context.fillText(data.label, bx + pad, data.y + size / 3);
}

// ── nodeReducer: dim non-neighbours when a node is selected ──────────────────
//   • selected node                  → full colour, highlighted
//   • direct neighbour or chunk's neighbour → full colour, label visible
//   • everything else                → very dark (fades into bg), label hidden
// Also applies zoom-adaptive sizing: nodes scale up when zoomed out so the
// graph doesn't collapse into an undifferentiated blob at low zoom levels.
function nodeReducer(node, data) {
  var res = Object.assign({}, data);

  // ── CC threshold filter ───────────────────────────────────────────────────
  if (data.group === 'external' && filterCiterCC > 0 && (data.cc || 0) < filterCiterCC) {
    res.hidden = true;
    return res;
  }
  if (data.group === 'reference' && filterRefCC > 0 && (data.cc || 0) < filterRefCC) {
    res.hidden = true;
    return res;
  }

  // ── Zoom-adaptive size ────────────────────────────────────────────────────
  // sigma node sizes are fixed screen-pixels, unaffected by camera zoom.
  // When zoomed OUT (ratio > 1): shrink nodes aggressively to prevent blob.
  // No enlargement when zoomed in — that causes a sudden-size bug on pan.
  // camera may be null on the very first call during renderer construction.
  if (camera) {
    var ratio = camera.getState().ratio;
    if (ratio > 1) {
      // Zoomed out: shrink. ratio=2→×0.57, ratio=4→×0.33, ratio=8→×0.19
      res.size = Math.max(0.5, data.size / Math.pow(ratio, 0.8));
    }
    // Hide labels for external/reference nodes when zoomed out — only show Zotero
    // item labels (group='zotero') so the screen doesn't fill with tiny text.
    if (ratio > 2 && data.group !== 'zotero' && data.group !== 'chunk') {
      res.label = '';
    }
  }

  // ── Selection dimming + citer/ref color highlight ────────────────────────
  // 選択時: 選択ノード自身は highlighted、隣接ノードは辺の向きで色分け。
  // Inactive nodes are hidden (not just dimmed) so they cannot cover active
  // nodes regardless of WebGL draw order.
  // _selectionT > 0 なら選択中 or 解除トランジション中
  if (selectedNode !== null || _selectionT > 0) {
    if (node === selectedNode) {
      res.highlighted = true;
      res.zIndex = 2;
    } else if (selectedNeighbors.has(node) || selectedChunkNeighbors.has(node)) {
      res.zIndex = 1;
      // 辺の向きに応じて色を上書き（chunk/zotero ノードは元色を保持）
      var grp = data.group;
      if (grp === 'external' || grp === 'reference') {
        if (selectedCiterNodes.has(node)) {
          res.color = THEME.nodeCiter;
        } else if (selectedRefNodes.has(node)) {
          res.color = THEME.nodeRef;
        }
      }
    } else {
      // 選択中: 完全適用なら hidden、トランジション中は色補間
      if (_selectionT >= 1) {
        res.hidden = true;
      } else {
        res.color = _hexLerp(data.color || res.color, _BG_COLOR, _selectionT);
        res.size  = res.size * (1 - _selectionT * 0.8);
      }
      res.label = '';
    }
  }
  // ── 起動フェードイン ────────────────────────────────────────────────────────
  if (_fadeInT < 1 && !res.hidden) {
    res.color = _hexLerp(_BG_COLOR, res.color || data.color, _fadeInT);
    res.size  = (res.size || data.size || 4) * _fadeInT;
    res.label = '';
  }
  return res;
}

// ── edgeReducer: show edges touching selected node OR any of its chunk nodes ──
// Chunk nodes are expanded children: their edges to citers/refs must be visible.
function edgeReducer(edge, data) {
  var res = Object.assign({}, data);
  // Hide edges to/from filtered-out nodes
  var src = graph.source(edge), tgt = graph.target(edge);
  try {
    var sa = graph.getNodeAttributes(src), ta = graph.getNodeAttributes(tgt);
    if ((sa.group === 'external'  && filterCiterCC > 0 && (sa.cc || 0) < filterCiterCC) ||
        (ta.group === 'external'  && filterCiterCC > 0 && (ta.cc || 0) < filterCiterCC) ||
        (sa.group === 'reference' && filterRefCC   > 0 && (sa.cc || 0) < filterRefCC)   ||
        (ta.group === 'reference' && filterRefCC   > 0 && (ta.cc || 0) < filterRefCC)) {
      res.hidden = true;
      return res;
    }
  } catch(_) {}
  if (selectedNode !== null) {
    if (src === selectedNode || tgt === selectedNode ||
        selectedChunkNodes.has(src) || selectedChunkNodes.has(tgt)) {
      res.zIndex = 1;
      // 辺の向きで色を上書き:
      //   other → selectedNode : citation edge (amber)
      //   selectedNode → other : reference edge (rose)
      if (tgt === selectedNode || selectedChunkNodes.has(tgt)) {
        res.color = THEME.edgeCitation;
      } else if (src === selectedNode || selectedChunkNodes.has(src)) {
        res.color = THEME.edgeReference;
      }
    } else {
      res.hidden = true;
    }
  }
  // ── 起動フェードイン ────────────────────────────────────────────────────────
  if (_fadeInT < 1 && !res.hidden) {
    res.color = _hexLerp(_BG_COLOR, res.color || data.color, _fadeInT);
  }
  return res;
}

const container = document.getElementById('sigma-container');
const renderer  = new SigmaClass(graph, container, {
  renderEdgeLabels: false,
  defaultEdgeType:  'arrow',
  labelFont:        'Inter, system-ui, sans-serif',
  labelSize:        12,
  labelWeight:      '500',
  labelThreshold:   7,
  labelColor:       { color: THEME.onSurfaceVariant },
  defaultNodeColor: THEME.nodeZotero,
  defaultEdgeColor: THEME.edgeDefault,
  minCameraRatio:   0.01,
  maxCameraRatio:   20,
  hoverRenderer:    customDrawHover,
  nodeReducer:      nodeReducer,
  edgeReducer:      edgeReducer,
});
var camera = renderer.getCamera();

// Fix sigma's normalisation to the pre-computed bbox so D3 movements
// produce stable framed-coordinate changes (visible animation).
// bboxR is computed at the top based on graph size.
renderer.setCustomBBox({ x: [-bboxR, bboxR], y: [-bboxR, bboxR] });
// フェードインループに refresh を接続
_fadeInRefresh = function() { renderer.refresh(); };
// debug hooks
window.__g = graph; window.__r = renderer; window.__sim = simulation; window.__bboxR = bboxR;
window.__d3nodes = d3nodes; window.__d3nById = d3nodeById;

/* ── 4. Tooltip (hover) + click-selection ───────────────── */
// selectedNode / selectedNeighbors are declared before the sigma constructor.
var mouseX = 0, mouseY = 0;
var tooltip = document.getElementById('rag-tooltip');

container.addEventListener('mousemove', function (e) {
  mouseX = e.clientX; mouseY = e.clientY;
  if (tooltip.style.display !== 'none') {
    var tw = tooltip.offsetWidth || 360, th = tooltip.offsetHeight || 80;
    tooltip.style.left = Math.min(mouseX + 15, window.innerWidth  - tw - 8) + 'px';
    tooltip.style.top  = Math.min(mouseY + 10, window.innerHeight - th - 8) + 'px';
  }
});

// Hover: show tooltip only (no dimming)
renderer.on('enterNode', function (ev) {
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
  tooltip.style.display = 'none';
});

// Click: toggle selection focus (same node again → deselect)
renderer.on('clickNode', function (ev) {
  // ドラッグ後のmouseupではなく、純粋なクリックのみ選択を行う
  if (hasDragged) return;
  if (selectedNode === ev.node) { clearSelection(); renderer.refresh(); }
  else {
    recomputeSelection(ev.node);
    renderer.refresh();
    if (window._panToNode) window._panToNode(ev.node);
  }
});

// Click on empty canvas → deselect
renderer.on('clickStage', function () {
  if (selectedNode !== null) { clearSelection(); renderer.refresh(); }
});

/* ── 5. Smooth zoom ─────────────────────────────────────── */
var zoomVel = 0, zoomId = null;
container.addEventListener('wheel', function (e) {
  e.preventDefault(); e.stopPropagation();
  zoomVel = Math.max(-0.06, Math.min(0.06, zoomVel + (e.deltaY < 0 ? 0.008 : -0.008)));
  if (!zoomId) (function az() {
    if (Math.abs(zoomVel) < 0.001) { zoomId = null; return; }
    var newRatio = Math.max(0.01, Math.min(20, camera.getState().ratio * (1 - zoomVel)));
    camera.setState(renderer.getViewportZoomedState({ x: mouseX, y: mouseY }, newRatio));
    zoomVel *= 0.72;
    zoomId = requestAnimationFrame(az);
  })();
}, { capture: true, passive: false });

/* ── 6. Pan inertia + Node drag ─────────────────────────── */
// draggedNode: ドラッグ中のノードID (null = パン中)
// isPanning:   カメラパン中フラグ
var isPanning = false, velCamX = 0, velCamY = 0, inertiaId = null;
var prevCamX = 0, prevCamY = 0;
var draggedNode = null;
var hasDragged  = false;  // ドラッグが実際に起きたかのフラグ（クリック判定用）
var dragDownX   = 0, dragDownY = 0;
var mc = renderer.getMouseCaptor();

// ノードのdown: ドラッグ開始準備
renderer.on('downNode', function (ev) {
  draggedNode = ev.node;
  hasDragged  = false;
  dragDownX   = ev.event.x;
  dragDownY   = ev.event.y;
  isPanning   = false;
  cancelAnimationFrame(inertiaId); inertiaId = null;
  // D3にノードを固定させてシミュレーションに引き戻されないようにする
  var d3n = d3nodeById[draggedNode];
  if (d3n) {
    var attrs = graph.getNodeAttributes(draggedNode);
    d3n.fx = attrs.x; d3n.fy = attrs.y;
  }
});

// キャンバスのdown: パン開始 (ノードドラッグ中は無視)
mc.on('mousedown', function () {
  if (draggedNode) return;
  cancelAnimationFrame(inertiaId); inertiaId = null;
  isPanning = true;
  velCamX = velCamY = 0;
  var s = camera.getState(); prevCamX = s.x; prevCamY = s.y;
});

// mousemovebody: ノードドラッグ or パン速度追跡
// sigma は mousemovebody でカメラを動かすので、ノードドラッグ時は
// ev.preventSigmaDefault() でそれを止める
mc.on('mousemovebody', function (ev) {
  if (draggedNode) {
    // 3px以上動いたら「ドラッグあり」とみなす
    if (!hasDragged) {
      var dx = ev.x - dragDownX, dy = ev.y - dragDownY;
      if (dx*dx + dy*dy > 9) hasDragged = true;
    }
    if (!hasDragged) return;  // 微小な動きは無視（クリック扱いにする）
    // ノードドラッグ: マウス位置をグラフ座標に変換してノードを移動
    var gpos = renderer.viewportToGraph({ x: ev.x, y: ev.y });
    graph.setNodeAttribute(draggedNode, 'x', gpos.x);
    graph.setNodeAttribute(draggedNode, 'y', gpos.y);
    var d3n = d3nodeById[draggedNode];
    if (d3n) { d3n.fx = gpos.x; d3n.fy = gpos.y; d3n.vx = 0; d3n.vy = 0; }
    // sigmaのカメラパンを抑制
    ev.preventSigmaDefault();
    ev.original.preventDefault();
    ev.original.stopPropagation();
    renderer.refresh();
    if (layoutDone) {
      layoutDone = false;
      simulation.alpha(0.2).restart();
      requestAnimationFrame(layoutStep);
    }
  } else if (isPanning) {
    // パン中: カメラ速度を追跡（イナーシャ用）
    var s = camera.getState(), a = 0.45;
    velCamX = velCamX * (1-a) + (s.x - prevCamX) * a;
    velCamY = velCamY * (1-a) + (s.y - prevCamY) * a;
    prevCamX = s.x; prevCamY = s.y;
  }
});

mc.on('mouseup', function () {
  if (draggedNode) {
    // ノードドラッグ終了
    var d3n = d3nodeById[draggedNode];
    if (d3n) { d3n.fx = null; d3n.fy = null; }
    draggedNode = null;
    return;
  }
  if (!isPanning) return;
  isPanning = false;
  if (Math.abs(velCamX) + Math.abs(velCamY) < 0.00005) return;
  (function glide() {
    velCamX *= 0.93; velCamY *= 0.93;
    if (Math.abs(velCamX) + Math.abs(velCamY) < 0.00003) { inertiaId = null; return; }
    var s = camera.getState();
    camera.setState({ x: s.x + velCamX, y: s.y + velCamY });
    inertiaId = requestAnimationFrame(glide);
  })();
});

// window mouseup: ブラウザ外でボタンを離した場合のフォールバック
window.addEventListener('mouseup', function () {
  if (draggedNode) {
    var d3n = d3nodeById[draggedNode];
    if (d3n) { d3n.fx = null; d3n.fy = null; }
    draggedNode = null;
  }
});

/* ── 7. D3-force layout ──────────────────────────────────────
   If the Python backend pre-computed positions (FA2), we skip the initial
   layout entirely and just build the D3 node/link arrays for drag support
   and the expand/collapse reheat.
   Otherwise we fall back to progressive batch loading. */

var gup = 8;
var chunkSz = 3;  // chunk node size — kept small so rings stay compact

var nNodes = GRAPH_DATA.nodes.length;
// ノードサイズに比例した斥力（大きいノードほど周囲を広く押しのける）
var chargeBase = nNodes < 100  ? -1200 :
                 nNodes < 500  ? -800  :
                 nNodes < 2000 ? -500  : -280;
function chargeStr(d) {
  var sz = 5;
  try { sz = graph.getNodeAttribute(d.id, 'size') || 5; } catch (_) {}
  // size は 1〜8 の範囲。基準サイズ 2.5 を 1.0 として二乗比例
  return chargeBase * (sz / 2.5) * (sz / 2.5);
}
var linkDist  = nNodes < 100  ?  300  :
                nNodes < 500  ?  260  :
                nNodes < 2000 ?  170  : 110;

// ── D3 node/link arrays (used for drag + expand/collapse reheat) ──
var d3nodes = [], d3nodeById = {};
var activeNodeIds = new Set();

function _addD3Nodes(ids) {
  ids.forEach(function (id) {
    if (d3nodeById[id]) return;
    var attrs = graph.getNodeAttributes(id);
    var n = { id: id, x: attrs.x, y: attrs.y };
    d3nodes.push(n); d3nodeById[id] = n; activeNodeIds.add(id);
  });
}

var d3links = [];
function _rebuildLinks() {
  d3links = [];
  graph.forEachEdge(function (e, attrs, src, tgt) {
    if (activeNodeIds.has(src) && activeNodeIds.has(tgt))
      d3links.push({ source: src, target: tgt });
  });
}

var simulation = d3.forceSimulation([])
  .force('charge', d3.forceManyBody().strength(chargeStr).theta(0.8))
  .force('link',   d3.forceLink([]).id(function (d) { return d.id; }).distance(linkDist))
  .force('collide', d3.forceCollide()
    .radius(function (d) {
      var sz = 5;
      try { sz = graph.getNodeAttribute(d.id, 'size') || 5; } catch (_) {}
      return sz * gup * 1.6;
    })
    .strength(0.85).iterations(3))
  .force('x', d3.forceX(0).strength(0.05))
  .force('y', d3.forceY(0).strength(0.05))
  .velocityDecay(0.45)
  .alphaDecay(0.012)
  .stop();

var badge    = document.getElementById('layout-badge');
var badgePct = document.getElementById('layout-pct');
var layoutDone = false;

function finishLayout() {
  if (layoutDone) return;
  layoutDone = true;
  badge.style.display = 'none';
  console.log('[RAG] layout ready');
  // 90% 未満で完了した場合（alphaMin到達）や最終フィット未実施ならここでフィット
  if (!_finalFitDone && !_layoutUserInteracted) _fitView(true);
  else renderer.refresh();
}

document.getElementById('layout-skip').addEventListener('click', finishLayout);

// ── レイアウト中のカメラフィット ──────────────────────────────────────────
// ユーザーがスクロール・パンした場合は自動フィットをスキップする。
var _layoutUserInteracted = false;
var _layoutFitting = false;   // 自分のコードによるカメラ変更中フラグ

// camera の updated イベント: 自分のコード以外によるカメラ変更を検出
camera.on('updated', function() {
  if (!_layoutFitting && !layoutDone) _layoutUserInteracted = true;
});

function _fitView(animated) {
  var minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  graph.forEachNode(function(id, attrs) {
    if (attrs.x != null) {
      if (attrs.x < minX) minX = attrs.x; if (attrs.x > maxX) maxX = attrs.x;
      if (attrs.y < minY) minY = attrs.y; if (attrs.y > maxY) maxY = attrs.y;
    }
  });
  if (!isFinite(minX)) return;
  var cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  var dim = renderer.getDimensions();
  var S = Math.min(dim.width, dim.height), ratio2 = 2 * bboxR;
  var newRatio = Math.max(
    (maxX - minX) * S / (0.88 * ratio2 * dim.width),
    (maxY - minY) * S / (0.88 * ratio2 * dim.height)
  );
  newRatio = Math.max(0.1, Math.min(newRatio, 5));
  var target = { x: cx / ratio2 + 0.5, y: cy / ratio2 + 0.5, ratio: newRatio };
  _layoutFitting = true;
  if (animated) {
    camera.animate(target, { duration: 600, easing: 'quadraticInOut' });
    setTimeout(function() { _layoutFitting = false; }, 650);
  } else {
    camera.setState(target);
    _layoutFitting = false;
  }
}

// 起動から 0.5 秒後に一度だけ即時フィット
setTimeout(function() { if (!layoutDone && !_layoutUserInteracted) _fitView(false); }, 500);

var _finalFitDone = false;

function layoutStep() {
  if (layoutDone) return;
  var tpf = nNodes < 500 ? 1 : nNodes < 2000 ? 2 : nNodes < 5000 ? 3 : 5;
  for (var i = 0; i < tpf; i++) simulation.tick();
  d3nodes.forEach(function (n) {
    if (isFinite(n.x) && isFinite(n.y)) {
      graph.setNodeAttribute(n.id, 'x', n.x);
      graph.setNodeAttribute(n.id, 'y', n.y);
    }
  });
  renderer.refresh();
  var pct = Math.round((1 - simulation.alpha()) * 100);
  badgePct.textContent = pct + '%';
  // 90% 到達時に最終フィット（ユーザー操作なしの場合のみ）
  if (!_finalFitDone && pct >= 80 && !_layoutUserInteracted) {
    _finalFitDone = true;
    _fitView(true);
  }
  if (simulation.alpha() <= simulation.alphaMin()) { finishLayout(); return; }
  requestAnimationFrame(layoutStep);
}

if (_hasPrecomputedLayout) {
  // Pre-computed positions give a good starting point.
  // Run the simulation from there — nodes are already near equilibrium so
  // it converges in a few seconds, giving a visible but brief settling animation.
  graph.forEachNode(function (id) { _addD3Nodes([id]); });
  _rebuildLinks();
  simulation.nodes(d3nodes);
  simulation.force('link').links(d3links);
  simulation.alpha(0.5);   // start from pre-computed → fast convergence
  badge.style.display = 'block';
  requestAnimationFrame(layoutStep);
  console.log('[RAG] pre-computed layout, running simulation from good start, nodes=' + d3nodes.length);
} else {
  // Fallback: progressive batch simulation.
  var _allNodes = { zotero: [], external: [], reference: [] };
  graph.forEachNode(function (id, attrs) {
    var g = attrs.group || 'external';
    if (_allNodes[g]) _allNodes[g].push(id); else _allNodes.external.push(id);
  });
  var BATCH_SIZE = 400, BATCH_TICKS = 40;
  var _batchQueue = [];
  var _arr1 = _allNodes.external, _arr2 = _allNodes.reference;
  for (var _bi = 0; _bi < _arr1.length; _bi += BATCH_SIZE)
    _batchQueue.push({ ids: _arr1.slice(_bi, _bi + BATCH_SIZE), phase: 'citers' });
  for (var _bi2 = 0; _bi2 < _arr2.length; _bi2 += BATCH_SIZE)
    _batchQueue.push({ ids: _arr2.slice(_bi2, _bi2 + BATCH_SIZE), phase: 'refs' });

  _addD3Nodes(_allNodes.zotero);
  _rebuildLinks();
  simulation.nodes(d3nodes);
  simulation.force('link').links(d3links);

  badge.style.display = 'block';
  var _tickCount = 0, _nextBatchAt = BATCH_TICKS * 2;
  var _origLayoutStep = layoutStep;
  layoutStep = function () {
    if (layoutDone) return;
    var tpf = nNodes < 500 ? 1 : nNodes < 2000 ? 2 : nNodes < 5000 ? 3 : 5;
    for (var i = 0; i < tpf; i++) { simulation.tick(); _tickCount++; }
    if (_batchQueue.length > 0 && _tickCount >= _nextBatchAt) {
      var batch = _batchQueue.shift();
      _addD3Nodes(batch.ids);
      _rebuildLinks();
      simulation.nodes(d3nodes); simulation.force('link').links(d3links);
      simulation.alpha(Math.max(simulation.alpha(), 0.25));
      _nextBatchAt = _tickCount + BATCH_TICKS;
    }
    d3nodes.forEach(function (n) {
      if (isFinite(n.x) && isFinite(n.y)) {
        graph.setNodeAttribute(n.id, 'x', n.x);
        graph.setNodeAttribute(n.id, 'y', n.y);
      }
    });
    renderer.refresh();
    if (_batchQueue.length > 0) {
      badgePct.textContent = Math.round(d3nodes.length / nNodes * 100) + '%';
    } else {
      badgePct.textContent = Math.round((1 - simulation.alpha()) * 100) + '%';
    }
    if (_batchQueue.length === 0 && simulation.alpha() <= simulation.alphaMin()) {
      finishLayout(); return;
    }
    requestAnimationFrame(layoutStep);
  };
  requestAnimationFrame(layoutStep);
  console.log('[RAG] fallback progressive layout, batches=' + _batchQueue.length);
}

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
  graph.setNodeAttribute(nodeId, 'color', THEME.nodeZoteroExpanded);

  // Ring radius: adjacent chunks just touch at their collision boundary.
  // collision radius = chunkSz * gup * 1.6 (same formula as forceCollide)
  // For n chunks in a ring:  r = n * collisionDiameter / (2π) = n * chunkSz * gup * 1.6 / π
  // minR: ring must be larger than the parent node + one chunk radius so chunks sit outside.
  var parentSz = attrs.size || 10;
  var minR = parentSz * gup + chunkSz * gup * 2;  // parent radius + 1 chunk diameter margin
  r = Math.max(chunkIds.length * chunkSz * gup * 1.6 / Math.PI, minR);

  // ── debug: expose position info ──────────────────────────────────────────
  var allXY = [];
  graph.forEachNode(function(n, a) { if (!n.startsWith('chunk:')) allXY.push({id:n.slice(0,20), x:Math.round(a.x), y:Math.round(a.y)}); });
  var dists = allXY.map(function(a) { var dx=a.x-pos.x, dy=a.y-pos.y; return Math.sqrt(dx*dx+dy*dy); });
  var avgDist = dists.length ? dists.reduce(function(s,d){return s+d;},0)/dists.length : 0;
  console.log('[expand] nodeId:', nodeId, '| pos:', JSON.stringify(pos), '| chunkCount:', chunkIds.length, '| computedR:', Math.round(r), '| avgNeighborDist:', Math.round(avgDist), '| gup:', gup);
  window.__lastExpand = { nodeId: nodeId, pos: pos, r: Math.round(r), chunkCount: chunkIds.length, avgDist: Math.round(avgDist), bboxR: bboxR };
  // ─────────────────────────────────────────────────────────────────────────

  var step = (2*Math.PI) / Math.max(chunkIds.length, 1);
  var addedNodes = [], addedEdges = [];

  chunkIds.forEach(function (cid, i) {
    var angle = step*i - Math.PI/2;
    var cnid  = 'chunk:' + cid;
    if (graph.hasNode(cnid)) return;
    var cx = pos.x + r*Math.cos(angle);
    var cy = pos.y + r*Math.sin(angle);
    graph.addNode(cnid, {
      x: cx, y: cy,
      size: chunkSz, color: THEME.nodeChunk,
      label: chunkLabel(cid), tooltip: chunkTooltip(cid), group: 'chunk',
    });
    addedNodes.push(cnid);
    var seid = graph.addEdge(nodeId, cnid, {
      size: 0.5, color: THEME.edgeStructural, type: 'line', _structural: true,
    });
    addedEdges.push(seid);
  });

  data.in_edges.forEach(function (e) {
    if (!graph.hasNode(e[0]) || !graph.hasNode('chunk:'+e[1])) return;
    try { addedEdges.push(graph.addEdge(e[0], 'chunk:'+e[1], { size:e[2], color:THEME.edgeDefault, type:'arrow' })); } catch(_){}
  });
  data.out_edges.forEach(function (e) {
    if (!graph.hasNode('chunk:'+e[0]) || !graph.hasNode(e[1])) return;
    try { addedEdges.push(graph.addEdge('chunk:'+e[0], e[1], { size:e[2], color:THEME.edgeDefault, type:'arrow' })); } catch(_){}
  });

  expandedItems.add(itemKey);
  expandedState.set(itemKey, { addedNodes:addedNodes, addedEdges:addedEdges,
                               hiddenEdges:hiddenEdges, origColor:origColor,
                               ringR: r });
  // If this node is currently selected, include the new chunk nodes in the selection
  if (selectedNode === nodeId) { recomputeSelection(nodeId); renderer.refresh(); }

  // ── Pin the parent node so the ring stays centered on it during reheat ──
  var parentD3 = d3nodeById[nodeId];
  if (parentD3) { parentD3.fx = parentD3.x; parentD3.fy = parentD3.y; }

  // ── Add chunk nodes to D3 simulation with fixed positions ──────────────
  // Chunks are pinned (fx/fy) so charge/collide cannot scatter them.
  var newD3Nodes = [];
  addedNodes.forEach(function(cnid) {
    if (!d3nodeById[cnid] && graph.hasNode(cnid)) {
      var a = graph.getNodeAttributes(cnid);
      var dn = { id: cnid, x: a.x, y: a.y, fx: a.x, fy: a.y };  // pinned
      d3nodes.push(dn); d3nodeById[cnid] = dn;
      newD3Nodes.push(dn);
    }
  });
  if (newD3Nodes.length > 0) {
    d3links = [];
    graph.forEachEdge(function(edge, attrs, src, tgt) {
      if (graph.hasNode(src) && graph.hasNode(tgt)) {
        d3links.push({ source: src, target: tgt });
      }
    });
    simulation.nodes(d3nodes);
    simulation.force('link').links(d3links);
    // ── Ring exclusion force: push non-chunk nodes outside the ring ──────
    // This runs during reheat and keeps other nodes from drifting into
    // the expanded ring area.
    _updateRingExclusionForce();
    // Gentle reheat so citers/refs can adjust without major layout change.
    console.log('[expand] firing reheat, newD3Nodes=' + newD3Nodes.length + ' layoutDone-before=' + layoutDone);
    layoutDone = false;
    simulation.alpha(0.3);
    badge.style.display = 'block';
    requestAnimationFrame(layoutStep);
    console.log('[expand] after RAF, layoutDone=' + layoutDone + ' alpha=' + simulation.alpha().toFixed(3));
  }
}

// ── Ring exclusion force ────────────────────────────────────────────────────
// Keeps non-chunk nodes outside all active ring radii.
function _updateRingExclusionForce() {
  simulation.force('ringExclude', function(alpha) {
    // Build list of active rings from expandedState
    expandedState.forEach(function(st, ikey) {
      var parentId = 'item:' + ikey;
      if (!d3nodeById[parentId]) return;
      var pd = d3nodeById[parentId];
      var ringR = st.ringR || 0;
      var buffer = chunkSz * gup * 1.6;  // one collision radius as buffer
      var minDist = ringR + buffer;
      d3nodes.forEach(function(nd) {
        if (nd.id === parentId) return;          // skip parent itself
        if (nd.id && nd.id.startsWith('chunk:')) return;  // skip chunks
        if (nd.fx != null) return;               // skip other pinned nodes
        var dx = nd.x - pd.x, dy = nd.y - pd.y;
        var dist = Math.sqrt(dx*dx + dy*dy) || 1;
        if (dist < minDist) {
          // Push the node radially outward past the ring
          var force = (minDist - dist) / dist * alpha * 8;
          nd.vx += dx * force;
          nd.vy += dy * force;
        }
      });
    });
  });
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
  // Unpin the parent node so it can drift back to equilibrium after collapse.
  var parentD3 = d3nodeById[nodeId];
  if (parentD3) { delete parentD3.fx; delete parentD3.fy; }
  expandedItems.delete(itemKey); expandedState.delete(itemKey);
  // If this node is selected, refresh selection now that chunks are gone
  if (selectedNode === nodeId) { recomputeSelection(nodeId); renderer.refresh(); }
  // Rebuild D3 links after node removal and do a short reheat
  d3links = [];
  graph.forEachEdge(function(edge, attrs, src, tgt) {
    if (graph.hasNode(src) && graph.hasNode(tgt)) {
      d3links.push({ source: src, target: tgt });
    }
  });
  simulation.nodes(d3nodes);
  simulation.force('link').links(d3links);
  // Update ring exclusion force (removes this ring; keeps others if any).
  // If no rings remain, remove the force entirely so it doesn't waste CPU.
  if (expandedState.size === 0) {
    simulation.force('ringExclude', null);
  } else {
    _updateRingExclusionForce();
  }
  // Reheat so nodes around the collapsed item can drift back into place.
  layoutDone = false;
  simulation.alpha(0.3);
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

// ── CC filter inputs ──────────────────────────────────────────────────────────
(function() {
  function _applyFilter() {
    filterCiterCC = parseInt(document.getElementById('filter-citer-cc').value) || 0;
    filterRefCC   = parseInt(document.getElementById('filter-ref-cc').value)   || 0;
    renderer.refresh();

    // 表示数カウント更新（被引用元・参照先それぞれ）
    var visCiter = 0, visRef = 0, visTotal = 0;
    graph.forEachNode(function(n, a) {
      if (a.group === 'zotero' || a.group === 'chunk') { visTotal++; return; }
      if (a.group === 'external') {
        if (filterCiterCC > 0 && (a.cc||0) < filterCiterCC) return;
        visCiter++; visTotal++;
      } else if (a.group === 'reference') {
        if (filterRefCC > 0 && (a.cc||0) < filterRefCC) return;
        visRef++; visTotal++;
      }
    });
    var ec = document.getElementById('stat-citer-vis');
    var er = document.getElementById('stat-ref-vis');
    var en = document.getElementById('stat-nodes');
    if (ec) ec.textContent = visCiter;
    if (er) er.textContent = visRef;
    if (en) en.textContent = visTotal;

    // フィルタ変更後に物理演算をリスタートして再レイアウト
    if (layoutDone) {
      layoutDone = false;
      badge.style.display = 'block';
      badgePct.textContent = '0%';
      simulation.alpha(0.4).restart();
      requestAnimationFrame(layoutStep);
    } else {
      simulation.alpha(Math.max(simulation.alpha(), 0.4));
    }
  }
  ['filter-citer-cc', 'filter-ref-cc'].forEach(function(id) {
    var el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('change', _applyFilter);
    el.addEventListener('keydown', function(e) { if (e.key === 'Enter') _applyFilter(); });
  });

  // 起動時にフィルターを適用（input の初期値を変数に反映して描画を更新）
  _applyFilter();
})();

/* ── 8. Sidebar: 資料一覧 ────────────────────────────────── */
(function() {
  var sbNodes = GRAPH_DATA.nodes.filter(function(n) {
    return n.group === 'zotero' || n.group === 'external' || n.group === 'reference';
  });
  var sortCol = 'citations';
  var sortDir = -1;   // -1=降順, 1=昇順
  var filterQ = '';
  var filterZoteroOnly = false;
  var sbActiveId = null;

  var toggle       = document.getElementById('sb-toggle');
  var sbCountEl    = document.getElementById('sb-count');
  var searchEl     = document.getElementById('sb-search');
  var zoteroOnlyEl = document.getElementById('sb-zotero-only');
  var tbody        = document.getElementById('sb-list-body');
  var detail       = document.getElementById('sb-detail');

  // ── トグル ──────────────────────────────────────────────
  toggle.addEventListener('click', function() {
    document.body.classList.toggle('sb-collapsed');
    toggle.textContent = document.body.classList.contains('sb-collapsed') ? '‹' : '›';
    setTimeout(function() { renderer.refresh(); }, 260);
  });

  // ── 列ヘッダーでソート ────────────────────────────────
  document.querySelectorAll('#sb-list thead th').forEach(function(th) {
    th.addEventListener('click', function() {
      var col = th.dataset.col;
      if (sortCol === col) { sortDir *= -1; }
      else { sortCol = col; sortDir = (col === 'title') ? 1 : -1; }
      document.querySelectorAll('#sb-list thead th').forEach(function(t) { t.className = ''; });
      th.className = sortDir === 1 ? 'sort-asc' : 'sort-desc';
      renderList();
    });
  });

  // ── 絞り込み ─────────────────────────────────────────
  searchEl.addEventListener('input', function() {
    filterQ = searchEl.value.toLowerCase();
    renderList();
  });
  zoteroOnlyEl.addEventListener('change', function() {
    filterZoteroOnly = zoteroOnlyEl.checked;
    renderList();
  });

  // ── ソートキー取得 ────────────────────────────────────
  function getVal(n, col) {
    if (col === 'title')     return (n.fullTitle || n.label || '').toLowerCase();
    if (col === 'year')      return parseInt(n.year) || 0;
    if (col === 'citations') return n.citations != null ? n.citations : (n.cc || 0);
    if (col === 'refCount')  return n.refCount  || 0;
    if (col === 'inZotero')  return n.group === 'zotero' ? 1 : 0;
    return '';
  }

  // ── HTMLエスケープ ────────────────────────────────────
  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  // ── リスト描画 ────────────────────────────────────────
  function renderList() {
    var filtered = sbNodes.filter(function(n) {
      if (filterZoteroOnly && n.group !== 'zotero') return false;
      if (!filterQ) return true;
      return (n.fullTitle || n.label || '').toLowerCase().indexOf(filterQ) !== -1 ||
             (n.authors   || '').toLowerCase().indexOf(filterQ) !== -1;
    });
    filtered.sort(function(a, b) {
      var av = getVal(a, sortCol), bv = getVal(b, sortCol);
      if (typeof av === 'string') return av < bv ? -sortDir : av > bv ? sortDir : 0;
      return (av - bv) * sortDir;
    });

    sbCountEl.textContent = '(' + filtered.length + '件)';
    tbody.innerHTML = '';
    filtered.forEach(function(n) {
      var tr = document.createElement('tr');
      if (n.id === sbActiveId) tr.className = 'sb-active';
      var titleStr = n.fullTitle || n.label || '';
      var shortTitle = titleStr.length > 34 ? titleStr.slice(0, 34) + '…' : titleStr;
      var citVal = n.citations != null ? n.citations : (n.cc != null ? n.cc : null);
      var inZ = n.group === 'zotero';
      tr.innerHTML =
        '<td style="max-width:145px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + esc(shortTitle) + '</td>' +
        '<td class="sb-num">' + esc(n.year || '—') + '</td>' +
        '<td class="sb-num">' + (citVal != null ? Number(citVal).toLocaleString() : '—') + '</td>' +
        '<td class="sb-num">' + (n.refCount ? Number(n.refCount).toLocaleString() : '—') + '</td>' +
        '<td class="sb-num" style="color:' + (inZ ? 'var(--node-zotero)' : 'var(--on-surface-variant)') + '">' + (inZ ? '✓' : '—') + '</td>';
      tr.addEventListener('click', function() { selectFromSidebar(n.id); });
      tbody.appendChild(tr);
    });
  }

  // ── サイドバーからノード選択 ──────────────────────────
  function selectFromSidebar(nodeId) {
    sbActiveId = nodeId;
    recomputeSelection(nodeId);
    renderer.refresh();
    renderList();
    showDetail(nodeId);
    window._panToNode(nodeId);
    // アクティブ行をスクロールで表示
    setTimeout(function() {
      var activeRow = tbody.querySelector('tr.sb-active');
      if (activeRow) activeRow.scrollIntoView({ block: 'nearest' });
    }, 0);
  }

  // ── カメラをノードに向ける ────────────────────────────
  // viewportToGraph を2点で使い、「フレーム座標/グラフ座標」の正規化係数を逆算。
  // これでグラフ座標→フレーム座標の変換ができ、カメラ中心に設定する。
  // ── ノーマライゼーション係数キャッシュ ─────────────────
  // sigma のカメラ座標（framed unit）と graph 座標の変換係数を一度だけキャッシュする。
  // camera.setState 後に viewportToGraph が信頼できなくなるため、
  // カメラが初期位置(0.5,0.5)にある時点で計算する。
  //
  // sigma の framed 座標は「ビューポート幅を1とした割合」単位。
  //   viewportToGraph(center+100px) の差分 = scalePixel = graph/px
  //   normRatio = scalePixel × dim.width = graph / framed_unit = 2 × bboxR
  // sigma の normalizationFunction: apply(gx) = gx/(2×bboxR) + 0.5
  // bboxR はメインスコープで定義済み（setCustomBBox に渡した値）。
  // viewportToGraph に依存しないため、カメラ移動後でも正確。

  // sigma の normalizationFunction: apply(gx) = gx/(2×bboxR) + 0.5
  // bboxR はメインスコープで定義済み（setCustomBBox に渡した値）。
  // viewportToGraph に依存しないため、カメラ移動後でも正確。
  function _graphToFramed(gx, gy) {
    var ratio2 = 2 * bboxR;
    return {
      x: gx / ratio2 + 0.5,
      y: gy / ratio2 + 0.5,
    };
  }

  window._panToNode = function panToNode(nodeId) {
    try {
      var nx = graph.getNodeAttribute(nodeId, 'x');
      var ny = graph.getNodeAttribute(nodeId, 'y');
      if (nx == null || ny == null) { console.warn('[panToNode] no position'); return; }

      // 選択ノードを中心に固定し、表示中の隣接ノードの最大距離でズームを決める。
      // CCフィルターで非表示になっているノードは除外する。
      function _isVisible(n) {
        var attrs = graph.getNodeAttributes(n);
        if (attrs.group === 'external'  && filterCiterCC > 0 && (attrs.cc || 0) < filterCiterCC) return false;
        if (attrs.group === 'reference' && filterRefCC   > 0 && (attrs.cc || 0) < filterRefCC)   return false;
        return true;
      }

      // 選択ノード中心からの最大距離（X/Y 別）を収集
      var halfW = 0, halfH = 0;
      function _updateHalf(n) {
        if (!_isVisible(n)) return;
        var x = graph.getNodeAttribute(n, 'x'), y = graph.getNodeAttribute(n, 'y');
        if (x != null) halfW = Math.max(halfW, Math.abs(x - nx));
        if (y != null) halfH = Math.max(halfH, Math.abs(y - ny));
      }
      selectedNeighbors.forEach(_updateHalf);
      selectedChunkNeighbors.forEach(_updateHalf);

      // 選択ノードを画面中心へ
      var framed = _graphToFramed(nx, ny);

      // ratio 計算: 選択ノード中心で、隣接ノードが画面 80% に収まるよう
      // visible_graph_half_X = bboxR * dim.width * ratio / min(dim.width, dim.height)
      // → ratio = halfW * min / (0.8 * bboxR * dim.width) × 0.5  [half なので ×0.5 不要、full span = 2*half]
      var dim = renderer.getDimensions();
      var S   = Math.min(dim.width, dim.height);
      var minSpan = bboxR * 0.03;  // 隣接なし時の最小表示範囲
      var spanW = Math.max(halfW * 2, minSpan);
      var spanH = Math.max(halfH * 2, minSpan);
      var newRatio = Math.max(
        spanW * S / (0.8 * 2 * bboxR * dim.width),
        spanH * S / (0.8 * 2 * bboxR * dim.height)
      );
      newRatio = Math.max(0.03, Math.min(newRatio, 2));

      console.log('[panToNode] center=(%f,%f) half=(%f,%f) framed=(%f,%f) ratio=%f',
        nx, ny, halfW, halfH, framed.x, framed.y, newRatio);

      camera.animate({ x: framed.x, y: framed.y, ratio: newRatio },
                     { duration: 400, easing: 'quadraticInOut' });
    } catch(e) { console.error('[panToNode] error:', e); }
  }

  // ── 詳細パネル表示 ────────────────────────────────────
  function showDetail(nodeId) {
    var n = GRAPH_DATA.nodes.find(function(nd) { return nd.id === nodeId; });
    if (!n) { detail.className = ''; detail.innerHTML = ''; return; }
    var doi  = n.doi  || '';
    var isbn = n.isbn || '';
    var doiHtml  = doi
      ? '<a href="https://doi.org/' + esc(doi) + '" target="_blank">' + esc(doi) + '</a>'
      : '—';
    var isbnHtml = isbn
      ? '<a href="https://openlibrary.org/isbn/' + esc(isbn) + '" target="_blank">' + esc(isbn) + '</a>'
      : '—';
    detail.className = 'sb-active';
    detail.innerHTML =
      '<div id="sb-detail-title">' + esc(n.fullTitle || n.label) + '</div>' +
      kv('著者',   n.authors || '—') +
      kv('年',     n.year    || '—') +
      kvH('DOI',  doiHtml) +
      kvH('ISBN', isbnHtml) +
      kv('被引用数', n.citations != null ? Number(n.citations).toLocaleString() : '—') +
      kv('参照数',   n.refCount  ? Number(n.refCount).toLocaleString() : '—') +
      kv('Key',   n.itemKey || '');
  }
  function kv(k, v) {
    return '<div class="sb-kv"><span class="sb-k">' + esc(k) + '</span><span class="sb-v">' + esc(v) + '</span></div>';
  }
  function kvH(k, vHtml) {
    return '<div class="sb-kv"><span class="sb-k">' + esc(k) + '</span><span class="sb-v">' + vHtml + '</span></div>';
  }

  // ── グラフクリックとサイドバーを同期 ─────────────────
  renderer.on('clickNode', function(ev) {
    if (hasDragged) return;
    if (ev.node && graph.getNodeAttribute(ev.node, 'group') === 'zotero') {
      // 主ハンドラー実行後、selectedNode が更新されてから動く
      setTimeout(function() {
        if (selectedNode === ev.node) {
          sbActiveId = ev.node;
          showDetail(ev.node);
        } else {
          sbActiveId = null;
          detail.className = '';
          detail.innerHTML = '';
        }
        renderList();
      }, 0);
    }
  });
  renderer.on('clickStage', function() {
    sbActiveId = null;
    detail.className = '';
    detail.innerHTML = '';
    renderList();
  });

  // 初期ソート表示（引用数降順）
  document.querySelector('#sb-list thead th[data-col="citations"]').className = 'sort-desc';
  renderList();
})();

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

  <!-- Sidebar toggle button -->
  <button id="sb-toggle" title="資料一覧を表示/非表示">›</button>

  <!-- Sidebar: 資料一覧 + 詳細 -->
  <div id="sidebar">
    <div id="sb-header">
      <span>資料一覧</span>
      <span id="sb-count"></span>
    </div>
    <div style="padding:6px 10px 8px;flex-shrink:0;border-bottom:1px solid var(--outline-variant)">
      <input id="sb-search" type="text" placeholder="タイトル・著者で絞り込み…">
      <label style="display:flex;align-items:center;gap:5px;margin-top:5px;font-size:11.5px;color:var(--on-surface-variant);cursor:pointer">
        <input id="sb-zotero-only" type="checkbox"> Zotero所収のみ
      </label>
    </div>
    <div id="sb-list-wrap">
      <table id="sb-list">
        <thead>
          <tr>
            <th data-col="title">タイトル</th>
            <th data-col="year">年</th>
            <th data-col="citations">引用</th>
            <th data-col="refCount">参照</th>
            <th data-col="inZotero" title="Zotero所収">Z</th>
          </tr>
        </thead>
        <tbody id="sb-list-body"></tbody>
      </table>
    </div>
    <div id="sb-detail"></div>
  </div>

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
    item_ref_counts: dict[str, int] | None = None,
    chunk_data: dict | None = None,
) -> None:
    import json as _json
    import re  as _re

    meta       = item_meta or {}
    item_rcnt  = item_ref_counts or {}
    chunk_data = chunk_data or {}

    # ── Color palette (single source of truth) ──────────────────────────────
    # Edit values here.  CSS :root variables and the JS THEME object are
    # generated automatically from this dict — nothing else needs changing.
    #
    # Surface/text/outline values follow the M3 dark scheme spec exactly.
    # Node/edge colors use M3 tonal-palette tone-80 (the recommended primary
    # tone for dark backgrounds) for each hue family.
    PALETTE: dict[str, str] = {
        # ── Visualization node colors ────────────────────────────────────────
        # Dark-background data-viz palette: high chroma at medium lightness
        # so colors pop clearly against #141218.
        # Zotero + Chunk share the same blue hue (different brightness).
        # Citer (warm amber) and Reference (vivid rose) are strongly distinct.
        "nodeZotero":         "#C97090",   # Rose  hsl(338,44%,62%) — Zotero items
        "nodeExternal":       "#A0A0A0",   # Grey — unified external (citer+ref) normal state
        "nodeCiter":          "#DFA040",   # Amber hsl(38,72%,56%)  — citer highlight (selection)
        "nodeRef":            "#7498DC",   # Blue  hsl(220,60%,65%) — ref highlight (selection)
        "nodeChunk":          "#E8A0B0",   # Rose lighter — chunks (same hue, lighter)
        "nodeZoteroExpanded": "#E8C0CC",   # Rose lightest — expanded
        "nodeUnknown":        "#8E8A98",   # Neutral muted
        "nodeDim":            "#2B2930",   # surface-container-high — non-selected fade
        # ── Visualization edge colors ────────────────────────────────────────
        "edgeDefault":        "#707070",   # Grey — unified edge normal state
        "edgeCitation":       "#C8882A",   # Amber-brown — citer edge (selection highlight)
        "edgeReference":      "#7498DC",   # Blue  — ref edge (selection highlight)
        "edgeStructural":     "rgba(201,112,144,0.12)",  # Rose @ 12 %
        # ── M3 dark surface system (spec values) ────────────────────────────
        # surface          = Neutral tone  6  (#141218)
        # surface-container-low  = Neutral tone 10  (#1D1B20) — legend / badges
        # surface-container-high = Neutral tone 17  (#2B2930) — tooltip
        # outline-variant  = Neutral-Variant tone 30 (#49454F) — dividers
        "surface":                "#141218",
        "surfaceContainerLow":    "#1D1B20",
        "surfaceContainerHigh":   "#2B2930",
        "outlineVariant":         "#49454F",
        # ── M3 dark text/icon roles ──────────────────────────────────────────
        # on-surface         = Neutral tone 90 (#E6E0E9) — high emphasis
        # on-surface-variant = Neutral-Variant tone 80 (#CAC4D0) — medium
        "onSurface":          "#E6E0E9",
        "onSurfaceVariant":   "#CAC4D0",
        "textDis":            "rgba(230,224,233,0.38)",  # disabled / de-emphasized
    }

    def _css_var(key: str) -> str:
        """camelCase → CSS custom-property name: nodeZotero → --node-zotero"""
        return "--" + _re.sub(r'([A-Z])', lambda m: "-" + m.group(1).lower(), key)

    # CSS :root block injected into <style>
    _css_root = ":root {\n" + "\n".join(
        f"  {_css_var(k)}: {v};"
        for k, v in PALETTE.items()
    ) + "\n}"

    # JS THEME constant prepended to the logic script
    _js_theme = "const THEME = " + _json.dumps(PALETTE, indent=2) + ";"

    # Convenience aliases used in graph data generation below
    C_ZOTERO    = PALETTE["nodeZotero"]
    C_EXTERNAL  = PALETTE["nodeExternal"]   # 通常時の統一外部ノード色
    C_CITER     = PALETTE["nodeCiter"]      # 選択時ハイライト用（後方互換）
    C_REF       = PALETTE["nodeRef"]        # 選択時ハイライト用（後方互換）
    C_UNK       = PALETTE["nodeUnknown"]

    # ── Server-side layout (FA2 + sector placement) ─────────────────────────
    import sys as _sys, time as _time
    _t0 = _time.time()
    _print = lambda *a, **kw: print(*a, file=_sys.stderr, **kw)
    _print("Computing layout (FA2)…", end=" ", flush=True)
    item_keys_for_layout = [d["item_key"] for d in items]
    layout_positions = compute_layout(item_keys_for_layout, citers, refs)
    _print(f"done in {_time.time()-_t0:.1f}s  ({len(layout_positions)} nodes placed)")

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
        # Year: S2 データ優先、なければ ChromaDB の year
        year_val = str(d.get("s2_year") or m.get("year") or "")
        # S2 総被引用数（s2_citation_count）が取得済みならそちらを表示
        s2_cc    = d.get("s2_citation_count")
        rcount   = item_rcnt.get(key, 0)
        color    = C_ZOTERO if status in ("mapped", "s2_done") else C_UNK

        doi_val  = d.get("doi")  or ""
        isbn_val = d.get("isbn") or ""

        extra = [("Key", key)]
        if creators:
            extra.append(("Authors", creators))
        extra += [
            ("Year",       year_val or "—"),
            ("DOI",        doi_val  or "—"),
            ("ISBN",       isbn_val or "—"),
            ("Citations",  f"{s2_cc:,}" if s2_cc is not None else (f"{count:,}" if count else "—")),
            ("References", f"{rcount:,}" if rcount else "—"),
            ("Contexts",   f"{d['context_count']:,}"),
            ("Status",     status),
        ]

        _nid = f"item:{key}"
        _xy  = layout_positions.get(_nid)
        nodes.append({
            "id":        _nid,
            "label":     _short(full, 28) if full else key,
            "size":      _node_size(count),
            "color":     color,
            "tooltip":   _tooltip(full or key, extra),
            "group":     "zotero",
            # Sidebar list/detail 用の構造化データ
            "fullTitle": full or key,
            "authors":   creators or "",
            "year":      year_val or "",
            "doi":       doi_val or "",
            "isbn":      isbn_val or "",
            "citations": s2_cc if s2_cc is not None else count,
            "refCount":  rcount,
            "itemKey":   key,
            **( {"x": round(_xy[0], 1), "y": round(_xy[1], 1)} if _xy else {} ),
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
            _xy = layout_positions.get(nid)
            nodes.append({
                "id":      nid,
                "label":   _short(title, 24),
                "size":    _node_size(cc),
                "color":   C_EXTERNAL,
                "tooltip": _tooltip(title, [
                    ("Year",       year or "—"),
                    ("DOI",        d.get("citing_doi") or "—"),
                    ("Citations",  f"{cc:,}" if cc else "—"),
                    ("References", "—"),
                ]),
                "group":     "external",
                "cc":        cc,
                "fullTitle": title,
                "year":      year,
                "doi":       d.get("citing_doi") or "",
                "authors":   "",
                **( {"x": round(_xy[0], 1), "y": round(_xy[1], 1)} if _xy else {"x": 0.0, "y": 0.0} ),
            })

        edges.append({
            "id":     _eid(),
            "source": nid,
            "target": f"item:{d['cited_item_key']}",
            "size":   max(0.5, min(4.0, d["context_count"] * 0.4)),
            "color":  PALETTE["edgeDefault"],
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
            _xy = layout_positions.get(nid)
            nodes.append({
                "id":      nid,
                "label":   _short(title, 24),
                "size":    _node_size(cc),
                "color":   C_EXTERNAL,
                "tooltip": _tooltip(title, [
                    ("Year",       year or "—"),
                    ("DOI",        d.get("cited_doi") or "—"),
                    ("Citations",  f"{cc:,}" if cc else "—"),
                    ("References", "—"),
                ]),
                "group":     "reference",
                "cc":        cc,
                "fullTitle": title,
                "year":      year,
                "doi":       d.get("cited_doi") or "",
                "authors":   "",
                **( {"x": round(_xy[0], 1), "y": round(_xy[1], 1)} if _xy else {"x": 0.0, "y": 0.0} ),
            })

        edges.append({
            "id":     _eid(),
            "source": f"item:{d['citing_item_key']}",
            "target": nid,
            "size":   max(0.5, min(3.0, d["context_count"] * 0.3)),
            "color":  PALETTE["edgeDefault"],
            "type":   "arrow",
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
        palette=PALETTE,
        css_root=_css_root,
        js_theme=_js_theme,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    print(f"\n✅ Graph saved: {output_path}")
    print(f"   {n_nodes} nodes  |  {n_edges} edges")
    print(f"   Rose={len(items)} Zotero items  Grey={n_citer} citers  Grey={n_ref} refs")


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
    parser.add_argument("--top",     type=int, default=9999,
                        help="表示する上位アイテム数 (default: 全件)")
    parser.add_argument("--citers",  type=int, default=9999,
                        help="1アイテムあたりの被引用論文数 (default: 全件)")
    parser.add_argument("--refs",    type=int, default=9999,
                        help="1アイテムあたりの参照先論文数 (default: 全件)")
    parser.add_argument("--min-cc",  type=int, default=10,
                        help="引用元・参照先の最低被引用数フィルタ (default: 10, 0=フィルタなし)")
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
    citers    = get_citers(conn, item_keys, args.citers, min_cc=args.min_cc)
    refs      = [] if args.no_refs else get_refs(conn, item_keys, args.refs, min_cc=args.min_cc)

    # ChromaDB からタイトル・著者・年を取得
    item_meta  = get_item_meta(item_keys)
    item_rcnt  = get_item_ref_counts(conn, item_keys)
    titled     = sum(1 for k in item_keys if item_meta.get(k, {}).get("title"))
    auth       = sum(1 for k in item_keys if item_meta.get(k, {}).get("creators"))

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
               item_meta=item_meta, item_ref_counts=item_rcnt, chunk_data=chunk_data)

    if not args.no_open:
        _serve(output_path)


if __name__ == "__main__":
    main()
