# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "networkx",
#   "fa2-modified",
#   "fastapi",
#   "uvicorn[standard]",
#   "requests",
#   "python-dotenv",
#   "chromadb",
#   "umap-learn",
#   "scikit-learn",
#   "scipy",
# ]
# ///
"""
Citation network visualizer for zotero-local-rag.
Uses sigma.js (WebGL) + graphology for scalable rendering of 1000–10000+ nodes.

Usage:
  uv run scripts/show_citation_graph.py               # デフォルト（全件）
  uv run scripts/show_citation_graph.py --top 100     # 上位100件
  uv run scripts/show_citation_graph.py --item KEY    # 1アイテムに絞る
  uv run scripts/show_citation_graph.py --no-refs     # 参照先を非表示
  uv run scripts/show_citation_graph.py --no-open     # ブラウザを自動で開かない
  uv run scripts/show_citation_graph.py --port 7234   # ポート指定（デフォルト: 7234）

  グラフは http://localhost:PORT でリアルタイム配信されます。
  Ctrl+C でサーバーを停止します。

Node colors:
  Blue   = Zotero アイテム（自分のライブラリ）
  Amber  = 外部論文（被引用: 自分の文献を引用している論文）
  Green  = 外部論文（参照先: 自分の文献が引用している論文）

Edge colors:
  Red   = 外部論文 → Zotero アイテム（被引用）
  Green = Zotero アイテム → 参照先論文（参照）
"""
import argparse
import json
import math
import os
import random
import re
import sqlite3
import sys
import time
import webbrowser
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.item_vectors import get_item_vectors as _shared_get_item_vectors
from src.chunk_store import get_item_chunks
load_dotenv(PROJECT_ROOT / ".env")

DB_PATH         = os.environ.get("RELATIONS_DB_PATH", str(PROJECT_ROOT / "data" / "relations.db"))
CHROMA_DB       = os.environ.get("CHROMA_DIR", str(PROJECT_ROOT / "data" / "chroma")) + "/chroma.sqlite3"
ZOTERO_SQLITE   = os.environ.get(
    "ZOTERO_SQLITE",
    str(Path.home() / "Zotero" / "zotero.sqlite"),
)
AZURE_TRANSLATOR_KEY    = os.environ.get("AZURE_TRANSLATOR_KEY", "")
AZURE_TRANSLATOR_REGION = os.environ.get("AZURE_TRANSLATOR_REGION", "japaneast")
DEFAULT_PORT    = 7234

# English stopwords for keyword extraction (lowercase)
_STOPWORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "has", "have", "been", "were",
    "its", "who", "whom", "which", "what", "when", "where", "why", "how",
    "from", "with", "this", "that", "these", "those", "they", "them",
    "their", "will", "would", "could", "should", "may", "might", "shall",
    "also", "into", "than", "then", "just", "about", "over", "each",
    "some", "any", "such", "only", "other", "more", "most", "very",
    "much", "well", "new", "using", "based", "used", "use", "via",
    "does", "did", "both", "after", "before", "between", "under",
    "during", "within", "without", "through", "however", "therefore",
    "here", "there", "still", "already", "often", "since", "while",
    "case", "due", "make", "made", "doing", "done", "many", "two",
    "three", "see", "yet", "own", "set", "part", "way", "get", "got",
    "first", "like", "now", "say", "said",
})


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


def get_item_vectors(item_keys: list[str]) -> dict[str, list[float]]:
    """各 Zotero アイテムの意味ベクトル（チャンク埋め込みの平均・正規化済み）を返す。

    ChromaDB からチャンクベクトルを読み出して平均する。結果は
    data/item_vectors_cache.json にキャッシュされ、未知のアイテムだけ追加計算する
    （文献の本文は通常変わらないため）。
    埋め込みモデルを変更した場合は次元数不一致によりキャッシュが自動無効化される。
    失敗時は空 dict を返し、レイアウトは引用エッジのみで計算される（表示は壊れない）。
    """
    return _shared_get_item_vectors(item_keys)


def get_item_ref_counts(conn: sqlite3.Connection, item_keys: list[str]) -> dict[str, int]:
    """各 Zotero アイテムが持つ参照先論文数（global_references）を返す。"""
    if not item_keys:
        return {}
    placeholders = ",".join("?" * len(item_keys))
    rows = conn.execute(f"""
        SELECT citing_item_key, COUNT(DISTINCT cited_paper_id) AS ref_count
        FROM global_references
        WHERE citing_item_key IN ({placeholders})
          AND NOT EXISTS (
              SELECT 1 FROM relation_reports rr
              WHERE rr.direction = 'references'
                AND rr.item_key = global_references.citing_item_key
                AND rr.external_paper_id = global_references.cited_paper_id
                AND rr.status = 'disabled'
          )
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
            WHERE NOT EXISTS (
                SELECT 1 FROM relation_reports rr
                WHERE rr.direction = 'citations'
                  AND rr.item_key = global_citations.cited_item_key
                  AND rr.external_paper_id = global_citations.citing_paper_id
                  AND rr.status = 'disabled')
            UNION
            SELECT citing_item_key AS item_key FROM global_references
            WHERE NOT EXISTS (
                SELECT 1 FROM relation_reports rr
                WHERE rr.direction = 'references'
                  AND rr.item_key = global_references.citing_item_key
                  AND rr.external_paper_id = global_references.cited_paper_id
                  AND rr.status = 'disabled')
        ) k
        LEFT JOIN (
            SELECT
                cited_item_key                      AS item_key,
                COUNT(DISTINCT citing_paper_id)     AS citer_count,
                COUNT(*)                            AS context_count
            FROM global_citations
            WHERE NOT EXISTS (
                SELECT 1 FROM relation_reports rr
                WHERE rr.direction = 'citations'
                  AND rr.item_key = global_citations.cited_item_key
                  AND rr.external_paper_id = global_citations.citing_paper_id
                  AND rr.status = 'disabled')
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
              AND NOT EXISTS (
                  SELECT 1 FROM relation_reports rr
                  WHERE rr.direction = 'citations'
                    AND rr.item_key = global_citations.cited_item_key
                    AND rr.external_paper_id = global_citations.citing_paper_id
                    AND rr.status = 'disabled')
            GROUP BY cited_item_key
        ) cit ON 1=1
        LEFT JOIN item_citation_status ics ON ics.item_key = ?
        WHERE cit.item_key IS NOT NULL
           OR EXISTS (
               SELECT 1 FROM global_references
               WHERE citing_item_key = ?
                 AND NOT EXISTS (
                     SELECT 1 FROM relation_reports rr
                     WHERE rr.direction = 'references'
                       AND rr.item_key = global_references.citing_item_key
                       AND rr.external_paper_id = global_references.cited_paper_id
                       AND rr.status = 'disabled'))
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
            MAX(citing_authors) AS citing_authors,
            COUNT(*) AS context_count
        FROM global_citations
        WHERE cited_item_key IN ({placeholders})
          AND citing_paper_id IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM relation_reports rr
              WHERE rr.direction = 'citations'
                AND rr.item_key = global_citations.cited_item_key
                AND rr.external_paper_id = global_citations.citing_paper_id
                AND rr.status = 'disabled'
          )
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
            MAX(cited_authors) AS cited_authors,
            COUNT(*) AS context_count
        FROM global_references
        WHERE citing_item_key IN ({placeholders})
          AND cited_paper_id IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM relation_reports rr
              WHERE rr.direction = 'references'
                AND rr.item_key = global_references.citing_item_key
                AND rr.external_paper_id = global_references.cited_paper_id
                AND rr.status = 'disabled'
          )
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


def get_contexts_for_edge(db_path: str, src_id: str, tgt_id: str) -> list[dict]:
    """エッジ(src_id → tgt_id)の引用コンテキストをDB から全件取得する。

    ノードID形式:
      paper:PAPERID  – 外部論文（citing_paper_id / cited_paper_id）
      ref:PAPERID    – 参照先外部論文（cited_paper_id）
      item:ITEMKEY   – Zoteroアイテム
    """
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        results = []

        def _pid(node_id: str) -> str:
            return node_id.split(":", 1)[1]

        if src_id.startswith(("paper:", "ref:")) and tgt_id.startswith("item:"):
            # 外部論文 → Zoteroアイテム（global_citations）
            rows = conn.execute("""
                SELECT context_snippet, page_hint FROM global_citations
                WHERE citing_paper_id = ? AND cited_item_key = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM relation_reports rr
                      WHERE rr.direction = 'citations'
                        AND rr.item_key = global_citations.cited_item_key
                        AND rr.external_paper_id = global_citations.citing_paper_id
                        AND rr.status = 'disabled')
                  AND (context_snippet IS NOT NULL OR page_hint IS NOT NULL)
                ORDER BY CASE WHEN context_snippet IS NOT NULL THEN 0 ELSE 1 END
            """, (_pid(src_id), _pid(tgt_id))).fetchall()
        elif src_id.startswith("item:") and tgt_id.startswith(("paper:", "ref:")):
            # Zoteroアイテム → 外部論文（global_references）
            rows = conn.execute("""
                SELECT context_snippet, page_hint FROM global_references
                WHERE citing_item_key = ? AND cited_paper_id = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM relation_reports rr
                      WHERE rr.direction = 'references'
                        AND rr.item_key = global_references.citing_item_key
                        AND rr.external_paper_id = global_references.cited_paper_id
                        AND rr.status = 'disabled')
                  AND (context_snippet IS NOT NULL OR page_hint IS NOT NULL)
                ORDER BY CASE WHEN context_snippet IS NOT NULL THEN 0 ELSE 1 END
            """, (_pid(src_id), _pid(tgt_id))).fetchall()
        else:
            rows = []

        for r in rows:
            results.append({"snippet": r["context_snippet"] or "", "page": r["page_hint"] or ""})
        return results
    except Exception:
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ── server-side FA2 layout ───────────────────────────────────────────────────

# レイアウトアルゴリズムのバージョン。物理パラメータや意味エッジの仕様を変えたら
# 上げること（layout_cache.json のキーに含まれ、変更時に再計算がトリガーされる）。
# 注: fa2_modified は linLogMode 未実装のため線形引力モデルのまま
# （クラスタ分離の強化は意味エッジ + 将来の OpenOrd/DrL 検討で対応）。
LAYOUT_VERSION = "v2-sem-noverlap"

# 意味的kNN仮想エッジの設定（②A）
SEM_KNN_K = 5            # 各Zoteroアイテムから張る意味エッジの本数
SEM_SIM_THRESHOLD = 0.35 # コサイン類似度がこれ未満の近傍にはエッジを張らない
SEM_EDGE_WEIGHT = 0.4    # 引用エッジ(weight=1.0)に対する意味エッジの相対重み係数

# 意味空間マップ用の次元削減手法
SEMANTIC_LAYOUT_METHODS = ["umap", "tsne", "pca", "mds"]


def compute_semantic_layout(
    vectors: dict[str, list[float]],
    method: str = "umap",
    *,
    random_state: int = 42,
) -> dict[str, tuple[float, float]]:
    """次元削減で資料ベクトルを2次元に縮約し、意味空間マップ座標を返す。

    Args:
        vectors: {item_key: embedding_vector} の辞書
        method: "umap" | "tsne" | "pca" | "mds"
        random_state: 乱数シード

    Returns:
        {item_key: (x, y)} の辞書。座標はスケーリング済み。
    """
    import numpy as np

    if not vectors or len(vectors) < 2:
        return {}

    keys = list(vectors.keys())
    X = np.array([vectors[k] for k in keys], dtype=np.float32)

    _print = lambda *a, **kw: print(*a, file=sys.stderr, **kw)
    t0 = time.time()
    _print(f"  [semantic-layout] method={method}, n={len(keys)}")

    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=random_state)
        coords = reducer.fit_transform(X)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=2, perplexity=min(30, len(keys) - 1),
            random_state=random_state, metric="cosine", n_jobs=1,
        )
        coords = reducer.fit_transform(X)
    elif method == "mds":
        from sklearn.manifold import MDS
        # コサイン距離で前計算 (正規化済みベクトルなら dot で類似度 → 1 - similarity = cosine distance)
        # ただしメモリ節約のため、metric MDS にユークリッド距離を渡す（PCA初期化 + cosine距離の代替）
        reducer = MDS(
            n_components=2, random_state=random_state,
            dissimilarity="precomputed", normalized_stress="auto",
            n_init=1, max_iter=300,
        )
        # コサイン距離行列
        D = 1.0 - (X @ X.T)
        np.fill_diagonal(D, 0.0)
        D = np.maximum(D, 0.0)  # 浮動小数点誤差対策
        coords = reducer.fit_transform(D)
    else:  # umap (default)
        from umap import UMAP
        reducer = UMAP(
            n_components=2, n_neighbors=min(15, len(keys) - 1),
            min_dist=0.1, metric="cosine", random_state=random_state,
            verbose=False,
        )
        coords = reducer.fit_transform(X)

    elapsed = time.time() - t0
    _print(f"  [semantic-layout] method={method} done in {elapsed:.1f}s")

    # スケーリング: sigma.js の bboxR ≈ span * 0.65 で正規化されるので、
    # 十分な広がりを持たせる（FA2 と同様に ~24000 units）
    cx = (coords[:, 0].max() + coords[:, 0].min()) / 2
    cy = (coords[:, 1].max() + coords[:, 1].min()) / 2
    span = max(
        coords[:, 0].max() - coords[:, 0].min(),
        coords[:, 1].max() - coords[:, 1].min(),
        1.0,
    )
    scale = 24000.0 / span

    result: dict[str, tuple[float, float]] = {}
    for i, key in enumerate(keys):
        x = float((coords[i, 0] - cx) * scale)
        y = float((coords[i, 1] - cy) * scale)
        result[key] = (round(x, 1), round(y, 1))

    return result


def compute_clusters(
    vectors: dict[str, list[float]],
    positions: dict[str, tuple[float, float]],
    *,
    n_clusters: int = 8,
) -> list[dict]:
    """Zotero 資料の埋め込みベクトルをクラスタリングし、ConvexHull 領域を返す。

    AgglomerativeClustering (cosine距離, average linkage) を使い、
    指定された数のクラスタに分割する。

    Args:
        vectors: {item_key: embedding_vector}
        positions: {item_key: (x, y)}  — 2D 座標（次元削減後）
        n_clusters: 目標クラスタ数。データ数を超える場合は自動調整。

    Returns:
        [{"id": 0, "label": "A", "item_keys": [...], "hull": [[x,y],...], "color": "#xxx"}, ...]
    """
    import numpy as np
    from sklearn.cluster import KMeans
    from scipy.spatial import ConvexHull

    if len(vectors) < 5:
        return []

    keys = list(vectors.keys())
    X = np.array([vectors[k] for k in keys], dtype=np.float32)

    # Clamp n_clusters to valid range
    n = min(n_clusters, len(X) - 1)
    n = max(n, 2)

    # KMeans on L2-normalized vectors ≈ spherical k-means on unit hypersphere.
    # Naturally produces balanced clusters (each centroid attracts items within
    # its Voronoi region), unlike agglomerative methods which can create one
    # giant cluster when most items are in a dense region.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clustering = KMeans(n_clusters=n, n_init="auto", random_state=42).fit(X)
    labels_arr = clustering.labels_

    # ── Build cluster groups (same format as before) ──
    groups: dict[int, list[str]] = {}
    for key, label in zip(keys, labels_arr):
        groups.setdefault(int(label), []).append(key)

    # Sort by size descending (largest = A)
    sorted_clusters = sorted(groups.items(), key=lambda x: -len(x[1]))
    n = len(sorted_clusters)
    print(f"  [clustering] k-means → {n} clusters "
          f"(sizes: {', '.join(str(len(m)) for _, m in sorted_clusters[:10])}"
          + ("…" if n > 10 else "") + ")", file=sys.stderr)

    # カラーパレット（視認性の高い色）
    palette = [
        "#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B",
        "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC",
        "#A0CBE8", "#FFBE7D", "#F28E2B", "#86BCB6", "#59A14F",
        "#8CD17D", "#B6992D", "#499894", "#D37295", "#F1CE63",
    ]

    result = []
    for idx, (gid, members) in enumerate(sorted_clusters):
        if len(result) >= len(palette):
            break

        if idx < 26:
            label = chr(65 + idx)  # A-Z
        else:
            label = chr(65 + (idx - 26) // 26) + chr(65 + (idx - 26) % 26)

        # ConvexHull を 2D 座標から計算
        hull: list[list[float]] = []
        cluster_2d = np.array([positions[k] for k in members if k in positions], dtype=np.float32)
        if len(cluster_2d) >= 3:
            try:
                ch = ConvexHull(cluster_2d)
                # 頂点数を制限（最大32点）
                verts = cluster_2d[ch.vertices]
                if len(verts) > 32:
                    step = max(1, len(verts) // 32)
                    verts = verts[::step]
                hull = [[round(float(v[0]), 1), round(float(v[1]), 1)] for v in verts]
            except Exception:
                hull = []

        result.append({
            "id": idx,
            "label": label,
            "item_keys": members,
            "hull": hull,
            "color": palette[idx],
            "keywords": [],  # filled in second pass below
        })

    # ── Second pass: keyword extraction using existing item vectors ──
    # For each candidate term, compute its relevance to a cluster as the
    # average cosine similarity of items (whose titles contain that term)
    # to the cluster centroid.  No new embeddings needed.
    if len(result) >= 2:
        import re

        # Get titles for all items
        all_keys = [k for c in result for k in c["item_keys"]]
        all_metas = get_item_meta(all_keys)

        # Build per-cluster info: centroid + candidate terms from top items
        cluster_info: list[dict] = []
        all_terms: set[str] = set()

        for c in result:
            members_arr = np.array(c["item_keys"])
            # Use all item vectors for the centroid (stable)
            member_vecs = np.array([vectors[k] for k in c["item_keys"] if k in vectors], dtype=np.float32)
            if len(member_vecs) >= 1:
                centroid = member_vecs.mean(axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
                # top-5 items closest to centroid → candidate term source
                sims = member_vecs @ centroid
                top_k = min(5, len(c["item_keys"]))
                top_idx = np.argsort(-sims)[:top_k]
                top_keys = members_arr[top_idx].tolist()
            else:
                centroid = np.zeros(1, dtype=np.float32)
                top_keys = c["item_keys"][:5]

            c_terms: list[str] = []
            for k in top_keys:
                title = all_metas.get(k, {}).get("title", "")
                tokens = re.split(r'[\s,.:;!?()\[\]「」『』【】《》""''/\\|　、。，．：；！？]+', title)
                for tok in tokens:
                    tok = tok.strip().lower()
                    if not (3 <= len(tok) <= 20):
                        continue
                    if not re.search(r'[a-z぀-ゟ゠-ヿ一-鿿]', tok):
                        continue
                    if re.match(r'^[\d\W]+$', tok):
                        continue
                    if tok in _STOPWORDS:
                        continue
                    if tok not in all_terms:
                        all_terms.add(tok)
                    c_terms.append(tok)

            cluster_info.append({
                "centroid": centroid,
                "terms": list(dict.fromkeys(c_terms)),
            })

        # For each candidate term, find which items contain it in their title.
        # The term's similarity to a cluster centroid is the average cosine
        # similarity of those items' vectors to the centroid.
        # Pre-compute: term → list of item vectors (normalized)
        term_item_vecs: dict[str, list[np.ndarray]] = {t: [] for t in all_terms}
        for k in all_keys:
            v = vectors.get(k)
            if v is None:
                continue
            vn = np.asarray(v, dtype=np.float32)
            vn = vn / (np.linalg.norm(vn) + 1e-10)
            title = all_metas.get(k, {}).get("title", "")
            tokens = set(re.split(r'[\s,.:;!?()\[\]「」『』【】《》""''/\\|　、。，．：；！？]+', title.lower()))
            for tok in tokens:
                if tok in term_item_vecs:
                    term_item_vecs[tok].append(vn)

        # Collect all (cluster_idx, term, score) globally, then assign each
        # term to only the highest-scoring cluster — no duplicates across clusters.
        global_scores: list[tuple[int, str, float]] = []
        for i, c in enumerate(result):
            info = cluster_info[i]
            cent = info["centroid"]
            if len(cent) < 2:
                continue

            for tok in info["terms"]:
                tvecs = term_item_vecs.get(tok)
                if not tvecs:
                    continue
                sims = [float(tv @ cent) for tv in tvecs]
                score = np.mean(sims) * np.log(len(tvecs) + 1)
                global_scores.append((i, tok, score))

        global_scores.sort(key=lambda x: -x[2])

        assigned: dict[str, str] = {}  # term → assigned to which cluster (label)
        cluster_kw: dict[int, list[str]] = {i: [] for i in range(len(result))}
        for ci, tok, score in global_scores:
            if tok in assigned:
                continue  # already claimed by a higher-scoring cluster
            if len(cluster_kw[ci]) >= 3:
                continue  # this cluster already has 3 keywords
            assigned[tok] = result[ci]["label"]
            cluster_kw[ci].append(tok)

        for i, c in enumerate(result):
            c["keywords"] = cluster_kw[i]

    return result

    return result


def _get_semantic_layout_cache_path(method: str) -> Path:
    return PROJECT_ROOT / "data" / f"semantic_layout_{method}.json"


def _load_semantic_layout_cache(method: str) -> dict[str, tuple[float, float]] | None:
    cache_path = _get_semantic_layout_cache_path(method)
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text())
        return {k: tuple(v) for k, v in data.get("positions", {}).items()}
    except Exception:
        return None


def _save_semantic_layout_cache(method: str, positions: dict[str, tuple[float, float]]) -> None:
    cache_path = _get_semantic_layout_cache_path(method)
    try:
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "method": method,
            "positions": {k: list(v) for k, v in positions.items()},
        }))
        tmp.replace(cache_path)
    except Exception:
        pass


def compute_layout(
    item_keys: list[str],
    citer_rows: list[dict],
    ref_rows: list[dict],
    warm_start: dict[str, tuple[float, float]] | None = None,
    semantic_vectors: dict[str, list[float]] | None = None,
    node_sizes: dict[str, float] | None = None,
) -> dict[str, tuple[float, float]]:
    """
    三段階ForceAtlas2（LinLogモード）でレイアウトを計算する。
    Phase 1a: Zotero + Ref のみ → Zoteroノードを十分に離し、Refを近くに配置
    Phase 1b: Zotero + Citer のみ → CiterをZoteroの周囲に配置
    Phase 2:  全ノード → 微調整
    仕上げ:   noverlap（ノード半径を考慮した重なり除去）

    semantic_vectors: {item_key: 正規化済みベクトル} を渡すと、内容が近い
    Zoteroアイテム間に弱い仮想バネ（意味的kNNエッジ）を張り、引用関係が
    無くてもテーマが近い資料が空間的に近くに配置される。
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
        # edgeWeightInfluence=1.0: エッジ重み（引用=1.0, 意味エッジ=0.4×類似度）を
        # 引力にそのまま反映させる。linLogMode は fa2_modified 未実装のため不使用。
        return ForceAtlas2(
            outboundAttractionDistribution=True,
            edgeWeightInfluence=1.0,
            barnesHutOptimize=True,
            barnesHutTheta=1.2,
            scalingRatio=scalingRatio,
            gravity=gravity,
            verbose=False,
        )

    # ── 意味的kNN仮想エッジ（Zoteroアイテム間のみ）────────────────────────
    semantic_edges: list[tuple[str, str, float]] = []
    if semantic_vectors:
        import numpy as _np
        # NaN/Inf を含むベクトル（抽出失敗チャンク由来）は除外
        sem_keys = [
            k for k in item_keys
            if k in semantic_vectors
            and _np.isfinite(_np.asarray(semantic_vectors[k], dtype=_np.float32)).all()
        ]
        if len(sem_keys) >= 3:
            V = _np.array([semantic_vectors[k] for k in sem_keys], dtype=_np.float32)
            # ベクトルは保存時に正規化済みだが念のため再正規化
            norms = _np.linalg.norm(V, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            V = V / norms
            S = V @ V.T  # cosine similarity matrix
            _np.fill_diagonal(S, -1.0)
            k_eff = min(SEM_KNN_K, len(sem_keys) - 1)
            seen: set[tuple[str, str]] = set()
            for i, key_i in enumerate(sem_keys):
                top = _np.argpartition(-S[i], k_eff)[:k_eff]
                for j in top:
                    sim = float(S[i, j])
                    if sim < SEM_SIM_THRESHOLD:
                        continue
                    pair = tuple(sorted((key_i, sem_keys[int(j)])))
                    if pair in seen:
                        continue
                    seen.add(pair)
                    semantic_edges.append(
                        (f"item:{pair[0]}", f"item:{pair[1]}", SEM_EDGE_WEIGHT * sim)
                    )
            print(f"  [layout] semantic kNN edges: {len(semantic_edges)} "
                  f"(k={k_eff}, threshold={SEM_SIM_THRESHOLD})", file=_stderr)

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
        G1a.add_edge(f"item:{r['citing_item_key']}", _ref_nid(pid), weight=1.0)
    for u, v, w in semantic_edges:
        # 既に引用エッジがある場合は意味エッジの重みを加算（両方の関係を反映）
        if G1a.has_edge(u, v):
            G1a[u][v]["weight"] += w
        else:
            G1a.add_edge(u, v, weight=w)

    # warm_start が渡された場合: 既知ノードはキャッシュ座標, 新規ノードは隣接ノードの近くに配置
    pos1a_init: dict[str, tuple[float, float]] | None = None
    _is_warm = bool(warm_start)
    if warm_start:
        ws_cx = sum(v[0] for v in warm_start.values()) / max(len(warm_start), 1)
        ws_cy = sum(v[1] for v in warm_start.values()) / max(len(warm_start), 1)
        pos1a_init = {}
        for nid in G1a.nodes():
            if nid in warm_start:
                pos1a_init[nid] = warm_start[nid]
            else:
                # 隣接ノードのキャッシュ座標の重心近くにランダム配置
                nbr_pos = [warm_start[nb] for nb in G1a.neighbors(nid) if nb in warm_start]
                if nbr_pos:
                    bx = sum(p[0] for p in nbr_pos) / len(nbr_pos)
                    by = sum(p[1] for p in nbr_pos) / len(nbr_pos)
                else:
                    bx, by = ws_cx, ws_cy
                pos1a_init[nid] = (bx + _rng.uniform(-1.0, 1.0),
                                   by + _rng.uniform(-1.0, 1.0))

    # warm_start 時はイテレーション数を大幅削減（既に良い初期値があるため早く収束）
    _iter1a = 800  if _is_warm else 3000
    _iter1b = 600  if _is_warm else 2000
    _iter2  = 600  if _is_warm else 2000

    print(f"  [layout] phase1a (Zotero+Ref): {G1a.number_of_nodes()} nodes "
          f"{'[warm]' if _is_warm else ''}…", file=_stderr)
    pos1a = _run_until_convergence(
        _fa2(scalingRatio=10.0, gravity=1.0), G1a, pos1a_init,
        batch=50, max_iter=_iter1a, tol_frac=0.001, consec=3, label="1a"
    )

    # ── Phase 1b: Zotero + Citer（Zoteroの初期配置はPhase1aの結果を使用）────
    G1b = nx.Graph()
    G1b.add_nodes_from(f"item:{k}" for k in item_keys)
    for r in citer_rows:
        pid = r.get("citing_paper_id")
        if pid:
            G1b.add_edge(f"paper:{pid}", f"item:{r['cited_item_key']}", weight=1.0)

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

    print(f"  [layout] phase1b (Zotero+Citer): {G1b.number_of_nodes()} nodes "
          f"{'[warm]' if _is_warm else ''}…", file=_stderr)
    pos1b = _run_until_convergence(
        _fa2(scalingRatio=3.0, gravity=1.0), G1b, pos1b_init,
        batch=50, max_iter=_iter1b, tol_frac=0.0002, label="1b"
    )

    # ── Phase 2: 全ノード統合 ─────────────────────────────────────────────
    G2 = nx.Graph()
    G2.add_nodes_from(G1a.nodes())
    G2.add_nodes_from(G1b.nodes())
    for u, v, d in G1a.edges(data=True):
        G2.add_edge(u, v, weight=d.get("weight", 1.0))
    for u, v, d in G1b.edges(data=True):
        G2.add_edge(u, v, weight=d.get("weight", 1.0))

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

    print(f"  [layout] phase2 (all): {G2.number_of_nodes()} nodes "
          f"{'[warm]' if _is_warm else ''}…", file=_stderr)
    pos = _run_until_convergence(
        _fa2(scalingRatio=2.0, gravity=1.5), G2, pos2_init,
        batch=50, max_iter=_iter2, tol_frac=0.0002, label="2 "
    )

    # 座標を graph-units にスケーリング（広めに取る）
    xs = [v[0] for v in pos.values()]
    ys = [v[1] for v in pos.values()]
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1)
    scale = 24000 / span  # 広めにスケーリング

    scaled = {nid: ((x - cx) * scale, (y - cy) * scale) for nid, (x, y) in pos.items()}

    # ── 仕上げ: noverlap（重なり除去）────────────────────────────────────
    scaled = _noverlap(scaled, node_sizes or {}, _stderr)
    return scaled


def _noverlap(
    pos: dict[str, tuple[float, float]],
    node_sizes: dict[str, float],
    _stderr,
    *,
    px_per_unit: float = 900.0 / 24000.0,  # 全体表示時のおおよその px/graph-unit
    margin: float = 1.15,
    iterations: int = 40,
) -> dict[str, tuple[float, float]]:
    """ノードの描画半径（全体表示ズーム基準）を考慮し、重なったノード同士を
    最小限の移動で押し離す後処理。空間ハッシュグリッドで近傍ペアのみ判定する。
    node_sizes が無いノードはデフォルト半径 2px とみなす。
    """
    import math as _math
    if not pos:
        return pos

    # px → graph-unit 変換（margin で少し余白を持たせる）
    radii = {
        nid: (node_sizes.get(nid, 2.0) / px_per_unit) * margin / 2.0
        for nid in pos
    }
    max_r = max(radii.values())
    cell = max_r * 2.0
    if cell <= 0:
        return pos

    cur = {nid: [x, y] for nid, (x, y) in pos.items()}
    nids = list(cur.keys())

    moved_total = 0
    for it in range(iterations):
        # 空間ハッシュグリッド構築
        grid: dict[tuple[int, int], list[str]] = {}
        for nid in nids:
            x, y = cur[nid]
            grid.setdefault((int(x // cell), int(y // cell)), []).append(nid)

        moved = 0
        for (gx, gy), bucket in grid.items():
            # 9近傍セルの候補を集め、b > a（ID順）の組のみ処理して二重判定を回避
            cand: list[str] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cand.extend(grid.get((gx + dx, gy + dy), []))
            for a in bucket:
                ax, ay = cur[a]
                ra = radii[a]
                for b in cand:
                    if b <= a:
                        continue
                    bx, by = cur[b]
                    min_d = ra + radii[b]
                    dx_, dy_ = bx - ax, by - ay
                    d2 = dx_ * dx_ + dy_ * dy_
                    if d2 >= min_d * min_d:
                        continue
                    d = _math.sqrt(d2)
                    if d < 1e-9:
                        # 完全に同一座標 → 決定論的な方向に少しずらす
                        dx_, dy_, d = 1.0, 0.5, 1.118
                    push = (min_d - d) / 2.0
                    ux, uy = dx_ / d, dy_ / d
                    cur[a][0] -= ux * push
                    cur[a][1] -= uy * push
                    cur[b][0] += ux * push
                    cur[b][1] += uy * push
                    ax, ay = cur[a]
                    moved += 1
        moved_total += moved
        if moved == 0:
            print(f"  [layout] noverlap converged at iter={it + 1} "
                  f"(total pushes={moved_total})", file=_stderr)
            break
    else:
        print(f"  [layout] noverlap finished {iterations} iters "
              f"(total pushes={moved_total})", file=_stderr)

    return {nid: (xy[0], xy[1]) for nid, xy in cur.items()}


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
    グラフデータは /api/graph から fetch するため埋め込まない。
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

  /* ── View mode tabs (bottom-left floating pill) ── */
  #view-tabs {
    position: fixed; bottom: 16px; left: 16px; z-index: 600;
    display: flex; gap: 0;
    background: var(--surface-container-high);
    border: 1px solid var(--outline-variant);
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.5);
  }
  .view-tab {
    padding: 6px 14px; font-size: 11.5px; cursor: pointer;
    border: none; background: none; color: var(--on-surface-variant);
    transition: color 0.15s, background 0.15s;
    white-space: nowrap;
  }
  .view-tab + .view-tab { border-left: 1px solid var(--outline-variant); }
  .view-tab.active {
    color: #fff; background: var(--node-zotero); font-weight: 500;
  }
  .view-tab:hover:not(.active) { color: var(--on-surface); background: var(--surface-container-low); }
  /* 意味マップモード用プルダウン（凡例カード内） */
  #semantic-method-wrap {
    display: none; margin-top: 12px;
  }
  #semantic-method-wrap.show { display: block; }
  #semantic-method-wrap label {
    display: block; font-size: 10.5px; color: var(--text-dis); margin-bottom: 4px;
  }
  #semantic-method-select {
    width: 100%; padding: 4px 6px; font-size: 11.5px;
    background: var(--surface-container-high); color: var(--on-surface);
    border: 1px solid var(--outline-variant); border-radius: 4px; cursor: pointer;
    outline: none;
  }
  #semantic-method-select:focus { border-color: var(--node-zotero); }
  /* クラスタ数スライダー */
  #semantic-clusters-wrap {
    display: none; margin-top: 12px;
  }
  #semantic-clusters-wrap.show { display: block; }
  #semantic-clusters-wrap label {
    display: block; font-size: 10.5px; color: var(--text-dis); margin-bottom: 4px;
  }
  #nclusters-slider {
    width: 100%; margin: 0; cursor: pointer;
    accent-color: var(--node-zotero);
  }

  /* ── Cluster overlay canvas ── */
  #cluster-overlay {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; /* behind sigma canvas */
  }

  /* ── Cluster legend (in legend card) ── */
  #cluster-legend { display: none; margin-top: 12px; }
  #cluster-legend.show { display: block; }
  #cluster-legend .cl-header {
    font-size: 10.5px; color: var(--text-dis); margin-bottom: 6px;
  }
  #cluster-legend .cl-row {
    display: flex; align-items: center; flex-wrap: wrap; gap: 6px; margin-bottom: 3px;
    cursor: pointer; padding: 2px 4px; border-radius: 3px;
    font-size: 11px; color: var(--on-surface-variant);
  }
  #cluster-legend .cl-row:hover { background: var(--surface-container-high); }
  #cluster-legend .cl-row.active { background: var(--surface-container-high); color: var(--on-surface); }
  #cluster-legend .cl-dot {
    width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;
  }
  #cluster-legend .cl-label { flex: 0 0 auto; }
  #cluster-legend .cl-count { font-size: 10.5px; color: var(--text-dis); }
  #cluster-legend .cl-keywords {
    flex-basis: 100%; margin-top: 1px;
    font-size: 10px; color: var(--text-dis);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    padding-left: 16px;
  }

  /* ── Collapsible sections ── */
  .sect-header {
    display: flex; align-items: center; justify-content: space-between;
    cursor: pointer; user-select: none;
    font-size: 11px; color: var(--text-dis); margin: 8px 0 4px; padding: 2px 0;
  }
  .sect-header:hover { color: var(--on-surface); }
  .sect-arrow { font-size: 8px; transition: transform 0.2s; }
  .sect-header.open .sect-arrow { transform: rotate(90deg); }
  .sect-body { overflow: hidden; }
  .sect-body:not(.open) { display: none; }

  /* ── Sidebar ── */
  #sidebar {
    position: fixed; right: 0; top: 0; bottom: 0; width: var(--sb-width);
    background: var(--surface);
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
  #sb-count { font-weight: 400; font-size: 11px; color: var(--text-dis); }
  #sb-search {
    display: block; width: 100%; box-sizing: border-box;
    background: var(--surface-container-high);
    border: 1px solid var(--outline-variant); border-radius: 4px;
    color: var(--on-surface); font-size: 12px;
    padding: 5px 8px; outline: none;
  }
  #sb-search:focus { border-color: var(--node-zotero); }
  #sb-tab-list { display: flex; flex-direction: column; flex: 1 1 0; min-height: 0; overflow: hidden; }
  #sb-list-wrap { flex: 1 1 0; overflow-y: auto; min-height: 0; scrollbar-gutter: stable; }

  /* MD3 スクロールバー */
  #sb-list-wrap, #sb-detail-body {
    scrollbar-width: thin;
    scrollbar-color: transparent transparent;
    transition: scrollbar-color 0.2s;
  }
  #sb-list-wrap:hover, #sb-list-wrap:focus-within,
  #sb-detail-body:hover, #sb-detail-body:focus-within {
    scrollbar-color: var(--outline-variant) transparent;
  }
  #sb-list-wrap::-webkit-scrollbar,
  #sb-detail-body::-webkit-scrollbar { width: 4px; }
  #sb-list-wrap::-webkit-scrollbar-track,
  #sb-detail-body::-webkit-scrollbar-track { background: transparent; }
  #sb-list-wrap::-webkit-scrollbar-thumb,
  #sb-detail-body::-webkit-scrollbar-thumb {
    background: transparent;
    border-radius: 9999px;
    transition: background 0.2s;
  }
  #sb-list-wrap:hover::-webkit-scrollbar-thumb,
  #sb-list-wrap:focus-within::-webkit-scrollbar-thumb,
  #sb-detail-body:hover::-webkit-scrollbar-thumb,
  #sb-detail-body:focus-within::-webkit-scrollbar-thumb {
    background: var(--outline-variant);
  }
  #sb-list-wrap::-webkit-scrollbar-thumb:hover,
  #sb-detail-body::-webkit-scrollbar-thumb:hover {
    background: var(--on-surface-variant);
  }
  /* ヘッダーをスクロール容器の外に分離 */
  #sb-list-head {
    flex-shrink: 0;
    overflow: hidden;
    border-bottom: 1px solid var(--outline-variant);
    background: var(--surface-container-high);
    /* スクロールバー幅（4px）＋ scrollbar-gutter stable 分を右に確保 */
    padding-right: 4px;
  }
  #sb-list-head-table, #sb-list {
    width: 100%; border-collapse: collapse; font-size: 11.5px;
    table-layout: fixed;
  }
  /* 列幅固定（ヘッダーとボディを揃える） */
  #sb-list-head-table col:nth-child(1), #sb-list col:nth-child(1) { width: auto; }
  #sb-list-head-table col:nth-child(2), #sb-list col:nth-child(2) { width: 38px; }
  #sb-list-head-table col:nth-child(3), #sb-list col:nth-child(3) { width: 46px; }
  #sb-list-head-table col:nth-child(4), #sb-list col:nth-child(4) { width: 40px; }
  #sb-list-head-table col:nth-child(5), #sb-list col:nth-child(5) { width: 24px; }
  #sb-list-head-table th {
    background: var(--surface-container-high);
    padding: 5px 6px; text-align: left;
    color: var(--on-surface-variant); font-weight: 500;
    cursor: pointer; user-select: none; white-space: nowrap;
  }
  #sb-list-head-table th:hover { color: var(--on-surface); }
  #sb-list-head-table th.sort-asc::after  { content: ' ↑'; }
  #sb-list-head-table th.sort-desc::after { content: ' ↓'; }
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
  /* ── Sidebar resize handle (left edge) ── */
  #sb-resize-x {
    position: absolute; left: 0; top: 0; bottom: 0; width: 5px;
    cursor: ew-resize; z-index: 10;
  }
  #sb-resize-x:hover, #sb-resize-x.dragging { background: var(--on-surface-variant); opacity: 0.4; }

  /* ── Detail panel resize handle ── */
  #sb-resize-y {
    flex-shrink: 0; height: 5px; cursor: ns-resize;
    border-top: 1px solid var(--outline-variant);
    display: none;
  }
  #sb-resize-y.sb-active { display: block; }
  #sb-resize-y:hover, #sb-resize-y.dragging { background: var(--on-surface-variant); opacity: 0.4; }
  #sb-detail {
    flex-shrink: 0; height: 260px;
    font-size: 12px; display: none;
    flex-direction: column; overflow: hidden;
  }
  #sb-detail.sb-active { display: flex; }
  #sb-detail-header {
    flex-shrink: 0;
    padding: 10px 14px 8px;
    border-bottom: 1px solid var(--outline-variant);
    background: var(--surface-container);
  }
  #sb-detail-body {
    flex: 1 1 0; overflow-y: auto; min-height: 0;
    padding: 10px 14px 12px;
  }
  /* タイトル行（タイトル + 右側トグル） */
  .sb-detail-title-row {
    display: flex; align-items: flex-start; gap: 8px;
  }
  #sb-detail-title {
    flex: 1;
    font-size: 13px; font-weight: 600; color: var(--on-surface);
    line-height: 1.4;
  }
  .ctx-translate-wrap {
    flex-shrink: 0; display: flex; align-items: center; gap: 5px; align-self: center;
  }
  .sb-kv { display: flex; gap: 6px; margin-bottom: 3px; line-height: 1.5; }
  .sb-k  { color: var(--text-dis); flex-shrink: 0; min-width: 64px; }
  .sb-v  { color: var(--on-surface-variant); word-break: break-all; }
  .sb-v a { color: var(--node-zotero); text-decoration: none; }
  .sb-v a:hover { text-decoration: underline; }
  /* インライン識別子編集 */
  .id-edit-btn, a.id-edit-btn, a.id-edit-btn:visited, a.id-edit-btn:link {
    display: inline-block; margin-left: 5px; vertical-align: middle;
    color: var(--text-dis) !important; cursor: pointer; font-size: 11px;
    background: none; border: none; padding: 0 2px; line-height: 1;
    text-decoration: none !important;
  }
  .id-edit-btn:hover, a.id-edit-btn:hover {
    color: var(--on-surface) !important; text-decoration: none !important;
  }
  .id-edit-input {
    font-size: 12px; background: var(--surface-container-high);
    border: 1px solid var(--node-zotero); border-radius: 3px;
    color: var(--on-surface); padding: 1px 5px;
    width: calc(100% - 28px); outline: none; vertical-align: middle;
  }
  .id-cancel-btn {
    background: none; border: none; cursor: pointer; vertical-align: middle;
    color: var(--text-dis); font-size: 13px; padding: 0 3px; line-height: 1;
  }
  .id-cancel-btn:hover { color: var(--on-surface-variant); }
  /* Zotero編集ヒントモーダル */
  #zotero-edit-hint {
    display: none; position: fixed; inset: 0; z-index: 2000;
    align-items: center; justify-content: center;
    background: rgba(0,0,0,0.45);
  }
  #zotero-edit-hint.show { display: flex; }
  #zotero-edit-hint-box {
    background: var(--surface-container-high);
    border: 1px solid var(--outline-variant); border-radius: 8px;
    padding: 20px 22px; max-width: 340px; width: 90%;
    font-size: 13px; color: var(--on-surface); line-height: 1.6;
  }
  #zotero-edit-hint-box ol {
    margin: 10px 0 14px 18px; padding: 0; color: var(--on-surface-variant);
  }
  #zotero-edit-hint-box ol li { margin-bottom: 4px; }
  .zeh-actions {
    display: flex; align-items: center; justify-content: space-between; gap: 8px; flex-wrap: wrap;
  }
  .zeh-open-btn {
    padding: 5px 14px; border-radius: 5px; border: none; cursor: pointer;
    background: var(--node-zotero); color: #fff; font-size: 12px;
  }
  .zeh-open-btn:hover { opacity: 0.85; }
  .zeh-close-btn {
    padding: 5px 12px; border-radius: 5px; cursor: pointer; font-size: 12px;
    background: none; border: 1px solid var(--outline-variant); color: var(--on-surface-variant);
  }
  .zeh-close-btn:hover { border-color: var(--on-surface-variant); color: var(--on-surface); }
  .zeh-suppress {
    display: flex; align-items: center; gap: 5px;
    font-size: 11px; color: var(--text-dis); cursor: pointer; margin-top: 12px;
  }
  .zeh-suppress input[type=checkbox] { cursor: pointer; }

  /* ── 重複ID警告バナー ── */
  #dup-warn {
    display: none; position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
    z-index: 3000; background: var(--surface-container-high);
    border: 1px solid #c97c00; border-radius: 6px;
    padding: 10px 16px; max-width: 420px; width: 90%;
    font-size: 12px; color: var(--on-surface); box-shadow: 0 4px 16px rgba(0,0,0,0.5);
    display: none; align-items: flex-start; gap: 10px;
  }
  #dup-warn.show { display: flex; }
  #dup-warn .dup-icon { color: #c97c00; font-size: 16px; flex-shrink: 0; line-height: 1.4; }
  #dup-warn .dup-body { flex: 1; line-height: 1.5; }
  #dup-warn .dup-body strong { display: block; margin-bottom: 2px; color: #c97c00; }
  #dup-warn .dup-close { background: none; border: none; cursor: pointer; color: var(--text-dis);
    font-size: 15px; padding: 0 2px; flex-shrink: 0; line-height: 1; align-self: flex-start; }
  #dup-warn .dup-close:hover { color: var(--on-surface); }

  /* メタデータ修正バナー */
  #meta-edit-banner {
    display: none; background: #3b2a00; border: 1px solid #c97c00;
    border-radius: 6px; padding: 8px 12px; margin: 0 0 8px;
    font-size: 11.5px; color: #ffd977; line-height: 1.5;
  }
  #meta-edit-banner.show { display: block; }
  /* メタデータ手動編集フォーム */
  #meta-edit-form {
    display: none; background: var(--surface-container-low);
    border: 1px solid var(--outline-variant); border-radius: 6px;
    padding: 10px 12px; margin: 4px 0 8px; font-size: 12px;
  }
  #meta-edit-form.show { display: block; }
  #meta-edit-form .me-row { margin-bottom: 7px; }
  #meta-edit-form .me-label {
    display: block; font-size: 10.5px; color: var(--on-surface-variant); margin-bottom: 3px;
  }
  #meta-edit-form input[type=text] {
    width: 100%; box-sizing: border-box; padding: 4px 7px; font-size: 12px;
    background: var(--surface-container-high); color: var(--on-surface);
    border: 1px solid var(--outline-variant); border-radius: 4px;
  }
  #meta-edit-form input[type=text]:focus {
    outline: none; border-color: var(--node-zotero);
  }
  #meta-edit-form .me-actions {
    display: flex; gap: 6px; margin-top: 8px; justify-content: flex-end;
  }
  #meta-edit-form .me-save {
    background: var(--node-zotero); color: #fff; border: none;
    border-radius: 4px; padding: 4px 14px; font-size: 12px; cursor: pointer;
  }
  #meta-edit-form .me-save:hover { filter: brightness(1.15); }
  #meta-edit-form .me-cancel {
    background: none; color: var(--on-surface-variant); border: none;
    border-radius: 4px; padding: 4px 10px; font-size: 12px; cursor: pointer;
  }

  .edge-ctx {
    border-left: 2px solid var(--outline-variant); margin-bottom: 10px;
    padding-left: 10px;
  }
  .edge-ctx-page { font-size: 11px; color: var(--text-dis); margin-bottom: 3px; }
  .edge-ctx-snippet { font-size: 12px; color: var(--on-surface-variant); line-height: 1.6; }
  .edge-ctx-loading { font-size: 12px; color: var(--text-dis); padding: 8px 0; }
  .edge-ctx-translation {
    display: none; margin-top: 5px; padding-top: 5px;
    border-top: 1px dashed var(--outline-variant);
    font-size: 12px; color: var(--on-surface-variant); line-height: 1.6;
  }
  .ctx-translate-label { font-size: 11px; color: var(--text-dis); }
  .ctx-toggle-btn {
    width: 34px; height: 19px; border-radius: 10px; border: none; cursor: pointer;
    background: var(--outline-variant); position: relative;
    transition: background 0.2s; flex-shrink: 0; padding: 0;
  }
  .ctx-toggle-btn.on { background: var(--node-zotero); }
  .ctx-toggle-btn::after {
    content: ''; position: absolute; width: 15px; height: 15px;
    border-radius: 50%; background: #fff; top: 2px; left: 2px;
    transition: left 0.2s;
  }
  .ctx-toggle-btn.on::after { left: 17px; }

  /* ── Sidebar tabs ── */
  #sb-tabs {
    display: flex; flex-shrink: 0;
    border-bottom: 1px solid var(--outline-variant);
  }
  .sb-tab {
    flex: 1; padding: 7px 6px; font-size: 12px; cursor: pointer;
    border: none; background: none; color: var(--on-surface-variant);
    border-bottom: 2px solid transparent; transition: color 0.15s;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .sb-tab.active { color: var(--node-zotero); border-bottom-color: var(--node-zotero); font-weight: 500; }
  .sb-tab:hover:not(.active) { color: var(--on-surface); }

  /* ── Context/Abstract pane ── */
  #sb-context-pane {
    flex: 1 1 0; overflow-y: auto; min-height: 0;
    padding: 10px 14px 14px; display: none;
    scrollbar-width: thin; scrollbar-color: transparent transparent;
    transition: scrollbar-color 0.2s;
  }
  #sb-context-pane.active { display: block; }
  #sb-context-pane:hover, #sb-context-pane:focus-within {
    scrollbar-color: var(--outline-variant) transparent;
  }
  #sb-context-pane::-webkit-scrollbar { width: 4px; }
  #sb-context-pane::-webkit-scrollbar-track { background: transparent; }
  #sb-context-pane::-webkit-scrollbar-thumb {
    background: transparent; border-radius: 9999px;
  }
  #sb-context-pane:hover::-webkit-scrollbar-thumb,
  #sb-context-pane:focus-within::-webkit-scrollbar-thumb { background: var(--outline-variant); }
  #sb-context-pane::-webkit-scrollbar-thumb:hover { background: var(--on-surface-variant); }
  /* Context pane empty state */
  .ctx-pane-empty {
    font-size: 12px; color: var(--text-dis); padding: 12px 0; text-align: center;
  }
  /* Context pane header (for edge: translate toggle; for node: title) */
  .ctx-pane-header {
    display: flex; align-items: flex-start; gap: 8px; margin-bottom: 10px; flex-shrink: 0;
  }
  .ctx-pane-title {
    flex: 1; font-size: 12.5px; font-weight: 600; color: var(--on-surface); line-height: 1.4;
  }
  /* Abstract */
  .abstract-text {
    font-size: 12px; color: var(--on-surface-variant); line-height: 1.7;
  }
  .abstract-translation {
    margin-top: 8px; padding-top: 8px; border-top: 1px dashed var(--outline-variant);
    font-size: 12px; color: var(--on-surface-variant); line-height: 1.7; display: none;
  }
  .abstract-translation.show { display: block; }
  /* Summary */
  .summary-section {
    margin-top: 14px; border-top: 1px solid var(--outline-variant); padding-top: 10px;
  }
  .summary-section-label {
    font-size: 10.5px; font-weight: 600; color: var(--on-surface-variant);
    letter-spacing: .04em; text-transform: uppercase; margin-bottom: 6px;
  }
  .summary-text {
    font-size: 12px; color: var(--on-surface-variant); line-height: 1.7; white-space: pre-wrap;
  }
  .summary-actions {
    display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap;
  }
  .summary-btn {
    font-size: 11.5px; padding: 3px 10px; border-radius: 4px; cursor: pointer;
    border: 1px solid var(--outline-variant); background: none; color: var(--on-surface-variant);
  }
  .summary-btn:hover { border-color: var(--on-surface-variant); color: var(--on-surface); }
  .summary-btn.primary {
    background: var(--node-zotero); color: #fff; border-color: transparent;
  }
  .summary-btn.primary:hover { filter: brightness(1.1); }
  .summary-btn:disabled { opacity: 0.5; cursor: default; }
  .summary-model-select {
    font-size: 11px; padding: 3px 6px; border-radius: 4px; cursor: pointer;
    border: 1px solid var(--outline-variant); background: var(--surface-container-high);
    color: var(--on-surface-variant); outline: none; max-width: 150px;
  }
  .summary-model-select:focus { border-color: var(--node-zotero); }
  .summary-textarea {
    width: 100%; box-sizing: border-box; min-height: 100px;
    font-size: 12px; line-height: 1.6;
    background: var(--surface-container-high); border: 1px solid var(--outline-variant);
    border-radius: 4px; color: var(--on-surface); padding: 6px 8px;
    resize: vertical; outline: none; margin-top: 6px;
  }
  .summary-textarea:focus { border-color: var(--node-zotero); }
  .summary-status {
    font-size: 11px; color: var(--text-dis); margin-top: 4px; display: none;
  }
  .summary-status.show { display: block; }

  /* ── Loading overlay ── */
  #loading {
    display: none; position: fixed; inset: 0; z-index: 9999;
    flex-direction: column; align-items: center; justify-content: center;
    background: var(--surface); color: var(--on-surface-variant);
    gap: 16px; font-size: 13px;
  }
  .loading-spinner {
    width: 32px; height: 32px;
    border: 3px solid var(--outline-variant);
    border-top-color: var(--node-zotero);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

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
  #col-layout-badge {
    display: none;
    position: fixed; bottom: 16px; right: calc(var(--sb-width) + 16px); z-index: 1000;
    transition: right 0.25s ease;
    background: var(--surface-container-low);
    border: 1px solid var(--outline-variant);
    border-radius: 4px; padding: 6px 14px;
    font-size: 11.5px; color: var(--on-surface-variant);
    box-shadow: 0 2px 8px rgba(0,0,0,0.5);
  }
  body.sb-collapsed #col-layout-badge { right: 16px; }
  #col-layout-badge span { color: var(--node-zotero); font-weight: 500; margin-left: 4px; }
  #col-layout-badge.show { display: block; }

  /* ── Legend card  (surface dp1) ── */
  #rag-legend {
    position: fixed; top: 16px; left: 16px; z-index: 1000;
    background: var(--surface-container-low);
    border: 1px solid var(--outline-variant);
    border-radius: 4px; padding: 18px 20px 14px;
    color: var(--on-surface); font-size: 13px; min-width: 210px;
    max-height: calc(100vh - 100px); overflow: hidden;
    display: flex; flex-direction: column;
    box-shadow: 0 4px 8px rgba(0,0,0,0.5);
    transition: padding 0.38s ease, min-width 0.38s ease;
  }
  #legend-body {
    overflow-y: auto; overflow-x: hidden;
    flex: 1; min-height: 0;
    transition: max-height 0.38s ease, opacity 0.38s ease;
    max-height: 2000px; opacity: 1;
    /* Thin scrollbar matching sidebar panels */
    scrollbar-width: thin;
    scrollbar-color: transparent transparent;
  }
  #rag-legend.minimized #legend-body { max-height: 0; opacity: 0; }
  #legend-body:hover, #legend-body:focus-within {
    scrollbar-color: var(--outline-variant) transparent;
  }
  #legend-body::-webkit-scrollbar { width: 4px; }
  #legend-body::-webkit-scrollbar-track { background: transparent; }
  #legend-body::-webkit-scrollbar-thumb {
    background: transparent; border-radius: 9999px; transition: background 0.2s;
  }
  #legend-body:hover::-webkit-scrollbar-thumb,
  #legend-body:focus-within::-webkit-scrollbar-thumb {
    background: var(--outline-variant);
  }
  #legend-body::-webkit-scrollbar-thumb:hover {
    background: var(--on-surface-variant);
  }
  #rag-legend h3 { margin: 0 0 14px; font-size: 14px; font-weight: 500; flex-shrink: 0;
                   color: var(--on-surface); letter-spacing: .01em; padding-right: 20px; }
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
  .rl-filter-val { font-size: 11px; color: var(--on-surface); margin-left: auto;
                   font-variant-numeric: tabular-nums; min-width: 36px; text-align: right; }
  .rl-slider {
    width: 100%; margin: 2px 0 0; accent-color: var(--on-surface-variant);
    cursor: pointer;
  }

  /* ── Legend minimize ── */
  #rag-legend.minimized { padding: 10px 14px; min-width: unset; }
  #rag-legend.minimized h3 { margin: 0; }
  #legend-minimize {
    position: absolute; top: 8px; right: 8px;
    width: 20px; height: 20px; border-radius: 4px;
    background: none; border: none; cursor: pointer;
    color: var(--on-surface-variant); font-size: 14px; line-height: 1;
    display: flex; align-items: center; justify-content: center;
  }
  #legend-minimize:hover { color: var(--on-surface); background: var(--surface-container-high); }

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
  <button id="legend-minimize" title="最小化">−</button>
  <h3>Citation Network</h3>
  <div id="legend-body">
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
  <div class="sect-header open" data-section="legend">
    凡例<span class="sect-arrow">▶</span>
  </div>
  <div class="sect-body open" data-section="legend">
  <div style="margin:0 0 4px;font-size:11px;color:var(--text-dis)">ノード選択時の色分け</div>
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
  </div>
  <hr class="rl-divider">
  <div class="sect-header open" data-section="filter">
    フィルタ<span class="sect-arrow">▶</span>
  </div>
  <div class="sect-body open" data-section="filter">
  <div class="rl-filter">
    <label style="display:flex;align-items:center;justify-content:space-between">
      コレクションフィルタ
      <button id="col-reload" title="コレクションを再読み込み" style="
        background:none;border:none;cursor:pointer;font-size:13px;
        color:var(--on-surface-variant);padding:0 2px;line-height:1">↻</button>
    </label>
    <select id="col-filter" style="
      width:100%;margin-top:5px;padding:4px 6px;font-size:11.5px;
      background:var(--surface-container-high);color:var(--on-surface);
      border:1px solid var(--outline-variant);border-radius:4px;cursor:pointer">
      <option value="">（読み込み中…）</option>
    </select>
    <div id="col-filter-msg" style="
      margin-top:5px;font-size:10.5px;color:#f87171;display:none"></div>
  </div>
  <hr class="rl-divider">
  <div class="rl-filter">
    <label>最低被引用数フィルタ</label>
    <div class="rl-filter-row">
      <span class="rl-dot" style="background:{palette['nodeExternal']}"></span>
      <span class="rl-filter-lbl">被引用元 ≥</span>
      <span class="rl-filter-val" id="filter-citer-cc-val">50</span>
    </div>
    <input type="range" class="rl-slider" id="filter-citer-cc" min="0" max="100" value="42">
    <div class="rl-filter-row" style="margin-top:6px">
      <span class="rl-dot" style="background:{palette['nodeExternal']}"></span>
      <span class="rl-filter-lbl">参照先 ≥</span>
      <span class="rl-filter-val" id="filter-ref-cc-val">50</span>
    </div>
    <input type="range" class="rl-slider" id="filter-ref-cc" min="0" max="100" value="42">
  </div>
  </div>
  <hr class="rl-divider">
  <div class="rl-stats">
    <span id="stat-nodes">{n_nodes}</span> nodes &nbsp;·&nbsp; <span>{n_edges}</span> edges
  </div>
  <div class="rl-hint">
    スクロールでズーム · ドラッグで移動<br>
    ホバーで詳細 · クリックでノード選択
  </div>
  <div id="semantic-method-wrap">
    <hr class="rl-divider">
    <label>次元削減手法</label>
    <select id="semantic-method-select">
      <option value="umap">UMAP</option>
      <option value="tsne">t-SNE</option>
      <option value="pca">PCA</option>
      <option value="mds">MDS (cosine)</option>
    </select>
  </div>
  <div id="semantic-clusters-wrap">
    <hr class="rl-divider">
    <label>クラスタ数: <strong id="nclusters-label">8</strong></label>
    <input type="range" id="nclusters-slider" min="2" max="20" value="8" step="1">
  </div>
  <div id="cluster-legend">
    <hr class="rl-divider">
    <div class="cl-header">クラスタ（クリックで絞り込み）</div>
    <div id="cluster-legend-list"></div>
  </div>
  </div>
</div>"""

    # ── sigma.js logic (plain string – no Python vars, no brace escaping) ─────
    # グラフデータは fetch('/api/graph') でオンデマンド取得する。
    logic_script = "<script>\n(function () {\n'use strict';\n\n" + js_theme + """
document.getElementById('loading').style.display = 'flex';

// 経過時間に応じてローディングメッセージを更新する
// キャッシュヒット時はレイアウト再計算が不要なのでシンプルなメッセージにする
var _loadingMsg = document.querySelector('#loading span');
var _loadingMsgsFull = [
  [0,   'グラフデータを構築中…'],
  [5,   'レイアウトを計算中…（初回やノード追加時は数分かかることがあります）'],
  [30,  'レイアウト計算中… しばらくお待ちください'],
  [90,  'もう少しです… FA2 レイアウト最適化中'],
];
var _loadingMsgsCache = [
  [0,   'グラフデータを読み込み中…'],
];
var _loadingMsgs = _loadingMsgsFull;  // キャッシュ状況が判明するまでフル版を使用
var _loadStart = Date.now();
var _loadTimer = setInterval(function() {
  var elapsed = (Date.now() - _loadStart) / 1000;
  var msg = _loadingMsgs[0][1];
  for (var i = 0; i < _loadingMsgs.length; i++) {
    if (elapsed >= _loadingMsgs[i][0]) msg = _loadingMsgs[i][1];
  }
  if (_loadingMsg) _loadingMsg.textContent = msg;
}, 1000);

fetch('/api/graph')
  .then(function(r) { return r.json(); })
  .then(function(GRAPH_DATA) {
    clearInterval(_loadTimer);
    document.getElementById('loading').style.display = 'none';

    // ── ブラウザ閉じ検知（pagehide で close シグナルを送信）─────────────────
    // ページロード時に生存通知を送る（リフレッシュ時に close キャンセル）。
    fetch('/api/heartbeat', {method: 'POST'}).catch(function(){});
    // タブ・ウィンドウを閉じるときに sendBeacon で close シグナルを送信。
    // sendBeacon は pagehide で確実に届く（fetch/XHR は届かないことがある）。
    window.addEventListener('pagehide', function() {
      navigator.sendBeacon('/api/close');
    });

    // 凡例カウントをグラフデータから更新
    var _nAll   = GRAPH_DATA.nodes.length;
    var _nEdges = GRAPH_DATA.edges.length;
    var _nZ  = GRAPH_DATA.nodes.filter(function(n){return n.group==='zotero';}).length;
    var _nC  = GRAPH_DATA.nodes.filter(function(n){return n.group==='external';}).length;
    var _nR  = GRAPH_DATA.nodes.filter(function(n){return n.group==='reference';}).length;
    var _elStatNodes = document.getElementById('stat-nodes');
    var _elStatEdges = document.querySelector('#stat-nodes + *') || null;
    if (_elStatNodes) _elStatNodes.textContent = _nAll;
    var _elCiterVis = document.getElementById('stat-citer-vis');
    var _elRefVis   = document.getElementById('stat-ref-vis');
    if (_elCiterVis) { _elCiterVis.textContent = _nC; _elCiterVis.parentElement.lastChild.textContent = '('+_nC+')'; }
    if (_elRefVis)   { _elRefVis.textContent   = _nR; _elRefVis.parentElement.lastChild.textContent   = '('+_nR+')'; }

    // ── Semantic map state ─────────────────────────────────────────────────
    var _viewMode = 'citation';                     // 'citation' | 'semantic'
    var _semanticMethod = 'umap';                   // 選択中の次元削減手法
    var _nClusters = 8;                             // クラスタ数 (2-20)
    var _semanticPositions = {};                    // { node_id: [x, y] }
    var _citationNodePositions = {};                // 引用ネットワークのノード位置バックアップ
    var _semanticAbort = null;                      // 進行中の fetch をキャンセルするための AbortController
    var _semanticLayoutReady = false;
    var _semanticReqId = 0;                         // リクエスト識別子（古いレスポンスを破棄）
    var _clusterData = [];                          // クラスタ情報 [{id, label, item_keys, hull, color}, ...]
    var _selectedClusterId = -1;                    // -1 = 未選択
    var _clusterItemMap = {};                       // item_key → cluster_id

    // ── Save citation positions immediately (deep copy from graphology) ──
    // Wait for graph to be available, then save.  This runs after the graph is built.
    function _saveCitationPositionsNow() {
      _citationNodePositions = {};
      try {
        graph.forEachNode(function (id, attrs) {
          if (attrs.x != null) {
            _citationNodePositions[id] = [attrs.x, attrs.y];
          }
        });
        console.log('[RAG] saved citation positions: ' + Object.keys(_citationNodePositions).length + ' nodes');
      } catch(e) {
        // graph not ready yet — will retry on first use
        console.warn('[RAG] could not save citation positions (graph not ready):', e);
      }
    }

    function _loadSemanticLayout(method) {
      // 進行中のリクエストがあればキャンセル
      if (_semanticAbort) { _semanticAbort.abort(); }
      _semanticAbort = new AbortController();
      var signal = _semanticAbort.signal;
      var reqId = ++_semanticReqId;

      // badgePct の参照を壊さないよう、label テキストは data-label 属性で管理
      var badge = document.getElementById('layout-badge');
      var badgePctEl = document.getElementById('layout-pct');
      if (badge) {
        badge.style.display = 'block';
        badge.setAttribute('data-label', '意味マップ計算中 (' + method.toUpperCase() + ')');
        if (badgePctEl) badgePctEl.textContent = badge.getAttribute('data-label');
      }
      console.log('[RAG] loading semantic layout: method=' + method + ' n=' + _nClusters + ' reqId=' + reqId);
      fetch('/api/semantic-layout?method=' + encodeURIComponent(method) + '&n_clusters=' + _nClusters, { signal: signal })
        .then(function(r) { return r.json(); })
        .then(function(data) {
          // 古いリクエストの結果は無視
          if (reqId !== _semanticReqId) {
            console.log('[RAG] discarding stale response: reqId=' + reqId + ' current=' + _semanticReqId);
            return;
          }
          _semanticPositions = data.positions || {};
          _semanticLayoutReady = true;
          _semanticMethod = method;

          // クラスタデータ
          _clusterData = data.clusters || [];
          _clusterItemMap = {};
          _clusterData.forEach(function(c) {
            c.item_keys.forEach(function(ik) { _clusterItemMap[ik] = c.id; });
          });
          _buildClusterLegend();

          console.log('[RAG] semantic layout loaded: method=' + method +
                      ' totalPositions=' + Object.keys(_semanticPositions).length +
                      ' nItems=' + (data.n_items || 0) +
                      ' clusters=' + _clusterData.length);
          if (_viewMode === 'semantic') {
            _applySemanticPositions();
          }
          // Draw hulls AFTER positions are applied to graph nodes
          _drawClusterHulls();
          var _b = document.getElementById('layout-badge');
          if (_b) _b.style.display = 'none';
        })
        .catch(function(e) {
          if (e.name === 'AbortError') return;  // キャンセルは無視
          console.error('[RAG] semantic layout fetch failed:', e);
          var _b = document.getElementById('layout-badge');
          if (_b) _b.style.display = 'none';
        });
    }

    function _applySemanticPositions() {
      // Save citation positions on first switch (if not already saved)
      if (Object.keys(_citationNodePositions).length === 0) {
        _saveCitationPositionsNow();
      }

      // 物理シミュレーションを停止（動いたままだと座標が上書きされる）
      if (typeof simulation !== 'undefined' && simulation) {
        simulation.stop();
      }
      if (!layoutDone) {
        layoutDone = true;
        badge.style.display = 'none';
      }

      var pos = _semanticPositions;
      var count = 0;
      graph.forEachNode(function (id, attrs) {
        if (pos[id]) {
          graph.setNodeAttribute(id, 'x', pos[id][0]);
          graph.setNodeAttribute(id, 'y', pos[id][1]);
          count++;
        }
      });
      console.log('[RAG] applied semantic positions: ' + count + ' nodes updated');
      renderer.refresh();

      // ビューポートフィット
      if (selectedNode && typeof window._panToNode === 'function') {
        window._panToNode(selectedNode);
      } else if (typeof _fitView === 'function') {
        _fitView(true);
      }
    }

    function _restoreCitationPositions() {
      // Ensure we have saved positions
      if (Object.keys(_citationNodePositions).length === 0) {
        console.warn('[RAG] no citation positions saved, cannot restore');
        return;
      }
      var cpos = _citationNodePositions;
      var count = 0;
      graph.forEachNode(function (id, attrs) {
        if (cpos[id]) {
          graph.setNodeAttribute(id, 'x', cpos[id][0]);
          graph.setNodeAttribute(id, 'y', cpos[id][1]);
          count++;
        }
      });
      console.log('[RAG] restored citation positions: ' + count + ' nodes');
      renderer.refresh();

      // D3 シミュレーションを再開し、物理演算レイアウトを収束させる
      if (typeof simulation !== 'undefined' && simulation && typeof _addD3Nodes === 'function') {
        // バッジラベルを「再レイアウト中」に
        var _rb = document.getElementById('layout-badge');
        if (_rb) {
          _rb.setAttribute('data-label', '再レイアウト中');
          var _rp = document.getElementById('layout-pct');
          if (_rp) _rp.textContent = '再レイアウト中 0%';
        }
        // d3 state を全ノードで再構築（コレクションフィルタ後でも正しく動作）
        d3nodeById = {};
        d3nodes = [];
        activeNodeIds = new Set();
        graph.forEachNode(function (id) { _addD3Nodes([id]); });
        _rebuildLinks();
        simulation.nodes(d3nodes);
        simulation.force('link').links(d3links);
        layoutDone = false;
        simulation.alpha(0.3);
        var badge = document.getElementById('layout-badge');
        if (badge) badge.style.display = 'block';
        requestAnimationFrame(layoutStep);
      } else {
        if (selectedNode && typeof window._panToNode === 'function') {
          window._panToNode(selectedNode);
        } else if (typeof _fitView === 'function') {
          _fitView(true);
        }
      }
    }

    function switchViewMode(mode) {
      if (mode === _viewMode) return;
      _viewMode = mode;

      document.querySelectorAll('.view-tab').forEach(function(t) {
        t.classList.toggle('active', t.dataset.view === mode);
      });

      var methodWrap = document.getElementById('semantic-method-wrap');
      var clustersWrap = document.getElementById('semantic-clusters-wrap');
      var clusterLegend = document.getElementById('cluster-legend');
      if (mode === 'semantic') {
        // 物理シミュレーションを即座に停止（非同期ロード中も動き続けるのを防ぐ）
        if (typeof simulation !== 'undefined' && simulation) {
          simulation.stop();
        }
        if (!layoutDone) {
          layoutDone = true;
          badge.style.display = 'none';
        }

        // 意味マップでは外部論文が非表示のため、サイドバーもZoteroのみに強制
        var _zoEl = document.getElementById('sb-zotero-only');
        var _zlEl = document.getElementById('sb-zotero-only-label');
        if (_zoEl) {
          if (!('_savedZoteroOnly' in window)) window._savedZoteroOnly = _zoEl.checked;
          _zoEl.checked = true;
          _zoEl.disabled = true;
          if (_zlEl) _zlEl.style.opacity = '0.45';
        }
        if (typeof window._renderList === 'function') window._renderList();

        if (methodWrap) methodWrap.classList.add('show');
        if (clustersWrap) clustersWrap.classList.add('show');
        if (clusterLegend && _clusterData.length > 0) clusterLegend.classList.add('show');
        _drawClusterHulls();
        if (_semanticLayoutReady) {
          _applySemanticPositions();
        } else {
          _loadSemanticLayout(_semanticMethod);
        }
      } else {
        // 引用ネットワークに戻すときはZotero-onlyフィルタを元に戻す
        var _czEl = document.getElementById('sb-zotero-only');
        var _czLbl = document.getElementById('sb-zotero-only-label');
        if (_czEl) {
          _czEl.disabled = false;
          _czEl.checked = window._savedZoteroOnly || false;
          if (_czLbl) _czLbl.style.opacity = '';
        }
        if (typeof window._renderList === 'function') window._renderList();

        if (methodWrap) methodWrap.classList.remove('show');
        if (clustersWrap) clustersWrap.classList.remove('show');
        if (clusterLegend) clusterLegend.classList.remove('show');
        // Clear overlay when switching to citation mode
        var canvas = document.getElementById('cluster-overlay');
        if (canvas) {
          var ggl = canvas.__gl;
          if (ggl) { ggl.clear(ggl.COLOR_BUFFER_BIT); }
          else { var cc = canvas.getContext('2d'); if (cc) cc.clearRect(0, 0, canvas.width, canvas.height); }
        }
        _selectedClusterId = -1;
        if (window._colItemKeys !== null) {
          // クラスタフィルタを解除（コレクションフィルタはそのまま）
          window._colItemKeys = null;
          if (typeof _applyColFilter === 'function') _applyColFilter();
        }
        _restoreCitationPositions();
      }
    }

    // ── Plain Voronoi overlay ──────────────────────────────────────────
    // Each cell is flat-filled with its cluster colour (no stroke).

    function _drawClusterHulls() {
      if (!_clusterData.length) return;

      var canvas = document.getElementById('cluster-overlay');
      if (!canvas) return;
      var dim = renderer.getDimensions();
      if (dim.width <= 0 || dim.height <= 0) return;
      canvas.width = dim.width; canvas.height = dim.height;
      var ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, dim.width, dim.height);

      // Collect screen-space points with cluster colour
      var pts = [];
      _clusterData.forEach(function(c) {
        if (!c.item_keys) return;
        c.item_keys.forEach(function(ik) {
          var pos = _semanticPositions['item:' + ik];
          if (pos) {
            var vp = renderer.graphToViewport({ x: pos[0], y: pos[1] });
            pts.push({ x: vp.x, y: vp.y, color: c.color });
          }
        });
      });
      if (pts.length < 3) return;

      // Delaunay → Voronoi
      var delaunay = d3.Delaunay.from(pts, function(d) { return d.x; }, function(d) { return d.y; });
      var voronoi = delaunay.voronoi([0, 0, dim.width, dim.height]);

      ctx.globalAlpha = 0.35;
      for (var i = 0; i < pts.length; i++) {
        var poly = voronoi.cellPolygon(i);
        if (!poly || poly.length < 3) continue;
        ctx.beginPath();
        ctx.moveTo(poly[0][0], poly[0][1]);
        for (var j = 1; j < poly.length; j++) ctx.lineTo(poly[j][0], poly[j][1]);
        ctx.closePath();
        ctx.fillStyle = pts[i].color;
        ctx.fill();
      }
      ctx.globalAlpha = 1.0;
    }

    // ── Cluster legend builder ─────────────────────────────────────────────
    function _buildClusterLegend() {
      var wrap = document.getElementById('cluster-legend');
      var list = document.getElementById('cluster-legend-list');
      if (!wrap || !list) return;
      if (_viewMode === 'semantic' && _clusterData.length > 0) {
        wrap.classList.add('show');
      } else {
        wrap.classList.remove('show');
        return;
      }
      list.innerHTML = '';
      _clusterData.forEach(function(c) {
        var kw = c.keywords && c.keywords.length ? c.keywords.join(', ') : '';

        var row = document.createElement('div');
        row.className = 'cl-row' + (c.id === _selectedClusterId ? ' active' : '');
        row.innerHTML = '<span class="cl-dot" style="background:' + c.color + '"></span>' +
                        '<span class="cl-label">' + c.label + '</span>' +
                        '<span class="cl-count">' + c.item_keys.length + '</span>' +
                        (kw ? '<span class="cl-keywords">' + kw + '</span>' : '');
        row.addEventListener('click', function() {
          _selectCluster(c.id === _selectedClusterId ? -1 : c.id);
        });
        list.appendChild(row);
      });
    }

    function _selectCluster(clusterId) {
      _selectedClusterId = clusterId;
      _buildClusterLegend();
      _drawClusterHulls();

      if (clusterId >= 0) {
        var cluster = _clusterData.find(function(c) { return c.id === clusterId; });
        if (cluster && cluster.item_keys) {
          window._colItemKeys = new Set(cluster.item_keys);
        }
      } else {
        window._colItemKeys = null;
      }
      renderer.refresh();
      if (typeof window._renderList === 'function') window._renderList();
    }

    // ── Cluster click detection (via sigma clickStage) ──────────────────────
    // Canvas overlay は pointer-events: none のまま。パン/ズームは sigma が処理する。
    var _lastCanvasClick = { x: 0, y: 0 };
    document.getElementById('sigma-container').addEventListener('click', function(ev) {
      _lastCanvasClick.x = ev.offsetX;
      _lastCanvasClick.y = ev.offsetY;
    });

    // Camera-follow handler registered below, after sigma initialization.

    function _pointInPolygon(x, y, polygon) {
      var inside = false;
      for (var i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        var xi = polygon[i][0], yi = polygon[i][1];
        var xj = polygon[j][0], yj = polygon[j][1];
        if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
          inside = !inside;
        }
      }
      return inside;
    }

    // View tab click handlers
    document.querySelectorAll('.view-tab').forEach(function(tab) {
      tab.addEventListener('click', function() {
        switchViewMode(tab.dataset.view);
      });
    });

    // Semantic method dropdown handler
    var _semanticMethodSelect = document.getElementById('semantic-method-select');
    if (_semanticMethodSelect) {
      _semanticMethodSelect.addEventListener('change', function() {
        _semanticMethod = this.value;
        _semanticLayoutReady = false;  // キャッシュがなければ再計算
        _loadSemanticLayout(_semanticMethod);
      });
    }

    // ── Cluster count slider ──
    var _nClustersSlider = document.getElementById('nclusters-slider');
    var _nClustersLabel = document.getElementById('nclusters-label');
    var _nClustersTimer = null;
    if (_nClustersSlider && _nClustersLabel) {
      _nClustersSlider.addEventListener('input', function() {
        _nClusters = parseInt(this.value);
        _nClustersLabel.textContent = _nClusters;
        // Debounce: wait 400ms after last change before re-fetching
        if (_nClustersTimer) clearTimeout(_nClustersTimer);
        _nClustersTimer = setTimeout(function() {
          if (_viewMode === 'semantic') {
            // Re-fetch with new n_clusters — layout cache still applies,
            // only clustering is recomputed server-side
            _loadSemanticLayout(_semanticMethod);
          }
        }, 400);
      });
    }

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

// Save initial FA2/phyllotaxis positions for citation-network restore
_saveCitationPositionsNow();

/* ── 3. Create sigma renderer ───────────────────────────── */
const SigmaClass = (typeof Sigma === 'function') ? Sigma
                 : (Sigma && typeof Sigma.Sigma === 'function') ? Sigma.Sigma
                 : null;
if (!SigmaClass) { console.error('[RAG] sigma.js not loaded'); return; }

// ── コレクションフィルタ状態（nodeReducer から参照）────────────────────────
window._colItemKeys = null;  // null = フィルタなし、Set = 表示対象 item_key

// ── エッジホバー状態 ─────────────────────────────────────────────────────────
var hoveredEdge = null;

// ── 翻訳トグル状態（エッジ切替後も保持）──────────────────────────────────────
var _translateToggle = false;

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
var _activeEdge            = null;       // クリックされたエッジ（コンテキスト表示中は強調維持）

// ── 選択トランジション ───────────────────────────────────────────────────────
// _selectionT: 0 = 選択直後（未適用）→ 1 = 完全適用
// nodeReducer でこの値を使い、非選択ノードの色を背景色へ補間する。
var _selectionT    = 0;
var _selectionAnimId = null;
var _SELECTION_DUR = 380;  // ms
var _BG_COLOR      = THEME.surface || '#141218';

// 解除アニメーション中に直前の選択状態を保持する変数
var _prevSelectedNode    = null;
var _prevSelectedNeighbors   = new Set();
var _prevCiterNodes      = new Set();
var _prevRefNodes        = new Set();

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
  // 直前の選択状態を保存（解除アニメーション中に参照）
  _prevSelectedNode      = selectedNode;
  _prevSelectedNeighbors = new Set(selectedNeighbors);
  _prevCiterNodes        = new Set(selectedCiterNodes);
  _prevRefNodes          = new Set(selectedRefNodes);
  selectedNode           = null;
  selectedNeighbors      = new Set();
  selectedChunkNodes     = new Set();
  selectedChunkNeighbors = new Set();
  selectedCiterNodes     = new Set();
  selectedRefNodes       = new Set();
  // すでに完全解除済みなら即終了
  if (startT <= 0) { _selectionT = 0; _prevSelectedNode = null; return; }
  function step() {
    var raw = Math.min(1, (Date.now() - start) / (_SELECTION_DUR * 2));
    var eased = 1 - (1 - raw) * (1 - raw);
    _selectionT = startT * (1 - eased);
    renderer.refresh();
    if (raw < 1) { _selectionAnimId = requestAnimationFrame(step); }
    else { _selectionT = 0; _selectionAnimId = null; _prevSelectedNode = null; }
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
// Group base z-index: Zotero items always render in front of external nodes (sigma v3 zIndex).
var _Z_BASE = { zotero: 20, chunk: 20, external: 10, reference: 10 };

function nodeReducer(node, data) {
  var res = Object.assign({}, data);

  // ── Base z-index by group ─────────────────────────────────────────────────
  res.zIndex = _Z_BASE[data.group] || 10;

  // ── Collection filter ────────────────────────────────────────────────────
  if (window._colItemKeys !== null) {
    if (data.group === 'zotero' || data.group === 'chunk') {
      // Zotero アイテム：コレクション外なら非表示
      // itemKey フィールド優先、なければノードID "item:XXXXX" から抽出
      var nkey = data.itemKey || (node.indexOf(':') >= 0 ? node.split(':')[1] : node);
      if (!window._colItemKeys.has(nkey)) {
        res.hidden = true;
        return res;
      }
    } else {
      // 外部論文：フィルタ後の Zotero ノードに接続していなければ非表示
      var hasVisibleNeighbor = false;
      try {
        graph.forEachNeighbor(node, function(nid, ndata) {
          if (ndata.group === 'zotero' || ndata.group === 'chunk') {
            var nk = ndata.itemKey || nid;
            if (window._colItemKeys.has(nk)) { hasVisibleNeighbor = true; }
          }
        });
      } catch(_) {}
      if (!hasVisibleNeighbor) {
        res.hidden = true;
        return res;
      }
    }
  }

  // ── CC threshold filter ───────────────────────────────────────────────────
  if (data.group === 'external' && filterCiterCC > 0 && (data.cc || 0) < filterCiterCC) {
    res.hidden = true;
    return res;
  }
  if (data.group === 'reference' && filterRefCC > 0 && (data.cc || 0) < filterRefCC) {
    res.hidden = true;
    return res;
  }

  // ── Semantic map mode: hide external/reference nodes, color by cluster ────
  if (_viewMode === 'semantic') {
    if (data.group === 'external' || data.group === 'reference') {
      res.hidden = true;
      return res;
    }
    // In semantic mode, use same colors as citation mode
    if (data.group === 'zotero') {
      res.color = THEME.nodeZotero;
    }
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
    if (selectedNode !== null) {
      // ── 選択中 ──────────────────────────────────────────────
      if (node === selectedNode) {
        res.highlighted = true;
        res.zIndex = (res.zIndex || 10) + 10;
      } else if (selectedNeighbors.has(node) || selectedChunkNeighbors.has(node)) {
        res.zIndex = (res.zIndex || 10) + 5;
        var grp = data.group;
        if (grp === 'external' || grp === 'reference') {
          if (selectedCiterNodes.has(node)) res.color = THEME.nodeCiter;
          else if (selectedRefNodes.has(node)) res.color = THEME.nodeRef;
        }
      } else {
        if (_selectionT >= 1) { res.hidden = true; }
        else {
          res.color = _hexLerp(data.color || res.color, _BG_COLOR, _selectionT);
          res.size  = res.size * (1 - _selectionT * 0.8);
        }
        res.label = '';
      }
    } else {
      // ── 解除アニメーション中（selectedNode === null, _selectionT > 0）──
      // 直前の選択ノード・隣接は表示したまま色だけ元に戻す
      if (node === _prevSelectedNode) {
        res.zIndex = (res.zIndex || 10) + 10;
        // highlighted色 → 通常色へ補間（_selectionT: 1→0）
      } else if (_prevSelectedNeighbors.has(node)) {
        res.zIndex = (res.zIndex || 10) + 5;
        var grp2 = data.group;
        if (grp2 === 'external' || grp2 === 'reference') {
          var fromColor = _prevCiterNodes.has(node) ? THEME.nodeCiter
                        : _prevRefNodes.has(node)   ? THEME.nodeRef
                        : data.color;
          res.color = _hexLerp(data.color || res.color, fromColor, _selectionT);
        }
      } else {
        // 非選択ノード: 背景色から通常色へフェードイン
        res.color = _hexLerp(data.color || res.color, _BG_COLOR, _selectionT);
        res.size  = res.size * (1 - _selectionT * 0.8);
        res.label = '';
      }
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

  // ── Semantic map mode: hide edges involving external nodes ────────────────
  if (_viewMode === 'semantic') {
    try {
      var _sga = graph.getNodeAttributes(src), _tga = graph.getNodeAttributes(tgt);
      if (_sga.group === 'external' || _sga.group === 'reference' ||
          _tga.group === 'external' || _tga.group === 'reference') {
        res.hidden = true;
        return res;
      }
    } catch(_) {}
  }

  if (selectedNode !== null) {
    // ── 選択中 ──────────────────────────────────────────────────────────────
    if (src === selectedNode || tgt === selectedNode ||
        selectedChunkNodes.has(src) || selectedChunkNodes.has(tgt)) {
      res.zIndex = 1;
      if (tgt === selectedNode || selectedChunkNodes.has(tgt)) {
        res.color = THEME.edgeCitation;
      } else if (src === selectedNode || selectedChunkNodes.has(src)) {
        res.color = THEME.edgeReference;
      }
      // アクティブエッジ（コンテキスト表示中）はシアンで太く強調
      if (edge === _activeEdge) {
        res.color  = '#00E5FF';
        res.size   = (data.size || 1) * 3.0;
        res.zIndex = 3;
      }
      // ホバー中エッジを白+太く強調（アクティブより優先）
      if (edge === hoveredEdge) {
        res.color  = '#ffffff';
        res.size   = (data.size || 1) * 3.5;
        res.zIndex = 4;
      }
    } else {
      res.hidden = true;
    }
  } else if (_prevSelectedNode !== null && _selectionT > 0) {
    // ── 解除アニメーション中: 直前の選択エッジは表示したまま色を戻す ──────
    if (src === _prevSelectedNode || tgt === _prevSelectedNode) {
      res.zIndex = 1;
      var fromColor = (tgt === _prevSelectedNode) ? THEME.edgeCitation
                    : (src === _prevSelectedNode)  ? THEME.edgeReference
                    : data.color;
      res.color = _hexLerp(data.color || res.color, fromColor, _selectionT);
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
  zIndex:                true,
  renderEdgeLabels:      false,
  defaultEdgeType:       'arrow',
  labelFont:             'Inter, system-ui, sans-serif',
  labelSize:             12,
  labelWeight:           '500',
  labelThreshold:        7,
  labelColor:            { color: THEME.onSurfaceVariant },
  defaultNodeColor:      THEME.nodeZotero,
  defaultEdgeColor:      THEME.edgeDefault,
  minCameraRatio:        0.01,
  maxCameraRatio:        20,
  // sigma 3.x ではエッジイベント検出は単一の enableEdgeEvents で制御する。
  // v2 の enableEdgeClickEvents / enableEdgeHoverEvents は設定キーとして残るが
  // 当たり判定を有効化しないため、これを true にしないとエッジをクリックできない。
  enableEdgeEvents: true,
  defaultDrawNodeHover:  customDrawHover,
  nodeReducer:           nodeReducer,
  edgeReducer:           edgeReducer,
});
var camera = renderer.getCamera();

// ── Camera-follow for Voronoi overlay ──
var _hullRedrawTimer = null;
camera.on('updated', function() {
  if (_viewMode !== 'semantic' || !_clusterData.length) return;
  if (_hullRedrawTimer) clearTimeout(_hullRedrawTimer);
  _hullRedrawTimer = setTimeout(_drawClusterHulls, 50);
});

// Fix sigma's normalisation to the pre-computed bbox so D3 movements
// produce stable framed-coordinate changes (visible animation).
// bboxR is computed at the top based on graph size.
renderer.setCustomBBox({ x: [-bboxR, bboxR], y: [-bboxR, bboxR] });
// フェードインループに refresh を接続
_fadeInRefresh = function() { renderer.refresh(); };
// debug hooks
window.__g = graph; window.__r = renderer; window.__bboxR = bboxR;
window.__d3nodes = d3nodes; window.__d3nById = d3nodeById;
// window.__sim は simulation 宣言後に設定（下方で代入）


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

// Edge hover – only highlight when a node is selected (avoids misclicks in dense areas)
renderer.on('enterEdge', function (ev) {
  if (selectedNode === null) return;
  hoveredEdge = ev.edge;
  renderer.refresh();
});
renderer.on('leaveEdge', function () {
  if (hoveredEdge === null) return;
  hoveredEdge = null;
  renderer.refresh();
});

// Edge click – fetch citation contexts and show in context pane
renderer.on('clickEdge', function (ev) {
  if (selectedNode === null) return;
  var edge = ev.edge;
  _activeEdge = edge;
  renderer.refresh();
  var src  = graph.source(edge), tgt = graph.target(edge);
  var srcA = graph.getNodeAttributes(src);
  var tgtA = graph.getNodeAttributes(tgt);
  var edgeA = graph.getEdgeAttributes(edge);
  var myReq = ++_ctxPaneReq;
  var reportButton = edgeA.relationKey
    ? '<button id="relation-report-btn" style="margin-left:auto;background:none;' +
      'border:1px solid var(--outline-variant);border-radius:4px;color:var(--on-surface-variant);' +
      'padding:3px 7px;cursor:pointer;font-size:10.5px">誤りを報告</button>'
    : '';

  _showContextPane(
    '<div class="ctx-pane-header">' +
      '<div class="ctx-pane-title">' +
        esc(srcA.fullTitle || srcA.label) +
        ' <span style="opacity:.6;font-size:0.85em">→</span> ' +
        esc(tgtA.fullTitle || tgtA.label) +
      '</div>' +
      reportButton +
      '<div class="ctx-translate-wrap">' +
        '<span class="ctx-translate-label">翻訳</span>' +
        '<button class="ctx-toggle-btn' + (_translateToggle ? ' on' : '') + '" id="ctx-toggle"></button>' +
      '</div>' +
    '</div>' +
    '<div id="ctx-body"><div class="edge-ctx-loading">読み込み中…</div></div>'
  );

  var relationReportBtn = document.getElementById('relation-report-btn');
  if (relationReportBtn) {
    relationReportBtn.addEventListener('click', function() {
      var details = window.prompt(
        '誤りと思う具体的な根拠を入力してください。\n' +
        '分野が違うという印象だけでなく、原資料に存在しない、別著作の識別子である、などを記載してください。'
      );
      if (!details || !details.trim()) return;
      relationReportBtn.disabled = true;
      relationReportBtn.textContent = '報告中…';
      fetch('/api/relation/report', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          direction: edgeA.direction,
          item_key: edgeA.itemKey,
          external_paper_id: edgeA.externalPaperId,
          external_title: edgeA.externalTitle || '',
          reason: 'other',
          details: details.trim()
        })
      }).then(function(r) { return r.json(); }).then(function(d) {
        if (d.error) throw new Error(d.error);
        relationReportBtn.textContent = '報告済み';
        relationReportBtn.title = 'メンテナンス時の確認待ちです';
      }).catch(function(e) {
        relationReportBtn.disabled = false;
        relationReportBtn.textContent = '誤りを報告';
        window.alert('報告できませんでした: ' + e.message);
      });
    });
  }

  fetch('/api/edge/contexts?src=' + encodeURIComponent(src) + '&tgt=' + encodeURIComponent(tgt))
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (myReq !== _ctxPaneReq) return;  // 別のノード/エッジに切り替え済み
      var ctxs = data.contexts || [];
      var ctxBody = document.getElementById('ctx-body');
      if (!ctxBody) return;

      function _renderContexts() {
        var html = '';
        if (ctxs.length === 0) {
          html = '<div style="padding:8px 0;opacity:.6;font-size:0.85em">引用コンテクスト情報なし</div>';
        } else {
          ctxs.forEach(function(c, i) {
            html += '<div class="edge-ctx">';
            if (c.page) html += '<div class="edge-ctx-page">p.' + esc(String(c.page)) + '</div>';
            html += '<div class="edge-ctx-snippet">' + esc(c.snippet || '') + '</div>';
            html += '<div class="edge-ctx-translation" data-idx="' + i + '" style="display:none">' +
                    (c.translation ? esc(c.translation) : '<span style="opacity:.5">翻訳中…</span>') +
                    '</div>';
            html += '</div>';
          });
        }
        ctxBody.innerHTML = html;

        var toggleBtn = document.getElementById('ctx-toggle');
        if (toggleBtn) {
          toggleBtn.addEventListener('click', function() {
            _translateToggle = !_translateToggle;
            toggleBtn.className = 'ctx-toggle-btn' + (_translateToggle ? ' on' : '');
            _applyToggle();
          });
        }
        if (_translateToggle) { _applyToggle(); }
      }

      function _applyToggle() {
        var ctxBodyEl = document.getElementById('ctx-body');
        if (!ctxBodyEl) return;
        if (!_translateToggle) {
          ctxBodyEl.querySelectorAll('.edge-ctx-translation').forEach(function(d) {
            d.style.display = 'none';
          });
          return;
        }
        ctxs.forEach(function(c, i) {
          var div = ctxBodyEl.querySelector('.edge-ctx-translation[data-idx="' + i + '"]');
          if (!div) return;
          if (c.translation) { div.textContent = c.translation; div.style.display = 'block'; }
        });
        var pending = [];
        ctxs.forEach(function(c, i) {
          if (!c.translation && (c.snippet || '')) pending.push({ idx: i, snippet: c.snippet });
        });
        if (pending.length === 0) return;
        pending.forEach(function(p) {
          var div = ctxBodyEl.querySelector('.edge-ctx-translation[data-idx="' + p.idx + '"]');
          if (div) { div.innerHTML = '<span style="opacity:.5">翻訳中…</span>'; div.style.display = 'block'; }
        });
        fetch('/api/translate/batch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ texts: pending.map(function(p) { return p.snippet; }) })
        })
          .then(function(r) { return r.json(); })
          .then(function(d) {
            if (d.error) {
              pending.forEach(function(p) {
                var div = ctxBodyEl.querySelector('.edge-ctx-translation[data-idx="' + p.idx + '"]');
                if (div) { div.style.color = '#f87171'; div.textContent = 'エラー: ' + d.error; }
              });
            } else {
              pending.forEach(function(p, j) {
                ctxs[p.idx].translation = d.translations[j];
                var div = ctxBodyEl.querySelector('.edge-ctx-translation[data-idx="' + p.idx + '"]');
                if (div) { div.style.color = ''; div.textContent = d.translations[j]; }
              });
            }
          })
          .catch(function(e) {
            pending.forEach(function(p) {
              var div = ctxBodyEl.querySelector('.edge-ctx-translation[data-idx="' + p.idx + '"]');
              if (div) { div.style.color = '#f87171'; div.textContent = 'エラー: ' + e.message; }
            });
          });
      }

      _renderContexts();
    })
    .catch(function(err) {
      if (myReq !== _ctxPaneReq) return;
      var ctxBody = document.getElementById('ctx-body');
      if (ctxBody) ctxBody.innerHTML =
        '<div style="color:#f87171;font-size:12px">エラー: ' + esc(err.message) + '</div>';
    });
});

// Click: toggle selection focus (same node again → deselect)
renderer.on('clickNode', function (ev) {
  if (hasDragged) return;
  _activeEdge = null;
  if (selectedNode === ev.node) { clearSelection(); }
  else {
    recomputeSelection(ev.node);
    if (window._panToNode) window._panToNode(ev.node);
    // Show abstract in context pane (zotero → 本文+AI要約 / 外部論文 → S2概要)
    var n = GRAPH_DATA.nodes.find(function(nd) { return nd.id === ev.node; });
    if (n && n.group === 'zotero' && n.itemKey) {
      _showNodeAbstract(ev.node);
    } else if (n && (n.group === 'external' || n.group === 'reference')) {
      _showExternalAbstract(ev.node);
    }
  }
});

// Click on empty canvas → deselect, or cluster-filter in semantic mode
renderer.on('clickStage', function () {
  _activeEdge = null;

  // 意味マップモード: クラスタクリック検出
  if (_viewMode === 'semantic' && _clusterData.length) {
    var dim = renderer.getDimensions();
    var rx = _lastCanvasClick.x / dim.width;
    var ry = _lastCanvasClick.y / dim.height;
    var gpos = renderer.viewportToGraph({ x: rx, y: ry });
    var hitCluster = false;
    for (var ci = 0; ci < _clusterData.length; ci++) {
      var hull = _clusterData[ci].hull;
      if (!hull || hull.length < 3) continue;
      if (_pointInPolygon(gpos.x, gpos.y, hull)) {
        _selectCluster(_selectedClusterId === _clusterData[ci].id ? -1 : _clusterData[ci].id);
        hitCluster = true;
        break;
      }
    }
    if (hitCluster) return;
    if (_selectedClusterId >= 0) _selectCluster(-1);
    // クラスタ外クリック時はノード選択解除にフォールスルー
  }

  if (selectedNode !== null) { clearSelection(); }
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

// ノードのdown: ドラッグ開始準備（意味マップモードでは無効）
renderer.on('downNode', function (ev) {
  if (_viewMode === 'semantic') return;  // 意味マップではドラッグ不可
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
    if (layoutDone && _viewMode !== 'semantic') {
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

window.__sim = simulation;  // simulation 宣言後にグローバル公開

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
  var _label = badge.getAttribute('data-label') || 'レイアウト計算中';
  badgePct.textContent = _label + ' ' + pct + '%';
  // skipボタンはレイアウト中のみ表示
  var _skip = document.getElementById('layout-skip');
  if (_skip) _skip.style.display = '';
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
    var _lbl2 = badge.getAttribute('data-label') || 'レイアウト計算中';
    if (_batchQueue.length > 0) {
      badgePct.textContent = _lbl2 + ' ' + Math.round(d3nodes.length / nNodes * 100) + '%';
    } else {
      badgePct.textContent = _lbl2 + ' ' + Math.round((1 - simulation.alpha()) * 100) + '%';
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

// ── Legend minimize ──────────────────────────────────────────────────────────
(function() {
  var legend = document.getElementById('rag-legend');
  var btn    = document.getElementById('legend-minimize');
  btn.addEventListener('click', function() {
    var minimized = legend.classList.toggle('minimized');
    btn.textContent = minimized ? '+' : '−';
    btn.title       = minimized ? '展開' : '最小化';
  });

  // Section collapse/expand
  legend.querySelectorAll('.sect-header').forEach(function(hdr) {
    hdr.addEventListener('click', function() {
      var section = this.dataset.section;
      var body = legend.querySelector('.sect-body[data-section="' + section + '"]');
      var isOpen = this.classList.toggle('open');
      body.classList.toggle('open', isOpen);
    });
  });
})();

// ── Collection filter ────────────────────────────────────────────────────────
(function() {
  var sel     = document.getElementById('col-filter');
  var msg     = document.getElementById('col-filter-msg');
  var reloadBtn = document.getElementById('col-reload');

  function _applyColFilter() {
    if (window.__r) window.__r.refresh();
    if (window._renderList) window._renderList();
    // 表示ノードだけでレイアウトを再計算
    _relayoutVisible();
  }

  function _isNodeVisible(id, attrs) {
    // CCフィルタチェック
    if (attrs.group === 'external'  && filterCiterCC > 0 && (attrs.cc || 0) < filterCiterCC) return false;
    if (attrs.group === 'reference' && filterRefCC   > 0 && (attrs.cc || 0) < filterRefCC)   return false;

    var colKeys = window._colItemKeys;
    if (colKeys === null) return true;
    if (attrs.group === 'zotero' || attrs.group === 'chunk') {
      var nk = attrs.itemKey || (id.indexOf(':') >= 0 ? id.split(':')[1] : id);
      return colKeys.has(nk);
    }
    // 外部論文: フィルタ後の Zotero ノードに隣接していれば表示
    var ok = false;
    try {
      window.__g.forEachNeighbor(id, function(nid, nd) {
        if (nd.group === 'zotero' || nd.group === 'chunk') {
          var nk2 = nd.itemKey || (nid.indexOf(':') >= 0 ? nid.split(':')[1] : nid);
          if (colKeys.has(nk2)) ok = true;
        }
      });
    } catch(_) {}
    return ok;
  }

  var _colSimRaf = null;
  var _colBadge  = document.getElementById('col-layout-badge');
  var _colPct    = document.getElementById('col-layout-pct');

  function _relayoutVisible() {
    console.log('[col-relayout] start, __sim=', !!window.__sim, '__g=', !!window.__g);
    if (!window.__sim || !window.__g) return;
    var g   = window.__g;
    var sim = window.__sim;

    if (_colSimRaf) { cancelAnimationFrame(_colSimRaf); _colSimRaf = null; }
    _colBadge.classList.remove('show');
    sim.stop();

    var visIds = [];
    g.forEachNode(function(id, attrs) {
      if (_isNodeVisible(id, attrs)) visIds.push(id);
    });
    console.log('[col-relayout] visible nodes:', visIds.length, '/', g.order);
    if (visIds.length === 0) return;

    var visSet   = new Set(visIds);
    var visNodes = visIds.map(function(id) {
      var a = g.getNodeAttributes(id);
      return { id: id, x: a.x || 0, y: a.y || 0 };
    });
    var visLinks = [];
    g.forEachEdge(function(e, attrs, src, tgt) {
      if (visSet.has(src) && visSet.has(tgt))
        visLinks.push({ source: src, target: tgt });
    });
    console.log('[col-relayout] vis links:', visLinks.length, 'alpha before:', sim.alpha());

    sim.nodes(visNodes);
    sim.force('link').links(visLinks);
    sim.alpha(0.6);
    console.log('[col-relayout] alpha after set:', sim.alpha(), 'alphaMin:', sim.alphaMin());

    var _startAlpha = sim.alpha();
    _colBadge.classList.add('show');
    _colPct.textContent = '0%';

    var _tickCount = 0;
    var _fitTriggered = false;
    function _colTick() {
      for (var i = 0; i < 4; i++) sim.tick();
      _tickCount++;
      visNodes.forEach(function(n) {
        g.setNodeAttribute(n.id, 'x', n.x);
        g.setNodeAttribute(n.id, 'y', n.y);
      });
      window.__r.refresh();
      var pct = Math.min(99, Math.round((1 - sim.alpha() / _startAlpha) * 100));
      _colPct.textContent = pct + '%';
      if (_tickCount === 1) console.log('[col-relayout] first tick done, alpha=', sim.alpha());
      if (!_fitTriggered && pct >= 80) {
        _fitTriggered = true;
        _fitVisibleNodes(true);
      }
      if (sim.alpha() > sim.alphaMin()) {
        _colSimRaf = requestAnimationFrame(_colTick);
      } else {
        console.log('[col-relayout] converged after', _tickCount, 'frames');
        _colPct.textContent = '100%';
        setTimeout(function() { _colBadge.classList.remove('show'); }, 600);
        _colSimRaf = null;
      }
    }
    _colSimRaf = requestAnimationFrame(_colTick);
    console.log('[col-relayout] rAF scheduled, id=', _colSimRaf);
  }

  function _fitVisibleNodes(animated) {
    if (!window.__r || !window.__g) return;
    var minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    window.__g.forEachNode(function(id, attrs) {
      if (_isNodeVisible(id, attrs) && attrs.x != null) {
        if (attrs.x < minX) minX = attrs.x; if (attrs.x > maxX) maxX = attrs.x;
        if (attrs.y < minY) minY = attrs.y; if (attrs.y > maxY) maxY = attrs.y;
      }
    });
    if (!isFinite(minX)) return;
    var bboxR2 = window.__bboxR || 1;
    var cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    var dim = window.__r.getDimensions();
    var ratio2 = 2 * bboxR2;
    var newRatio = Math.max(
      (maxX - minX) / (0.85 * ratio2),
      (maxY - minY) * dim.width / (0.85 * ratio2 * dim.height)
    );
    newRatio = Math.max(0.1, Math.min(newRatio, 8));
    var target = { x: cx / ratio2 + 0.5, y: cy / ratio2 + 0.5, ratio: newRatio };
    var cam = window.__r.getCamera();
    if (animated) cam.animate(target, { duration: 500, easing: 'quadraticInOut' });
    else cam.setState(target);
  }

  function _loadCollections() {
    sel.innerHTML = '<option value="">（読み込み中…）</option>';
    msg.style.display = 'none';
    fetch('/api/collections').then(function(r) { return r.json(); }).then(function(data) {
      if (data.error) {
        sel.innerHTML = '<option value="">（取得失敗）</option>';
        msg.textContent = data.message ||
          'Zotero を起動し、ローカルサーバーが立ち上がっているか確認してください。';
        msg.style.display = 'block';
        return;
      }
      var cols = data.collections || [];
      sel.innerHTML = '<option value="">すべてのコレクション</option>';
      cols.forEach(function(c) {
        var opt = document.createElement('option');
        opt.value = c.id;
        opt.textContent = c.path;
        opt.dataset.keys = JSON.stringify(c.item_keys);
        sel.appendChild(opt);
      });
    }).catch(function(e) {
      sel.innerHTML = '<option value="">（取得失敗）</option>';
      msg.textContent = 'Zotero を起動し、データベースにアクセスできるか確認してください。';
      msg.style.display = 'block';
    });
  }

  sel.addEventListener('change', function() {
    var opt = sel.options[sel.selectedIndex];
    if (!opt || opt.value === '') {
      window._colItemKeys = null;
      _applyColFilter();
      // フィルタ解除: 全ノードでレイアウト再計算
      if (window.__sim && window.__g) {
        var allNodes = [];
        var allById  = {};
        window.__g.forEachNode(function(id, attrs) {
          var n = { id: id, x: attrs.x || 0, y: attrs.y || 0 };
          allNodes.push(n); allById[id] = n;
        });
        var allLinks = [];
        window.__g.forEachEdge(function(e, attrs, src, tgt) {
          allLinks.push({ source: src, target: tgt });
        });
        window.__sim.stop();
        window.__sim.nodes(allNodes);
        window.__sim.force('link').links(allLinks);
        window.__sim.alpha(0.4).restart();
      }
    } else {
      var keys = JSON.parse(opt.dataset.keys || '[]');
      window._colItemKeys = new Set(keys);
      _applyColFilter();
    }
  });

  reloadBtn.addEventListener('click', _loadCollections);

  // グラフデータが届いたあとに読み込む（GRAPH_DATA が確定してから）
  // フォールバックとして DOMContentLoaded 後に即時実行
  _loadCollections();
  window._relayoutVisible = _relayoutVisible;
})();

// ── CC filter sliders (対数スケール: slider 0-100 → value 0-10000) ───────────
(function() {
  // slider=0 → 0, slider=25 → 10, slider=50 → 100, slider=75 → 1000, slider=100 → 10000
  function _sliderToVal(s) {
    s = parseInt(s);
    return s <= 0 ? 0 : Math.round(Math.pow(10, s / 25));
  }
  function _updateLabel(sliderId, labelId) {
    var v = _sliderToVal(document.getElementById(sliderId).value);
    var el = document.getElementById(labelId);
    if (el) el.textContent = v.toLocaleString();
    return v;
  }
  function _applyFilter() {
    filterCiterCC = _sliderToVal(document.getElementById('filter-citer-cc').value);
    filterRefCC   = _sliderToVal(document.getElementById('filter-ref-cc').value);
    renderer.refresh();

    // 表示数カウント更新（被引用元・参照先それぞれ）
    var visCiter = 0, visRef = 0, visTotal = 0;
    graph.forEachNode(function(n, a) {
      if (a.group === 'zotero') { visTotal++; return; }
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

    if (layoutDone && window._relayoutVisible) window._relayoutVisible();
  }
  // ラベルリアルタイム更新＋変更確定時にフィルタ適用
  document.getElementById('filter-citer-cc').addEventListener('input', function() {
    _updateLabel('filter-citer-cc', 'filter-citer-cc-val');
  });
  document.getElementById('filter-ref-cc').addEventListener('input', function() {
    _updateLabel('filter-ref-cc', 'filter-ref-cc-val');
  });
  document.getElementById('filter-citer-cc').addEventListener('change', _applyFilter);
  document.getElementById('filter-ref-cc').addEventListener('change', _applyFilter);
  // 初期ラベル設定
  _updateLabel('filter-citer-cc', 'filter-citer-cc-val');
  _updateLabel('filter-ref-cc', 'filter-ref-cc-val');

  // 起動時にフィルターを適用（input の初期値を変数に反映して描画を更新）
  _applyFilter();
})();

// ── 詳細パネル・共通ヘルパー（イベントハンドラとサイドバー両方から参照） ──
var detail          = document.getElementById('sb-detail');
var detailHeader    = document.getElementById('sb-detail-header');
var detailBody      = document.getElementById('sb-detail-body');
var _metaBanner     = document.getElementById('meta-edit-banner');
var _metaForm       = document.getElementById('meta-edit-form');
var _meTitle        = document.getElementById('me-title');
var _meAuthors      = document.getElementById('me-authors');
var _meYear         = document.getElementById('me-year');
var _meCitations    = document.getElementById('me-citations');
var _metaNodeId     = null;  // フォームが対象としているノードID
var detailResizeHnd = document.getElementById('sb-resize-y');
function _setDetailActive(active) {
  detail.className          = active ? 'sb-active' : '';
  detailResizeHnd.className = active ? 'sb-active' : '';
  if (!active) { detailHeader.innerHTML = ''; detailBody.innerHTML = ''; }
}
function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
function kv(k, v) {
  return '<div class="sb-kv"><span class="sb-k">' + esc(k) + '</span><span class="sb-v">' + esc(v) + '</span></div>';
}
function kvH(k, vHtml) {
  return '<div class="sb-kv"><span class="sb-k">' + esc(k) + '</span><span class="sb-v">' + vHtml + '</span></div>';
}
// ── Zotero編集ヒントモーダル ───────────────────────────────────────────────
var _zehModal   = document.getElementById('zotero-edit-hint');
var _zehOpen    = document.getElementById('zeh-open');
var _zehClose   = document.getElementById('zeh-close');
var _zehSuppress = document.getElementById('zeh-suppress-chk');
var _zehKey     = '';
var _LS_KEY     = 'hideZoteroEditHint';

function _showZoteroHint(itemKey) {
  if (localStorage.getItem(_LS_KEY) === '1') {
    // 非表示設定済み → そのままZoteroを開く
    window.location.href = 'zotero://select/library/items/' + itemKey;
    return;
  }
  _zehKey = itemKey;
  _zehSuppress.checked = false;
  _zehModal.classList.add('show');
}

_zehOpen.addEventListener('click', function() {
  if (_zehSuppress.checked) localStorage.setItem(_LS_KEY, '1');
  _zehModal.classList.remove('show');
  window.location.href = 'zotero://select/library/items/' + _zehKey;
});
_zehClose.addEventListener('click', function() {
  if (_zehSuppress.checked) localStorage.setItem(_LS_KEY, '1');
  _zehModal.classList.remove('show');
});
// 背景クリックで閉じる
_zehModal.addEventListener('click', function(e) {
  if (e.target === _zehModal) _zehModal.classList.remove('show');
});

// 識別子（DOI/ISBN）行を生成するヘルパー
// isZotero=true → 編集ボタンはZoteroディープリンク
// isZotero=false → 編集ボタンはインライン入力に切り替え
function _kvIdentifier(label, linkHtml, rawVal, nodeId, field, isZotero, itemKey) {
  var editBtn;
  if (isZotero) {
    editBtn = '<button class="id-edit-btn"' +
              ' data-zotero-key="' + esc(itemKey) + '"' +
              ' title="Zoteroで編集">✎</button>';
  } else {
    editBtn = '<button class="id-edit-btn"' +
              ' data-node="' + esc(nodeId) + '"' +
              ' data-field="' + esc(field) + '"' +
              ' data-val="'   + esc(rawVal)  + '"' +
              ' title="編集">✎</button>';
  }
  return '<div class="sb-kv"><span class="sb-k">' + esc(label) + '</span>' +
         '<span class="sb-v" data-id-field="' + esc(field) + '">' +
           linkHtml + editBtn +
         '</span></div>';
}

// インライン編集を起動（外部論文専用）
function _startIdEdit(btn) {
  var nodeId = btn.dataset.node;
  var field  = btn.dataset.field;
  var curVal = btn.dataset.val || '';
  var span   = btn.parentElement;
  span.innerHTML =
    '<input class="id-edit-input" value="' + esc(curVal) + '">' +
    '<button class="id-cancel-btn" data-node="' + esc(nodeId) + '">✕</button>';
  var input = span.querySelector('.id-edit-input');
  input.focus(); input.select();
  input.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      _saveIdentifier(nodeId, field, input.value.trim());
    } else if (e.key === 'Escape') {
      showDetail(nodeId);
    }
  });
  span.querySelector('.id-cancel-btn').addEventListener('click', function() {
    showDetail(nodeId);
  });
}

// 識別子を保存してパネルを更新
// 重複警告バナー
var _dupWarn      = document.getElementById('dup-warn');
var _dupWarnTitle = document.getElementById('dup-warn-title');
var _dupWarnMsg   = document.getElementById('dup-warn-msg');
document.getElementById('dup-warn-close').addEventListener('click', function() {
  _dupWarn.classList.remove('show');
});
function _normIsbn(s) {
  return s.replace(/[-\s]/g, '').toLowerCase();
}
function _isbnSet(raw) {
  // スペース区切りで複数ISBNが入る場合があるため、各要素をnormして Set にする
  return new Set((raw || '').split(/\s+/).map(_normIsbn).filter(Boolean));
}
function _checkDuplicate(nodeId, field, newVal) {
  if (!newVal || field === 'title') return;
  var dups;
  if (field === 'doi') {
    var norm = newVal.trim().toLowerCase();
    dups = GRAPH_DATA.nodes.filter(function(nd) {
      if (nd.id === nodeId) return false;
      return (nd.doi || '').trim().toLowerCase() === norm;
    });
  } else {
    // ISBN: 入力値・保存値ともにスペース区切り複数対応 + ハイフン正規化
    var inputSet = _isbnSet(newVal);
    dups = GRAPH_DATA.nodes.filter(function(nd) {
      if (nd.id === nodeId) return false;
      var ndSet = _isbnSet(nd.isbn || '');
      for (var v of inputSet) { if (ndSet.has(v)) return true; }
      return false;
    });
  }
  if (dups.length === 0) { _dupWarn.classList.remove('show'); return; }
  var fieldLabel = field === 'doi' ? 'DOI' : 'ISBN';
  var names = dups.map(function(nd) { return '「' + (nd.fullTitle || nd.label || nd.id) + '」'; }).join('、');
  _dupWarnTitle.textContent = fieldLabel + ' が他のノードと重複しています';
  _dupWarnMsg.textContent = names + ' と同じ ' + fieldLabel + ' です。ブラウザをリロードすると同一ノードに統合されます。';
  _dupWarn.classList.add('show');
}

function _saveIdentifier(nodeId, field, newVal) {
  fetch('/api/node/identifier', {
    method:  'POST',
    headers: {'Content-Type': 'application/json'},
    body:    JSON.stringify({node_id: nodeId, field: field, value: newVal}),
  }).then(function(r) { return r.json(); })
    .then(function(d) {
      if (d.ok) {
        // GRAPH_DATA（詳細パネル用）を更新
        var node = GRAPH_DATA.nodes.find(function(nd) { return nd.id === nodeId; });
        if (node) {
          node[field] = newVal;
          if (field === 'title') { node.fullTitle = newVal; }
        }
        // sigma の graph オブジェクト（ラベル・ツールチップ）も更新
        if (graph.hasNode(nodeId)) {
          graph.setNodeAttribute(nodeId, field, newVal);
          if (field === 'title') {
            graph.setNodeAttribute(nodeId, 'fullTitle', newVal);
            graph.setNodeAttribute(nodeId, 'label', newVal.slice(0, 24));
          }
          renderer.refresh();
        }
        _checkDuplicate(nodeId, field, newVal);
        showDetail(nodeId);
        // DOI / ISBN 修正時はメタデータ確認バナーとフォームを表示
        if (field === 'doi' || field === 'isbn') {
          _showMetaEditForm(nodeId, true);
        }
      }
    });
}

// ── メタデータ手動編集フォーム ────────────────────────────────────────────────
function _showMetaEditForm(nodeId, showBanner) {
  var n = GRAPH_DATA.nodes.find(function(nd) { return nd.id === nodeId; });
  if (!n) return;
  _metaNodeId          = nodeId;
  _meTitle.value       = n.fullTitle || n.label || '';
  _meAuthors.value     = n.authors || '';
  _meYear.value        = n.year || '';
  var _ccVal = n.citations != null ? n.citations : (n.cc != null ? n.cc : '');
  _meCitations.value   = _ccVal !== '' ? String(_ccVal) : '';
  _metaForm.classList.add('show');
  if (showBanner) _metaBanner.classList.add('show');
}

function _hideMetaForm() {
  _metaForm.classList.remove('show');
  _metaBanner.classList.remove('show');
  _metaNodeId = null;
}

function _saveMetaField(nodeId, field, val) {
  return fetch('/api/node/identifier', {
    method:  'POST',
    headers: {'Content-Type': 'application/json'},
    body:    JSON.stringify({node_id: nodeId, field: field, value: val}),
  }).then(function(r) { return r.json(); }).then(function(d) {
    if (d.ok) {
      var node = GRAPH_DATA.nodes.find(function(nd) { return nd.id === nodeId; });
      if (node) {
        node[field] = val;
        if (field === 'title')     { node.fullTitle = val; }
        if (field === 'citations') { node.citations = val ? parseInt(val) : null;
                                     node.cc        = node.citations; }
      }
      if (graph.hasNode(nodeId)) {
        graph.setNodeAttribute(nodeId, field, val);
        if (field === 'title') {
          graph.setNodeAttribute(nodeId, 'fullTitle', val);
          graph.setNodeAttribute(nodeId, 'label', val.slice(0, 24));
        }
        if (field === 'citations') {
          var ccNum = val ? parseInt(val) : 0;
          graph.setNodeAttribute(nodeId, 'cc', ccNum);
          // ノードサイズを再計算（被引用数に比例）
          var newSize = Math.max(4, Math.min(28, 4 + Math.sqrt(ccNum) * 1.2));
          graph.setNodeAttribute(nodeId, 'size', newSize);
        }
      }
    }
    return d;
  });
}

document.getElementById('me-save').addEventListener('click', function() {
  if (!_metaNodeId) return;
  var nid = _metaNodeId;
  Promise.all([
    _saveMetaField(nid, 'title',     _meTitle.value.trim()),
    _saveMetaField(nid, 'authors',   _meAuthors.value.trim()),
    _saveMetaField(nid, 'year',      _meYear.value.trim()),
    _saveMetaField(nid, 'citations', _meCitations.value.trim()),
  ]).then(function() {
    renderer.refresh();
    showDetail(nid);
    _hideMetaForm();
  });
});

document.getElementById('me-cancel').addEventListener('click', _hideMetaForm);

function showDetail(nodeId) {
  var n = GRAPH_DATA.nodes.find(function(nd) { return nd.id === nodeId; });
  if (!n) { _setDetailActive(false); return; }
  // 別ノードへ切り替えたらメタフォームを閉じる
  if (_metaNodeId && _metaNodeId !== nodeId) { _hideMetaForm(); }
  var doi  = n.doi  || '';
  var isbn = n.isbn || '';
  var isZotero   = (n.group === 'zotero');
  var isExternal = (n.group === 'external' || n.group === 'reference');
  var itemKey    = n.itemKey || '';

  var doiLinkHtml = doi
    ? '<a href="https://doi.org/' + esc(doi) + '" target="_blank">' + esc(doi) + '</a>'
    : '<span style="opacity:.5">—</span>';
  var isbnLinkHtml = isbn
    ? isbn.trim().split(/\s+/).map(function(i) {
        return '<a href="https://worldcat.org/isbn/' + esc(i) + '" target="_blank">' + esc(i) + '</a>';
      }).join('<span style="color:var(--text-dis)"> / </span>')
    : '<span style="opacity:.5">—</span>';

  _setDetailActive(true);

  // ヘッダー: タイトル（外部論文はタイトルも編集可）
  if (isExternal) {
    var titleEditBtn = '<button class="id-edit-btn"' +
      ' data-node="' + esc(n.id) + '"' +
      ' data-field="title"' +
      ' data-val="' + esc(n.fullTitle || n.label) + '"' +
      ' title="タイトルを編集">✎</button>';
    var metaEditBtn = '<button id="meta-edit-open" title="著者・年・タイトルを手動編集" style="' +
      'background:none;border:1px solid var(--outline-variant);border-radius:4px;' +
      'cursor:pointer;font-size:11px;color:var(--on-surface-variant);padding:2px 7px;' +
      'margin-left:6px;flex-shrink:0">メタデータ編集</button>';
    detailHeader.innerHTML =
      '<div class="sb-detail-title-row">' +
        '<div id="sb-detail-title" style="flex:1">' + esc(n.fullTitle || n.label) + titleEditBtn + '</div>' +
        metaEditBtn +
      '</div>';
    document.getElementById('meta-edit-open').addEventListener('click', function() {
      _showMetaEditForm(n.id, false);
    });
  } else {
    detailHeader.innerHTML =
      '<div class="sb-detail-title-row">' +
        '<div id="sb-detail-title">' + esc(n.fullTitle || n.label) + '</div>' +
      '</div>';
  }

  var body = '';
  body += kv('著者', n.authors || '—');
  body += kv('年', n.year || '—');
  body += _kvIdentifier('DOI',  doiLinkHtml,  doi,  n.id, 'doi',  isZotero, itemKey);
  body += _kvIdentifier('ISBN', isbnLinkHtml, isbn, n.id, 'isbn', isZotero, itemKey);
  body += kv('被引用数', n.citations != null ? Number(n.citations).toLocaleString()
                       : n.cc        != null ? Number(n.cc).toLocaleString() : '—');
  if (!isExternal) {
    body += kv('参照数', n.refCount ? Number(n.refCount).toLocaleString() : '—');
    body += kv('Key', n.itemKey || '');
  }
  detailBody.innerHTML = body;

  // 編集ボタンにイベントを登録（ヘッダー・ボディ両方）
  [detailHeader, detailBody].forEach(function(root) {
    // 外部論文：インライン編集
    root.querySelectorAll('.id-edit-btn[data-node]').forEach(function(btn) {
      btn.addEventListener('click', function(e) {
        e.preventDefault();
        _startIdEdit(btn);
      });
    });
    // Zoteroアイテム：ヒントモーダル → Zoteroを開く
    root.querySelectorAll('.id-edit-btn[data-zotero-key]').forEach(function(btn) {
      btn.addEventListener('click', function(e) {
        e.preventDefault();
        _showZoteroHint(btn.dataset.zoteroKey);
      });
    });
  });
}

/* ── 8a. Tab switching ───────────────────────────────────── */
var _ctxPane   = document.getElementById('sb-context-pane');
var _tabList   = document.getElementById('sb-tab-list');

function _switchTab(name) {
  document.querySelectorAll('.sb-tab').forEach(function(t) {
    t.classList.toggle('active', t.dataset.tab === name);
  });
  if (name === 'list') {
    _tabList.style.display = '';
    _ctxPane.classList.remove('active');
  } else {
    _tabList.style.display = 'none';
    _ctxPane.classList.add('active');
  }
}

document.querySelectorAll('.sb-tab').forEach(function(tab) {
  tab.addEventListener('click', function() { _switchTab(tab.dataset.tab); });
});

/* ── 8b. Context pane: abstract + summary (node) / contexts (edge) ── */
var _abstrTranslateOn = false;
var _ctxPaneReq       = 0;  // 連続クリック時に古いfetch応答が上書きするのを防ぐトークン

function _showContextPane(html) {
  _ctxPane.innerHTML = html;
  _switchTab('context');
}

function _showNodeAbstract(nodeId) {
  var n = GRAPH_DATA.nodes.find(function(nd) { return nd.id === nodeId; });
  if (!n || !n.itemKey) return;
  var myReq = ++_ctxPaneReq;

  _showContextPane(
    '<div class="ctx-pane-header">' +
      '<div class="ctx-pane-title">' + esc(n.fullTitle || n.label) + '</div>' +
      '<div class="ctx-translate-wrap" id="abs-trans-wrap" style="display:none">' +
        '<span class="ctx-translate-label">翻訳</span>' +
        '<button class="ctx-toggle-btn' + (_abstrTranslateOn ? ' on' : '') + '" id="abs-toggle"></button>' +
      '</div>' +
    '</div>' +
    '<div id="abs-loading" style="font-size:12px;color:var(--text-dis)">読み込み中…</div>' +
    '<div class="summary-section" id="abs-summary-section" style="display:none">' +
      '<div class="summary-section-label">AI 要約</div>' +
      '<div id="abs-summary-body"></div>' +
    '</div>'
  );

  // 取得済みアブストラクトを描画（翻訳トグル込み）
  function _renderAbstractInto(targetEl, abstract) {
    var isEnglish = abstract.length > 20 && abstract.charCodeAt(0) < 0x100;
    var transWrap = document.getElementById('abs-trans-wrap');
    if (transWrap && isEnglish) transWrap.style.display = '';
    targetEl.outerHTML =
      '<div class="abstract-text" id="abs-text">' + esc(abstract) + '</div>' +
      '<div class="abstract-translation" id="abs-translation"></div>';
    var toggleBtn = document.getElementById('abs-toggle');
    if (toggleBtn) {
      toggleBtn.addEventListener('click', function() {
        _abstrTranslateOn = !_abstrTranslateOn;
        toggleBtn.className = 'ctx-toggle-btn' + (_abstrTranslateOn ? ' on' : '');
        _applyAbstractToggle(abstract);
      });
    }
    if (_abstrTranslateOn) _applyAbstractToggle(abstract);
  }

  // Zotero local API から abstractNote を取得してキャッシュ→描画（ノードを開くと自動実行）。
  // 失敗時（Zotero 未起動・概要なし）はメッセージと再取得ボタンを出す。
  function _fetchAbstractFromZotero(targetEl) {
    targetEl.outerHTML =
      '<div id="abs-fetch-wrap">' +
        '<div class="summary-status show" id="abs-fetch-status" style="color:var(--text-dis)">Zotero から概要を取得中…</div>' +
      '</div>';
    fetch('/api/node/fetch-abstract', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ item_key: n.itemKey })
    })
      .then(function(r) { return r.json().then(function(d){ return { ok: r.ok, d: d }; }); })
      .then(function(res) {
        if (myReq !== _ctxPaneReq) return;  // 別のノード/エッジに切り替え済み
        var d = res.d || {};
        if (d.abstract) {
          var wrap = document.getElementById('abs-fetch-wrap');
          if (wrap) _renderAbstractInto(wrap, d.abstract);
        } else {
          _renderFetchFailure(d.error || 'この資料には概要情報がありませんでした');
        }
      })
      .catch(function(e) {
        if (myReq !== _ctxPaneReq) return;
        _renderFetchFailure('エラー: ' + e.message);
      });
  }

  // 取得失敗時のメッセージ＋再取得ボタン（Zotero を起動してから再試行できるように）
  function _renderFetchFailure(msg) {
    var wrap = document.getElementById('abs-fetch-wrap');
    if (!wrap) return;
    wrap.innerHTML =
      '<div class="summary-status show" style="color:#f87171">' + esc(msg) + '</div>' +
      '<button class="summary-btn" id="abs-retry-btn" style="margin-top:6px">再取得</button>';
    var rb = document.getElementById('abs-retry-btn');
    if (rb) rb.addEventListener('click', function() { _fetchAbstractFromZotero(wrap); });
  }

  fetch('/api/node/abstract?key=' + encodeURIComponent(n.itemKey))
    .then(function(r) { return r.json(); })
    .then(function(d) {
      if (myReq !== _ctxPaneReq) return;  // 別のノード/エッジに切り替え済み
      var absEl = document.getElementById('abs-loading');
      if (!absEl) return;
      var abstract = d.abstract || '';
      // キャッシュ済みなら描画、無ければ Zotero から自動取得
      if (abstract) { _renderAbstractInto(absEl, abstract); }
      else          { _fetchAbstractFromZotero(absEl); }
      // Summary section
      _renderSummarySection(n.itemKey, n.fullTitle || n.label || '', d.summary);
    })
    .catch(function(e) {
      if (myReq !== _ctxPaneReq) return;
      var absEl = document.getElementById('abs-loading');
      if (absEl) absEl.textContent = 'エラー: ' + e.message;
    });
}

function _applyAbstractToggle(abstract) {
  var transDiv = document.getElementById('abs-translation');
  if (!transDiv) return;
  if (!_abstrTranslateOn) {
    transDiv.classList.remove('show');
    return;
  }
  if (transDiv.dataset.cached) {
    transDiv.textContent = transDiv.dataset.cached;
    transDiv.classList.add('show');
    return;
  }
  transDiv.innerHTML = '<span style="opacity:.5">翻訳中…</span>';
  transDiv.classList.add('show');
  fetch('/api/translate/batch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ texts: [abstract] })
  })
    .then(function(r) { return r.json(); })
    .then(function(d) {
      if (d.translations && d.translations[0]) {
        transDiv.dataset.cached = d.translations[0];
        transDiv.textContent = d.translations[0];
      } else {
        transDiv.textContent = d.error || '翻訳失敗';
      }
    })
    .catch(function(e) {
      transDiv.style.color = '#f87171';
      transDiv.textContent = 'エラー: ' + e.message;
    });
}

// 外部論文ノード（group=external/reference）の概要を取得して表示。
// Crossref（Abstract）優先 → S2（abstract + tldr）フォールバック。翻訳トグル付き。
// 本文チャンクが無いため AI 要約は出さない。両ソースに情報が無ければ「概要情報なし」を明示。
function _showExternalAbstract(nodeId) {
  var n = GRAPH_DATA.nodes.find(function(nd) { return nd.id === nodeId; });
  if (!n) return;
  var myReq = ++_ctxPaneReq;
  var paperId = (n.id || '').replace(/^(paper|ref):/, '');
  var doi = n.doi || '';

  _showContextPane(
    '<div class="ctx-pane-header">' +
      '<div class="ctx-pane-title">' + esc(n.fullTitle || n.label) + '</div>' +
      '<div class="ctx-translate-wrap" id="ext-trans-wrap" style="display:none">' +
        '<span class="ctx-translate-label">翻訳</span>' +
        '<button class="ctx-toggle-btn' + (_abstrTranslateOn ? ' on' : '') + '" id="ext-toggle"></button>' +
      '</div>' +
    '</div>' +
    '<div id="ext-loading" style="font-size:12px;color:var(--text-dis)">概要を取得中…（Crossref / Semantic Scholar）</div>'
  );

  // Crossref は DOI、S2 フォールバックは paper_id を使うため両方渡す。
  var parts = [];
  if (doi)     parts.push('doi='      + encodeURIComponent(doi));
  if (paperId) parts.push('paper_id=' + encodeURIComponent(paperId));
  fetch('/api/node/external-abstract?' + parts.join('&'))
    .then(function(r) { return r.json().then(function(d){ return { ok: r.ok, d: d }; }); })
    .then(function(res) {
      if (myReq !== _ctxPaneReq) return;
      var loadEl = document.getElementById('ext-loading');
      if (!loadEl) return;
      var d = res.d || {};
      if (!res.ok || d.error) {
        loadEl.innerHTML = '<span style="color:#f87171">' + esc(d.error || '取得に失敗しました') + '</span>';
        return;
      }
      var abstract = d.abstract || '';
      var tldr     = d.tldr || '';
      if (!abstract && !tldr) {
        loadEl.outerHTML =
          '<div style="font-size:12px;color:var(--text-dis);line-height:1.7">' +
            'Crossref・Semantic Scholar のいずれにも概要情報がありません。<br>' +
            '参照できる情報がないため、要約は生成できません。' +
          '</div>';
        return;
      }
      // 概要本文を組み立て（tldr → abstract の順。両方あれば両方表示）
      var html = '';
      if (tldr) {
        html +=
          '<div style="margin-bottom:10px">' +
            '<div class="summary-section-label">S2 TL;DR</div>' +
            '<div class="abstract-text" id="ext-tldr-text">' + esc(tldr) + '</div>' +
            '<div class="abstract-translation" id="ext-tldr-trans"></div>' +
          '</div>';
      }
      if (abstract) {
        html +=
          '<div>' +
            '<div class="summary-section-label">Abstract</div>' +
            '<div class="abstract-text" id="ext-abs-text">' + esc(abstract) + '</div>' +
            '<div class="abstract-translation" id="ext-abs-trans"></div>' +
          '</div>';
      }
      loadEl.outerHTML = html;

      // 翻訳対象（英語と思われる場合だけトグルを出す）
      var transTargets = [];
      if (tldr)     transTargets.push({ src: tldr,     transId: 'ext-tldr-trans' });
      if (abstract) transTargets.push({ src: abstract, transId: 'ext-abs-trans' });
      var firstText = (tldr || abstract || '');
      var isEnglish = firstText.length > 20 && firstText.charCodeAt(0) < 0x100;
      var transWrap = document.getElementById('ext-trans-wrap');
      if (transWrap && isEnglish) transWrap.style.display = '';

      var toggleBtn = document.getElementById('ext-toggle');
      if (toggleBtn) {
        toggleBtn.addEventListener('click', function() {
          _abstrTranslateOn = !_abstrTranslateOn;
          toggleBtn.className = 'ctx-toggle-btn' + (_abstrTranslateOn ? ' on' : '');
          _applyExternalToggle(transTargets);
        });
      }
      if (_abstrTranslateOn) _applyExternalToggle(transTargets);
    })
    .catch(function(e) {
      if (myReq !== _ctxPaneReq) return;
      var loadEl = document.getElementById('ext-loading');
      if (loadEl) loadEl.innerHTML = '<span style="color:#f87171">エラー: ' + esc(e.message) + '</span>';
    });
}

// 外部論文の abstract / tldr をまとめて翻訳・表示する。
function _applyExternalToggle(targets) {
  if (!_abstrTranslateOn) {
    targets.forEach(function(t) {
      var div = document.getElementById(t.transId);
      if (div) div.classList.remove('show');
    });
    return;
  }
  // キャッシュ済みは即表示し、未翻訳のものだけ1リクエストで翻訳
  var pending = [];
  targets.forEach(function(t) {
    var div = document.getElementById(t.transId);
    if (!div) return;
    if (div.dataset.cached) {
      div.textContent = div.dataset.cached;
      div.classList.add('show');
    } else {
      div.innerHTML = '<span style="opacity:.5">翻訳中…</span>';
      div.classList.add('show');
      pending.push(t);
    }
  });
  if (pending.length === 0) return;
  fetch('/api/translate/batch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ texts: pending.map(function(t) { return t.src; }) })
  })
    .then(function(r) { return r.json(); })
    .then(function(d) {
      pending.forEach(function(t, j) {
        var div = document.getElementById(t.transId);
        if (!div) return;
        if (d.error) { div.style.color = '#f87171'; div.textContent = 'エラー: ' + d.error; }
        else {
          var tr = d.translations[j];
          div.dataset.cached = tr;
          div.style.color = '';
          div.textContent = tr;
        }
      });
    })
    .catch(function(e) {
      pending.forEach(function(t) {
        var div = document.getElementById(t.transId);
        if (div) { div.style.color = '#f87171'; div.textContent = 'エラー: ' + e.message; }
      });
    });
}

// 選択可能な要約モデル（バックエンドの SUMMARY_MODELS と対応）
var _SUMMARY_MODELS = [
  ['deepseek-v4-pro', 'DeepSeek V4 Pro'],
];
var _SUMMARY_MODEL_DEFAULT = 'deepseek-v4-pro';

function _getSummaryModel() {
  var saved = localStorage.getItem('aiSummaryModel');
  var valid = _SUMMARY_MODELS.some(function(m) { return m[0] === saved; });
  return valid ? saved : _SUMMARY_MODEL_DEFAULT;
}

function _modelLabel(id) {
  for (var i = 0; i < _SUMMARY_MODELS.length; i++) {
    if (_SUMMARY_MODELS[i][0] === id) return _SUMMARY_MODELS[i][1];
  }
  return id;
}

function _modelSelectHtml() {
  // モデルが1つだけならドロップダウンは出さない（将来増えたら自動で表示）
  if (_SUMMARY_MODELS.length <= 1) return '';
  var cur = _getSummaryModel();
  var html = '<select class="summary-model-select" id="sum-model-select" title="要約に使うモデル">';
  _SUMMARY_MODELS.forEach(function(m) {
    html += '<option value="' + m[0] + '"' + (m[0] === cur ? ' selected' : '') + '>' + m[1] + '</option>';
  });
  html += '</select>';
  return html;
}

function _fmtSummaryDate(iso) {
  if (!iso) return '';
  // SQLite CURRENT_TIMESTAMP は UTC ("YYYY-MM-DD HH:MM:SS")。Zが無ければ付与してパース。
  var s = iso.replace(' ', 'T');
  if (!/Z|[+-]\d\d:?\d\d$/.test(s)) s += 'Z';
  var d = new Date(s);
  return isNaN(d) ? '' : d.toLocaleDateString('ja-JP');
}

function _renderSummarySection(itemKey, title, summaryData) {
  var sec = document.getElementById('abs-summary-section');
  if (!sec) return;
  sec.style.display = '';
  var bodyEl = document.getElementById('abs-summary-body');
  if (!bodyEl) return;

  function _renderSummaryBody(sd) {
    if (!sd) {
      bodyEl.innerHTML =
        '<div class="summary-actions">' +
          '<button class="summary-btn primary" id="sum-gen-btn">AI要約を生成</button>' +
          _modelSelectHtml() +
        '</div>' +
        '<div class="summary-status" id="sum-status"></div>';
    } else {
      var modelLabel = sd.model && sd.model !== 'manual' ? ' (' + esc(_modelLabel(sd.model)) + ')' : '';
      var dateStr = _fmtSummaryDate(sd.updated_at);
      bodyEl.innerHTML =
        '<div class="summary-text" id="sum-text">' + esc(sd.summary) + '</div>' +
        '<div style="font-size:10.5px;color:var(--text-dis);margin-top:4px">' +
          esc(dateStr) + modelLabel +
        '</div>' +
        '<div class="summary-actions" id="sum-actions">' +
          '<button class="summary-btn" id="sum-regen-btn">再生成</button>' +
          '<button class="summary-btn" id="sum-edit-btn">編集</button>' +
          _modelSelectHtml() +
        '</div>' +
        '<div class="summary-status" id="sum-status"></div>';
    }
    // Wire buttons
    var genBtn   = document.getElementById('sum-gen-btn');
    var regenBtn = document.getElementById('sum-regen-btn');
    var editBtn  = document.getElementById('sum-edit-btn');
    var modelSel = document.getElementById('sum-model-select');

    if (modelSel) {
      modelSel.addEventListener('change', function() {
        localStorage.setItem('aiSummaryModel', modelSel.value);
      });
    }

    function _startGenerate(force) {
      var statusEl = document.getElementById('sum-status');
      if (statusEl) {
        statusEl.style.color = '';
        statusEl.textContent = 'AI要約を生成中…（' + _modelLabel(_getSummaryModel()) + '）';
        statusEl.classList.add('show');
      }
      if (genBtn)   genBtn.disabled = true;
      if (regenBtn) regenBtn.disabled = true;
      fetch('/api/node/summary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          item_key: itemKey,
          title:    title || '',
          force:    !!force,
          model:    _getSummaryModel()
        })
      })
        .then(function(r) { return r.json(); })
        .then(function(d) {
          if (d.error) {
            if (statusEl) { statusEl.style.color = '#f87171'; statusEl.textContent = 'エラー: ' + d.error; }
            if (genBtn)   genBtn.disabled = false;
            if (regenBtn) regenBtn.disabled = false;
          } else {
            _renderSummaryBody({ summary: d.summary, model: d.model, updated_at: d.updated_at });
          }
        })
        .catch(function(e) {
          if (statusEl) { statusEl.style.color = '#f87171'; statusEl.textContent = 'エラー: ' + e.message; }
          if (genBtn)   genBtn.disabled = false;
          if (regenBtn) regenBtn.disabled = false;
        });
    }

    if (genBtn)   genBtn.addEventListener('click', function() { _startGenerate(false); });
    if (regenBtn) regenBtn.addEventListener('click', function() { _startGenerate(true); });
    if (editBtn) {
      editBtn.addEventListener('click', function() {
        var curText = document.getElementById('sum-text');
        var cur = curText ? curText.textContent : '';
        var actionsDiv = document.getElementById('sum-actions');
        if (actionsDiv) actionsDiv.style.display = 'none';
        if (curText) curText.style.display = 'none';
        var editArea = document.createElement('textarea');
        editArea.className = 'summary-textarea';
        editArea.value = cur;
        bodyEl.appendChild(editArea);
        var saveBtn = document.createElement('button');
        saveBtn.className = 'summary-btn primary';
        saveBtn.textContent = '保存';
        var cancelBtn = document.createElement('button');
        cancelBtn.className = 'summary-btn';
        cancelBtn.textContent = 'キャンセル';
        var editActions = document.createElement('div');
        editActions.className = 'summary-actions';
        editActions.appendChild(saveBtn);
        editActions.appendChild(cancelBtn);
        bodyEl.appendChild(editActions);
        editArea.focus();
        saveBtn.addEventListener('click', function() {
          var newText = editArea.value.trim();
          if (!newText) return;
          fetch('/api/node/summary', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ item_key: itemKey, summary: newText })
          })
            .then(function(r) { return r.json(); })
            .then(function() {
              _renderSummaryBody({ summary: newText, model: 'manual', updated_at: new Date().toISOString() });
            });
        });
        cancelBtn.addEventListener('click', function() {
          _renderSummaryBody({ summary: cur, model: sd ? sd.model : '', updated_at: sd ? sd.updated_at : '' });
        });
      });
    }
  }

  _renderSummaryBody(summaryData);
}

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

  // ── トグル ──────────────────────────────────────────────
  toggle.addEventListener('click', function() {
    document.body.classList.toggle('sb-collapsed');
    toggle.textContent = document.body.classList.contains('sb-collapsed') ? '‹' : '›';
    setTimeout(function() { renderer.refresh(); }, 260);
  });

  // ── 列ヘッダーでソート ────────────────────────────────
  document.querySelectorAll('#sb-list-head-table th').forEach(function(th) {
    th.addEventListener('click', function() {
      var col = th.dataset.col;
      if (sortCol === col) { sortDir *= -1; }
      else { sortCol = col; sortDir = (col === 'title') ? 1 : -1; }
      document.querySelectorAll('#sb-list-head-table th').forEach(function(t) { t.className = ''; });
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

  // ── リスト描画 ────────────────────────────────────────
  function _sbNodeVisible(n) {
    // CCフィルタ
    if (n.group === 'external'  && filterCiterCC > 0 && (n.cc || 0) < filterCiterCC) return false;
    if (n.group === 'reference' && filterRefCC   > 0 && (n.cc || 0) < filterRefCC)   return false;
    // コレクションフィルタ
    var colKeys = window._colItemKeys;
    if (colKeys === null) return true;
    if (n.group === 'zotero') {
      var nk = n.itemKey || (n.id.indexOf(':') >= 0 ? n.id.split(':')[1] : n.id);
      return colKeys.has(nk);
    }
    var edges = graph.edges(n.id);
    for (var ei = 0; ei < edges.length; ei++) {
      var nbr = graph.source(edges[ei]) === n.id
                ? graph.target(edges[ei]) : graph.source(edges[ei]);
      var nd = graph.getNodeAttributes(nbr);
      if (nd.group === 'zotero') {
        var nk2 = nd.itemKey || (nbr.indexOf(':') >= 0 ? nbr.split(':')[1] : nbr);
        if (colKeys.has(nk2)) return true;
      }
    }
    return false;
  }

  function renderList() {
    var sortFn = function(a, b) {
      var av = getVal(a, sortCol), bv = getVal(b, sortCol);
      if (typeof av === 'string') return av < bv ? -sortDir : av > bv ? sortDir : 0;
      return (av - bv) * sortDir;
    };

    var baseNodes = sbNodes.filter(function(n) {
      if (zoteroOnlyEl.checked && n.group !== 'zotero') return false;
      if (!filterQ) return true;
      return (n.fullTitle || n.label || '').toLowerCase().indexOf(filterQ) !== -1 ||
             (n.authors   || '').toLowerCase().indexOf(filterQ) !== -1;
    });

    var visNodes   = baseNodes.filter(function(n) { return  _sbNodeVisible(n); });
    var grayNodes  = baseNodes.filter(function(n) { return !_sbNodeVisible(n); });
    visNodes.sort(sortFn);
    grayNodes.sort(sortFn);

    sbCountEl.textContent = ' (' + visNodes.length + ')';
    tbody.innerHTML = '';

    function makeRow(n, grayed) {
      var tr = document.createElement('tr');
      if (n.id === sbActiveId) tr.className = 'sb-active';
      if (grayed) tr.style.opacity = '0.38';
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
      if (grayed) {
        tr.addEventListener('click', function() {
          sbActiveId = n.id;
          renderList();
          showDetail(n.id);
        });
      } else {
        tr.addEventListener('click', function() { selectFromSidebar(n.id); });
      }
      tbody.appendChild(tr);
    }

    visNodes.forEach(function(n)  { makeRow(n, false); });
    grayNodes.forEach(function(n) { makeRow(n, true);  });
  }

  // ── サイドバーからノード選択 ──────────────────────────
  function selectFromSidebar(nodeId) {
    sbActiveId = nodeId;
    recomputeSelection(nodeId);
    renderer.refresh();
    renderList();
    showDetail(nodeId);
    window._panToNode(nodeId);
    var _n = GRAPH_DATA.nodes.find(function(nd) { return nd.id === nodeId; });
    if (_n && _n.group === 'zotero' && _n.itemKey) { _showNodeAbstract(nodeId); }
    else if (_n && (_n.group === 'external' || _n.group === 'reference')) { _showExternalAbstract(nodeId); }
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
  // ── グラフクリックとサイドバーを同期 ─────────────────
  renderer.on('clickNode', function(ev) {
    if (hasDragged) return;
    if (ev.node) {
      // 主ハンドラー実行後、selectedNode が更新されてから動く
      setTimeout(function() {
        if (selectedNode === ev.node) {
          sbActiveId = ev.node;
          showDetail(ev.node);
        } else {
          sbActiveId = null;
          _setDetailActive(false);
        }
        renderList();
      }, 0);
    }
  });
  renderer.on('clickStage', function() {
    sbActiveId = null;
    _setDetailActive(false);
    renderList();
  });

  // 初期ソート表示（引用数降順）
  document.querySelector('#sb-list-head-table th[data-col="citations"]').className = 'sort-desc';
  renderList();

  // コレクションフィルタから呼び出せるように公開
  window._renderList = renderList;

  // ── サイドバー幅リサイズ ──────────────────────────────────────────────────
  (function() {
    var hnd = document.getElementById('sb-resize-x');
    var sb  = document.getElementById('sidebar');
    var MIN_W = 200, MAX_W = 700;
    hnd.addEventListener('mousedown', function(e) {
      e.preventDefault();
      hnd.classList.add('dragging');
      var startX = e.clientX;
      var startW = sb.offsetWidth;
      function onMove(e) {
        var w = Math.max(MIN_W, Math.min(MAX_W, startW - (e.clientX - startX)));
        document.documentElement.style.setProperty('--sb-width', w + 'px');
      }
      function onUp() {
        hnd.classList.remove('dragging');
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup',   onUp);
        // ドラッグ完了後にキャンバスサイズを更新して正しい領域に再描画する
        if (window.__r) { window.__r.resize(); window.__r.refresh(); }
      }
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup',   onUp);
    });
  })();

  // ── 詳細パネル高さリサイズ ───────────────────────────────────────────────
  (function() {
    var hnd   = detailResizeHnd;
    var MIN_H = 80, MAX_H = 600;
    hnd.addEventListener('mousedown', function(e) {
      e.preventDefault();
      hnd.classList.add('dragging');
      var startY = e.clientY;
      var startH = detail.offsetHeight;
      function onMove(e) {
        var h = Math.max(MIN_H, Math.min(MAX_H, startH - (e.clientY - startY)));
        detail.style.height = h + 'px';
      }
      function onUp() {
        hnd.classList.remove('dragging');
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup',   onUp);
      }
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup',   onUp);
    });
  })();
})();

console.log('[RAG] sigma ready – nodes:', graph.order, 'edges:', graph.size,
            '· D3-force layout starting…');

  }) // end fetch .then(GRAPH_DATA)
  .catch(function(e) {
    var el = document.getElementById('loading');
    if (el) el.innerHTML =
      '<div style="color:#f87171;font-size:13px">グラフ読み込みエラー: ' + e.message + '</div>';
    console.error('[RAG] fetch /api/graph failed:', e);
  });

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
  <script src="https://cdn.jsdelivr.net/npm/sigma@3.0.3/dist/sigma.min.js"></script>
  <!-- D3-force for graph layout (includes all required sub-modules) -->
  <script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
  {css}
</head>
<body>
  <!-- ── View mode tabs ── -->
  <div id="view-tabs">
    <button class="view-tab active" data-view="citation">引用ネットワーク</button>
    <button class="view-tab" data-view="semantic">意味マップ</button>
  </div>

  <!-- sigma.js renders into this div -->
  <div id="sigma-container">
    <canvas id="cluster-overlay"></canvas>
  </div>

  <!-- Sidebar toggle button -->
  <button id="sb-toggle" title="資料一覧を表示/非表示">›</button>

  <!-- Sidebar: 資料一覧 + 詳細 -->
  <div id="sidebar">
    <div id="sb-resize-x"></div>
    <div id="sb-tabs">
      <button class="sb-tab active" data-tab="list">資料一覧<span id="sb-count"></span></button>
      <button class="sb-tab" data-tab="context">概要/コンテキスト</button>
    </div>
    <div id="sb-tab-list">
    <div style="padding:6px 10px 8px;flex-shrink:0;border-bottom:1px solid var(--outline-variant)">
      <input id="sb-search" type="text" placeholder="タイトル・著者で絞り込み…">
      <label id="sb-zotero-only-label" style="display:flex;align-items:center;gap:5px;margin-top:5px;font-size:11.5px;color:var(--on-surface-variant);cursor:pointer">
        <input id="sb-zotero-only" type="checkbox"> Zotero所収のみ
      </label>
    </div>
    <div id="sb-list-head">
      <table id="sb-list-head-table">
        <colgroup>
          <col><col style="width:38px"><col style="width:46px"><col style="width:40px"><col style="width:24px">
        </colgroup>
        <thead>
          <tr>
            <th data-col="title">タイトル</th>
            <th data-col="year">年</th>
            <th data-col="citations">引用</th>
            <th data-col="refCount">参照</th>
            <th data-col="inZotero" title="Zotero所収">Z</th>
          </tr>
        </thead>
      </table>
    </div>
    <div id="sb-list-wrap">
      <table id="sb-list">
        <colgroup>
          <col><col style="width:38px"><col style="width:46px"><col style="width:40px"><col style="width:24px">
        </colgroup>
        <tbody id="sb-list-body"></tbody>
      </table>
    </div>
    </div><!-- #sb-tab-list -->
    <div id="sb-context-pane">
      <div class="ctx-pane-empty">ノードまたはエッジを選択してください</div>
    </div>
    <div id="sb-resize-y"></div>
    <div id="sb-detail">
      <div id="sb-detail-header"></div>
      <div id="meta-edit-banner">⚠ 識別子が修正されました。著者・年・タイトルに誤りがある場合は下のフォームで修正できます。</div>
      <div id="meta-edit-form">
        <div class="me-row">
          <label class="me-label">タイトル</label>
          <input type="text" id="me-title" placeholder="タイトル">
        </div>
        <div class="me-row">
          <label class="me-label">著者</label>
          <input type="text" id="me-authors" placeholder="著者名（カンマ区切り）">
        </div>
        <div class="me-row">
          <label class="me-label">年</label>
          <input type="text" id="me-year" placeholder="出版年">
        </div>
        <div class="me-row">
          <label class="me-label">被引用数</label>
          <input type="text" id="me-citations" placeholder="整数">
        </div>
        <div class="me-actions">
          <button class="me-cancel" id="me-cancel">キャンセル</button>
          <button class="me-save" id="me-save">保存</button>
        </div>
      </div>
      <div id="sb-detail-body"></div>
    </div>
  </div>

  <!-- Layout progress badge (bottom-right corner, hidden after layout done) -->
  <!-- Zotero編集ヒントモーダル -->
  <div id="zotero-edit-hint">
    <div id="zotero-edit-hint-box">
      <div style="font-weight:600;margin-bottom:6px">Zoteroで編集してください</div>
      <ol>
        <li>Zoteroで該当アイテムの DOI / ISBN を修正する</li>
        <li><code>Maintenance-Widget.command</code> でCitation Network更新を実行する</li>
        <li>ブラウザをリロードする</li>
      </ol>
      <div class="zeh-actions">
        <button class="zeh-open-btn" id="zeh-open">Zoteroで開く</button>
        <button class="zeh-close-btn" id="zeh-close">閉じる</button>
      </div>
      <label class="zeh-suppress">
        <input type="checkbox" id="zeh-suppress-chk">
        以降このメッセージを表示しない
      </label>
    </div>
  </div>

  <div id="layout-badge" data-label="レイアウト計算中">
    <span id="layout-pct">0%</span>
    <button id="layout-skip" style="margin-left:10px;background:none;border:1px solid rgba(59,130,246,0.4);border-radius:4px;color:#93c5fd;font-size:10px;padding:2px 7px;cursor:pointer;">スキップ</button>
  </div>
  <div id="col-layout-badge">再レイアウト中<span id="col-layout-pct">0%</span></div>

  <!-- 重複ID警告バナー -->
  <div id="dup-warn">
    <span class="dup-icon">⚠</span>
    <div class="dup-body">
      <strong id="dup-warn-title">DOI / ISBN が重複しています</strong>
      <span id="dup-warn-msg"></span>
    </div>
    <button class="dup-close" id="dup-warn-close">✕</button>
  </div>

  <!-- Loading overlay (hidden after /api/graph fetch completes) -->
  <div id="loading">
    <div class="loading-spinner"></div>
    <span>グラフデータを読み込み中…</span>
  </div>

  {legend}

  <!-- Tooltip (positioned by JS) -->
  <div id="rag-tooltip"></div>

  {logic_script}
</body>
</html>"""


# ── Color palette (module-level single source of truth) ──────────────────────
import re as _re_palette, json as _json_palette

_PALETTE: dict[str, str] = {
    "nodeZotero":         "#C97090",
    "nodeExternal":       "#A0A0A0",
    "nodeCiter":          "#DFA040",
    "nodeRef":            "#7498DC",
    "nodeUnknown":        "#8E8A98",
    "nodeDim":            "#2B2930",
    "edgeDefault":        "#707070",
    "edgeCitation":       "#C8882A",
    "edgeReference":      "#7498DC",
    "surface":                "#141218",
    "surfaceContainerLow":    "#1D1B20",
    "surfaceContainerHigh":   "#2B2930",
    "outlineVariant":         "#49454F",
    "onSurface":          "#E6E0E9",
    "onSurfaceVariant":   "#CAC4D0",
    "textDis":            "rgba(230,224,233,0.38)",
}

def _palette_css_var(key: str) -> str:
    return "--" + _re_palette.sub(r'([A-Z])', lambda m: "-" + m.group(1).lower(), key)

_CSS_ROOT = ":root {\n" + "\n".join(
    f"  {_palette_css_var(k)}: {v};" for k, v in _PALETTE.items()
) + "\n}"
_JS_THEME = "const THEME = " + _json_palette.dumps(_PALETTE, indent=2) + ";"


# ── build_graph_data: assemble node/edge data for the API ─────────────────────

def build_graph_data(
    items: list[dict],
    citers: list[dict],
    refs: list[dict],
    item_meta: dict[str, dict] | None = None,
    item_ref_counts: dict[str, int] | None = None,
    db_path: str | None = None,
) -> dict:
    """ノード・エッジを構築し、FastAPI で返却するデータ dict を返す。
    エッジコンテキストはオンデマンド取得のため埋め込まない。
    """
    import json as _json
    import re  as _re

    meta      = item_meta or {}
    item_rcnt = item_ref_counts or {}

    # モジュールレベルの _PALETTE / _CSS_ROOT / _JS_THEME を参照
    PALETTE   = _PALETTE
    _css_root = _CSS_ROOT
    _js_theme = _JS_THEME

    C_ZOTERO   = PALETTE["nodeZotero"]
    C_EXTERNAL = PALETTE["nodeExternal"]
    C_CITER    = PALETTE["nodeCiter"]
    C_REF      = PALETTE["nodeRef"]
    C_UNK      = PALETTE["nodeUnknown"]

    # ── Server-side layout (FA2 + sector placement) ─────────────────────────
    import sys as _sys, time as _time, hashlib as _hashlib, json as _json
    _t0 = _time.time()
    _print = lambda *a, **kw: print(*a, file=_sys.stderr, **kw)
    item_keys_for_layout = [d["item_key"] for d in items]

    # キャッシュキー: レイアウトバージョン + ノードIDのソート済みリストをSHA1ハッシュ化
    # （LAYOUT_VERSION を含めることで、物理パラメータ変更時に再計算がトリガーされる）
    _all_node_ids = sorted(
        [f"item:{d['item_key']}" for d in items] +
        [f"paper:{d['citing_paper_id']}" for d in citers] +
        list({f"paper:{d['cited_paper_id']}" if f"paper:{d['cited_paper_id']}" not in
              {f"paper:{c['citing_paper_id']}" for c in citers} else f"ref:{d['cited_paper_id']}"
              for d in refs})
    )
    _cache_key = _hashlib.sha1(
        (LAYOUT_VERSION + "\n" + "\n".join(_all_node_ids)).encode()
    ).hexdigest()[:16]
    _cache_path = Path(__file__).parent.parent / "data" / "layout_cache.json"

    layout_positions: dict = {}
    _cache_hit = False
    _stale_positions: dict = {}  # キャッシュキーが変わった場合の旧座標（warm_start 用）
    if _cache_path.exists():
        try:
            _cached = _json.loads(_cache_path.read_text())
            if _cached.get("key") == _cache_key:
                layout_positions = {k: tuple(v) for k, v in _cached["positions"].items()}
                _cache_hit = True
                _print(f"Layout cache hit ({len(layout_positions)} nodes, key={_cache_key})")
            else:
                # キャッシュキーは変わったが旧座標を warm_start として再利用する
                _stale_positions = {k: tuple(v) for k, v in _cached["positions"].items()}
                _print(f"Layout cache stale (key changed) – warm-starting from {len(_stale_positions)} nodes")
        except Exception as _e:
            _print(f"Layout cache load error: {_e}")

    if not _cache_hit:
        # 意味ベクトル（②A: 内容が近いアイテムを近くに配置するための仮想エッジ用）
        _sem_vectors = get_item_vectors(item_keys_for_layout)
        _print(f"Semantic vectors: {len(_sem_vectors)}/{len(item_keys_for_layout)} items")

        # noverlap 用のノード描画サイズ（build時と同じ式で先に計算）
        _citer_pids = {f"paper:{c['citing_paper_id']}" for c in citers if c.get("citing_paper_id")}
        _node_sizes: dict[str, float] = {}
        for d in items:
            _node_sizes[f"item:{d['item_key']}"] = _node_size(d.get("citer_count"))
        for r in citers:
            pid = r.get("citing_paper_id")
            if pid:
                _node_sizes[f"paper:{pid}"] = _node_size(r.get("citing_citation_count"))
        for r in refs:
            pid = r.get("cited_paper_id")
            if pid:
                nid = f"paper:{pid}" if f"paper:{pid}" in _citer_pids else f"ref:{pid}"
                _node_sizes.setdefault(nid, _node_size(r.get("cited_citation_count")))

        _print("Computing layout (FA2 LinLog + semantic edges)…", end=" ", flush=True)
        layout_positions = compute_layout(
            item_keys_for_layout, citers, refs,
            warm_start=_stale_positions or None,
            semantic_vectors=_sem_vectors or None,
            node_sizes=_node_sizes,
        )
        _print(f"done in {_time.time()-_t0:.1f}s  ({len(layout_positions)} nodes placed)")
        # アトミック書き込み: tempファイルに書いてからrename（中断時に既存キャッシュを壊さない）
        try:
            _tmp = _cache_path.with_suffix(".tmp")
            _tmp.write_text(_json.dumps({
                "key":       _cache_key,
                "positions": {k: list(v) for k, v in layout_positions.items()},
            }))
            _tmp.replace(_cache_path)  # POSIX では atomic
            _print(f"Layout cache saved → {_cache_path.name}")
        except Exception as _e:
            _print(f"Layout cache save error: {_e}")

    nodes: list[dict] = []
    edges: list[dict] = []
    added_papers: set[str] = set()
    edge_counter = 0

    def _eid() -> str:
        nonlocal edge_counter
        edge_counter += 1
        return f"e{edge_counter}"

    # ── node_identifier_overrides テーブルからオーバーライドを読み込む ─────────
    _id_overrides: dict[str, dict] = {}
    _ov_path = db_path or DB_PATH
    try:
        with sqlite3.connect(_ov_path) as _ov_conn:
            _ensure_override_table(_ov_conn)
            for row in _ov_conn.execute(
                "SELECT node_id, doi, isbn, title, year, authors, citations FROM node_identifier_overrides"
            ):
                _id_overrides[row[0]] = {
                    "doi": row[1], "isbn": row[2], "title": row[3],
                    "year": row[4], "authors": row[5], "citations": row[6],
                }
    except Exception as _ov_err:
        print(f"[warn] override load failed: {_ov_err}", file=__import__("sys").stderr)

    def _apply_id_override(
        nid: str, doi_val: str, isbn_val: str = "", title_val: str = "",
        year_val: str = "", authors_val: str = "", citations_val: str = ""
    ) -> tuple[str, str, str, str, str, str]:
        """オーバーライドがあればDOI/ISBN/タイトル/年/著者/被引用数を上書きして返す。"""
        ov = _id_overrides.get(nid)
        if ov:
            if ov.get("doi")       is not None: doi_val       = ov["doi"]
            if ov.get("isbn")      is not None: isbn_val      = ov["isbn"]
            if ov.get("title")     is not None: title_val     = ov["title"]
            if ov.get("year")      is not None: year_val      = ov["year"]
            if ov.get("authors")   is not None: authors_val   = ov["authors"]
            if ov.get("citations") is not None: citations_val = ov["citations"]
        return doi_val, isbn_val, title_val, year_val, authors_val, citations_val

    def _norm_isbn(s: str) -> str:
        """ISBN を正規化（ハイフン・スペース除去、小文字化）して比較キーにする。"""
        return s.replace("-", "").replace(" ", "").strip().lower()

    # ── DOI / ISBN → item:KEY マップ（外部論文との重複排除に使用）─────────
    doi_to_item_nid: dict[str, str] = {}
    isbn_to_item_nid: dict[str, str] = {}
    for d in items:
        item_nid = f"item:{d['item_key']}"
        # overrides が適用されたDOI/ISBNを使う
        doi_ov, isbn_ov, *_ = _apply_id_override(
            item_nid, d.get("doi") or "", d.get("isbn") or ""
        )
        raw_doi = doi_ov.strip().lower()
        if raw_doi:
            doi_to_item_nid[raw_doi] = item_nid
        for isbn_part in (isbn_ov or "").split():
            nk = _norm_isbn(isbn_part)
            if nk:
                isbn_to_item_nid[nk] = item_nid

    # 外部論文間のDOI/ISBN重複排除マップ（初出のnidを正規IDとして記録）
    doi_to_ext_nid:  dict[str, str] = {}
    isbn_to_ext_nid: dict[str, str] = {}

    def _resolve_external_nid(nid_candidate: str, doi_raw: str, isbn_raw: str = "") -> str:
        """外部論文のノードIDを返す。
        1. DOI/ISBNがZoteroアイテムと一致 → item:KEY に統合
        2. DOI/ISBNが既出の外部論文と一致 → 先に追加された外部論文のnidに統合
        """
        if doi_raw:
            doi_key = doi_raw.strip().lower()
            if doi_key in doi_to_item_nid: return doi_to_item_nid[doi_key]
            if doi_key in doi_to_ext_nid:  return doi_to_ext_nid[doi_key]
        if isbn_raw:
            for isbn_part in isbn_raw.split():
                nk = _norm_isbn(isbn_part)
                if not nk: continue
                if nk in isbn_to_item_nid: return isbn_to_item_nid[nk]
                if nk in isbn_to_ext_nid:  return isbn_to_ext_nid[nk]
        return nid_candidate

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

        _cc_str = str(s2_cc) if s2_cc is not None else str(count)
        doi_val, isbn_val, title_ov_z, year_ov_z, authors_ov_z, citations_ov_z = _apply_id_override(
            f"item:{d['item_key']}", d.get("doi") or "", d.get("isbn") or "",
            full or "", year_val, creators, _cc_str,
        )
        if title_ov_z:    full     = title_ov_z
        if year_ov_z:     year_val = year_ov_z
        if authors_ov_z:  creators = authors_ov_z
        if citations_ov_z:
            try: s2_cc = int(citations_ov_z)
            except ValueError: pass

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
        authors = d.get("citing_authors") or ""
        raw_doi, raw_isbn, title_ov, year_ov, authors_ov, citations_ov = _apply_id_override(
            f"paper:{pid}", d.get("citing_doi") or "", "", title, year,
            authors, str(cc),
        )
        if title_ov:    title   = title_ov
        if year_ov:     year    = year_ov
        if authors_ov:  authors = authors_ov
        if citations_ov:
            try: cc = int(citations_ov)
            except ValueError: pass
        nid   = _resolve_external_nid(f"paper:{pid}", raw_doi, raw_isbn)

        if nid not in added_papers and not nid.startswith("item:"):
            added_papers.add(nid)
            if raw_doi:
                doi_to_ext_nid.setdefault(raw_doi.strip().lower(), nid)
            for _ip in (raw_isbn or "").split():
                _nik = _norm_isbn(_ip)
                if _nik: isbn_to_ext_nid.setdefault(_nik, nid)
            _xy = layout_positions.get(nid)
            nodes.append({
                "id":      nid,
                "label":   _short(title, 24),
                "size":    _node_size(cc),
                "color":   C_EXTERNAL,
                "tooltip": _tooltip(title, [
                    ("Authors",    authors or "—"),
                    ("Year",       year or "—"),
                    ("DOI",        raw_doi or "—"),
                    ("Citations",  f"{cc:,}" if cc else "—"),
                    ("References", "—"),
                ]),
                "group":     "external",
                "cc":        cc,
                "fullTitle": title,
                "year":      year,
                "doi":       raw_doi,
                "isbn":      raw_isbn,
                "authors":   authors,
                **( {"x": round(_xy[0], 1), "y": round(_xy[1], 1)} if _xy else {"x": 0.0, "y": 0.0} ),
            })
        elif nid.startswith("item:"):
            added_papers.add(f"paper:{pid}")  # paper:PID は処理済みとしてマーク

        target_nid = f"item:{d['cited_item_key']}"
        if nid != target_nid:  # 自己ループを防止
            edges.append({
                "id":     _eid(),
                "source": nid,
                "target": target_nid,
                "size":   max(0.5, min(4.0, d["context_count"] * 0.4)),
                "color":  PALETTE["edgeDefault"],
                "type":   "arrow",
                "direction": "citations",
                "itemKey": d["cited_item_key"],
                "externalPaperId": pid,
                "externalTitle": title,
                "relationKey": f"citations:{d['cited_item_key']}:{pid}",
            })

    # ── reference-paper nodes + edges ─────────────────────────────────────
    n_citer = len(added_papers)
    ref_set: set[str] = set()

    for d in refs:
        pid     = d["cited_paper_id"]
        title   = d["cited_title"] or pid
        year    = str(d["cited_year"] or "")
        cc      = d["cited_citation_count"] or 0
        base_nid = f"paper:{pid}" if f"paper:{pid}" in added_papers else f"ref:{pid}"
        authors  = d.get("cited_authors") or ""
        raw_doi, raw_isbn, title_ov, year_ov, authors_ov, citations_ov = _apply_id_override(
            base_nid, d.get("cited_doi") or "", "", title, year,
            authors, str(cc),
        )
        if title_ov:    title   = title_ov
        if year_ov:     year    = year_ov
        if authors_ov:  authors = authors_ov
        if citations_ov:
            try: cc = int(citations_ov)
            except ValueError: pass
        nid      = _resolve_external_nid(base_nid, raw_doi, raw_isbn)

        if nid not in added_papers and not nid.startswith("item:"):
            added_papers.add(nid)
            ref_set.add(nid)
            if raw_doi:
                doi_to_ext_nid.setdefault(raw_doi.strip().lower(), nid)
            for _ip in (raw_isbn or "").split():
                _nik = _norm_isbn(_ip)
                if _nik: isbn_to_ext_nid.setdefault(_nik, nid)
            _xy = layout_positions.get(nid)
            nodes.append({
                "id":      nid,
                "label":   _short(title, 24),
                "size":    _node_size(cc),
                "color":   C_EXTERNAL,
                "tooltip": _tooltip(title, [
                    ("Authors",    authors or "—"),
                    ("Year",       year or "—"),
                    ("DOI",        raw_doi or "—"),
                    ("Citations",  f"{cc:,}" if cc else "—"),
                    ("References", "—"),
                ]),
                "group":     "reference",
                "cc":        cc,
                "fullTitle": title,
                "year":      year,
                "doi":       raw_doi,
                "isbn":      raw_isbn,
                "authors":   authors,
                **( {"x": round(_xy[0], 1), "y": round(_xy[1], 1)} if _xy else {"x": 0.0, "y": 0.0} ),
            })
        elif nid.startswith("item:"):
            added_papers.add(base_nid)  # 元のIDを処理済みとしてマーク

        source_nid = f"item:{d['citing_item_key']}"
        if nid != source_nid:  # 自己ループを防止
            edges.append({
                "id":     _eid(),
                "source": source_nid,
                "target": nid,
                "size":   max(0.5, min(3.0, d["context_count"] * 0.3)),
                "color":  PALETTE["edgeDefault"],
                "type":   "arrow",
                "direction": "references",
                "itemKey": d["citing_item_key"],
                "externalPaperId": pid,
                "externalTitle": title,
                "relationKey": f"references:{d['citing_item_key']}:{pid}",
            })

    n_nodes = len(nodes)
    n_edges = len(edges)
    n_ref   = len(ref_set)

    print(f"   {n_nodes} nodes  |  {n_edges} edges")
    print(f"   Rose={len(items)} Zotero  Grey={n_citer} citers  Grey={n_ref} refs")

    return {
        "nodes":     nodes,
        "edges":     edges,
        "cache_hit": _cache_hit,
        "meta": {
            "n_items":  len(items),
            "n_nodes":  n_nodes,
            "n_edges":  n_edges,
            "n_citer":  n_citer,
            "n_ref":    n_ref,
            "palette":  PALETTE,
            "css_root": _css_root,
            "js_theme": _js_theme,
        },
    }


# ── FastAPI app ───────────────────────────────────────────────────────────────

import threading as _threading

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# サーバー起動時に main() が詰める in-memory state
_state: dict = {
    "graph":      None,   # {"nodes": [...], "edges": [...]}
    "html":       None,   # str – HTML シェル
    "db_path":    None,   # str – DB ファイルパス
    "translator": None,   # Azure Translator config (or None if key not set)
    "args":       None,   # argparse.Namespace – 起動時引数（リビルドに使用）
    "cache_hit":  False,  # 最後のビルドでレイアウトキャッシュがヒットしたか
}

# リビルド完了を /api/graph に通知するためのイベント
_rebuild_done         = _threading.Event()
_rebuild_lock         = _threading.Lock()
_initial_build_done   = _threading.Event()  # main() の初期ビルドが終わるまで False

# ── ブラウザ閉じ検知（/api/close シグナル方式）──────────────────────────────
import time as _time_hb

_close_pending_at: float = 0.0        # /api/close を受信した時刻（0 = 未受信）
_browser_opened   = False             # /api/heartbeat を受信したか（ブラウザが開いた）
_CLOSE_GRACE      = 10.0              # close 後この秒数以内に heartbeat が来たらキャンセル

def _close_watcher() -> None:
    """close シグナルを受信後、猶予時間内に heartbeat が来なければ終了する。"""
    global _close_pending_at, _browser_opened
    while True:
        _time_hb.sleep(1)
        if not _browser_opened or _close_pending_at == 0.0:
            continue
        if _time_hb.time() - _close_pending_at > _CLOSE_GRACE:
            # リビルド中なら完了を待ってから終了（キャッシュ保存を確実に行う）
            if _rebuild_lock.locked():
                print("\nブラウザが閉じられました。リビルド完了後に終了します…", flush=True)
                _rebuild_done.wait(timeout=180)
            print("\nブラウザが閉じられました。サーバーを終了します。", flush=True)
            import os as _os
            _os.kill(_os.getpid(), __import__("signal").SIGINT)
            return

_threading.Thread(target=_close_watcher, daemon=True).start()

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


def _rebuild_graph() -> None:
    """DB を読み直してグラフデータを再構築し _state["graph"] を更新する。"""
    args = _state["args"]
    conn = sqlite3.connect(_state["db_path"])
    conn.row_factory = sqlite3.Row
    try:
        if args.item:
            item_row = get_item_row(conn, args.item)
            items = [item_row] if item_row else []
        else:
            items = get_top_items(conn, args.top)
        if not items:
            return
        item_keys = [d["item_key"] for d in items]
        citers    = get_citers(conn, item_keys, args.citers, min_cc=args.min_cc)
        refs      = [] if args.no_refs else get_refs(conn, item_keys, args.refs, min_cc=args.min_cc)
        item_meta = get_item_meta(item_keys)
        item_rcnt = get_item_ref_counts(conn, item_keys)
    finally:
        conn.close()
    result = build_graph_data(items, citers, refs,
                              item_meta=item_meta, item_ref_counts=item_rcnt,
                              db_path=_state["db_path"])
    m = result["meta"]
    _state["graph"]     = {"nodes": result["nodes"], "edges": result["edges"]}
    _state["cache_hit"] = result.get("cache_hit", False)
    _state["html"]      = _build_sigma_html(
        n_items=m["n_items"], n_nodes=m["n_nodes"], n_edges=m["n_edges"],
        n_citer=m["n_citer"], n_ref=m["n_ref"],
        palette=m["palette"], css_root=m["css_root"], js_theme=m["js_theme"],
    )


def _rebuild_graph_bg() -> None:
    """バックグラウンドスレッドでリビルドを実行し、完了を通知する。"""
    with _rebuild_lock:
        _rebuild_done.clear()
        try:
            _rebuild_graph()
        finally:
            _rebuild_done.set()


@app.get("/", response_class=HTMLResponse)
def _route_index() -> str:
    # 初期ビルド完了後かつ別のリビルドが走っていない場合のみ再ビルドを起動
    # （初期ビルド前に叩かれた場合は /api/graph が _rebuild_done.wait() で待機する）
    if _initial_build_done.is_set() and not _rebuild_lock.locked():
        t = _threading.Thread(target=_rebuild_graph_bg, daemon=False)
        t.start()
    return _state["html"]


@app.get("/api/graph")
def _route_graph() -> JSONResponse:
    # リビルドが実行中の場合は完了まで待機（最大120秒）
    _rebuild_done.wait(timeout=120)
    payload = dict(_state["graph"])
    payload["cache_hit"] = _state.get("cache_hit", False)
    return JSONResponse(payload)


@app.post("/api/heartbeat")
def _route_heartbeat() -> JSONResponse:
    """ページロード時の生存通知。close が pending 中でもキャンセルする。"""
    global _browser_opened, _close_pending_at
    _browser_opened    = True
    _close_pending_at  = 0.0   # close をキャンセル（リフレッシュ等）
    return JSONResponse({"ok": True})


@app.post("/api/close")
def _route_close() -> JSONResponse:
    """タブ・ウィンドウが閉じられたときのシグナル。猶予後に終了する。"""
    global _close_pending_at
    _close_pending_at = _time_hb.time()
    return JSONResponse({"ok": True})


@app.get("/api/collections")
def _route_collections() -> JSONResponse:
    """Zotero SQLite からコレクション一覧と所属 item_key を返す。"""
    if not os.path.exists(ZOTERO_SQLITE):
        return JSONResponse(
            {"error": "zotero_not_found",
             "message": f"Zotero データベースが見つかりません（{ZOTERO_SQLITE}）。"
                        "Zotero を起動してください。"},
            status_code=503,
        )
    try:
        uri = f"file:{ZOTERO_SQLITE}?mode=ro&nolock=1"
        zconn = sqlite3.connect(uri, uri=True, timeout=5)
        zconn.row_factory = sqlite3.Row

        # コレクション全件（ID, 名前, 親ID）
        cols = zconn.execute(
            "SELECT collectionID, collectionName, parentCollectionID "
            "FROM collections ORDER BY collectionName"
        ).fetchall()

        # 親IDをたどってパスを構築
        col_by_id: dict = {r["collectionID"]: dict(r) for r in cols}
        def _path(cid: int) -> str:
            parts = []
            visited = set()
            while cid and cid not in visited:
                visited.add(cid)
                c = col_by_id.get(cid)
                if not c:
                    break
                parts.append(c["collectionName"])
                cid = c["parentCollectionID"]
            return " / ".join(reversed(parts))

        # collectionItems → items.key のマッピング
        rows = zconn.execute("""
            SELECT ci.collectionID, i.key AS item_key
            FROM collectionItems ci
            JOIN items i ON i.itemID = ci.itemID
        """).fetchall()
        zconn.close()

        # コレクションごとに item_key リストを集約
        from collections import defaultdict as _dd
        col_items: dict = _dd(list)
        for r in rows:
            col_items[r["collectionID"]].append(r["item_key"])

        result = [
            {
                "id":        cid,
                "path":      _path(cid),
                "item_keys": col_items.get(cid, []),
            }
            for cid in col_by_id
        ]
        result.sort(key=lambda x: x["path"])
        return JSONResponse({"collections": result})

    except Exception as e:
        return JSONResponse(
            {"error": "read_failed",
             "message": f"Zotero データベースの読み込みに失敗しました: {e}。"
                        "Zotero が起動中の場合はしばらく待ってから再試行してください。"},
            status_code=503,
        )


@app.get("/api/semantic-layout")
def _route_semantic_layout(method: str = "umap", n_clusters: int = 8) -> JSONResponse:
    """資料ベクトルを次元削減して2D意味空間マップ座標を返す。

    Query params:
        method: "umap" (default) | "tsne" | "pca" | "mds"
        n_clusters: クラスタ数 (default: 8, range: 2-20)

    Returns:
        { positions: { item_key: [x, y], ... }, method, n_items, cached }
    """
    if method not in SEMANTIC_LAYOUT_METHODS:
        return JSONResponse(
            {"error": "invalid_method",
             "message": f"Unknown method '{method}'. Supported: {', '.join(SEMANTIC_LAYOUT_METHODS)}"},
            status_code=400,
        )

    # グラフ構築中なら待機
    _rebuild_done.wait(timeout=120)
    if _state["graph"] is None:
        return JSONResponse(
            {"error": "no_graph", "message": "Graph not yet built. Please wait."},
            status_code=503,
        )

    # Zotero アイテムキーのセット（キャッシュ検証用）
    _graph = _state["graph"]
    _zotero_item_keys = set()
    for n in _graph.get("nodes", []):
        if n.get("group") == "zotero" and n.get("itemKey"):
            _zotero_item_keys.add(n["itemKey"])

    # キャッシュ確認（node ID 形式 "item:XXX" で保存されている）
    cached = _load_semantic_layout_cache(method)
    if cached:
        # キャッシュから Zotero の itemKey だけ抽出して検証
        _cached_zotero = set()
        for k in cached:
            if k.startswith("item:"):
                _cached_zotero.add(k.split(":", 1)[1])
        if _cached_zotero == _zotero_item_keys:
            # キャッシュヒット時もクラスタリングは毎回計算（軽量）
            item_keys_for_cluster = sorted(_zotero_item_keys)
            vectors_cached = get_item_vectors(item_keys_for_cluster)
            positions_cached = {k: v for k, v in cached.items() if k.startswith("item:")}
            _raw_positions = {k.split(":", 1)[1]: v for k, v in positions_cached.items()}
            clusters_cached = compute_clusters(vectors_cached, _raw_positions, n_clusters=n_clusters) if len(vectors_cached) >= 5 else []

            return JSONResponse({
                "positions": {k: list(v) for k, v in cached.items()},
                "clusters": clusters_cached,
                "method": method,
                "n_items": len(_cached_zotero),
                "cached": True,
            })
        print(f"  [semantic-layout] cache miss (item set changed), recomputing {method}",
              file=sys.stderr)

    # ベクトルロード
    item_keys_for_layout = sorted(_zotero_item_keys)
    vectors = get_item_vectors(item_keys_for_layout)
    if len(vectors) < 2:
        return JSONResponse({
            "positions": {},
            "method": method,
            "n_items": 0,
            "cached": False,
            "warning": "Not enough item vectors (need >= 2). Run the indexer first.",
        })

    positions = compute_semantic_layout(vectors, method=method)
    if not positions:
        return JSONResponse({
            "positions": {},
            "method": method,
            "n_items": 0,
            "cached": False,
            "warning": "Dimensionality reduction produced no results.",
        })

    # item:KEY → (x, y) のマップを構築（node ID 形式で統一）
    # 外部論文は意味マップモードでは非表示のため、Zotero 資料のみ返す
    _item_pos: dict[str, tuple[float, float]] = {}
    for item_key, (x, y) in positions.items():
        _item_pos[f"item:{item_key}"] = (x, y)

    # キャッシュに保存
    if _item_pos:
        _save_semantic_layout_cache(method, _item_pos)

    # クラスタリング
    clusters = compute_clusters(vectors, positions, n_clusters=n_clusters)

    return JSONResponse({
        "positions": {k: list(v) for k, v in _item_pos.items()},
        "clusters": clusters,
        "method": method,
        "n_items": len(_item_pos),
        "cached": False,
    })


class _TranslateBatchRequest(BaseModel):
    texts: list[str]
    target: str = "ja"


@app.post("/api/translate/batch")
def _route_translate_batch(req: _TranslateBatchRequest) -> JSONResponse:
    """Azure Cognitive Services Translator で複数テキストを一括翻訳して返す。"""
    if not _state.get("translator"):
        return JSONResponse(
            {"error": "翻訳未設定（AZURE_TRANSLATOR_KEY を .env に追加してください）"},
            status_code=503,
        )
    if not req.texts:
        return JSONResponse({"translations": []})
    import requests as _req
    cfg = _state["translator"]
    try:
        resp = _req.post(
            "https://api.cognitive.microsofttranslator.com/translate",
            params={"api-version": "3.0", "to": req.target},
            headers={
                "Ocp-Apim-Subscription-Key":    cfg["key"],
                "Ocp-Apim-Subscription-Region": cfg["region"],
                "Content-Type": "application/json",
            },
            json=[{"text": t} for t in req.texts],
            timeout=30,
        )
        resp.raise_for_status()
        translations = [item["translations"][0]["text"] for item in resp.json()]
        return JSONResponse({"translations": translations})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/edge/contexts")
def _route_edge_contexts(src: str, tgt: str) -> dict:
    """エッジ(src → tgt)の引用コンテキストを全件返す。"""
    contexts = get_contexts_for_edge(_state["db_path"], src, tgt)
    return {"contexts": contexts}


class _RelationReportRequest(BaseModel):
    direction: str
    item_key: str
    external_paper_id: str
    external_title: str = ""
    reason: str = "other"
    details: str


@app.post("/api/relation/report")
def _route_relation_report(body: _RelationReportRequest) -> JSONResponse:
    """Queue a graph relation for human review; do not hide it immediately."""
    try:
        from db_relations import submit_relation_report
        report = submit_relation_report(
            direction=body.direction,
            item_key=body.item_key,
            external_paper_id=body.external_paper_id,
            external_title=body.external_title,
            reason=body.reason,
            details=body.details,
            reporter="citation-graph",
        )
        return JSONResponse({"ok": True, "report": report})
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


_SRC_DIR = str(PROJECT_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# オンデマンド要約で選択可能なモデル（key: APIモデルID, value: 表示名）。
SUMMARY_MODELS = {
    "deepseek-v4-pro": "DeepSeek V4 Pro",
}
SUMMARY_DEFAULT_MODEL = "deepseek-v4-pro"


@app.get("/api/node/abstract")
def _route_node_abstract(key: str) -> JSONResponse:
    """アイテムのアブストラクトとキャッシュ済み要約を返す。"""
    from db_relations import get_item_abstract, get_item_summary
    abstract = get_item_abstract(key)
    summary  = get_item_summary(key)
    return JSONResponse({
        "abstract": abstract,
        "summary":  summary,
    })


class _FetchAbstractRequest(BaseModel):
    item_key: str


@app.post("/api/node/fetch-abstract")
def _route_fetch_abstract(body: _FetchAbstractRequest) -> JSONResponse:
    """Zotero local API から abstractNote を取得して DB にキャッシュする。

    Zotero 起動中のみ利用可能（SQLite は起動中ロックされるため HTTP API を使う）。
    取得できた概要は item_citation_status.abstract に保存し、以降は DB から表示する。
    """
    from db_relations import get_item_abstract, update_item_citation_status

    # 既にキャッシュ済みなら即返す
    cached = get_item_abstract(body.item_key)
    if cached:
        return JSONResponse({"abstract": cached, "cached": True})

    import urllib.request as _ur, urllib.error as _ue, urllib.parse as _uq
    # Zotero アイテムキーは英数字のみだが念のためエスケープ
    url = f"http://127.0.0.1:23119/api/users/0/items/{_uq.quote(body.item_key)}"
    try:
        req = _ur.Request(url, headers={"Accept": "application/json"})
        with _ur.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
    except _ue.HTTPError as e:
        if e.code == 404:
            return JSONResponse(
                {"abstract": None, "error": "Zotero ライブラリにアイテムが見つかりません"},
                status_code=404)
        return JSONResponse(
            {"abstract": None, "error": f"Zotero API エラー {e.code}"}, status_code=502)
    except Exception:
        return JSONResponse(
            {"abstract": None,
             "error": "Zotero に接続できません（Zotero を起動してください）"},
            status_code=503)

    item = data.get("data", data) if isinstance(data, dict) else {}
    abstract = (item.get("abstractNote") or "").strip() or None
    if abstract:
        update_item_citation_status(body.item_key, "mapped", abstract=abstract)
        return JSONResponse({"abstract": abstract, "cached": False})
    return JSONResponse({"abstract": None, "found": False})


@app.get("/api/node/external-abstract")
def _route_external_abstract(paper_id: str = "", doi: str = "") -> JSONResponse:
    """外部論文の概要を取得して返す。Crossref（DOI）を優先し、無ければ S2 にフォールバック。

    取得順:
      1) Crossref を DOI で引いて Abstract を取得（書誌系の主ソース）
      2) Crossref に Abstract が無い／DOI が無い場合は S2 にフォールバックし、
         abstract と tldr（S2 自身の AI 要約）を取得
    どちらでも得られなければ status='none'。外部論文には本文チャンクが無いため
    AI 要約は生成しない（情報不足時のハルシネーション回避）。
    """
    from db_relations import get_external_abstract, save_external_abstract

    paper_id = (paper_id or "").strip()
    doi = (doi or "").strip()
    if not paper_id and not doi:
        return JSONResponse({"error": "paper_id か doi が必要です"}, status_code=400)

    cache_key = f"DOI:{doi.lower()}" if doi else paper_id

    # キャッシュ確認（found / none はそのまま返す。エラーは行が無いので再取得される）
    cached = get_external_abstract(cache_key)
    if cached:
        return JSONResponse({
            "abstract": cached["abstract"], "tldr": cached["tldr"],
            "status": cached["status"], "cached": True,
        })

    abstract = None
    tldr = None
    transient_error = False  # 一時失敗があれば（none でも）キャッシュせず再試行可能にする

    # 1) Crossref 優先（DOI 必須）
    if doi:
        from crossref_client import fetch_crossref_by_doi, CrossrefError
        try:
            meta = fetch_crossref_by_doi(doi)
            abstract = (meta or {}).get("abstract")
        except CrossrefError:
            transient_error = True  # S2 で補える可能性があるので続行

    # 2) Crossref で Abstract が得られなければ S2 にフォールバック（abstract + tldr）
    if not abstract:
        from citation_mapper import s2_request
        import urllib.parse as _up
        if paper_id:
            s2_url = f"https://api.semanticscholar.org/graph/v1/paper/{_up.quote(paper_id)}?fields=abstract,tldr"
        elif doi:
            s2_url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{_up.quote(doi)}?fields=abstract,tldr"
        else:
            s2_url = None
        if s2_url:
            data = s2_request(s2_url, max_retries=3)
            if data is None:
                transient_error = True
            else:
                if not abstract:
                    abstract = (data.get("abstract") or "").strip() or None
                tldr_obj = data.get("tldr") or {}
                tldr = (tldr_obj.get("text") or "").strip() or None if isinstance(tldr_obj, dict) else None

    # 何も得られず一時エラーがあった → キャッシュせず再試行可能なエラーを返す
    if not abstract and not tldr and transient_error:
        return JSONResponse(
            {"abstract": None, "tldr": None, "status": "error",
             "error": "概要を取得できませんでした（時間をおいて再試行してください）"},
            status_code=502,
        )

    status = "found" if (abstract or tldr) else "none"
    save_external_abstract(cache_key, abstract, tldr, status)
    return JSONResponse({
        "abstract": abstract, "tldr": tldr, "status": status, "cached": False,
    })


class _SummaryRequest(BaseModel):
    item_key: str
    title:    str = ""
    force:    bool = False
    model:    str = SUMMARY_DEFAULT_MODEL


def _natural_key(s: str) -> list:
    """'a:p10:para2' 形式のIDを数値考慮で並べるためのソートキー。"""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


@app.post("/api/node/summary")
def _route_generate_summary(body: _SummaryRequest) -> JSONResponse:
    """ChromaDB チャンクからDeepSeekで要約を生成してキャッシュする。"""
    from db_relations import get_item_summary, save_item_summary
    from build_summaries import _excluded_from_llm
    from llm_client import DeepSeekClient, LLMError

    # キャッシュ済みで force=False なら即返す
    if not body.force:
        cached = get_item_summary(body.item_key)
        if cached:
            return JSONResponse({
                "summary": cached["summary"], "model": cached["model"],
                "updated_at": cached["updated_at"], "cached": True,
            })

    excluded, exclusion_reason = _excluded_from_llm(body.item_key)
    if excluded:
        return JSONResponse(
            {"error": f"クラウド要約対象外です: {exclusion_reason}"}, status_code=403,
        )

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return JSONResponse({"error": "DEEPSEEK_API_KEY が .env に設定されていません"}, status_code=503)

    model = body.model if body.model in SUMMARY_MODELS else SUMMARY_DEFAULT_MODEL

    try:
        chunks_text = [chunk["text"] for chunk in get_item_chunks(body.item_key) if chunk["text"]]
    except Exception as e:
        return JSONResponse({"error": f"ChromaDB 読み取りエラー: {e}"}, status_code=500)

    if not chunks_text:
        return JSONResponse({"error": "チャンクが見つかりません（インデックスを確認してください）"}, status_code=404)

    # チャンクを連結（最大 60,000 文字）
    combined = "\n\n".join(chunks_text)[:60000]

    title_hint = f"（タイトル: {body.title}）" if body.title else ""
    prompt = (
        f"次に示すのは学術資料のテキスト断片です{title_hint}。\n"
        "この資料の内容を研究者向けに日本語で 400〜600 字程度に要約してください。\n"
        "主張・方法・結論を含め、簡潔かつ正確にまとめます。前置きや見出しは不要で、"
        "要約本文のみを書いてください。\n\n"
        f"=== 資料ここから ===\n{combined}\n=== 資料ここまで ===\n\n"
        "日本語要約："
    )
    try:
        summary_text = DeepSeekClient(model).generate_text(
            prompt, max_tokens=2048, timeout=120,
        )
    except LLMError as e:
        return JSONResponse({"error": f"DeepSeek API エラー: {e}"}, status_code=502)

    stored_model = f"deepseek:{model}"
    save_item_summary(body.item_key, summary_text, stored_model)
    saved = get_item_summary(body.item_key)
    return JSONResponse({
        "summary": summary_text, "model": stored_model,
        "updated_at": saved["updated_at"] if saved else None, "cached": False,
    })


class _SummarySaveRequest(BaseModel):
    item_key: str
    summary:  str


@app.put("/api/node/summary")
def _route_save_summary(body: _SummarySaveRequest) -> JSONResponse:
    """手動編集した要約を保存する。"""
    from db_relations import save_item_summary
    save_item_summary(body.item_key, body.summary, "manual")
    return JSONResponse({"ok": True})


class _IdentifierUpdate(BaseModel):
    node_id: str
    field: str   # "doi" or "isbn"
    value: str


def _ensure_override_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS node_identifier_overrides (
            node_id    TEXT PRIMARY KEY,
            doi        TEXT,
            isbn       TEXT,
            title      TEXT,
            year       TEXT,
            authors    TEXT,
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)
    # 既存テーブルへのカラム追加（マイグレーション）
    for col, coltype in [("isbn", "TEXT"), ("title", "TEXT"), ("year", "TEXT"), ("authors", "TEXT"), ("citations", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE node_identifier_overrides ADD COLUMN {col} {coltype}")
        except Exception:
            pass  # 既に存在する場合は無視


@app.post("/api/node/identifier")
def _route_update_identifier(body: _IdentifierUpdate) -> JSONResponse:
    _ALLOWED = ("doi", "isbn", "title", "year", "authors", "citations")
    if body.field not in _ALLOWED:
        return JSONResponse({"error": "invalid field"}, status_code=400)
    db_path = _state["db_path"]
    with sqlite3.connect(db_path) as conn:
        _ensure_override_table(conn)
        conn.execute(f"""
            INSERT INTO node_identifier_overrides (node_id, {body.field}, updated_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(node_id) DO UPDATE SET
                {body.field} = excluded.{body.field},
                updated_at   = excluded.updated_at
        """, (body.node_id, body.value or None))
        conn.commit()
    # in-memory graph を即時更新
    for node in (_state["graph"] or {}).get("nodes", []):
        if node["id"] == body.node_id:
            node[body.field] = body.value
            if body.field == "title":
                node["fullTitle"] = body.value
                node["label"]     = body.value[:24] if body.value else node["label"]
            break
    return JSONResponse({"ok": True})


def _find_free_port(start: int) -> int:
    import socket
    for p in range(start, start + 20):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", p))
                return p
        except OSError:
            continue
    return start  # fallback (uvicorn will error if truly busy)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import threading
    import uvicorn

    parser = argparse.ArgumentParser(
        description="Visualize citation network – serves at http://localhost:PORT")
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
    parser.add_argument("--port",    type=int, default=DEFAULT_PORT,
                        help=f"HTTPサーバーのポート番号 (default: {DEFAULT_PORT})")
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        print(f"Error: DB not found: {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    # Apply backward-compatible DB migrations before the visualizer's direct
    # SQLite queries (notably the relation_reports exception table).
    from src import db_relations as _db_relations
    _db_relations.DB_PATH = DB_PATH
    _migration_conn = _db_relations.get_db_connection()
    _migration_conn.close()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # ── DBからデータ取得 ──────────────────────────────────────────────────
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

    item_meta = get_item_meta(item_keys)
    item_rcnt = get_item_ref_counts(conn, item_keys)
    conn.close()

    titled        = sum(1 for k in item_keys if item_meta.get(k, {}).get("title"))
    auth          = sum(1 for k in item_keys if item_meta.get(k, {}).get("creators"))
    unique_citers = len(set(d["citing_paper_id"] for d in citers))
    unique_refs   = len(set(d["cited_paper_id"]  for d in refs))
    print(f"Building graph: {len(items)} Zotero items "
          f"({titled} with titles, {auth} with authors), "
          f"{unique_citers} unique citers, {unique_refs} unique refs…")

    # Azure Translator 初期化
    if AZURE_TRANSLATOR_KEY:
        _state["translator"] = {
            "key":    AZURE_TRANSLATOR_KEY,
            "region": AZURE_TRANSLATOR_REGION,
        }
        print(f"  Azure Translator: ready (region={AZURE_TRANSLATOR_REGION})")
    else:
        print("  Azure Translator: AZURE_TRANSLATOR_KEY not set (翻訳機能は無効)")

    # ── サーバーを先に起動してブラウザを開く ──────────────────────────────
    # HTML シェルをプレースホルダーカウントで即時生成（FA2 完了前でも表示可能）
    _state["html"]    = _build_sigma_html(
        n_items=0, n_nodes=0, n_edges=0, n_citer=0, n_ref=0,
        palette=_PALETTE, css_root=_CSS_ROOT, js_theme=_JS_THEME,
    )
    _state["db_path"]    = DB_PATH
    _state["chroma_dir"] = os.environ.get("CHROMA_DIR", str(PROJECT_ROOT / "data" / "chroma"))
    _state["args"]       = args
    # _rebuild_done はクリア状態 → /api/graph はビルド完了まで待機する

    port = _find_free_port(args.port)
    url  = f"http://localhost:{port}"

    # uvicorn をバックグラウンドスレッドで起動
    _uvicorn_server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning"
    ))
    _srv_thread = threading.Thread(target=_uvicorn_server.run, daemon=True)
    _srv_thread.start()

    # ポートが開くまで待機してからブラウザを開く（最大3秒）
    import socket as _socket
    for _ in range(30):
        try:
            with _socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            threading.Event().wait(0.1)

    if not args.no_open:
        webbrowser.open(url)
    print(f"\n  → {url}")
    print("  グラフを計算中… ブラウザにローディング画面が表示されます")
    print("  終了するには Ctrl+C を押してください\n")

    # ── グラフデータ構築（FA2 含む）─────────────────────────────────────
    result = build_graph_data(items, citers, refs,
                               item_meta=item_meta, item_ref_counts=item_rcnt,
                               db_path=DB_PATH)
    m = result["meta"]
    _state["graph"]     = {"nodes": result["nodes"], "edges": result["edges"]}
    _state["cache_hit"] = result.get("cache_hit", False)
    _state["html"]      = _build_sigma_html(
        n_items=m["n_items"], n_nodes=m["n_nodes"], n_edges=m["n_edges"],
        n_citer=m["n_citer"], n_ref=m["n_ref"],
        palette=m["palette"], css_root=m["css_root"], js_theme=m["js_theme"],
    )
    _rebuild_done.set()       # ビルド完了 → /api/graph がレスポンスを返す
    _initial_build_done.set() # 初期ビルド完了 → リロード時の再ビルドを許可

    # メインスレッドでサーバースレッドの終了を待つ（Ctrl+C まで）
    try:
        _srv_thread.join()
    except KeyboardInterrupt:
        _uvicorn_server.should_exit = True
        _srv_thread.join(timeout=3)


if __name__ == "__main__":
    main()
