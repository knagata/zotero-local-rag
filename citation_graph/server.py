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
  uv run citation_graph/server.py               # デフォルト（全件）
  uv run citation_graph/server.py --top 100     # 上位100件
  uv run citation_graph/server.py --item KEY    # 1アイテムに絞る
  uv run citation_graph/server.py --no-refs     # 参照先を非表示
  uv run citation_graph/server.py --no-open     # ブラウザを自動で開かない
  uv run citation_graph/server.py --port 7234   # ポート指定（デフォルト: 7234）

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
import os
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
from citation_graph.graph_service import GraphBuildService
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
    for key, label in zip(keys, labels_arr, strict=False):
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
    for idx, (_gid, members) in enumerate(sorted_clusters):
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
        for i, _c in enumerate(result):
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
        for ci, tok, _score in global_scores:
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
    """sigma.js (WebGL) ベースの HTML シェルを生成する。

    生成するのは外枠と凡例だけ。スタイルと動作の本体は
    citation_graph/static/app.css・app.js にあり、ここでは <link> と
    <script src> で参照する。**CSS/JSをこの関数の文字列リテラルへ戻さないこと**
    （理由は citation_graph/README.md）。グラフデータも同様に埋め込まず
    /api/graph から fetch する。

    Python側から渡すのは _PALETTE 由来の2つだけ: css_root（:root のCSS変数）と
    js_theme（window.__RAG_THEME__ への代入）。件数は f-string で凡例に入る。
    """

    # ── CSS ──────────────────────────────────────────────────────────────
    # 本体は citation_graph/static/app.css。ここでは :root のパレット変数だけを
    # インラインで出す（_PALETTE を単一の真実の源に保つため）。app.css は
    # var(--xxx) 経由で参照するので、この <style> は <link> より前に置くこと。
    css = "<style>\n" + css_root + "\n</style>"

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
  <link rel="stylesheet" href="/static/app.css">
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

  <script>{js_theme}</script>
  <script src="/static/app.js"></script>
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
# app.js は window.__RAG_THEME__ からこれを読む。_PALETTE を単一の真実の源に
# 保つため、パレットをJSファイル側へ複製せずサーバーから注入する。
_JS_THEME = ("window.__RAG_THEME__ = "
             + _json_palette.dumps(_PALETTE, indent=2) + ";")


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

    meta      = item_meta or {}
    item_rcnt = item_ref_counts or {}

    # モジュールレベルの _PALETTE / _CSS_ROOT / _JS_THEME を参照
    PALETTE   = _PALETTE
    _css_root = _CSS_ROOT
    _js_theme = _JS_THEME

    C_ZOTERO   = PALETTE["nodeZotero"]
    C_EXTERNAL = PALETTE["nodeExternal"]

    # ── Server-side layout (FA2 + sector placement) ─────────────────────────
    import sys as _sys, time as _time, hashlib as _hashlib
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
    # S2 paper ID → item:KEY マップ。DOI/ISBN より強い同一性の証拠であり、かつ
    # S2 の書籍レコードは DOI も ISBN も持たないことが多い（例: Bratton "The
    # Stack" は externalIds が MAG/CorpusId のみ）。これが無いと、所蔵資料が
    # 自分のノードと外部ノードの二重で描画される。
    paper_to_item_nid: dict[str, str] = {}
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
        pid_own = str(d.get("s2_paper_id") or "").strip()
        if pid_own:
            paper_to_item_nid[pid_own] = item_nid

    # 外部論文間のDOI/ISBN重複排除マップ（初出のnidを正規IDとして記録）
    doi_to_ext_nid:  dict[str, str] = {}
    isbn_to_ext_nid: dict[str, str] = {}

    # paper ID 経由で所蔵ノードに吸収された外部論文の記録。s2_paper_id は検証を
    # 経ていない保存値なので、誤同定なら「別作品を丸ごと飲み込む」形になり、しかも
    # 統合前なら見えていた重複ノードという症状ごと消える。何を吸収したかを必ず
    # 残し、ノード側からも stderr からも確認できるようにする。
    absorbed_by_item: dict[str, dict[str, str]] = {}

    def _resolve_external_nid(
        nid_candidate: str, doi_raw: str, isbn_raw: str = "", paper_id: str = ""
    ) -> str:
        """外部論文のノードIDを返す。
        1. S2 paper IDがZoteroアイテムのs2_paper_idと一致 → item:KEY に統合
        2. DOI/ISBNがZoteroアイテムと一致 → item:KEY に統合
        3. DOI/ISBNが既出の外部論文と一致 → 先に追加された外部論文のnidに統合
        """
        if paper_id:
            owned = paper_to_item_nid.get(paper_id.strip())
            if owned: return owned
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
        # Every row here is a Zotero-owned item, so it gets the "Zotero アイテム"
        # colour the legend names. Whether S2 could identify the work is a
        # property of S2's coverage, not of ownership; greying those items out
        # made owned books look like they belonged to no category at all (the
        # legend has no entry for grey).  The S2 outcome stays visible in the
        # tooltip's Status row.
        color    = C_ZOTERO

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
        nid   = _resolve_external_nid(f"paper:{pid}", raw_doi, raw_isbn, pid)

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
            if paper_to_item_nid.get(pid) == nid:
                absorbed_by_item.setdefault(nid, {})[pid] = title

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
        nid      = _resolve_external_nid(base_nid, raw_doi, raw_isbn, pid)

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
            if paper_to_item_nid.get(pid) == nid:
                absorbed_by_item.setdefault(nid, {})[pid] = title

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

    # 吸収された外部論文を所蔵ノードに残す。s2_paper_id が誤っていた場合、ここが
    # 「なぜこの資料が別作品の引用を持っているのか」を辿れる唯一の手掛かりになる。
    if absorbed_by_item:
        node_by_id = {n["id"]: n for n in nodes}
        for item_nid, absorbed in absorbed_by_item.items():
            node = node_by_id.get(item_nid)
            if node is None:
                continue
            node["absorbedPapers"] = [
                {"paperId": pid_a, "title": title_a}
                for pid_a, title_a in sorted(absorbed.items())
            ]
        _print(
            f"   Merged by s2_paper_id: {sum(len(v) for v in absorbed_by_item.values())} "
            f"external paper(s) into {len(absorbed_by_item)} owned item(s)"
        )
        for item_nid, absorbed in sorted(absorbed_by_item.items()):
            for pid_a, title_a in sorted(absorbed.items()):
                _print(f"     {item_nid} <- {pid_a[:12]} {title_a[:60]!r}")

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
from citation_graph.lifecycle import SingleFlight, StartOnce

from fastapi import APIRouter, FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
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

def _start_close_watcher() -> None:
    """Start lifecycle monitoring once, never as an import side effect."""
    _close_watcher_once.ensure_started()


_close_watcher_once = StartOnce(
    lambda: _threading.Thread(target=_close_watcher, daemon=True).start()
)

router = APIRouter()


def _rebuild_graph() -> None:
    """DB を読み直してグラフデータを再構築し _state["graph"] を更新する。"""
    service = GraphBuildService(
        get_item_row=get_item_row, get_top_items=get_top_items,
        get_citers=get_citers, get_refs=get_refs, get_item_meta=get_item_meta,
        get_item_ref_counts=get_item_ref_counts, build_graph_data=build_graph_data,
        build_html=_build_sigma_html,
    )
    snapshot = service.rebuild(_state["db_path"], _state["args"])
    if snapshot is None:
        return
    _state["graph"] = snapshot.graph
    _state["cache_hit"] = snapshot.cache_hit
    _state["html"] = snapshot.html


def _rebuild_graph_bg() -> None:
    """バックグラウンドスレッドでリビルドを実行し、完了を通知する。"""
    with _rebuild_lock:
        _rebuild_done.clear()
        try:
            _rebuild_graph()
        finally:
            _rebuild_done.set()


_rebuild_singleflight = SingleFlight(
    _rebuild_graph_bg,
    lambda target: _threading.Thread(target=target, daemon=False).start(),
    externally_busy=_rebuild_lock.locked,
)


def _schedule_rebuild() -> bool:
    return _rebuild_singleflight.schedule()


@router.get("/", response_class=HTMLResponse)
def _route_index() -> str:
    # 初期ビルド完了後かつ別のリビルドが走っていない場合のみ再ビルドを起動
    # （初期ビルド前に叩かれた場合は /api/graph が _rebuild_done.wait() で待機する）
    if _initial_build_done.is_set():
        _schedule_rebuild()
    return _state["html"]


#: 画面のCSS/JSはPython文字列ではなくこのディレクトリのファイルが正。
#: cwdではなく __file__ 基準で解決する（PROJECT_ROOT と同じ理由。どこから
#: 起動されても動く必要がある）。
_STATIC_DIR = Path(__file__).resolve().parent / "static"
#: media_type を明示するのは、Starlette が mimetypes を外したとき text/plain に
#: フォールバックするため。text/plain のスタイルシートはブラウザに拒否され、
#: ページが完全に無スタイルになる。OSのmimetypes DBに依存させない。
_STATIC_ASSETS = {
    "app.css": "text/css; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
}


@router.get("/static/{name}")
def _route_static_asset(name: str) -> Response:
    """許可した2ファイルだけを返す（allowlist方式でパストラバーサル面をゼロにする）。

    ``Cache-Control: no-cache`` は「キャッシュ可だが毎回再検証」。HTMLに ``?v=``
    を焼き込む方式を採らないのは、_route_index が *前回ビルドの* HTML を返して
    次回用のリビルドを裏で走らせる構造のため、埋め込んだバージョン文字列が常に
    1回分古くなり「編集しても2回リロードしないと反映されない」という別の混乱を
    作るから。レスポンス時に評価されるヘッダならその問題が起きない。
    """
    media_type = _STATIC_ASSETS.get(name)
    if media_type is None:
        return JSONResponse({"error": "not_found", "name": name}, status_code=404)
    return FileResponse(
        _STATIC_DIR / name, media_type=media_type,
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/api/graph")
def _route_graph() -> JSONResponse:
    # リビルドが実行中の場合は完了まで待機（最大120秒）。待っても終わらない
    # 場合（初回ビルドが120秒を超える、または他プロセスがDBを占有している
    # 場合など）、_state["graph"] はまだ None のままなので、そのまま
    # dict(None) すると 500 で落ちる。他のエンドポイント（/api/semantic-layout
    # 等）は明示的に None チェックしているのに、ここだけ抜けていた
    # （2026-08-03、実際のクラッシュから発見）。
    _rebuild_done.wait(timeout=120)
    if _state["graph"] is None:
        return JSONResponse(
            {"error": "no_graph", "message": "Graph not yet built. Please wait."},
            status_code=503,
        )
    payload = dict(_state["graph"])
    payload["cache_hit"] = _state.get("cache_hit", False)
    return JSONResponse(payload)


@router.post("/api/heartbeat")
def _route_heartbeat() -> JSONResponse:
    """ページロード時の生存通知。close が pending 中でもキャンセルする。"""
    global _browser_opened, _close_pending_at
    _browser_opened    = True
    _close_pending_at  = 0.0   # close をキャンセル（リフレッシュ等）
    return JSONResponse({"ok": True})


@router.post("/api/close")
def _route_close() -> JSONResponse:
    """タブ・ウィンドウが閉じられたときのシグナル。猶予後に終了する。"""
    global _close_pending_at
    _close_pending_at = _time_hb.time()
    return JSONResponse({"ok": True})


@router.get("/api/collections")
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


@router.get("/api/semantic-layout")
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


@router.post("/api/translate/batch")
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


@router.get("/api/edge/contexts")
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


@router.post("/api/relation/report")
def _route_relation_report(body: _RelationReportRequest) -> JSONResponse:
    """Queue a graph relation for human review; do not hide it immediately."""
    try:
        from src.db_relations import submit_relation_report
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


@router.get("/api/node/abstract")
def _route_node_abstract(key: str) -> JSONResponse:
    """アイテムのアブストラクトとキャッシュ済み要約を返す。"""
    from src.db_relations import get_item_abstract, get_item_root_summary
    abstract = get_item_abstract(key)
    summary = get_item_root_summary(key, searchable_only=True)
    return JSONResponse({
        "abstract": abstract,
        "summary":  summary,
    })


@router.get("/api/node/insights")
def _route_node_insights(key: str) -> JSONResponse:
    """Return the lightweight hierarchy overview for one Zotero item."""
    from citation_graph.insights import get_item_insights
    try:
        return JSONResponse(get_item_insights(key))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.get("/api/node/processing-status")
def _route_node_processing_status(key: str) -> JSONResponse:
    """Return stage-by-stage status and available fallbacks for one item."""
    from citation_graph.insights import get_processing_overview
    try:
        return JSONResponse(get_processing_overview(key))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.get("/api/node/outline")
def _route_node_outline(key: str) -> JSONResponse:
    from citation_graph.insights import get_document_outline
    try:
        return JSONResponse(get_document_outline(key))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.get("/api/processing-status/summary")
def _route_processing_status_summary() -> JSONResponse:
    from src.db_relations import get_processing_status_summary
    return JSONResponse({"items": get_processing_status_summary()})


@router.get("/api/node/sections")
def _route_node_sections(
    key: str, q: str = "", cursor: str = "", limit: int = 50,
) -> JSONResponse:
    from citation_graph.insights import list_sections
    try:
        return JSONResponse(list_sections(
            key, query=q, cursor=cursor or None, limit=limit,
        ))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.get("/api/node/section-source")
def _route_node_section_source(key: str, section_id: str) -> JSONResponse:
    from citation_graph.insights import get_section_source
    try:
        return JSONResponse(get_section_source(key, section_id))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except KeyError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


class _QualityReportRequest(BaseModel):
    target_type: str
    item_key: str = ""
    section_id: str = ""
    reason: str
    details: str
    evidence_chunk_ids: list[str] = []


@router.post("/api/quality-report")
def _route_quality_report(body: _QualityReportRequest) -> JSONResponse:
    """Queue a summary issue without immediately hiding its target."""
    from src.db_relations import submit_summary_quality_report
    try:
        if body.target_type in {"item_summary", "section_summary"}:
            report = submit_summary_quality_report(
                item_key=body.item_key,
                section_id=body.section_id if body.target_type == "section_summary" else None,
                reason=body.reason, details=body.details,
                evidence_chunk_ids=body.evidence_chunk_ids, reporter="citation-graph",
            )
        else:
            raise ValueError("target_type must be item_summary or section_summary.")
        return JSONResponse({"ok": True, "report": report})
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


class _FetchAbstractRequest(BaseModel):
    item_key: str


@router.post("/api/node/fetch-abstract")
def _route_fetch_abstract(body: _FetchAbstractRequest) -> JSONResponse:
    """Zotero local API から abstractNote を取得して DB にキャッシュする。

    Zotero 起動中のみ利用可能（SQLite は起動中ロックされるため HTTP API を使う）。
    取得できた概要は item_citation_status.abstract に保存し、以降は DB から表示する。
    """
    from src.db_relations import (
        get_item_abstract, get_item_citation_status, update_item_citation_status,
    )

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
        # Caching an abstract must not restate the S2 outcome. Passing a literal
        # "mapped" here promoted items S2 had never identified, which then made
        # update_citations.py skip them forever. Keep whatever status the
        # citation pipeline recorded.
        update_item_citation_status(
            body.item_key,
            get_item_citation_status(body.item_key) or "pending",
            abstract=abstract,
        )
        return JSONResponse({"abstract": abstract, "cached": False})
    return JSONResponse({"abstract": None, "found": False})


@router.get("/api/node/external-abstract")
def _route_external_abstract(paper_id: str = "", doi: str = "") -> JSONResponse:
    """外部論文の概要を取得して返す。Crossref（DOI）を優先し、無ければ S2 にフォールバック。

    取得順:
      1) Crossref を DOI で引いて Abstract を取得（書誌系の主ソース）
      2) Crossref に Abstract が無い／DOI が無い場合は S2 にフォールバックし、
         abstract と tldr（S2 自身の AI 要約）を取得
    どちらでも得られなければ status='none'。外部論文には本文チャンクが無いため
    階層要約は生成しない（情報不足時のハルシネーション回避）。
    """
    from src.db_relations import get_external_abstract, save_external_abstract

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
        from src.crossref_client import fetch_crossref_by_doi, CrossrefError
        try:
            meta = fetch_crossref_by_doi(doi)
            abstract = (meta or {}).get("abstract")
        except CrossrefError:
            transient_error = True  # S2 で補える可能性があるので続行

    # 2) Crossref で Abstract が得られなければ S2 にフォールバック（abstract + tldr）
    if not abstract:
        from src.citation_mapper import s2_request
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


@router.post("/api/node/identifier")
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


def create_app() -> FastAPI:
    """Build the ASGI application without starting background workers."""
    application = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    application.include_router(router)
    return application


# Import-compatible ASGI target. Runtime workers remain explicitly started by
# main(), so importing this module is side-effect free.
app = create_app()


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

    _start_close_watcher()

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

    # 画面のCSS/JSは同梱ファイルが正。欠けたまま起動すると、ブラウザは
    # /static/app.js の404を1行コンソールに出すだけでローディング表示のまま
    # 固まり、原因が分からない。ここで止めて理由を明示する。
    # parse_args() の *後* に置くこと: --help は parse_args() 内で終了するため、
    # チェックがそこを壊さない（tests/test_show_citation_graph_entrypoint.py）。
    _missing_assets = [name for name in _STATIC_ASSETS if not (_STATIC_DIR / name).is_file()]
    if _missing_assets:
        print(
            f"Error: static asset(s) missing from {_STATIC_DIR}: "
            f"{', '.join(sorted(_missing_assets))}",
            file=sys.stderr,
        )
        sys.exit(1)

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
