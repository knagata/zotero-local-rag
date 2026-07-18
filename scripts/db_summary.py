#!/usr/bin/env python3
"""
Citation DB サマリーを表示する。
Citation Network更新後のDB概要確認に使用する。
"""
import os
import sqlite3
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.environ.get("RELATIONS_DB_PATH",
                         os.path.join(PROJECT_ROOT, "data", "relations.db"))

if not os.path.exists(DB_PATH):
    print(f"[summary] DB not found: {DB_PATH}", file=sys.stderr)
    sys.exit(0)

conn = sqlite3.connect(DB_PATH)

# ── item_citation_status ──────────────────────────────────────────────────────
total_items = conn.execute("SELECT COUNT(*) FROM item_citation_status").fetchone()[0]
statuses = conn.execute(
    "SELECT s2_status, COUNT(*) AS n FROM item_citation_status "
    "GROUP BY s2_status ORDER BY n DESC"
).fetchall()

# ── global_citations（被引用データ）────────────────────────────────────────────
cit_items   = conn.execute(
    "SELECT COUNT(DISTINCT cited_item_key) FROM global_citations").fetchone()[0]
cit_papers  = conn.execute(
    "SELECT COUNT(DISTINCT citing_paper_id) FROM global_citations").fetchone()[0]
cit_rows    = conn.execute("SELECT COUNT(*) FROM global_citations").fetchone()[0]

# ── global_references（参照データ）─────────────────────────────────────────────
ref_items   = conn.execute(
    "SELECT COUNT(DISTINCT citing_item_key) FROM global_references").fetchone()[0]
ref_papers  = conn.execute(
    "SELECT COUNT(DISTINCT cited_paper_id) FROM global_references").fetchone()[0]
ref_rows    = conn.execute("SELECT COUNT(*) FROM global_references").fetchone()[0]

# ── 両方あるアイテム ─────────────────────────────────────────────────────────────
both_items  = conn.execute("""
    SELECT COUNT(*) FROM (
        SELECT cited_item_key AS k FROM global_citations
        INTERSECT
        SELECT citing_item_key AS k FROM global_references
    )
""").fetchone()[0]

only_cit    = cit_items - both_items
only_ref    = ref_items - both_items
no_data     = conn.execute("""
    SELECT COUNT(*) FROM item_citation_status
    WHERE item_key NOT IN (
        SELECT cited_item_key FROM global_citations
        UNION
        SELECT citing_item_key FROM global_references
    )
""").fetchone()[0]

# ── チャンクマッチステータス内訳 ──────────────────────────────────────────────
cit_status = conn.execute("""
    SELECT chunk_status, COUNT(*) AS n FROM global_citations
    GROUP BY chunk_status ORDER BY n DESC
""").fetchall()

ref_status = conn.execute("""
    SELECT s2_status, COUNT(*) AS n FROM global_references
    GROUP BY s2_status ORDER BY n DESC
""").fetchall()

conn.close()

# ── 表示 ──────────────────────────────────────────────────────────────────────
SEP = "─" * 44
print()
print("╔" + "═" * 44 + "╗")
print("║         Citation DB サマリー            ║")
print("╚" + "═" * 44 + "╝")
print()
print(f"  登録アイテム総数     : {total_items:>6,} 件")
print(f"  {SEP}")
for s2_status, n in statuses:
    label = {
        "mapped":  "S2 マッピング済み",
        "failed":  "S2 取得失敗",
        "pending": "未処理",
        "skipped": "スキップ",
    }.get(s2_status or "", s2_status or "(不明)")
    pct = n / total_items * 100 if total_items else 0
    print(f"    {label:<22}: {n:>6,} 件  ({pct:.1f}%)")
print()
_status_label = {
    "matched":    "チャンク紐付け済み",
    "no_chunk":   "コンテキストあり・未紐付け（AI候補）",
    "no_context": "コンテキストなし（AI候補）",
    "ai_pending": "AI解析待ち",
    "ai_done":    "AI解析完了",
}
print(f"  ── 被引用データ（citations）──────────────")
print(f"    データあり Zotero アイテム: {cit_items:>5,} 件")
print(f"    ユニーク外部論文（被引用元): {cit_papers:>5,} 件")
print(f"    総レコード数              : {cit_rows:>6,} 件")
for st, n in cit_status:
    label = _status_label.get(st or "", st or "(不明)")
    pct = n / cit_rows * 100 if cit_rows else 0
    print(f"      {label:<30}: {n:>6,} 件  ({pct:.1f}%)")
print()
print(f"  ── 参照データ（references）───────────────")
print(f"    データあり Zotero アイテム: {ref_items:>5,} 件")
print(f"    ユニーク外部論文（参照先） : {ref_papers:>5,} 件")
print(f"    総レコード数              : {ref_rows:>6,} 件")
for st, n in ref_status:
    label = _status_label.get(st or "", st or "(不明)")
    pct = n / ref_rows * 100 if ref_rows else 0
    print(f"      {label:<30}: {n:>6,} 件  ({pct:.1f}%)")
print()
print(f"  ── アイテム別内訳 ────────────────────────")
print(f"    被引用＋参照 両方あり     : {both_items:>5,} 件")
print(f"    被引用データのみ          : {only_cit:>5,} 件")
print(f"    参照データのみ            : {only_ref:>5,} 件")
print(f"    データなし                : {no_data:>5,} 件")
print()
