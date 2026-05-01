#!/usr/bin/env python3
"""
PDFアウトラインのうち「意味のある章構造」を持つものを判別するスクリプト。

判定ロジック:
- アウトラインエントリのタイトルが .pdf / .jpg / .png などファイル名パターンに見える
  → 「ページリスト型」（スキャンPDF結合）と判定
- タイトルが自然言語の章・節名（"Chapter", "第X章", "Introduction" など）
  → 「意味的な章構造」と判定
"""

import json
import os
import re
import sys
from pathlib import Path

try:
    import fitz
except ImportError:
    print("ERROR: PyMuPDF がインストールされていません。")
    sys.exit(1)

MANIFEST_PATH = Path(__file__).parent.parent / "data" / "manifest.json"

# ファイル名パターン（スキャンPDF結合の目次エントリによく見られる）
RE_FILENAME_LIKE = re.compile(
    r"""
    (   \.(pdf|jpg|jpeg|png|tif|tiff|bmp|gif)$   # 拡張子で終わる
    |   ^[a-zA-Z0-9_\-]+_page_\d+                 # xxx_page_001 パターン
    |   ^page\s*\d+$                              # "page 1" のみ
    |   ^\d+[LR]?$                                # "001R" のみ（スキャン左右ページ）
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# 章・節らしさを示すパターン（これがあれば加点）
RE_CHAPTER_LIKE = re.compile(
    r"""
    (   \bchapter\b | \bsection\b | \bpart\b | \bintroduction\b
    |   \bconclusion\b | \bappendix\b | \bpreface\b | \bforeword\b
    |   \bcontents\b | \bbibliography\b | \breference\b | \bindex\b
    |   第[0-9０-９一二三四五六七八九十百]+[章節部編]
    |   [0-9]+\.\s+[^\d]        # "1. Introduction" スタイル
    |   [0-9]+\.[0-9]+\s+       # "1.1 Methods" スタイル
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def classify_toc(toc: list) -> str:
    """
    アウトラインを分類する。
    Returns:
        "meaningful"  - 意味のある章構造
        "page_list"   - スキャンPDFのページリスト
        "ambiguous"   - 判別困難
    """
    if not toc:
        return "none"

    titles = [entry[1] for entry in toc if len(entry) >= 2]
    if not titles:
        return "none"

    filename_count = sum(1 for t in titles if RE_FILENAME_LIKE.search(t))
    chapter_count = sum(1 for t in titles if RE_CHAPTER_LIKE.search(t))

    filename_ratio = filename_count / len(titles)
    chapter_ratio = chapter_count / len(titles)

    if filename_ratio >= 0.5:
        return "page_list"
    elif chapter_ratio >= 0.15 or (chapter_count >= 3 and filename_ratio < 0.2):
        return "meaningful"
    else:
        return "ambiguous"


def check_outlines_quality(manifest_path: Path):
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    files = manifest.get("files", {})

    results = {
        "meaningful": [],
        "ambiguous": [],
        "page_list": [],
        "no_outline": [],
        "missing": [],
    }

    for key, entry in files.items():
        pdf_path = entry.get("pdf_path", "")
        if not pdf_path.lower().endswith(".pdf"):
            continue

        size_mb = entry.get("size", 0) / (1024 * 1024)
        name = Path(pdf_path).name

        if not os.path.exists(pdf_path):
            results["missing"].append({"key": key, "name": name})
            continue

        try:
            doc = fitz.open(pdf_path)
            toc = doc.get_toc()
            doc.close()
        except Exception as e:
            results["no_outline"].append({"key": key, "name": name, "note": f"ERROR: {e}"})
            continue

        category = classify_toc(toc)
        if category == "none":
            category = "no_outline"
        info = {
            "key": key,
            "name": name[:70],
            "size_mb": round(size_mb, 1),
            "toc_count": len(toc),
            "toc_preview": [e[1][:60] for e in toc[:4]] if toc else [],
        }
        results[category].append(info)

    # ===== 表示 =====
    total_pdf = sum(len(v) for v in results.values())
    meaningful = len(results["meaningful"])
    ambiguous = len(results["ambiguous"])
    page_list = len(results["page_list"])
    no_outline = len(results["no_outline"])
    missing = len(results["missing"])

    print("=" * 65)
    print("📊 PDFアウトライン 品質分析")
    print("=" * 65)
    print(f"PDF総数:                   {total_pdf}")
    print(f"  ✅ 意味的な章構造あり:    {meaningful}  ({meaningful/total_pdf*100:.1f}%)")
    print(f"  🤔 判別困難 (ambiguous):  {ambiguous}  ({ambiguous/total_pdf*100:.1f}%)")
    print(f"  📄 ページリスト型:        {page_list}  ({page_list/total_pdf*100:.1f}%)")
    print(f"  ❌ アウトラインなし:      {no_outline}  ({no_outline/total_pdf*100:.1f}%)")
    print(f"  ⚠️  ファイル未存在:        {missing}")
    print()

    print(f"✅ 意味的な章構造あり ({meaningful}件) ——")
    for r in sorted(results["meaningful"], key=lambda x: -x["toc_count"]):
        print(f"  [{r['key']}] {r['name']}  ({r['toc_count']}エントリ, {r['size_mb']}MB)")
        for t in r["toc_preview"]:
            print(f"    - {t}")
    print()

    print(f"🤔 判別困難 ({ambiguous}件) ——")
    for r in results["ambiguous"]:
        print(f"  [{r['key']}] {r['name']}  ({r['toc_count']}エントリ)")
        for t in r["toc_preview"]:
            print(f"    - {t}")
    print()

    print(f"📄 ページリスト型 ({page_list}件) ——")
    for r in results["page_list"]:
        print(f"  [{r['key']}] {r['name']}  ({r['toc_count']}エントリ)")

if __name__ == "__main__":
    check_outlines_quality(MANIFEST_PATH)
