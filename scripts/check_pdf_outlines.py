#!/usr/bin/env python3
"""
Zoteroライブラリ内のPDFのうち、PDFアウトライン（目次）を持つものを調査するスクリプト。
manifest_v3.json を読み込み、各PDFに対して fitz.get_toc() を実行する。
"""

import json
import os
import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF がインストールされていません。`pip install pymupdf` を実行してください。")
    sys.exit(1)

MANIFEST_PATH = Path(__file__).parent.parent / "data" / "manifest_v3.json"

def check_outlines(manifest_path: Path):
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    files = manifest.get("files", {})

    total_pdf = 0
    has_outline = 0
    no_outline = 0
    error_count = 0
    missing_count = 0

    results_with = []
    results_without = []

    for key, entry in files.items():
        pdf_path = entry.get("pdf_path", "")
        if not pdf_path.lower().endswith(".pdf"):
            continue  # epub/html はスキップ

        total_pdf += 1
        size_mb = entry.get("size", 0) / (1024 * 1024)

        if not os.path.exists(pdf_path):
            missing_count += 1
            results_without.append({
                "key": key,
                "path": pdf_path,
                "status": "FILE_NOT_FOUND",
                "toc_entries": 0,
                "size_mb": round(size_mb, 2),
            })
            continue

        try:
            doc = fitz.open(pdf_path)
            toc = doc.get_toc()
            doc.close()

            name = Path(pdf_path).name
            entry_result = {
                "key": key,
                "name": name,
                "path": pdf_path,
                "toc_entries": len(toc),
                "size_mb": round(size_mb, 2),
                "toc_preview": toc[:5] if toc else [],
            }

            if toc:
                has_outline += 1
                results_with.append(entry_result)
            else:
                no_outline += 1
                results_without.append({**entry_result, "status": "NO_OUTLINE"})

        except Exception as e:
            error_count += 1
            results_without.append({
                "key": key,
                "path": pdf_path,
                "status": f"ERROR: {e}",
                "toc_entries": 0,
                "size_mb": round(size_mb, 2),
            })

    # ===== 結果表示 =====
    print("=" * 60)
    print("📊 PDFアウトライン調査結果")
    print("=" * 60)
    print(f"PDF総数:            {total_pdf}")
    print(f"アウトラインあり:    {has_outline}  ({has_outline/total_pdf*100:.1f}%)" if total_pdf else "")
    print(f"アウトラインなし:    {no_outline}  ({no_outline/total_pdf*100:.1f}%)" if total_pdf else "")
    print(f"ファイル未存在:      {missing_count}")
    print(f"エラー:              {error_count}")
    print()

    if results_with:
        print(f"✅ アウトラインあり ({len(results_with)}件):")
        for r in sorted(results_with, key=lambda x: -x["toc_entries"]):
            print(f"  [{r['key']}] {r['name'][:60]}")
            print(f"    → エントリ数: {r['toc_entries']}, サイズ: {r['size_mb']:.1f} MB")
            if r["toc_preview"]:
                for entry in r["toc_preview"]:
                    indent = "  " * entry[0]
                    print(f"      {indent}L{entry[0]}: {entry[1][:50]} (p.{entry[2]})")
            print()

    if results_without:
        print(f"❌ アウトラインなし / エラー ({len(results_without)}件):")
        for r in results_without:
            status = r.get("status", "NO_OUTLINE")
            name = Path(r["path"]).name
            print(f"  [{r['key']}] {name[:60]} ({status})")

if __name__ == "__main__":
    check_outlines(MANIFEST_PATH)
