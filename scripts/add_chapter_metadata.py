#!/usr/bin/env python3
"""
ChromaDB の既存チャンクに章情報メタデータを後付けするスクリプト。
ベクトルの再計算は行わない（メタデータのみ更新）。

付与するフィールド:
  chapter  : level-1 の章タイトル（例: "Chapter 2", "第3章 ..."）
  section  : level-2 の節タイトル（例: "2.1 Methods"）（PDFのみ・あれば）

対象:
  - PDF : fitz.get_toc() でアウトラインを取得し、ページ番号で章を対応付け
  - EPUB: ebooklib の book.toc でTOCを取得し、chapter_index で章を対応付け
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# src/ を import パスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chromadb

from chapter_detect import (
    build_pdf_page_chapter_lookup,
    get_epub_chapter_index_to_title,
    get_pdf_toc,
)
from v3_data_plane import V3_COLLECTION

MANIFEST_PATH = Path(__file__).parent.parent / "data" / "manifest_v3.json"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma"

UPDATE_BATCH = 500  # ChromaDB への一括 update サイズ


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="既存 ChromaDB チャンクに章情報メタデータ (chapter / section) を付与する。"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="プレビューのみ（ChromaDB への書き込みを行わない）",
    )
    p.add_argument(
        "--key",
        help="特定の attachmentKey のみ処理する（デバッグ用）",
        default=None,
    )
    args = p.parse_args()

    # ── manifest 読み込み ──
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    files: dict = manifest.get("files", {})

    # ── ChromaDB 接続 ──
    collection_name = V3_COLLECTION

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_collection(collection_name)

    print(f"コレクション: {collection_name}  (総チャンク数: {col.count():,})")
    print(f"対象ファイル数: {len(files)}")
    if args.dry_run:
        print("【DRY-RUN モード】ChromaDB への書き込みは行いません")
    print()

    stats = {"pdf_ok": 0, "epub_ok": 0, "no_toc": 0, "missing": 0, "chunks_updated": 0}

    for att_key, entry in files.items():
        if args.key and att_key != args.key:
            continue

        path = entry.get("pdf_path", "")
        if not path:
            continue

        path_lower = path.lower()
        is_pdf = path_lower.endswith(".pdf")
        is_epub = path_lower.endswith(".epub")

        if not (is_pdf or is_epub):
            continue

        if not os.path.exists(path):
            stats["missing"] += 1
            continue

        # ── ChromaDB から既存チャンクを取得 ──
        try:
            results = col.get(
                where={"attachmentKey": att_key},
                include=["metadatas"],
            )
        except Exception as e:
            print(f"  [WARN] {att_key}: ChromaDB get() 失敗 → {e}")
            continue

        ids: list[str] = results["ids"]
        metadatas: list[dict] = results["metadatas"]

        if not ids:
            continue

        fname = Path(path).name

        # ── PDF 処理 ──
        if is_pdf:
            toc = get_pdf_toc(path)
            lookup = build_pdf_page_chapter_lookup(toc) if toc else None

            if lookup is None:
                stats["no_toc"] += 1
                continue

            updated_ids: list[str] = []
            updated_metas: list[dict] = []

            for chunk_id, md in zip(ids, metadatas):
                page = md.get("page")
                if page is None:
                    continue
                chapter, section = lookup(int(page))
                if not chapter:
                    continue
                new_md = dict(md)
                new_md["chapter"] = chapter
                if section:
                    new_md["section"] = section
                updated_ids.append(chunk_id)
                updated_metas.append(new_md)

            if not updated_ids:
                stats["no_toc"] += 1
                continue

            print(f"  [PDF ] {att_key}  {fname[:55]}")
            print(f"         → {len(updated_ids)}/{len(ids)} チャンクに chapter 付与")

            if not args.dry_run:
                for i in range(0, len(updated_ids), UPDATE_BATCH):
                    col.update(
                        ids=updated_ids[i : i + UPDATE_BATCH],
                        metadatas=updated_metas[i : i + UPDATE_BATCH],
                    )

            stats["pdf_ok"] += 1
            stats["chunks_updated"] += len(updated_ids)

        # ── EPUB 処理 ──
        elif is_epub:
            chap_map = get_epub_chapter_index_to_title(path)

            if not chap_map:
                stats["no_toc"] += 1
                continue

            updated_ids = []
            updated_metas = []

            for chunk_id, md in zip(ids, metadatas):
                chap_idx = md.get("chapter_index")
                if chap_idx is None:
                    continue
                title = chap_map.get(int(chap_idx), "")
                if not title:
                    continue
                new_md = dict(md)
                new_md["chapter"] = title
                updated_ids.append(chunk_id)
                updated_metas.append(new_md)

            if not updated_ids:
                stats["no_toc"] += 1
                continue

            print(f"  [EPUB] {att_key}  {fname[:55]}")
            print(f"         → {len(updated_ids)}/{len(ids)} チャンクに chapter 付与")

            if not args.dry_run:
                for i in range(0, len(updated_ids), UPDATE_BATCH):
                    col.update(
                        ids=updated_ids[i : i + UPDATE_BATCH],
                        metadatas=updated_metas[i : i + UPDATE_BATCH],
                    )

            stats["epub_ok"] += 1
            stats["chunks_updated"] += len(updated_ids)

    # ── サマリー ──
    print()
    print("=" * 55)
    print("完了サマリー")
    print("=" * 55)
    print(f"  PDF 処理成功:        {stats['pdf_ok']} ファイル")
    print(f"  EPUB 処理成功:       {stats['epub_ok']} ファイル")
    print(f"  TOCなし/スキップ:    {stats['no_toc']} ファイル")
    print(f"  ファイル未存在:      {stats['missing']} ファイル")
    print(f"  chapter 付与チャンク: {stats['chunks_updated']:,} 件")
    if args.dry_run:
        print()
        print("（DRY-RUN モード: ChromaDB への書き込みは行いませんでした）")


if __name__ == "__main__":
    main()
