#!/usr/bin/env python3
"""Check the index against the original files.

Read-only. Run before trusting any claim that the library is intact, and as
the gate before switching a rebuilt collection into service.

Why this exists rather than another index-to-index audit: on 2026-07-28 the
cutover audit returned ``passed: true`` for 521 items while five documents had
lost body text (up to 40 pages each), 1,774 chunks belonged to no item, and one
65,000-character essay had disappeared from search. None of those are bugs in
that audit -- it compares V3 against legacy, the manifest against Chroma, and
an item against itself, and two stores missing the same text agree.

The page check reads the source PDF. The companion checks need only the index
but cover the populations a per-item comparison cannot reach.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_utils import load_dotenv_native  # noqa: E402

load_dotenv_native(ROOT)

from src.chunk_store import active_collection_name  # noqa: E402
from src.db_relations import get_db_connection  # noqa: E402
from src.manifest import load_manifest  # noqa: E402
from src.source_verification import (  # noqa: E402
    chunks_without_item, compare_document, dangling_node_ids, indexed_page_chars,
    source_page_chars, unretrievable_documents,
)
from src.v3_data_plane import V3_COLLECTION, chroma_dir, manifest_path  # noqa: E402

# .env は MANIFEST_PATH / LEXICAL_DB_PATH を *相対パス* で持つ。素の Path() で
# 受けるとCWD基準になり、プロジェクトルート以外から起動したときに存在しない
# ファイルを見にいく（2026-08-04に実測）。解決規則は
# resolve_configured_path が唯一の正典。
MANIFEST = manifest_path(ROOT)
CHROMA_DIR = chroma_dir(ROOT)


def _collection_rows(collection_name: str, *, chroma_dir: Path = CHROMA_DIR) -> list[dict[str, Any]]:
    """Every chunk in one collection, as {"id", "text", "metadata"}."""
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(collection_name)
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = collection.get(limit=5000, offset=offset, include=["metadatas", "documents"])
        ids = page.get("ids") or []
        if not ids:
            break
        documents = page.get("documents") or []
        metadatas = page.get("metadatas") or []
        for index, chunk_id in enumerate(ids):
            rows.append({
                "id": chunk_id,
                "text": documents[index] if index < len(documents) else "",
                "metadata": metadatas[index] if index < len(metadatas) else {},
            })
        offset += len(ids)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default=None, help="Defaults to the active collection.")
    parser.add_argument(
        "--manifest", type=Path, default=MANIFEST,
        help="V3 manifest to inspect (defaults to MANIFEST_PATH or data/manifest_v3.json).",
    )
    parser.add_argument(
        "--chroma-dir", type=Path, default=CHROMA_DIR,
        help="Chroma persistence directory (defaults to CHROMA_DIR or data/chroma).",
    )
    parser.add_argument("--attachment", action="append", help="Limit the page check; repeatable.")
    parser.add_argument("--limit", type=int, default=0, help="Check at most N attachments (0 = all).")
    parser.add_argument("--skip-pages", action="store_true", help="Companion invariants only.")
    parser.add_argument("--output", type=Path, help="Write the full report as JSON.")
    args = parser.parse_args()

    collection_name = args.collection or active_collection_name(chroma_dir=args.chroma_dir)
    if collection_name != V3_COLLECTION:
        parser.error(f"--collection must be {V3_COLLECTION!r}; the legacy data plane is retired")

    rows = _collection_rows(collection_name, chroma_dir=args.chroma_dir)
    by_attachment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metadata = row.get("metadata") or {}
        attachment = str(metadata.get("attachmentKey") or "")
        item = str(metadata.get("itemKey") or "")
        if attachment:
            by_attachment[attachment].append(row)
        if item:
            by_item[item].append(row)

    connection = get_db_connection()
    live_nodes = [r[0] for r in connection.execute("SELECT node_id FROM document_nodes")]

    report: dict[str, Any] = {
        "collection": collection_name,
        "manifest_path": str(args.manifest.resolve()),
        "chroma_dir": str(args.chroma_dir.resolve()),
        "chunks": len(rows),
        "orphan_chunks": chunks_without_item(rows),
        "dangling_node_chunks": dangling_node_ids(rows, live_nodes),
        "unretrievable_items": unretrievable_documents(by_item),
        "documents": [],
    }

    if not args.skip_pages:
        files = (load_manifest(args.manifest).get("files") or {})
        selected = args.attachment or sorted(files)
        if args.limit > 0:
            selected = selected[: args.limit]
        for attachment_key in selected:
            entry = files.get(attachment_key)
            if not isinstance(entry, dict):
                continue
            path = Path(str(entry.get("pdf_path") or ""))
            if path.suffix.lower() != ".pdf" or not path.exists():
                continue
            try:
                source = source_page_chars(path)
            except Exception as exc:  # noqa: BLE001 - an unreadable source is a finding
                report["documents"].append({
                    "attachment_key": attachment_key, "error": f"{type(exc).__name__}: {exc}",
                })
                continue
            verdict = compare_document(
                attachment_key, source, indexed_page_chars(by_attachment.get(attachment_key, [])),
            )
            if verdict.failed or verdict.unreadable_pages:
                report["documents"].append({
                    "attachment_key": attachment_key,
                    "title": entry.get("title"),
                    "source_pages": verdict.source_pages,
                    "lost_pages": verdict.lost_pages,
                    "unreadable_pages": len(verdict.unreadable_pages),
                    "source_chars": verdict.source_chars,
                    "indexed_chars": verdict.indexed_chars,
                })

    lost = [d for d in report["documents"] if d.get("lost_pages")]
    # A source PDF this script could not read (moved, corrupted, permission
    # error) means the one invariant this script exists to check was never
    # actually verified for that document -- that must fail the audit, not
    # be recorded as a footnote excluded from the pass/fail computation
    # (found in code review, fixed 2026-07-30).
    unreadable = [d for d in report["documents"] if d.get("error")]
    report["summary"] = {
        "documents_with_lost_pages": len(lost),
        "lost_pages_total": sum(len(d["lost_pages"]) for d in lost),
        "orphan_chunks": len(report["orphan_chunks"]),
        "dangling_node_chunks": len(report["dangling_node_chunks"]),
        "unretrievable_items": len(report["unretrievable_items"]),
        "unreadable_source_documents": len(unreadable),
    }
    report["passed"] = not any(report["summary"].values())

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], **report["summary"]}, ensure_ascii=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
