#!/usr/bin/env python3
"""Compare legacy EPUB DOM reference extraction vs V3 chunk-based extraction (R6).

Read-only validation gate before atticizing ``epub_reference_extractor``.  For
each item it reports the reference-candidate count from both paths and their
normalized-text overlap, so the double-parse can be retired only once the
chunk-based path is shown to match.  No S2 calls, no DB writes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from src.env_utils import load_dotenv_native
load_dotenv_native(ROOT)

from src.chunk_reference_extractor import extract_references_from_chunks
from src.epub_reference_extractor import extract_epub_reference_candidates
from src.update_citations import _unwrap_item, _zotero_request, resolve_epub_path


def _normalize(text: str) -> str:
    return " ".join(str(text or "").casefold().split())


def _resolve_epub_path(item_key: str, zotero_data_dir: str) -> str | None:
    children = _zotero_request(f"items/{item_key}/children")
    if not isinstance(children, list):
        return None
    for child in children:
        _, att_data = _unwrap_item(child)
        if str(att_data.get("filename", "")).lower().endswith(".epub"):
            path = resolve_epub_path(att_data, zotero_data_dir)
            return path if path and os.path.exists(path) else None
    return None


def compare_item(item_key: str, zotero_data_dir: str, collection: str | None) -> dict:
    new_refs = extract_references_from_chunks(item_key, collection_name=collection)
    new_texts = {_normalize(r["raw_reference_text"]) for r in new_refs}
    result: dict = {
        "item_key": item_key,
        "new_count": len(new_refs),
        "new_bibliography_only": True,
    }
    epub_path = _resolve_epub_path(item_key, zotero_data_dir)
    if not epub_path:
        result["old_status"] = "epub_not_found"
        return result
    old_cands = extract_epub_reference_candidates(epub_path)
    old_bib = [c for c in old_cands if c.get("source_zone") == "bibliography"]
    old_bib_texts = {_normalize(c["raw_reference_text"]) for c in old_bib}
    overlap = new_texts & old_bib_texts
    union = new_texts | old_bib_texts
    result.update({
        "old_status": "ok",
        "old_total_count": len(old_cands),
        "old_bibliography_count": len(old_bib),
        "old_zones": sorted({c.get("source_zone") for c in old_cands}),
        "overlap_bibliography": len(overlap),
        "jaccard_bibliography": round(len(overlap) / len(union), 4) if union else 1.0,
        "only_in_new": sorted(new_texts - old_bib_texts)[:5],
        "only_in_old_bibliography": sorted(old_bib_texts - new_texts)[:5],
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item", action="append", required=True, help="Item key; repeatable")
    parser.add_argument("--collection", default=None, help="Source collection (default: active)")
    args = parser.parse_args()
    zotero_data_dir = os.environ.get("ZOTERO_DATA_DIR", os.path.expanduser("~/Zotero"))
    report = [compare_item(key, zotero_data_dir, args.collection) for key in args.item]
    print(json.dumps({"items": report, "canonical_data_modified": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
