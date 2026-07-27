#!/usr/bin/env python3
"""Build or refresh canonical document structures from locally indexed chunks."""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chunk_store import get_item_chunks, list_item_keys
from src.db_relations import (
    get_document_structure, get_item_processing_status,
    mark_artifact_status,
    replace_document_structure,
)
from src.document_structure import STRUCTURE_VERSION, build_document_structure
from src.orphan_cleanup import note_only_item


def rebuild_item(
    item_key: str, *, dry_run: bool, force: bool, run_id: str,
    collection_name: str | None = None,
) -> dict:
    all_chunks = get_item_chunks(item_key, collection_name=collection_name)
    chunks = [
        row for row in all_chunks
        if str((row.get("metadata") or {}).get("source_type") or "") != "note"
    ]
    if not chunks:
        # Notes are searchable annotations but are deliberately outside the
        # canonical document tree, so an item whose only content is notes has
        # no structure to build *by design*. Recording that as ``blocked``
        # made it a permanent entry in the unresolved list, which the
        # maintenance summary then reports forever (FSIXT5VE, 2026-07-27).
        # ``excluded`` says the same thing without implying something is stuck.
        if note_only_item(all_chunks):
            if not dry_run:
                mark_artifact_status(
                    item_key, "structure", "excluded", reason_code="note_only_item",
                    message="Item has only Zotero notes, which are excluded from the "
                            "canonical document tree by design.",
                    run_id=run_id,
                )
            return {"item_key": item_key, "status": "excluded", "reason_code": "note_only_item"}
        if not dry_run:
            mark_artifact_status(
                item_key, "structure", "blocked", reason_code="no_chunks",
                message="No locally indexed source chunks are available.", run_id=run_id,
            )
        return {"item_key": item_key, "status": "blocked", "reason_code": "no_chunks"}

    built = build_document_structure(item_key, chunks)
    previous = get_document_structure(item_key)
    unchanged = bool(
        previous
        and previous.get("source_fingerprint") == built["source_fingerprint"]
        and previous.get("structure_version") == STRUCTURE_VERSION
    )
    result = {
        "item_key": item_key, "status": built["status"], "confidence": built["confidence"],
        "node_count": len(built["nodes"]), "diagnostics": built["diagnostics"],
        "changed": not unchanged,
    }
    if dry_run or (unchanged and not force):
        result["action"] = "dry_run" if dry_run else "skipped_unchanged"
        return result

    mark_artifact_status(
        item_key, "structure", "running", source_fingerprint=built["source_fingerprint"],
        processor_version=STRUCTURE_VERSION, run_id=run_id,
    )
    try:
        replace_document_structure(
            item_key, source_fingerprint=built["source_fingerprint"],
            structure_version=STRUCTURE_VERSION, status=built["status"],
            confidence=built["confidence"], nodes=built["nodes"], diagnostics=built["diagnostics"],
        )
        artifact_status = "success" if built["status"] in {"exact", "recovered"} else "degraded"
        mark_artifact_status(
            item_key, "structure", artifact_status,
            reason_code="flat_fallback" if built["status"] == "flat_fallback" else None,
            source_fingerprint=built["source_fingerprint"], processor_version=STRUCTURE_VERSION,
            counts={"nodes": len(built["nodes"]), "leaves": built["diagnostics"]["leaf_count"]},
            fallback_kind="contiguous_semantic_segments" if built["status"] == "flat_fallback" else None,
            run_id=run_id,
        )
        if not unchanged:
            for artifact_type in ("summary", "embeddings"):
                mark_artifact_status(
                    item_key, artifact_type, "stale", reason_code="structure_changed",
                    source_fingerprint=built["source_fingerprint"], run_id=run_id,
                )
        result["action"] = "rebuilt"
        return result
    except Exception as exc:
        mark_artifact_status(
            item_key, "structure", "failed", reason_code="structure_build_failed",
            message=str(exc)[:1000], retryable=True, run_id=run_id,
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--item", action="append", help="Zotero parent item key; repeatable")
    selector.add_argument("--all", action="store_true", help="Process every item with indexed chunks")
    parser.add_argument("--dry-run", action="store_true", help="Build and validate without writing derived data")
    parser.add_argument("--force", action="store_true", help="Rebuild even when source fingerprint is unchanged")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of selected items (0 = all)")
    parser.add_argument("--retry-failed", action="store_true", help="Select retryable failed structure items only")
    parser.add_argument(
        "--collection", help="Source Chroma collection; use zotero_paragraphs_v3 during parallel migration.",
    )
    args = parser.parse_args()
    keys = list(dict.fromkeys(args.item or list_item_keys(collection_name=args.collection))) if args.all or args.item else []
    if args.retry_failed:
        keys = [
            key for key in keys
            if any(
                row.get("artifact_type") == "structure"
                and row.get("status") == "failed" and bool(row.get("retryable"))
                for row in get_item_processing_status(key)
            )
        ]
    if args.limit > 0:
        keys = keys[: args.limit]
    run_id = f"structure-v3-{uuid.uuid4().hex[:12]}"
    results = []
    failed = 0
    for item_key in keys:
        try:
            results.append(rebuild_item(
                item_key, dry_run=args.dry_run, force=args.force, run_id=run_id,
                collection_name=args.collection,
            ))
        except Exception as exc:  # continue so one bad document does not stop maintenance
            failed += 1
            results.append({"item_key": item_key, "status": "failed", "error": str(exc)})
    print(json.dumps({"run_id": run_id, "dry_run": args.dry_run, "failed": failed, "items": results}, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
