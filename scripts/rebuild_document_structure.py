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

from src.chunk_store import active_collection_name, get_item_chunks, list_item_keys
from src.db_relations import (
    get_document_structure, get_item_processing_status,
    mark_artifact_status,
    replace_document_structure,
)
from src.document_structure import STRUCTURE_VERSION, build_document_structure
from src.orphan_cleanup import note_only_item


def _resync_chunk_metadata(item_key: str, collection_name: str | None) -> int:
    """Copy the new leaf identity and policies back onto the item's chunks."""
    import chromadb

    from src.db_relations import get_db_connection
    from src.structure_metadata_sync import desired_chunk_metadata, stale_chunk_updates

    connection = get_db_connection()
    nodes = [dict(row) for row in connection.execute(
        "SELECT node_id, zone, summary_policy, retrieval_policy, citation_policy "
        "FROM document_nodes WHERE item_key = ?", (item_key,))]
    mapping: dict[str, list[str]] = {}
    for node_id, chunk_id in connection.execute(
        "SELECT c.node_id, c.chunk_id FROM document_node_chunks c "
        "JOIN document_nodes n ON n.node_id = c.node_id WHERE n.item_key = ?", (item_key,)
    ):
        mapping.setdefault(str(node_id), []).append(str(chunk_id))
    desired = desired_chunk_metadata(nodes, mapping)
    if not desired:
        return 0
    name = collection_name or active_collection_name()
    collection = chromadb.PersistentClient(path=str(ROOT / "data" / "chroma")).get_collection(name)
    ids = list(desired)
    current: dict[str, dict] = {}
    for start in range(0, len(ids), 500):
        got = collection.get(ids=ids[start:start + 500], include=["metadatas"])
        for chunk_id, metadata in zip(got.get("ids") or [], got.get("metadatas") or []):
            current[str(chunk_id)] = dict(metadata or {})
    updates = stale_chunk_updates(current, desired)
    keys = list(updates)
    for start in range(0, len(keys), 500):
        batch = keys[start:start + 500]
        collection.update(ids=batch, metadatas=[updates[key] for key in batch])
    return len(updates)


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
        # Node ids are derived from content, so a rebuild renames them, and
        # retrieval filters candidate chunks on the copy of node_id held in
        # Chroma -- not on the database. Leaving that copy behind made the leaf
        # route, the highest-weighted of the three fused routes, match nothing
        # and return silently empty: 213,748 chunks (44.7%) were in that state
        # before this call existed (2026-07-28). The structure and the chunks
        # that point at it have to be written in the same breath.
        resynced = _resync_chunk_metadata(item_key, collection_name) if not dry_run else 0
        result["chunk_metadata_resynced"] = resynced
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
