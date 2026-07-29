#!/usr/bin/env python3
"""Audit and locally repair V3 derived indexes without re-embedding chunks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import chromadb

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from chunk_store import list_attachment_keys, list_chunk_ids, list_item_keys  # noqa: E402
from db_relations import get_document_structure  # noqa: E402
from document_structure import STRUCTURE_VERSION, source_fingerprint  # noqa: E402
from index_from_zotero import _flush_and_verify_hnsw  # noqa: E402
from lexical_index import (  # noqa: E402
    delete_by_chunk_ids, list_chunk_ids as list_lexical_chunk_ids, upsert_chunks,
)
from manifest import load_manifest, save_manifest  # noqa: E402
from scripts.audit_v3_cutover import source_rows  # noqa: E402
from scripts.rebuild_document_structure import rebuild_item  # noqa: E402
from chunk_store import get_item_chunks  # noqa: E402


SENTINEL_PREFIX = "__hnsw_flush_sentinel_"


def _batches(values: Iterable[str], size: int = 1000) -> Iterable[list[str]]:
    batch: list[str] = []
    for value in values:
        batch.append(value)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _repair_lexical(collection: Any, *, missing: set[str], extra: set[str], lexical_path: Path) -> None:
    delete_by_chunk_ids(extra, path=lexical_path)
    for ids in _batches(sorted(missing)):
        result = collection.get(ids=ids, include=["documents", "metadatas"])
        found_ids = [str(value) for value in (result.get("ids") or [])]
        if set(found_ids) != set(ids):
            raise RuntimeError("Chroma rows changed during lexical repair")
        upsert_chunks(
            found_ids,
            list(result.get("documents") or []),
            list(result.get("metadatas") or []),
            path=lexical_path,
        )


def _stale_structure_items(collection_name: str) -> list[str]:
    stale = []
    for item_key in list_item_keys(collection_name=collection_name):
        rows = source_rows(get_item_chunks(item_key, collection_name=collection_name))
        if not rows:
            continue
        structure = get_document_structure(item_key)
        if (
            not structure
            or str(structure.get("structure_version") or "") != STRUCTURE_VERSION
            or str(structure.get("source_fingerprint") or "") != source_fingerprint(list(rows))
        ):
            stale.append(item_key)
    return stale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply safe derived-index repairs")
    parser.add_argument("--repair-structures", action="store_true", help="Also rebuild stale structures")
    parser.add_argument("--collection", default="zotero_paragraphs_v3")
    parser.add_argument("--chroma-dir", type=Path, default=ROOT / "data" / "chroma")
    parser.add_argument("--lexical-db", type=Path, default=ROOT / "data" / "lexical_v3.sqlite3")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest_v3.json")
    args = parser.parse_args()
    if args.collection != "zotero_paragraphs_v3":
        parser.error("--collection must be 'zotero_paragraphs_v3'; the legacy data plane is retired")

    manifest = load_manifest(args.manifest)
    manifest_files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    chroma_ids = set(list_chunk_ids(chroma_dir=args.chroma_dir, collection_name=args.collection))
    lexical_ids = set(list_lexical_chunk_ids(path=args.lexical_db))
    sentinels = {value for value in chroma_ids if value.startswith(SENTINEL_PREFIX)}
    real_chroma_ids = chroma_ids - sentinels
    missing_lexical = real_chroma_ids - lexical_ids
    extra_lexical = lexical_ids - real_chroma_ids
    chroma_attachments = set(list_attachment_keys(
        chroma_dir=args.chroma_dir, collection_name=args.collection,
    ))
    manifest_attachments = set(str(value) for value in manifest_files)
    stale_structures = _stale_structure_items(args.collection) if args.repair_structures else []

    report = {
        "apply": args.apply,
        "chroma_chunks": len(real_chroma_ids),
        "lexical_chunks": len(lexical_ids),
        "stale_sentinels": len(sentinels),
        "missing_lexical": len(missing_lexical),
        "extra_lexical": len(extra_lexical),
        "manifest_only_attachments": sorted(manifest_attachments - chroma_attachments),
        "chroma_only_attachments": sorted(chroma_attachments - manifest_attachments),
        "stale_structures": len(stale_structures),
    }
    print(json.dumps(report, ensure_ascii=False))
    if not args.apply:
        coverage_mismatch = bool(
            report["manifest_only_attachments"] or report["chroma_only_attachments"]
        )
        return 2 if any(
            (sentinels, missing_lexical, extra_lexical, stale_structures, coverage_mismatch)
        ) else 0
    if report["manifest_only_attachments"] or report["chroma_only_attachments"]:
        raise RuntimeError(
            "Manifest/Chroma attachment coverage differs; refusing automatic adoption. "
            "Re-run the listed attachments through the indexer."
        )

    client = chromadb.PersistentClient(path=str(args.chroma_dir))
    collection = client.get_collection(args.collection)
    try:
        if sentinels:
            collection.delete(ids=sorted(sentinels))
            delete_by_chunk_ids(sentinels, path=args.lexical_db)
        _repair_lexical(
            collection, missing=missing_lexical,
            extra=extra_lexical | sentinels, lexical_path=args.lexical_db,
        )
        for item_key in stale_structures:
            rebuild_item(
                item_key, dry_run=False, force=True, run_id="v3-index-repair",
                collection_name=args.collection,
            )
        remaining_ids = set(list_chunk_ids(
            chroma_dir=args.chroma_dir, collection_name=args.collection,
        ))
        remaining_lexical = set(list_lexical_chunk_ids(path=args.lexical_db))
        if remaining_ids != remaining_lexical:
            raise RuntimeError("Chroma/FTS ID sets still differ after repair")
        sample_id = next(iter(remaining_ids), None)
        _flush_and_verify_hnsw(collection, sample_id)
        manifest["hnsw_validated"] = True
        save_manifest(args.manifest, manifest)
    except Exception:
        manifest["hnsw_validated"] = False
        save_manifest(args.manifest, manifest)
        raise
    finally:
        client.close()
    print(json.dumps({"repaired": True, "chunks": len(remaining_ids)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
