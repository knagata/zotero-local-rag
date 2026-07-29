"""Per-item adoption of a prepared re-OCR result into the isolated V3 index.

The expensive OCR step happens before this module is called.  Adoption is
deliberately narrow: exactly one attachment is replaced, and the previous
Chroma/FTS rows are retained in memory until every canonical write succeeds.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .db_relations import (
    delete_document_node_summaries,
    mark_artifact_status,
    replace_document_structure,
)
from .document_structure import attach_structure_metadata, build_document_structure
from .lexical_index import delete_by_attachment_keys, upsert_chunks
from .manifest import load_manifest, save_manifest
from .text_utils import detect_lang


def canonicalize_prepared_blocks(
    item_key: str,
    attachment_key: str,
    prepared: Mapping[str, Any],
    *,
    inherited_metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Normalize staged blocks and assign new, content-versioned chunk IDs."""
    blocks = prepared.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("prepared result must contain at least one block")
    engine = str(prepared.get("engine") or "unknown")
    version = str(prepared.get("version") or "unknown")
    digest = hashlib.sha256()
    digest.update(f"{item_key}\0{attachment_key}\0{engine}\0{version}".encode())
    for block in blocks:
        if not isinstance(block, Mapping):
            raise ValueError("prepared blocks must be objects")
        digest.update(str(block.get("text") or "").encode("utf-8"))
        digest.update(b"\0")
    revision = digest.hexdigest()[:12]

    output: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    base = dict(inherited_metadata or {})
    for ordinal, block in enumerate(blocks):
        text = str(block.get("text") or "").strip()
        if not text:
            continue
        metadata = dict(base)
        candidate_metadata = block.get("metadata")
        if isinstance(candidate_metadata, Mapping):
            metadata.update(candidate_metadata)
        metadata.update({
            "itemKey": item_key,
            "attachmentKey": attachment_key,
            "source_type": metadata.get("source_type") or "pdf",
            "extraction_engine": engine,
            "extraction_version": version,
            "lang": detect_lang(text, metadata.get("lang")),
        })
        chunk_id = f"{attachment_key}:v3ocr:{revision}:{ordinal:06d}"
        if chunk_id in seen_ids:
            raise ValueError(f"duplicate canonical chunk id: {chunk_id}")
        seen_ids.add(chunk_id)
        output.append({"id": chunk_id, "text": text, "metadata": metadata})
    if not output:
        raise ValueError("prepared result contains no non-empty text blocks")
    return output


def adopt_prepared_reocr(
    *,
    item_key: str,
    attachment_key: str,
    prepared: Mapping[str, Any],
    collection: Any,
    old_item_chunks: Sequence[Mapping[str, Any]],
    manifest_path: Path,
    lexical_path: Path,
    force: bool = False,
    gate_passed: bool = True,
    status_writer: Callable[..., Any] = mark_artifact_status,
) -> dict[str, Any]:
    """Replace one attachment and rebuild its item's canonical V3 structure.

    Chroma and FTS are compensated from the in-memory snapshot if a later write
    fails.  The manifest uses its atomic temporary-file replacement.  Relational
    structure is written only after both retrieval indexes and manifest succeed.
    """
    if not item_key or not attachment_key:
        raise ValueError("item_key and attachment_key are required")
    if not gate_passed and not force:
        raise ValueError("quality gate did not pass")

    old_rows = [
        {"id": str(row.get("id") or ""), "text": str(row.get("text") or ""),
         "metadata": dict(row.get("metadata") or {})}
        for row in old_item_chunks
    ]
    old_attachment = [
        row for row in old_rows
        if str(row["metadata"].get("attachmentKey") or "") == attachment_key
    ]
    if not old_attachment:
        raise ValueError("canonical attachment has no existing V3 chunks")
    inherited = old_attachment[0]["metadata"]
    new_attachment = canonicalize_prepared_blocks(
        item_key, attachment_key, prepared, inherited_metadata=inherited,
    )
    combined = [
        row for row in old_rows
        if str(row["metadata"].get("attachmentKey") or "") != attachment_key
    ] + new_attachment
    built = build_document_structure(item_key, combined)
    annotated = attach_structure_metadata(combined, built["nodes"])
    new_attachment = [
        row for row in annotated
        if str(row["metadata"].get("attachmentKey") or "") == attachment_key
    ]
    new_ids = [row["id"] for row in new_attachment]
    new_docs = [row["text"] for row in new_attachment]
    new_metas = [row["metadata"] for row in new_attachment]
    old_ids = [row["id"] for row in old_attachment]
    old_docs = [row["text"] for row in old_attachment]
    old_metas = [row["metadata"] for row in old_attachment]

    manifest_before = load_manifest(manifest_path)
    manifest_after = dict(manifest_before)
    manifest_after["files"] = dict(manifest_before.get("files") or {})
    previous_entry = dict(manifest_after["files"].get(attachment_key) or {})
    quality = dict(prepared.get("quality") or {})
    quality.update({
        "parser": str(prepared.get("engine") or "unknown"),
        "parser_version": str(prepared.get("version") or "unknown"),
        "reocr_adopted": True,
        "force_adopted": bool(force),
        "structure_v3": {
            "status": built["status"],
            "version": built["structure_version"],
            "nodes": len(built["nodes"]),
            "leaves": built["diagnostics"].get("leaf_count", 0),
            "zone_counts": built["diagnostics"].get("zone_counts", {}),
        },
    })
    previous_entry.update({"quality": quality, "source_fingerprint": built["source_fingerprint"]})
    manifest_after["files"][attachment_key] = previous_entry

    status_writer(item_key, "extraction", "running", attachment_key=attachment_key,
                  processor_version=str(prepared.get("version") or "unknown"))
    collection_changed = False
    lexical_changed = False
    manifest_changed = False
    try:
        collection.delete(where={"attachmentKey": attachment_key})
        collection.upsert(ids=new_ids, documents=new_docs, metadatas=new_metas)
        collection_changed = True
        delete_by_attachment_keys([attachment_key], path=lexical_path)
        upsert_chunks(new_ids, new_docs, new_metas, path=lexical_path)
        lexical_changed = True
        save_manifest(manifest_path, manifest_after)
        manifest_changed = True

        # Replacing the tree cascades removal of obsolete node summaries.
        replace_document_structure(
            item_key,
            source_fingerprint=built["source_fingerprint"],
            structure_version=built["structure_version"],
            status=built["status"],
            nodes=built["nodes"],
            diagnostics=built["diagnostics"],
            confidence=built["confidence"],
        )
        delete_document_node_summaries(item_key)
        status_writer(
            item_key, "extraction", "success", attachment_key=attachment_key,
            source_fingerprint=built["source_fingerprint"],
            processor_version=f"{prepared.get('engine') or 'unknown'}:{prepared.get('version') or 'unknown'}",
            counts={"chunks": len(new_ids), "reocr": True, "force_adopt": bool(force)},
        )
        status_writer(item_key, "structure", "success", source_fingerprint=built["source_fingerprint"],
                      processor_version=built["structure_version"], counts=built["diagnostics"])
        status_writer(item_key, "embeddings", "success", attachment_key=attachment_key,
                      source_fingerprint=built["source_fingerprint"], counts={"chunks": len(new_ids)})
        status_writer(item_key, "summary", "stale", reason_code="source_reocr_adopted",
                      source_fingerprint=built["source_fingerprint"])
        status_writer(item_key, "references", "stale", reason_code="source_reocr_adopted",
                      source_fingerprint=built["source_fingerprint"])
    except Exception:
        if collection_changed:
            collection.delete(where={"attachmentKey": attachment_key})
            collection.upsert(ids=old_ids, documents=old_docs, metadatas=old_metas)
        if lexical_changed or collection_changed:
            delete_by_attachment_keys([attachment_key], path=lexical_path)
            upsert_chunks(old_ids, old_docs, old_metas, path=lexical_path)
        if manifest_changed:
            save_manifest(manifest_path, manifest_before)
        status_writer(item_key, "extraction", "failed", attachment_key=attachment_key,
                      reason_code="reocr_adoption_failed", retryable=True)
        raise

    return {
        "item_key": item_key,
        "attachment_key": attachment_key,
        "old_chunk_count": len(old_ids),
        "new_chunk_count": len(new_ids),
        "source_fingerprint": built["source_fingerprint"],
        "structure_status": built["status"],
        "summary_status": "stale",
        "canonical_data_modified": True,
    }


__all__ = ["adopt_prepared_reocr", "canonicalize_prepared_blocks"]
