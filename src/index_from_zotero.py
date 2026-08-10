#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import atexit
import json
import os
import sys
import time
import uuid
import gc
from dataclasses import asdict, dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

from zotero_source_localapi import ZoteroLocalAPI, ZoteroAttachment

import indexing_lock
from embedder import get_collection
from html_extract import (
    extract_chunks_from_html_snapshot,
    extract_chunks_from_epub_snapshot,
)
from pdf_extract import (
    extract_chunks_from_pdf, recompute_scanned_quality_after_patch,
    recompute_corrupted_quality_after_patch, recompute_blank_pages_after_patch,
)
from docling_worker import DoclingWorker
from granite_worker import GraniteWorker
from ndlocr_extract import extract_chunks_from_pdf_with_ndlocr
from rapidocr_extract import extract_chunks_from_pdf_with_rapidocr
from local_ocr_pipeline import REPEAT_ARTIFACT_RE, run_local_ocr
from mistral_ocr_extract import (
    extract_chunks_from_mistral_ocr_result,
    extract_chunks_from_pdf_with_mistral_ocr,
    has_archived_result,
    mistral_ocr_available,
)
from epub_fallback import (
    add_fixed_layout_terminal_markers, remap_ocr_chunks_to_epub,
    save_fixed_layout_derivative,
)
from extraction_engine import (
    pymupdf_fast_path_passes, pymupdf_fast_path_rejection_reason,
    resolve_extraction_engine, summarize_extraction_quality,
)
from pdf_provenance import (
    BORN_DIGITAL, SCANNED_NO_TEXT, SCANNED_OCR_LAYER, SCAN_DERIVED_CLASSES,
    classify_pdf_source, detect_text_defects,
)
from ocr_layer_audit import (
    DEGRADED as OCR_LAYER_DEGRADED, DISABLED_REASON as OCR_LAYER_AUDIT_DISABLED,
    audit_hit_file_problem, audit_ocr_text_layer, audit_sample_too_small,
    audit_was_transient_failure, replacement_required,
)
from feature_gates import (
    mistral_batch_queue_enabled, ocr_layer_audit_enabled,
    pdf_structure_recovery_enabled, structure_engine_for, verify_enabled_features,
)
from orphan_cleanup import live_item_keys, stale_identity_keys
from pdf_toc_recovery import try_ai_toc_fast_path
from note_extract import index_notes
from text_utils import detect_lang, looks_like_gibberish
from lexical_index import delete_by_attachment_keys as delete_lexical_attachments
from lexical_index import delete_by_chunk_ids as delete_lexical_chunk_ids
from lexical_index import delete_by_note_key as delete_lexical_note
from lexical_index import chunk_ids_by_attachment_keys as lexical_chunk_ids_by_attachment_keys
from lexical_index import upsert_chunks as upsert_lexical_chunks
from index_batch import (
    AttachmentBatch, FlushOutcome, PendingIndexBatch, replace_attachment_batch,
)
from index_run import (
    DiscoveryResult, NoteIndexOutcome, PdfExtraction, PdfGatePlan, QualityWarning,
    ReparseDecision, SourceVerdict,
    ResolvedAttachmentSource,
)

from manifest import content_signature, load_manifest, save_manifest
from db_relations import (
    drop_stale_identity_rows, get_item_processing_status, mark_artifact_status,
    purge_artifact_status_for_attachments, purge_removed_items, replace_document_structure,
    reset_ingestion_derived_state,
)
from document_structure import attach_structure_metadata, build_document_structure
from chunk_store import get_item_chunks, list_chunk_ids, list_item_keys
from v3_data_plane import (
    chroma_dir, lexical_path,
    V3_COLLECTION, collection_name as v3_collection_name,
    enforce_environment as enforce_v3_environment, manifest_path as v3_manifest_path,
)
from v3_runtime import (
    assert_code_unchanged, bind_manifest_pipeline, code_fingerprint,
    ensure_pipeline_config, pipeline_payload,
)
from source_coverage import (
    coverage_from_extraction, coverage_gap_is_adoptable, coverage_shortfall,
    validate_source_coverage,
)



# ----------------------------
# Paths / Env
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]


from env_utils import load_dotenv_native

load_dotenv_native(PROJECT_ROOT)
enforce_v3_environment(PROJECT_ROOT)

CHROMA_COLLECTION_DEFAULT = V3_COLLECTION


@dataclass(frozen=True)
class IngestPaths:
    """Where one run reads and writes: resolved once, then passed around.

    These were module constants, computed at import. That made the process
    environment at import time the only environment the run could have, and it
    is why the ingestion net has to start a child process to point the indexer
    at a temporary data plane -- by the time a test could set ``CHROMA_DIR``,
    this module had already read it. Everything else in the plane
    (``v3_data_plane``, ``chunk_store``) resolves per call and follows an
    environment change; this module was the one that could not.

    Resolved once per run rather than per call, which is the property the
    constants had and worth keeping: a mid-run change to ``CHROMA_DIR`` must
    not leave half the documents in one collection and half in another.
    """

    project_root: Path
    data_dir: Path
    chroma_dir: Path
    pdf_cache_dir: Path
    manifest_path: Path
    indexing_lock_path: Path
    pipeline_config_path: Path
    collection_name: str
    zotero_data_dir: str | None

    @classmethod
    def from_environment(cls, project_root: Path = PROJECT_ROOT) -> "IngestPaths":
        # Asked of v3_data_plane rather than read from the environment here: it
        # owns both the default location and the rule that expands ~ and
        # resolves a relative value against the project root. Copies of that
        # rule have drifted twice, each time leaving one configuration naming
        # two databases.
        data_dir = project_root / "data"
        resolved_chroma = chroma_dir(project_root)
        return cls(
            project_root=project_root,
            data_dir=data_dir,
            chroma_dir=resolved_chroma,
            pdf_cache_dir=Path(
                os.environ.get("PDF_CACHE_DIR", str(data_dir / "pdf_cache"))
            ),
            manifest_path=v3_manifest_path(project_root),
            indexing_lock_path=indexing_lock.default_path(project_root),
            pipeline_config_path=resolved_chroma / "embedder_config_v3.json",
            collection_name=v3_collection_name(),
            # Required for local storage resolution in your pipeline.
            zotero_data_dir=os.environ.get("ZOTERO_DATA_DIR"),
        )


_PATHS: IngestPaths | None = None


def paths() -> IngestPaths:
    """The paths this run is using, resolved on first use.

    Deliberately not resolved at import: importing this module must not decide
    where the run writes, or a caller that sets the environment afterwards
    silently writes somewhere else.
    """
    global _PATHS
    if _PATHS is None:
        _PATHS = IngestPaths.from_environment()
    return _PATHS


@contextmanager
def use_paths(replacement: IngestPaths):
    """Run against a different data plane, in this process.

    The seam the ingestion net was missing. Restores the previous value even
    on failure, so one test cannot leave the next one pointed at a temporary
    directory that has since been deleted.
    """
    global _PATHS
    previous = _PATHS
    _PATHS = replacement
    try:
        yield replacement
    finally:
        _PATHS = previous

#: Guards the stale-attachment deletion below. A sync that would retire more
#: than this is reporting on a failed enumeration, not on the library.
STALE_DELETE_MAX_RATIO = 0.05
STALE_DELETE_MIN_KEYS = 10

RAG_EXCLUDE_TAG = (os.environ.get("ZOTERO_RAG_EXCLUDE_TAG") or "rag:exclude").strip().casefold()
RAG_PREFER_EPUB_TAG = (
    os.environ.get("ZOTERO_RAG_PREFER_EPUB_TAG") or "rag:prefer-epub"
).strip().casefold()


RUN_CODE_PATHS = tuple(
    PROJECT_ROOT / "src" / name for name in (
        "index_from_zotero.py", "embedder.py", "text_utils.py", "html_extract.py",
        "pdf_extract.py", "docling_extract.py", "docling_worker.py", "extraction_engine.py",
        "local_ocr_pipeline.py", "rapidocr_extract.py", "ndlocr_extract.py", "epub_fallback.py",
        "pdf_toc_recovery.py", "document_structure.py", "v3_migration.py", "v3_runtime.py",
        "pdf_provenance.py", "ocr_layer_audit.py", "source_coverage.py",
    )
)


# Batch sizing — two separate knobs:
#   FLUSH_SIZE: how many chunks to accumulate before flushing to ChromaDB.
#     Larger = fewer transactions, faster indexing.  Default 500.
#   UPSERT_BATCH_SIZE: sub-batch size for col.upsert() calls.  Controls peak
#     memory during embedding computation.  Default 128.
FLUSH_SIZE = int((os.environ.get("FLUSH_SIZE") or "500").strip())
UPSERT_BATCH_SIZE = int((os.environ.get("UPSERT_BATCH_SIZE") or "128").strip())
# Backward-compat: BATCH_SIZE acts as a fallback for FLUSH_SIZE if set alone
if "BATCH_SIZE" in os.environ and "FLUSH_SIZE" not in os.environ:
    FLUSH_SIZE = int(os.environ["BATCH_SIZE"].strip())
    UPSERT_BATCH_SIZE = FLUSH_SIZE


# ---------------------------------------------------------------------------
# Indexing lock — prevents MCP server from serving stale/inconsistent results
# while the indexer is writing to ChromaDB.
# ---------------------------------------------------------------------------


#: The lock this process is holding, if any. Kept here so the run can be
#: released by whoever notices it ended -- including the ``finally`` in
#: ``main_async``, which previously had no way to learn what the body acquired
#: and left a failed run holding the lock until the process exited.
_HELD_LOCK: dict[str, Any] | None = None


def _acquire_indexing_lock() -> dict:
    """Take the lock for this run's plane. Kept as a name callers already use."""
    global _HELD_LOCK
    _HELD_LOCK = indexing_lock.acquire(paths().indexing_lock_path)
    return _HELD_LOCK


def _release_indexing_lock(lock_data: dict[str, Any] | None = None) -> None:
    """Release a lock this run took, at the path it was taken at."""
    global _HELD_LOCK
    held = lock_data or _HELD_LOCK
    if held is None:
        return
    indexing_lock.release(held, lock_path=paths().indexing_lock_path)
    _HELD_LOCK = None


def _dedupe_by_id(
    ids: list[str],
    docs: list[str],
    metas: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Dedupe records by id, keeping the last occurrence."""
    if not (len(ids) == len(docs) == len(metas)):
        raise ValueError(
            "Chunk arrays must have equal lengths: "
            f"ids={len(ids)}, documents={len(docs)}, metadatas={len(metas)}"
        )
    uniq: dict[str, tuple[str, dict[str, Any]]] = {}
    for cid, doc, md in zip(ids, docs, metas, strict=True):
        uniq[cid] = (doc, md)
    out_ids = list(uniq.keys())
    out_docs = [uniq[i][0] for i in out_ids]
    out_metas = [uniq[i][1] for i in out_ids]
    return out_ids, out_docs, out_metas


def _delete_by_attachment_keys(
    col: Any, attachment_keys: Iterable[str], *, strict: bool = False,
) -> None:
    """Delete all vector and lexical rows for each attachment key."""
    keys = [key for key in attachment_keys if key]
    for dk in keys:
        try:
            col.delete(where={"attachmentKey": dk})
        except Exception:
            if strict:
                raise
    try:
        delete_lexical_attachments(keys)
    except Exception as exc:
        if strict:
            raise RuntimeError(f"Lexical index delete failed: {exc}") from exc
        print(f"[WARN] Lexical index delete failed: {exc}", file=sys.__stderr__)


def _pages_without_chunks(
    chunks: Iterable[tuple[str, str, dict[str, Any]]], expected_page_count: Any,
) -> list[int]:
    """Expected pages (1-based) that produced zero chunks.

    extraction=success is a per-attachment binary: an attachment can hold
    this status while individual pages inside it produced zero chunks, and
    nothing in the ledger says which. 583 attachments recorded success while
    539 pages across them had no indexed text (2026-07-28, found only by
    comparing against the original source files -- no index-vs-index check
    can see this). Returns [] when the expected page count is unknown rather
    than guessing.
    """
    if not isinstance(expected_page_count, (int, float)) or not expected_page_count:
        return []
    pages_with_chunks: set[int] = set()
    for _cid, _text, md in chunks:
        page = md.get("page")
        if isinstance(page, bool):
            continue
        if isinstance(page, (int, float)):
            pages_with_chunks.add(int(page))
        elif isinstance(page, str) and page.isdigit():
            pages_with_chunks.add(int(page))
    return sorted(set(range(1, int(expected_page_count) + 1)) - pages_with_chunks)


def relieve_memory_pressure() -> None:
    """Best-effort memory cleanup between large batches.

    Keep this local to the indexing pipeline (not embedder/text_utils) because it is
    primarily about batching/upsert behavior and memory spikes during indexing.

    Behavior:
      - Run Python GC.
      - If torch is installed, optionally clear CUDA/MPS caches.
    """
    try:
        gc.collect()
    except Exception:
        pass

    if (os.environ.get("TORCH_EMPTY_CACHE") or "1") != "1":
        return

    # Optional: free GPU/accelerator caches if torch is present.
    try:
        import torch  # type: ignore

        try:
            if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            if getattr(torch, "mps", None) is not None:
                # torch.mps.empty_cache exists on some builds
                empty = getattr(torch.mps, "empty_cache", None)
                if callable(empty):
                    empty()
        except Exception:
            pass

    except Exception:
        pass


def _upsert_in_subbatches(
    col: Any,
    ids: list[str],
    docs: list[str],
    metas: list[dict[str, Any]],
    *,
    subbatch_size: int,
    show_progress: bool,
    label: str,
    strict_lexical: bool = False,
) -> None:
    """Upsert in smaller sub-batches to reduce memory spikes."""
    if not (len(ids) == len(docs) == len(metas)):
        raise ValueError(
            "Chunk arrays must have equal lengths: "
            f"ids={len(ids)}, documents={len(docs)}, metadatas={len(metas)}"
        )
    total = len(ids)
    if total == 0:
        return
    # Chroma rejects an empty-list metadata value outright (raises ValueError
    # deep in validate_insert_record_set). Some PyMuPDF layout fallback paths
    # legitimately emit e.g. source_block_indices=[] for a standalone chunk
    # that never gets merged with a neighbor; drop such empty-list values here
    # rather than at every chunk-producing call site (pre-existing latent bug,
    # surfaced by scanned-page patch batches; E2c, dev-notes/current/77).
    for meta in metas:
        for key in [k for k, v in meta.items() if isinstance(v, list) and not v]:
            del meta[key]
    if show_progress:
        print(
            f"[PROGRESS] {label}: {total} chunks (embedding+write) | sub-batch={subbatch_size}",
            file=sys.__stderr__,
        )
    for start in range(0, total, subbatch_size):
        end = min(start + subbatch_size, total)
        if show_progress:
            print(
                f"[PROGRESS]   ↳ upsert sub-batch {start + 1}-{end}/{total}",
                file=sys.__stderr__,
            )
        col.upsert(
            ids=ids[start:end],
            documents=docs[start:end],
            metadatas=metas[start:end],
        )
        try:
            upsert_lexical_chunks(
                ids[start:end], docs[start:end], metas[start:end]
            )
        except Exception as exc:
            if strict_lexical:
                raise RuntimeError(f"Lexical index update failed: {exc}") from exc
            print(f"[WARN] Lexical index update failed: {exc}", file=sys.__stderr__)
        relieve_memory_pressure()


def _replace_attachment_batch(
    col: Any,
    *,
    attachment_keys: Iterable[str],
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    expected_ids: dict[str, set[str]],
    attachment_item_keys: dict[str, str],
    subbatch_size: int,
    show_progress: bool,
    label: str,
    context_label: str,
    strict_lexical: bool,
) -> None:
    """Replace a flush as a compensating unit of work.

    Chroma does not expose a multi-call transaction. Snapshot the active
    generation before the destructive boundary and restore it if any delete,
    sub-batch write, lexical write, or post-write validation fails.
    """
    batch = AttachmentBatch.create(
        attachment_keys=attachment_keys,
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        expected_ids=expected_ids,
        attachment_item_keys=attachment_item_keys,
        subbatch_size=subbatch_size,
        show_progress=show_progress,
        label=label,
        context_label=context_label,
        strict_lexical=strict_lexical,
    )
    replace_attachment_batch(
        col, batch,
        delete_batch=_delete_by_attachment_keys,
        upsert_batch=_upsert_in_subbatches,
        health_check=_fail_flush_on_unhealthy_collection,
        verify_written=_verify_written_attachments,
    )


def _fail_flush_on_unhealthy_collection(
    col: Any,
    attachment_item_keys: dict[str, str],
    *,
    context_label: str,
) -> None:
    """Cheap post-flush Chroma health check.

    A flush's writes can leave the Chroma collection in a broken state for
    any reason (not just a Docling crash -- this is a general safety net).
    Detect that within seconds by probing with the cheapest possible call,
    col.count(), immediately after each flush's upsert lands. If it fails,
    the run must not continue accumulating a manifest that claims success
    against a broken collection: mark every attachment in this flush as
    blocked/retryable and stop the run immediately.
    """
    try:
        col.count()
    except Exception as exc:
        keys = sorted(attachment_item_keys.keys())
        print(
            f"[ERROR] Chroma collection unhealthy after {context_label}; "
            f"attachments in this flush={keys} err={exc}",
            file=sys.__stderr__,
        )
        for attachment_key, item_key in attachment_item_keys.items():
            mark_artifact_status(
                item_key, "extraction", "blocked",
                attachment_key=attachment_key,
                reason_code="chroma_collection_unhealthy",
                message=f"Chroma health check failed after {context_label}: {exc}",
                retryable=True,
            )
        raise SystemExit(1) from exc


def _flush_pending_index_batch(
    col: Any,
    pending: PendingIndexBatch,
    *,
    manifest: dict[str, Any],
    files_manifest: dict[str, dict[str, Any]],
    run_code_fingerprint: str,
    show_progress: bool,
    label: str,
    context_label: str,
) -> FlushOutcome:
    """Validate, replace, commit, and clear one accumulated attachment batch."""
    if not pending.manifest_updates:
        return FlushOutcome()

    ids, documents, metadatas = _dedupe_by_id(
        pending.ids, pending.documents, pending.metadatas,
    )
    pending.ids[:] = ids
    pending.documents[:] = documents
    pending.metadatas[:] = metadatas
    expected_ids = pending.expected_ids()
    committed_item_keys = frozenset(pending.item_keys.values())

    assert_code_unchanged(RUN_CODE_PATHS, run_code_fingerprint)
    inflight = {
        str(value) for value in manifest.get("inflight_attachments", []) if value
    }
    inflight.update(pending.delete_attachment_keys)
    manifest["inflight_attachments"] = sorted(inflight)
    save_manifest(paths().manifest_path, manifest)

    _replace_attachment_batch(
        col,
        attachment_keys=pending.delete_attachment_keys,
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        expected_ids=expected_ids,
        attachment_item_keys=dict(pending.item_keys),
        subbatch_size=UPSERT_BATCH_SIZE,
        show_progress=show_progress,
        label=label,
        context_label=context_label,
        strict_lexical=True,
    )

    for item_key, status, status_kwargs in pending.extraction_statuses.values():
        mark_artifact_status(item_key, "extraction", status, **status_kwargs)
    files_manifest.update(pending.manifest_updates)
    manifest["inflight_attachments"] = [
        value for value in manifest.get("inflight_attachments", [])
        if value not in pending.delete_attachment_keys
    ]

    outcome = FlushOutcome(
        updated_pdf=sum(value == "pdf" for value in pending.source_types.values()),
        updated_html=sum(value == "html" for value in pending.source_types.values()),
        updated_epub=sum(value == "epub" for value in pending.source_types.values()),
        last_written_id=str(ids[-1]) if ids else None,
        committed_item_keys=committed_item_keys,
    )
    save_manifest(paths().manifest_path, manifest)
    _finalize_v3_pending(
        manifest, set(committed_item_keys),
        collection_name=str(paths().collection_name or "zotero_paragraphs_v3"),
        code_paths=RUN_CODE_PATHS,
        expected_code_fingerprint=run_code_fingerprint,
    )
    pending.clear()
    return outcome


def _source_document_chunks(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Canonical structures cover source attachments, never Zotero notes."""
    return [
        row for row in rows
        if str((row.get("metadata") or {}).get("source_type") or "") != "note"
    ]


def _finalize_v3_item(item_key: str, *, collection_name: str) -> dict[str, Any]:
    item_chunks = _source_document_chunks(get_item_chunks(
        item_key, chroma_dir=paths().chroma_dir, collection_name=collection_name,
    ))
    built = build_document_structure(item_key, item_chunks)
    replace_document_structure(
        item_key, source_fingerprint=built["source_fingerprint"],
        structure_version=built["structure_version"], status=built["status"],
        confidence=built["confidence"], nodes=built["nodes"],
        diagnostics=built["diagnostics"],
    )
    mark_artifact_status(
        item_key, "structure",
        "success" if built["status"] in {"exact", "recovered"} else "degraded",
        reason_code="flat_fallback" if built["status"] == "flat_fallback" else None,
        source_fingerprint=built["source_fingerprint"],
        processor_version=built["structure_version"],
        counts={"nodes": len(built["nodes"]), "leaves": built["diagnostics"].get("leaf_count", 0)},
    )
    mark_artifact_status(
        item_key, "embeddings", "success",
        source_fingerprint=built["source_fingerprint"],
        counts={"chunks": len(item_chunks), "collection": collection_name},
    )
    mark_artifact_status(
        item_key, "summary", "stale", reason_code="v3_source_changed",
        source_fingerprint=built["source_fingerprint"],
    )
    return built


def _finalize_v3_pending(
    manifest: dict[str, Any], item_keys: Iterable[str], *, collection_name: str,
    code_paths: Iterable[Path] | None = None, expected_code_fingerprint: str | None = None,
) -> None:
    """Finalize durable V3 rows and clear their crash-recovery checkpoint."""
    pending = {str(value) for value in manifest.get("post_index_pending", []) if value}
    pending.update(str(value) for value in item_keys if value)
    manifest["post_index_pending"] = sorted(pending)
    save_manifest(paths().manifest_path, manifest)
    for item_key in sorted(pending):
        if code_paths is not None and expected_code_fingerprint:
            assert_code_unchanged(code_paths, expected_code_fingerprint)
        try:
            _finalize_v3_item(item_key, collection_name=collection_name)
        except Exception as exc:
            mark_artifact_status(
                item_key, "structure", "failed", reason_code="v3_post_index_structure_failed",
                message=str(exc)[:1000], retryable=True,
            )
            raise
        manifest["post_index_pending"] = [
            value for value in manifest.get("post_index_pending", []) if value != item_key
        ]
        save_manifest(paths().manifest_path, manifest)


def _retire_indexed_attachments(
    col: Any, manifest: dict[str, Any], files_manifest: dict[str, dict[str, Any]],
    attachments: Iterable[ZoteroAttachment], *, collection_name: str,
    summary_client: Any = None, code_paths: Iterable[Path] | None = None,
    expected_code_fingerprint: str | None = None,
) -> tuple[int, set[str]]:
    """Atomically retire committed attachment rows, then refresh their items."""
    rows_by_key = {
        row.attachmentKey: row for row in attachments
        if row.attachmentKey in files_manifest
    }
    affected_items: set[str] = set()
    deleted = 0
    for attachment_key, row in rows_by_key.items():
        _delete_by_attachment_keys(col, [attachment_key], strict=True)
        files_manifest.pop(attachment_key, None)
        purge_artifact_status_for_attachments([attachment_key])
        affected_items.add(str(row.parentItemKey or row.attachmentKey))
        deleted += 1
    save_manifest(paths().manifest_path, manifest)
    if affected_items:
        _finalize_v3_pending(
            manifest, affected_items, collection_name=collection_name,
            code_paths=code_paths,
            expected_code_fingerprint=expected_code_fingerprint,
        )
        _delete_summary_embeddings_for_items(
            affected_items, collection_name=collection_name, client=summary_client,
        )
    return deleted, affected_items


def _sync_tag_exclusions_without_embedding_runtime(
    manifest: dict[str, Any], files_manifest: dict[str, dict[str, Any]],
    excluded_attachments: Iterable[ZoteroAttachment], *,
    preferred_pdf_attachments: Iterable[ZoteroAttachment],
    inventory_attachments: Iterable[ZoteroAttachment], show_progress: bool,
) -> int:
    """Retire safely tagged rows without loading or validating an embedder."""
    import chromadb

    collection_name = str(paths().collection_name or "zotero_paragraphs_v3")
    client = chromadb.PersistentClient(path=str(paths().chroma_dir))
    col = client.get_collection(collection_name)
    ready_preferred = _ready_preferred_pdfs(
        preferred_pdf_attachments, inventory_attachments, files_manifest,
    )
    deleted, affected_items = _retire_indexed_attachments(
        col, manifest, files_manifest,
        [*excluded_attachments, *ready_preferred],
        collection_name=collection_name, summary_client=client,
    )
    if show_progress:
        print(
            f"[PROGRESS] Tag exclusions synchronized: attachments={deleted}, "
            f"structures={len(affected_items)}",
            file=sys.__stderr__,
        )
    return deleted


def _delete_summary_embeddings_for_items(
    item_keys: Iterable[str], *, collection_name: str, client: Any = None,
) -> None:
    """Remove vectors derived from an item's pre-exclusion source scope."""
    if client is None:
        import chromadb
        client = chromadb.PersistentClient(path=str(paths().chroma_dir))
    try:
        summary_collection = client.get_collection(f"{collection_name}__sum_node")
    except Exception:
        return
    for item_key in set(item_keys):
        try:
            summary_collection.delete(where={"itemKey": item_key})
        except Exception as exc:
            print(
                f"[WARN] Failed to delete stale summary vectors for {item_key}: {exc}",
                file=sys.stderr,
            )


def _verify_written_attachments(
    col: Any, expected: dict[str, set[str]],
) -> None:
    """Verify exact Chroma and FTS ID sets before committing the manifest."""
    lexical = lexical_chunk_ids_by_attachment_keys(expected.keys())
    for attachment_key, expected_ids in expected.items():
        result = col.get(where={"attachmentKey": attachment_key}, include=[])
        chroma_ids = {str(value) for value in (result.get("ids") or [])}
        if chroma_ids != expected_ids:
            raise RuntimeError(
                f"Chroma attachment ID mismatch for {attachment_key}: "
                f"expected={len(expected_ids)} actual={len(chroma_ids)}"
            )
        lexical_ids = lexical.get(attachment_key, set())
        if lexical_ids != expected_ids:
            raise RuntimeError(
                f"Lexical attachment ID mismatch for {attachment_key}: "
                f"expected={len(expected_ids)} actual={len(lexical_ids)}"
            )


def _collection_embedding_dim(col: Any) -> int | None:
    result = col.get(limit=1, include=["embeddings"])
    embeddings = result.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        return None
    return int(len(embeddings[0]))


def _close_chroma_collection(col: Any) -> None:
    client = getattr(col, "_chroma_client", None)
    if client is None:
        return
    try:
        client.close()
    except Exception:
        pass


def _flush_and_verify_hnsw(col: Any, sample_id: str | None) -> None:
    """Force pending HNSW state durable, then execute a real vector query."""
    sample_embedding = None
    if sample_id:
        row = col.get(ids=[sample_id], include=["embeddings"])
        embeddings = row.get("embeddings")
        if embeddings is None or len(embeddings) != 1:
            raise RuntimeError(f"Unable to load HNSW smoke-test embedding: {sample_id}")
        sample_embedding = embeddings[0]

    configuration = getattr(col, "configuration", {}) or {}
    hnsw_configuration = configuration.get("hnsw") or {}
    original_sync_threshold = int(hnsw_configuration.get("sync_threshold", 100))
    col.modify(configuration={"hnsw": {"sync_threshold": 1}})
    try:
        sentinel_prefix = f"__hnsw_flush_sentinel_{uuid.uuid4().hex}"
        sentinels = [f"{sentinel_prefix}_{index}__" for index in range(3)]
        upsert_args: dict[str, Any] = {
            "ids": sentinels,
            "documents": ["hnsw flush sentinel"] * len(sentinels),
        }
        if sample_embedding is not None:
            # Reuse a vector already accepted by this collection. This avoids
            # invoking the embedder and cannot introduce a dimension mismatch.
            upsert_args["embeddings"] = [sample_embedding] * len(sentinels)
        col.upsert(**upsert_args)
        col.delete(ids=sentinels)
        if sample_embedding is None:
            return
        result = col.query(query_embeddings=[sample_embedding], n_results=1, include=["distances"])
        if not result.get("ids") or not result["ids"][0]:
            raise RuntimeError("HNSW smoke-test query returned no result")
    finally:
        col.modify(configuration={"hnsw": {"sync_threshold": original_sync_threshold}})


def _remove_stale_hnsw_sentinels(col: Any, *, collection_name: str) -> list[str]:
    """Remove sentinels left by a hard crash before normal cleanup."""
    stale = [
        chunk_id for chunk_id in list_chunk_ids(collection_name=collection_name)
        if chunk_id.startswith("__hnsw_flush_sentinel_")
    ]
    if not stale:
        return []
    col.delete(ids=stale)
    delete_lexical_chunk_ids(stale)
    remaining = set(list_chunk_ids(collection_name=collection_name)).intersection(stale)
    if remaining:
        raise RuntimeError(f"Unable to remove stale HNSW sentinels: {sorted(remaining)}")
    return stale




# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Index Zotero local PDFs/HTML snapshots into Chroma (paragraph-level).")
    p.add_argument("--collection", help="Zotero collection key to restrict by (optional).", default=None)
    p.add_argument("--item", action="append", help="Restrict to a parent itemKey; repeatable.")
    p.add_argument(
        "--attachment", action="append",
        help="Restrict to an exact attachment key; repeatable and intersected with --item.",
    )
    p.add_argument("--limit", type=int, default=0, help="Maximum number of parent items (0 = all).")
    p.add_argument("--dry-run", action="store_true", help="Resolve and report scope without changing indexes or ledgers.")
    p.add_argument(
        "--sync-rag-exclusions-only", action="store_true",
        help="Remove currently tag-excluded attachments and rebuild affected structures "
             "without loading or validating the embedding model.",
    )
    p.add_argument("--retry-failed", action="store_true", help="Process only items with retryable failed artifacts.")
    p.add_argument(
        "--force-reparse", action="store_true",
        help="Re-extract selected items even if unchanged (bypasses the mtime/pipeline skip). "
             "Use after an extraction-code change. Requires --item/--attachment/--limit/--source-type to scope.",
    )
    p.add_argument(
        "--refetch-ocr", action="store_true",
        help="Ignore the archived raw OCR response and call the engine again. "
             "Only needed when the OCR output itself is suspect -- a parsing or "
             "chunking fix does not need this, since chunks are always re-derived "
             "from the archive with current code.",
    )
    p.add_argument("--dump-attachments", action="store_true", help="Print resolved attachments list then proceed.")
    p.add_argument(
        "--progress",
        action="store_true",
        help="Print progress while indexing (also enabled by PROGRESS=1 env var).",
    )
    p.add_argument(
        "--require-data-dir",
        action="store_true",
        help="Fail fast if ZOTERO_DATA_DIR is not set or invalid.",
    )
    p.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild: delete Chroma DB and manifest, then re-index everything.",
    )
    p.add_argument(
        "--check-quality",
        action="store_true",
        help="Scan existing indexed files in the manifest for scanned/corrupted pages (skips Chroma re-indexing) and update manifest.",
    )
    p.add_argument(
        "--use-docling",
        action="store_true",
        help="Use high-fidelity IBM Docling instead of PyMuPDF to extract text from all PDFs in this run.",
    )
    p.add_argument(
        "--reparse-corrupted",
        action="store_true",
        help="Automatically use IBM Docling to re-parse and re-index scanned or corrupted PDFs tracked in the manifest.",
    )
    p.add_argument(
        "--reocr-candidates", type=Path,
        help="Reparse only attachments listed in list_reocr_candidates.py JSON; ja uses NDLOCR-Lite.",
    )
    p.add_argument(
        "--reocr-limit", type=int,
        help="Maximum ranked candidates to process with --reocr-candidates.",
    )
    p.add_argument(
        "--source-type", choices=("epub", "html", "pdf"),
        help="Restrict to a single source type (useful for batching fast types first).",
    )
    args = p.parse_args()
    if args.reocr_limit is not None and args.reocr_limit < 1:
        p.error("--reocr-limit must be positive")
    if args.reocr_limit is not None and not args.reocr_candidates:
        p.error("--reocr-limit requires --reocr-candidates")
    if args.limit < 0:
        p.error("--limit must be zero or positive")
    if args.force_reparse and not (args.item or args.attachment or args.limit or args.source_type or args.reocr_candidates):
        p.error("--force-reparse requires a scope (--item, --attachment, --limit, or --source-type) to avoid re-parsing the whole corpus")
    if args.sync_rag_exclusions_only and any((
        args.rebuild, args.item, args.attachment, args.limit, args.collection,
        args.source_type, args.reocr_candidates, args.retry_failed,
        args.reparse_corrupted, args.force_reparse,
    )):
        p.error("--sync-rag-exclusions-only cannot be combined with indexing or scope options")
    rebuild_scoped = bool(
        args.item or args.attachment or args.limit or args.collection
        or args.source_type or args.reocr_candidates or args.reocr_limit
        or args.retry_failed or args.reparse_corrupted
    )
    if args.rebuild and rebuild_scoped:
        p.error(
            "--rebuild must cover the complete library; do not combine it with "
            "--item/--attachment/--limit/--collection/--source-type/"
            "--reocr-candidates/--retry-failed/--reparse-corrupted"
        )
    return args


def _resolve_source_type(content_type: str | None, file_path: Path, source_type: str | None) -> str:
    """Resolve the canonical source type for filtering."""
    if content_type == "application/epub+zip" or file_path.suffix.lower() == ".epub":
        return "epub"
    if file_path.suffix.lower() in {".html", ".htm"}:
        return "html"
    return source_type if source_type in {"pdf", "html", "epub"} else "pdf"


def _select_item_scope(
    attachments: List[ZoteroAttachment], item_keys: Iterable[str] | None, limit: int,
) -> List[ZoteroAttachment]:
    """Select complete items in source order so attachments are never split."""
    requested = {str(value) for value in (item_keys or []) if value}
    selected_keys: list[str] = []
    for attachment in attachments:
        key = str(getattr(attachment, "parentItemKey", None) or getattr(attachment, "attachmentKey", ""))
        if not key or (requested and key not in requested) or key in selected_keys:
            continue
        selected_keys.append(key)
        if limit and len(selected_keys) >= limit:
            break
    allowed = set(selected_keys)
    return [
        attachment for attachment in attachments
        if str(getattr(attachment, "parentItemKey", None) or getattr(attachment, "attachmentKey", "")) in allowed
    ]


def _select_attachment_scope(
    attachments: List[ZoteroAttachment], attachment_keys: Iterable[str] | None,
) -> List[ZoteroAttachment]:
    """Apply an exact attachment scope before parent-item grouping.

    ``--item`` intentionally processes every attachment of a parent item.  A
    queue row, however, represents one attachment; filtering first prevents a
    one-row repair from silently reindexing its siblings.
    """
    requested = {str(value) for value in (attachment_keys or []) if value}
    if not requested:
        return attachments
    return [
        attachment for attachment in attachments
        if str(getattr(attachment, "attachmentKey", "")) in requested
    ]


def _retryable_failed(item_key: str) -> bool:
    return any(
        row.get("status") in {"failed", "degraded"} and bool(row.get("retryable"))
        for row in get_item_processing_status(item_key)
    )


MISTRAL_TOC_QUEUE_REASON = "awaiting_mistral_ocr_batch"
MISTRAL_TOC_QUEUE_PROCESSOR_VERSION = "mistral-ocr-queue-v3"
PDF_MISTRAL_OCR_BATCH_MIN_PAGES = 30


def _scanned_pdf_ocr_route(
    quality_info: dict[str, Any], *, total_pages: int, item_key: str,
) -> tuple[str | None, str]:
    """Choose the replacement route for scan-derived PDF text.

    Which engine handles a scan is choice (C) (note 80): the operator picks one
    per size bucket, because the three trade differently -- Docling free and
    fast, Granite free and more accurate but ~2.3x slower, Mistral fast and
    accurate but charged per page. This used to be hardcoded as "short means
    Docling, long means the Mistral queue".

    Mistral is reached through the non-canonical Batch queue rather than a
    direct call, so existing chunks stay untouched until a staged result is
    explicitly adopted. If the queue is switched off, a Mistral choice falls
    back to Docling rather than failing: the document still gets structured.

    Per-item cloud tag gating was removed (2026-07-27): anything in the index is
    returned by search and reaches the assistant regardless, so it protected
    nothing it was still in.
    """
    if not pdf_structure_recovery_enabled():
        # Choice (A) off: PDFs are indexed as plain text, so there is no
        # structuring engine to escalate to. Local OCR still runs for a PDF
        # with no text layer -- that is extraction, not structure recovery.
        return None, "structure_recovery_disabled"
    source_class = str(quality_info.get("source_class") or "")
    needs_replacement = (
        source_class == SCANNED_NO_TEXT
        or (source_class == SCANNED_OCR_LAYER and replacement_required(quality_info))
    )
    if not needs_replacement:
        return None, "not_scan_ocr_replacement"
    engine = structure_engine_for(total_pages)
    if engine == "mistral":
        if mistral_batch_queue_enabled():
            return "mistral_batch", "structure_engine_mistral"
        return "docling", "mistral_batch_queue_disabled"
    if engine == "granite":
        return "granite", "structure_engine_granite"
    return "docling", "structure_engine_docling"


def _initial_scanned_pdf_ocr_route(
    quality_info: dict[str, Any], *, total_pages: int, item_key: str,
) -> tuple[str | None, str]:
    """Route only an OCR-*absent* scan before the stage-2 audit.

    Keeping this separate makes the ordering contractual: an OCR-layer scan
    must retain its PyMuPDF chunks until ``audit_ocr_text_layer`` supplies the
    verdict consumed by ``_scanned_pdf_ocr_route``.
    """
    if str(quality_info.get("source_class") or "") != SCANNED_NO_TEXT:
        return None, "awaiting_ocr_layer_audit"
    return _scanned_pdf_ocr_route(
        quality_info, total_pages=total_pages, item_key=item_key,
    )


def _is_current_mistral_toc_candidate(
    item_key: str, attachment_key: str, *, mtime: float, size: int,
) -> bool:
    """True when this exact source file was already deferred to Mistral."""
    for row in get_item_processing_status(item_key):
        if (
            row.get("artifact_type") == "extraction"
            and str(row.get("attachment_key") or "") == attachment_key
            and row.get("status") == "blocked"
            and row.get("reason_code") == MISTRAL_TOC_QUEUE_REASON
            and row.get("processor_version") == MISTRAL_TOC_QUEUE_PROCESSOR_VERSION
        ):
            counts = row.get("counts") or {}
            return (
                int(counts.get("source_size") or -1) == size
                and float(counts.get("source_mtime") or -1) == mtime
            )
    return False


def _skip_current_mistral_toc_candidate(
    *, stype: str, reocr_route: dict[str, Any] | None, force_reparse: bool,
    reparse_corrupted: bool, use_docling: bool,
) -> bool:
    """True only for ordinary resumption of an already-deferred PDF.

    ``--force-reparse`` is an explicit repair request.  It must re-enter the
    gate path rather than preserve a stale Mistral deferral (note 79 U5).
    """
    return (
        stype == "pdf" and not reocr_route and not force_reparse
        and not reparse_corrupted and not use_docling
    )


# AI-TOC rejection reasons that mean the LLM itself concluded the document has
# (nearly) no headings -- i.e. there is no structure to recover, as opposed to
# structure that exists but couldn't be aligned (P1, note 78).
NO_STRUCTURE_AI_TOC_REASONS = frozenset({
    "insufficient_inferred_headings", "insufficient_body_headings",
})
# AI-TOC rejection reasons that mean headings likely *do* exist but couldn't be
# deterministically aligned/attached -- the document is indexed unstructured,
# which is a degraded outcome worth surfacing, not a confirmed no-structure one.
AI_TOC_ALIGNMENT_FAILURE_REASONS = frozenset({
    "body_coverage_below_threshold", "structured_chunk_ratio_below_threshold",
})


def _source_content_unchanged(
    prev: dict[str, Any] | None, *, mtime: float, size: int, signature: str | None,
) -> bool:
    """Whether ``prev``'s recorded mtime/size (and content, if known) still match.

    mtime and size alone cannot tell a file replaced at the same path apart
    from an unchanged one, if the replacement happens to land on the same
    byte count -- a corrected scan re-saved from the same source, a sync tool
    that does not always preserve mtime. ``signature`` is only meaningful
    when both sides have one: an old manifest row written before content
    signatures existed is not forced to re-parse solely for lacking one
    (2026-07-28).
    """
    if not prev:
        return False
    if float(prev.get("mtime", -1)) != mtime or int(prev.get("size", -1)) != size:
        return False
    prev_signature = prev.get("content_signature")
    if prev_signature and signature is not None:
        return signature == prev_signature
    return True


def _prior_no_structure_ai_toc_status(
    prev: dict | None, *, mtime: float, size: int,
) -> str | None:
    """Return the prior run's no-structure AI-TOC verdict for this same file.

    P1 (note 78, user approval 2026-07-26): when a previous ingest of the
    *identical* source file (same mtime and size) already ran the AI-TOC fast
    path and the LLM concluded the document has no headings
    (``NO_STRUCTURE_AI_TOC_REASONS``), that verdict is final for this file --
    re-running the fast path on a reingest would spend another DeepSeek call
    plus a full-document page-record sweep only to reach the same conclusion.
    Alignment failures and transient errors are NOT cached: alignment logic
    improves across code versions and errors are retryable, so those reasons
    return None and the fast path runs again.
    """
    if not isinstance(prev, dict):
        return None
    try:
        if float(prev.get("mtime", -1)) != float(mtime) or int(prev.get("size", -1)) != int(size):
            return None
    except (TypeError, ValueError):
        return None
    quality = prev.get("quality")
    if not isinstance(quality, dict):
        return None
    status = str(quality.get("ai_toc_recovery_status") or "")
    return status if status in NO_STRUCTURE_AI_TOC_REASONS else None


#: Audit fields carried forward when the source file is unchanged.
_OCR_LAYER_AUDIT_FIELDS = (
    "ocr_layer_quality", "ocr_layer_error_rate", "ocr_layer_verified_count",
    "ocr_layer_rejected_count", "ocr_layer_denominator", "ocr_layer_sampled_pages",
    "ocr_layer_needs_review", "ocr_layer_examples", "ocr_layer_audit_reason",
)

_DEFERRED_AUDIT_CONTEXT_FIELDS = (
    "source_class", "source_class_full_page_image_ratio",
    "source_class_pure_text_page_ratio", "source_class_text_page_ratio",
    "source_class_sampled_pages", "pdf_producer", "text_defects",
    "letter_spacing_ratio", "dropped_ligature_count", "dropped_ligature_ratio",
    "text_defect_scope_applicable", "latin_word_count",
)


def _merge_deferred_ocr_audit_quality(
    previous: dict | None, fresh_quality: dict[str, Any],
) -> dict[str, Any]:
    """Keep only audit/provenance evidence when canonical adoption is deferred.

    Mistral deferral intentionally leaves the current Chroma/FTS chunks in
    place.  The source itself has nevertheless been measured, so dropping the
    measurement would both hide the gate result and bill the same audit again
    on resume.  Retaining only these fields avoids mislabelling the old
    canonical extraction as the transient PyMuPDF/repair attempt.
    """
    old_quality = (previous or {}).get("quality")
    merged = dict(old_quality) if isinstance(old_quality, dict) else {}
    for key in (*_OCR_LAYER_AUDIT_FIELDS, *_DEFERRED_AUDIT_CONTEXT_FIELDS):
        if key in fresh_quality:
            merged[key] = fresh_quality[key]
    return merged


def _manifest_entry(
    *, mtime: float, size: int, source_path: Any, title: Any,
    quality: dict[str, Any], content_signature_value: str | None = None,
    pipeline_fingerprint: str | None = None,
) -> dict[str, Any]:
    """One attachment's manifest row: what was indexed, and from which file.

    The single definition of the row's shape. It used to be spelled out inline
    at each of the four sites that produce one, and they drifted: the EPUB
    deferral simply omitted its write, so ``--rebuild`` reported the attachment
    under ``missing_manifest_attachments`` while also counting it in
    ``deferred_extract`` -- the same queued file named twice, as if lost.
    """
    entry: dict[str, Any] = {
        "mtime": mtime, "size": size, "pdf_path": str(source_path),
        "title": title, "quality": quality,
    }
    if content_signature_value:
        entry["content_signature"] = content_signature_value
    if pipeline_fingerprint is not None:
        entry["pipeline_fingerprint"] = pipeline_fingerprint
    return entry


def _deferred_manifest_entry(
    previous: dict | None, *, quality: dict[str, Any], **fields: Any,
) -> dict[str, Any]:
    """The row for an attachment handed to the Mistral OCR batch.

    A deferral *continues* the previous row instead of replacing it: the
    attachment's current chunks stay searchable until an adopted batch result
    passes the gate, so anything the last completed ingest recorded is carried
    forward and only the audit/provenance part of the fresh measurement is
    merged in. A row that is not a dict (hand-edited or truncated manifest) is
    tolerated on both paths -- the audit merge would otherwise raise on .get
    and abort the run.
    """
    prior = previous if isinstance(previous, dict) else None
    return {
        **(prior or {}),
        **_manifest_entry(
            quality=_merge_deferred_ocr_audit_quality(prior, quality), **fields,
        ),
    }


def _cached_ocr_layer_audit(
    prev: dict | None, *, mtime: float, size: int,
) -> dict[str, Any] | None:
    """Return a prior *measured* OCR-layer verdict for this same source file.

    Reusing it keeps reingestion free of API calls (note 79). Only a completed
    measurement is cached: an audit that could not run (cloud policy, sampling
    failure, LLM error) must be retried, since those conditions change.
    """
    if not isinstance(prev, dict):
        return None
    try:
        if float(prev.get("mtime", -1)) != float(mtime) or int(prev.get("size", -1)) != int(size):
            return None
    except (TypeError, ValueError):
        return None
    quality = prev.get("quality")
    if not isinstance(quality, dict):
        return None
    if str(quality.get("ocr_layer_quality") or "") not in {"acceptable", "degraded"}:
        return None
    return {key: quality[key] for key in _OCR_LAYER_AUDIT_FIELDS if key in quality}


def _docling_escalation_acceptable(
    chunks: list[tuple[str, str, dict[str, Any]]],
) -> tuple[bool, dict[str, Any]]:
    """Minimal quality gate for full-document Docling escalation output.

    P2 (note 78, user approval 2026-07-26): the local-OCR route puts Docling
    output through ``evaluate_local_ocr_gate``, but the total-gate escalation
    (short PDF / no-cloud / queue disabled) used to adopt Docling output with
    no gate at all -- an asymmetry that let output quality depend on which
    branch reached the same engine. This applies the two content checks from
    that gate that don't need page-coverage bookkeeping (gibberish blocks and
    repeat artifacts, the two failure signatures observed in real rejected
    OCR output); coverage/length checks stay out because a text-layer PDF
    legitimately produces sparse output for figure-heavy documents.
    """
    texts = [str(text or "").strip() for _cid, text, _md in chunks if str(text or "").strip()]
    gibberish_blocks = sum(int(looks_like_gibberish(text)) for text in texts)
    repeat_artifacts = sum(len(REPEAT_ARTIFACT_RE.findall(text)) for text in texts)
    return (
        gibberish_blocks == 0 and repeat_artifacts == 0,
        {"gibberish_blocks": gibberish_blocks, "repeat_artifacts": repeat_artifacts},
    )


def _sort_chunks_in_reading_order(
    chunks: list[tuple[str, str, dict[str, Any]]],
) -> list[tuple[str, str, dict[str, Any]]]:
    """Re-sort a chunk list into (page, reading_order) reading order.

    P5 (note 78, user approval 2026-07-26): the E2c/E2d page patches used to
    append recovered chunks at the end of the list, so a patched page-102
    figure caption could physically sit after the final chapter. The AI-TOC
    reference splice already re-sorts this way (``_splice_reference_chunks``);
    this applies the same ordering after page patches. Sorting is stable, so
    chunks on the same page keep their relative order when reading_order ties.
    """
    return sorted(
        chunks,
        key=lambda row: (int(row[2].get("page") or 0), int(row[2].get("reading_order") or 0)),
    )


def _deduplicate_exact_chunk_records(
    chunks: list[tuple[str, str, dict[str, Any]]],
) -> list[tuple[str, str, dict[str, Any]]]:
    """Drop only byte-for-byte equivalent extractor duplicates.

    Page-repair stages can converge on the same already-normalized chunk.  A
    duplicate ID is not safe to pass to either the structure builder or Chroma,
    but silently choosing between divergent payloads would hide a producer bug.
    Keep the first equivalent record in source order and fail closed when the
    same ID carries different text or metadata.
    """
    output: list[tuple[str, str, dict[str, Any]]] = []
    signatures: dict[str, tuple[str, str]] = {}
    for chunk_id, text, metadata in chunks:
        key = str(chunk_id or "")
        normalized_metadata = dict(metadata or {})
        try:
            metadata_signature = json.dumps(
                normalized_metadata, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), default=str,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"chunk {key!r} has non-canonical metadata") from exc
        signature = (str(text or ""), metadata_signature)
        previous = signatures.get(key)
        if previous is None:
            signatures[key] = signature
            output.append((key, str(text or ""), normalized_metadata))
        elif previous != signature:
            raise ValueError(f"conflicting duplicate chunk id: {key}")
    return output


SCAN_PAGE_REPAIR_MIN_CHARS = int(os.environ.get("PDF_SCAN_PAGE_REPAIR_MIN_CHARS", "80"))


def _scan_pages_needing_repair(
    chunks: list[tuple[str, str, dict[str, Any]]], total_pages: int,
) -> list[int]:
    """Pages of a scan-derived document whose OCR produced nothing usable.

    Only meaningful when the text layer came from OCR (note 79, U2). There, a
    page with no or almost no text is a *failure* -- the page image certainly
    carried something -- whereas the same page in a born-digital document is
    simply a figure. Pages already carrying a corrupted/figure marker are left
    alone: they have been through repair once already, and re-attempting them
    every run would loop.

    Returns 1-based page numbers, capped at ``total_pages``.
    """
    if total_pages <= 0:
        return []
    chars_by_page: dict[int, int] = {}
    marked: set[int] = set()
    for _chunk_id, text, metadata in chunks:
        page = int((metadata or {}).get("page") or 0)
        if page <= 0:
            continue
        chars_by_page[page] = chars_by_page.get(page, 0) + len(text or "")
        if str((metadata or {}).get("block_type") or "") in {"figure", "corrupted_unresolved"}:
            marked.add(page)
    return [
        page for page in range(1, total_pages + 1)
        if page not in marked
        and chars_by_page.get(page, 0) < SCAN_PAGE_REPAIR_MIN_CHARS
    ]


def _audit_reused_ocr_chunks(
    chunks: list[tuple[str, str, dict[str, Any]]],
    quality_info: dict[str, Any],
    pdf_path: Path,
    *,
    item_key: str,
    prev: dict | None,
    mtime: float,
    size: int,
    show_progress: bool,
) -> tuple[list[tuple[str, str, dict[str, Any]]], dict[str, Any], bool]:
    """Apply the source-class, defect and OCR-quality checks to reused text.

    The legacy-OCR reuse path (``v3_migration.reuse_ocr_chunks_for_v3``) skips
    the V3 PDF routing block, and with it every quality check added in note 79.
    That is the wrong way round: reused text *is* OCR output, so it is the most
    likely of all inputs to be degraded, and accepting it unmeasured would carry
    old damage into V3 while saving nothing worth saving.

    Returns ``(chunks, quality_info, reuse_still_ok)``. When the text measures
    degraded, reuse is abandoned -- the caller then falls through to normal
    extraction and its escalation routes, which is what would have happened had
    the legacy text never existed.
    """
    quality_info = dict(quality_info)
    try:
        source_class = classify_pdf_source(pdf_path)
        quality_info.update(source_class.as_metadata())
    except Exception as exc:
        # Not "no class": a class that could not be computed. Absent, the two
        # are the same value downstream, and reocr_quality reads the absence as
        # "not a scan" -- so a scanned document whose classification failed
        # would never again be offered as a re-OCR candidate. The normal
        # extraction path already warns here; this one used to pass silently.
        quality_info["source_class_error"] = str(exc)
        print(
            f"[WARN] PDF source classification failed on reused OCR text: "
            f"item={item_key} err={exc}",
            file=sys.__stderr__,
        )
    quality_info.update(
        detect_text_defects("\n".join(text for (_id, text, _md) in chunks))
    )

    if not ocr_layer_audit_enabled():
        return chunks, quality_info, True

    cached = _cached_ocr_layer_audit(prev, mtime=mtime, size=size)
    if cached is not None:
        quality_info.update(cached)
        quality_info["ocr_layer_audit_cached"] = True
    else:
        quality_info.update(audit_ocr_text_layer(pdf_path, item_key))

    degraded = (
        str(quality_info.get("ocr_layer_quality") or "") == OCR_LAYER_DEGRADED
        or bool(quality_info.get("text_defects"))
    )
    if degraded:
        if show_progress:
            print(
                "[PROGRESS]   ↳ legacy OCR text measured degraded "
                f"(rate={quality_info.get('ocr_layer_error_rate')}, "
                f"defects={quality_info.get('text_defects')}); "
                "not reusing it, re-extracting instead",
                file=sys.__stderr__,
            )
        return [], quality_info, False
    return chunks, quality_info, True


def _attach_pdf_source_provenance(
    extraction_quality: dict[str, Any], source_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Carry source classification across every OCR extractor replacement."""
    result = dict(extraction_quality)
    result.update(source_metadata)
    return result


def _carry_ocr_layer_audit(
    extraction_quality: dict[str, Any], prior_quality: dict[str, Any],
) -> dict[str, Any]:
    """Retain the measured OCR-layer verdict across a replacement extractor."""
    result = dict(extraction_quality)
    for key in _OCR_LAYER_AUDIT_FIELDS:
        if key in prior_quality:
            result[key] = prior_quality[key]
    return result


def _structure_with_engine(
    engine: str,
    file_path: Path,
    attachment_key: str,
    meta_base: dict[str, Any],
    *,
    docling_worker: Any,
    granite_worker: Any,
) -> tuple[list[tuple[str, str, dict[str, Any]]], dict[str, Any]]:
    """Run the engine choice (C) selected for this document.

    Granite and Docling return the same shape because the Granite runner calls
    the same normalisation and chunking code, just with a VlmPipeline
    converter. Granite failures fall back to Docling rather than failing the
    document: the operator chose Granite for quality, not as a requirement, and
    an unstructured document helps nobody.
    """
    if engine != "granite":
        return docling_worker.extract(file_path, attachment_key, meta_base)
    try:
        return granite_worker.extract(file_path, attachment_key, meta_base)
    except Exception as exc:  # noqa: BLE001 - fall back rather than lose the document
        print(
            f"[WARN] Granite extraction failed ({exc}); falling back to Docling: "
            f"attachment={attachment_key}",
            file=sys.__stderr__,
        )
        return docling_worker.extract(file_path, attachment_key, meta_base)


def _merge_uncertain_reason(existing: Any, reason: str) -> str:
    """Append a reason without discarding one an earlier stage already set.

    A page-level OCR caveat (``short_ocr_page``) and a document-level one
    (``incomplete_source_coverage``) answer different questions, so the second
    must not overwrite the first. Reasons are comma-separated; a single reason
    keeps its own internal detail after ``:``.
    """
    tokens = [token for token in str(existing or "").split(",") if token]
    for token in str(reason or "").split(","):
        if token and token not in tokens:
            tokens.append(token)
    return ",".join(tokens)


def _adopt_with_quality_uncertain(
    chunks: list[tuple[str, str, dict[str, Any]]],
    quality_info: dict[str, Any],
    *,
    reason: str,
) -> tuple[list[tuple[str, str, dict[str, Any]]], dict[str, Any]]:
    """Adopt chunks the pipeline could not improve, tagged as uncertain.

    U4 (note 79, user decision 2026-07-26): when every remaining route is
    exhausted -- Docling could not beat the local extractor and cloud OCR is
    not permitted for this item -- the document is still indexed rather than
    dropped, carrying a tag that says its text may be degraded. Unlike
    ``zone="corrupted"`` these chunks stay searchable by default: the text is
    suspect, not known-bad, and being findable with a caveat beats being
    absent.
    """
    tagged = [
        (chunk_id, text, {
            **metadata, "quality_uncertain": True,
            "quality_uncertain_reason": _merge_uncertain_reason(
                metadata.get("quality_uncertain_reason"), reason,
            ),
        })
        for chunk_id, text, metadata in chunks
    ]
    updated = dict(quality_info)
    updated["quality_uncertain"] = True
    updated["quality_uncertain_reason"] = _merge_uncertain_reason(
        quality_info.get("quality_uncertain_reason"), reason,
    )
    return tagged, updated


def _epub_ocr_quality_from_mapping(
    quality_info: dict[str, Any],
    chunks: list[tuple[str, str, dict[str, Any]]],
    mapping: dict[str, Any],
    *,
    epub_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate derivative-PDF OCR evidence back to OPF spine units.

    The OCR engines report 1-based derivative PDF pages, while the canonical
    EPUB contract is 1-based OPF spine documents.  Keep that mapping explicit
    so an OCR result cannot become a healthy EPUB merely because it yielded a
    few remapped chunks.
    """
    page_to_spine: dict[int, int] = {}
    for row in mapping.get("pages") or []:
        if not isinstance(row, dict):
            continue
        try:
            page_to_spine[int(row["pdf_page_index"]) + 1] = int(row["spine_index"]) + 1
        except (KeyError, TypeError, ValueError):
            continue
    expected_spines = sorted(set(page_to_spine.values()))

    def _page_units(value: Any) -> set[int]:
        if isinstance(value, bool):
            return set()
        if isinstance(value, int):
            return set(range(1, max(0, value) + 1))
        if not isinstance(value, (list, tuple, set, range)):
            return set()
        result: set[int] = set()
        for raw in value:
            if isinstance(raw, bool):
                continue
            try:
                page = int(raw)
            except (TypeError, ValueError):
                continue
            if page > 0:
                result.add(page)
        return result

    # Mistral's ``ocr_pages`` is actual output coverage; local adapters use it
    # for their page attempts.  Either way, an absent page list must remain
    # unknown rather than being filled optimistically from the derivative.
    attempted_pages = _page_units(quality_info.get("attempted_pages"))
    if not attempted_pages:
        attempted_pages = _page_units(quality_info.get("ocr_pages"))
    if not attempted_pages:
        attempted_pages = _page_units(quality_info.get("processed_pages"))

    def _mapped_spines(*fields: str) -> list[int]:
        pages: set[int] = set()
        for field in fields:
            pages.update(_page_units(quality_info.get(field)))
        return sorted({page_to_spine[page] for page in pages if page in page_to_spine})

    text_spines: set[int] = set()
    for _chunk_id, text, metadata in chunks:
        if not str(text or "").strip():
            continue
        try:
            spine = int(metadata.get("chapter_index")) + 1
        except (TypeError, ValueError):
            continue
        if spine in expected_spines:
            text_spines.add(spine)

    profile = dict(epub_profile or quality_info.get("epub_profile") or {})
    profile.setdefault("classification", "fixed_layout_image")
    profile.setdefault("spine_document_count", len(expected_spines))
    updated = dict(quality_info)
    updated.pop("source_coverage", None)  # derivative PDF coverage has different units
    updated.update({
        "total_pages": len(expected_spines),
        "spine_document_count": len(expected_spines),
        "expected_spines": expected_spines,
        "attempted_spines": sorted({
            page_to_spine[page] for page in attempted_pages if page in page_to_spine
        }),
        "text_spines": sorted(text_spines),
        "blank_spines": _mapped_spines("blank_pages", "empty_pages"),
        "failed_spines": _mapped_spines(
            "missing_pages", "extraction_failure_pages", "invalid_json_pages",
        ),
        "epub_profile": profile,
    })
    updated["source_coverage"] = coverage_from_extraction("epub", chunks, updated)
    return updated


def _reset_rebuild_target() -> None:
    """Reset only the isolated V3 target; the retired data plane is untouched."""
    import chromadb

    client = chromadb.PersistentClient(path=str(paths().chroma_dir))
    target = str(paths().collection_name or "zotero_paragraphs_v3")
    listed = client.list_collections()
    names = {
        str(value if isinstance(value, str) else getattr(value, "name", ""))
        for value in listed
    }
    # Paragraphs and every searchable summary layer form one rebuild
    # generation.  Retaining an old ``__sum_*`` collection would route new
    # searches through node/item IDs from the previous corpus.
    targets = (
        target,
        f"{target}__sum_node",
        f"{target}__sum_item",
        f"{target}__sum_section",
    )
    try:
        for collection_name in targets:
            if collection_name in names:
                # Fail closed: manifest/config/FTS still describe the existing
                # generation until Chroma confirms all target collections are gone.
                client.delete_collection(collection_name)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    reset_ingestion_derived_state()
    if paths().manifest_path.exists():
        paths().manifest_path.unlink()
    if paths().pipeline_config_path.exists():
        paths().pipeline_config_path.unlink()
    lexical_v3 = lexical_path(PROJECT_ROOT)
    if lexical_v3.exists():
        lexical_v3.unlink()

def _load_reocr_routes(path: Path | None, limit: int | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("candidates", [])
    if not isinstance(rows, list):
        raise ValueError("re-OCR candidate JSON must contain a candidates array")
    selected = rows[:limit] if limit is not None else rows
    return {
        str(row.get("attachment_key")): row
        for row in selected if isinstance(row, dict) and row.get("attachment_key")
    }


def _zotero_data_dir_is_valid(zotero_data_dir: Optional[str]) -> bool:
    if not zotero_data_dir:
        return False
    zdd = Path(zotero_data_dir).expanduser()
    return bool(zdd.exists() and (zdd / "storage").exists() and (zdd / "zotero.sqlite").exists())


def _validate_zotero_data_dir_or_exit():
    if not paths().zotero_data_dir:
        raise SystemExit(
            "ERROR: ZOTERO_DATA_DIR is not set.\n"
            "Set it to your Zotero data directory (must contain 'storage/' and 'zotero.sqlite').\n"
        )
    zdd = Path(paths().zotero_data_dir).expanduser()
    if not (zdd.exists() and (zdd / "storage").exists() and (zdd / "zotero.sqlite").exists()):
        raise SystemExit(
            f"ERROR: ZOTERO_DATA_DIR looks invalid: {zdd}\n"
            "Expected to find 'storage/' and 'zotero.sqlite' inside it.\n"
        )


def _apply_rag_tag_policy(
    attachments: Iterable[ZoteroAttachment],
) -> tuple[list[ZoteroAttachment], list[ZoteroAttachment], list[ZoteroAttachment]]:
    """Split indexable and explicitly excluded Zotero attachments.

    ``rag:exclude`` is attachment-local: putting it on a parent must not
    accidentally suppress every format. ``rag:prefer-epub`` is parent-local
    marks PDF siblings as retirement candidates. Candidates stay indexable
    until a sibling EPUB has an extraction committed in the manifest.
    """
    rows = list(attachments)
    epub_parents = {
        str(row.parentItemKey)
        for row in rows
        if row.parentItemKey and row.source_type == "epub"
        and RAG_EXCLUDE_TAG not in set(getattr(row, "tags", ()))
    }
    included: list[ZoteroAttachment] = []
    excluded: list[ZoteroAttachment] = []
    preferred_pdfs: list[ZoteroAttachment] = []
    for row in rows:
        tags = set(getattr(row, "tags", ()))
        parent_tags = set(getattr(row, "parentTags", ()))
        explicitly_excluded = RAG_EXCLUDE_TAG in tags
        prefer_epub_pdf = (
            row.source_type == "pdf"
            and bool(row.parentItemKey)
            and str(row.parentItemKey) in epub_parents
            and RAG_PREFER_EPUB_TAG in parent_tags
        )
        if explicitly_excluded:
            excluded.append(row)
        else:
            included.append(row)
            if prefer_epub_pdf:
                preferred_pdfs.append(row)
    return included, excluded, preferred_pdfs


def _ready_preferred_pdfs(
    candidates: Iterable[ZoteroAttachment],
    inventory: Iterable[ZoteroAttachment],
    files_manifest: dict[str, dict[str, Any]],
) -> list[ZoteroAttachment]:
    """PDF candidates whose usable EPUB sibling is durably committed."""
    committed_epub_parents = {
        str(row.parentItemKey)
        for row in inventory
        if row.parentItemKey and row.source_type == "epub"
        and RAG_EXCLUDE_TAG not in set(getattr(row, "tags", ()))
        and row.attachmentKey in files_manifest
    }
    return [
        row for row in candidates
        if row.parentItemKey and str(row.parentItemKey) in committed_epub_parents
    ]


async def _discover_attachments(
    args: argparse.Namespace,
    *,
    api: ZoteroLocalAPI,
    manifest: dict[str, Any],
    files_manifest: dict[str, dict[str, Any]],
    reocr_routes: dict[str, dict[str, Any]],
    show_progress: bool,
) -> DiscoveryResult:
    """Resolve the immutable source scope before any destructive index action."""
    zotero_data_dir: Optional[str] = None
    if _zotero_data_dir_is_valid(paths().zotero_data_dir):
        zotero_data_dir = paths().zotero_data_dir
    elif paths().zotero_data_dir:
        if args.require_data_dir:
            _validate_zotero_data_dir_or_exit()
        else:
            print(
                f"[WARN] ZOTERO_DATA_DIR looks invalid: {Path(paths().zotero_data_dir).expanduser()}\n"
                "      Falling back to Zotero Local API file download into PDF_CACHE_DIR.",
                file=sys.__stderr__,
            )
    elif args.require_data_dir:
        _validate_zotero_data_dir_or_exit()
    else:
        print(
            "[WARN] ZOTERO_DATA_DIR is not set. Falling back to Zotero Local API "
            "file download into PDF_CACHE_DIR.",
            file=sys.__stderr__,
        )

    if show_progress:
        print(
            "[PROGRESS] Fetching attachment metadata from Zotero (this may take a minute)...",
            file=sys.__stderr__,
        )
    attachments: List[ZoteroAttachment] = await api.list_normalized_attachments(
        zotero_data_dir=zotero_data_dir,
        pdf_cache_dir=str(paths().pdf_cache_dir),
        collection_key=args.collection,
        require_complete=bool(args.rebuild),
    )
    attachments = [a for a in attachments if getattr(a, "pdf_path", None)]
    inventory_attachments = list(attachments)
    attachments, excluded_attachments, preferred_pdf_attachments = (
        _apply_rag_tag_policy(attachments)
    )
    already_ready_pdfs = _ready_preferred_pdfs(
        preferred_pdf_attachments, inventory_attachments, files_manifest,
    )
    already_ready_keys = {row.attachmentKey for row in already_ready_pdfs}
    if already_ready_keys:
        attachments = [
            row for row in attachments if row.attachmentKey not in already_ready_keys
        ]
        excluded_attachments.extend(already_ready_pdfs)
        preferred_pdf_attachments = [
            row for row in preferred_pdf_attachments
            if row.attachmentKey not in already_ready_keys
        ]
    preflight_notes = None
    if args.rebuild:
        preflight_notes = await api.list_notes(
            collection_key=args.collection, require_complete=True,
        )

    attachments = _select_attachment_scope(attachments, args.attachment)
    attachments = _select_item_scope(attachments, args.item, 0)
    if args.retry_failed:
        inflight = {
            str(value) for value in manifest.get("inflight_attachments", []) if value
        }
        retryable_items = {
            str(a.parentItemKey or a.attachmentKey) for a in attachments
            if a.attachmentKey in inflight
            or _retryable_failed(str(a.parentItemKey or a.attachmentKey))
        }
        attachments = [
            a for a in attachments
            if str(a.parentItemKey or a.attachmentKey) in retryable_items
        ]
    if args.source_type:
        keep_type = str(args.source_type)
        attachments = [
            a for a in attachments
            if _resolve_source_type(
                getattr(a, "contentType", None), Path(getattr(a, "pdf_path", "")),
                getattr(a, "source_type", None),
            ) == keep_type
        ]
    if reocr_routes:
        attachments = [a for a in attachments if a.attachmentKey in reocr_routes]
    if args.reparse_corrupted:
        attachments = [
            a for a in attachments
            if (
                (previous := files_manifest.get(a.attachmentKey))
                and isinstance(previous.get("quality"), dict)
                and (
                    previous["quality"].get("is_scanned")
                    or previous["quality"].get("is_corrupted")
                )
                and previous["quality"].get("parser") != "docling"
            )
        ]
    return DiscoveryResult(
        attachments=attachments,
        preflight_notes=preflight_notes,
        excluded_attachments=excluded_attachments,
        inventory_attachments=inventory_attachments,
        preferred_pdf_attachments=preferred_pdf_attachments,
    )


async def _index_notes_phase(
    args: argparse.Namespace,
    *,
    api: ZoteroLocalAPI,
    col: Any,
    notes_manifest: dict[str, dict[str, Any]],
    preflight_notes: list[dict[str, Any]] | None,
    partial_scope: bool,
    processing_item_keys: set[str],
    attachments: list[ZoteroAttachment],
    show_progress: bool,
) -> NoteIndexOutcome:
    """Index the note inventory using the same preflight snapshot as rebuild."""
    try:
        notes = (
            preflight_notes
            if preflight_notes is not None
            else await api.list_notes(collection_key=args.collection)
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to enumerate Zotero notes; refusing to treat an unknown inventory as empty: {exc}"
        ) from exc

    if partial_scope:
        scoped_item_keys = set(processing_item_keys) if args.limit else {
            str(attachment.parentItemKey or attachment.attachmentKey)
            for attachment in attachments
        }
        notes = [
            note for note in notes
            if str(note.get("parentItemKey") or "") in scoped_item_keys
        ]

    updated_manifest, stats = index_notes(
        notes,
        col=col,
        notes_manifest=notes_manifest,
        batch_size=UPSERT_BATCH_SIZE,
        show_progress=show_progress,
        dedupe_fn=_dedupe_by_id,
        upsert_fn=_upsert_in_subbatches,
        lexical_delete_fn=delete_lexical_note,
        delete_stale=not partial_scope,
        strict_lexical=True,
    )
    return NoteIndexOutcome(
        manifest=updated_manifest,
        updated=int(stats.get("updated_notes", 0)),
        skipped=int(stats.get("skipped_notes", 0)),
        deleted_stale=int(stats.get("deleted_stale_notes", 0)),
    )


def _finalize_index_storage(
    col: Any,
    *,
    manifest: dict[str, Any],
    files_manifest: dict[str, dict[str, Any]],
    notes_manifest: dict[str, dict[str, Any]],
    last_written_id: str | None,
) -> str | None:
    """Flush and validate the physical index before committing final manifests."""
    try:
        if last_written_id is None:
            sample = col.get(limit=1, include=[])
            sample_ids = sample.get("ids") or []
            last_written_id = str(sample_ids[0]) if sample_ids else None
        _flush_and_verify_hnsw(col, last_written_id)
        manifest["hnsw_validated"] = True
        print("[INFO] HNSW final flush and query smoke test complete.", file=sys.__stderr__)
    except Exception as exc:
        manifest["hnsw_validated"] = False
        save_manifest(paths().manifest_path, manifest)
        raise RuntimeError(f"HNSW final validation failed: {exc}") from exc
        print(f"[WARN] HNSW final flush failed (non-fatal): {exc}", file=sys.__stderr__)

    manifest["notes"] = notes_manifest
    manifest["files"] = files_manifest
    save_manifest(paths().manifest_path, manifest)
    return last_written_id


def _quality_warnings(
    files_manifest: dict[str, dict[str, Any]],
) -> list[QualityWarning]:
    """Build structured warnings for unresolved extraction-quality damage."""
    warnings: list[QualityWarning] = []
    for attachment_key, entry in files_manifest.items():
        quality = entry.get("quality")
        if not isinstance(quality, dict):
            continue
        is_scanned = bool(quality.get("is_scanned"))
        is_corrupted = bool(quality.get("is_corrupted"))
        scanned_pages = quality.get("scanned_pages") or []
        corrupted_pages = quality.get("corrupted_pages") or []
        coverage_gap = (
            quality.get("source_coverage_shortfall")
            if quality.get("source_coverage_adopted")
            and isinstance(quality.get("source_coverage_shortfall"), dict)
            else None
        )
        if not (is_scanned or is_corrupted or scanned_pages or corrupted_pages or coverage_gap):
            continue
        reasons: list[str] = []
        if is_scanned:
            reasons.append("scanned/empty pages")
        elif scanned_pages:
            reasons.append(f"{len(scanned_pages)} unresolved scanned page(s) (within tolerance)")
        if is_corrupted:
            reasons.append("text layout/character encoding corruption")
        elif corrupted_pages:
            reasons.append(
                f"{len(corrupted_pages)} unresolved corrupted page(s) (within tolerance)"
            )
        if coverage_gap:
            reasons.append(
                "indexed with partial source coverage "
                f"({coverage_gap.get('accounted_units')}/{coverage_gap.get('expected_units')} "
                f"{coverage_gap.get('unit_kind')}s, tagged quality-uncertain)"
            )
        warnings.append(QualityWarning(
            attachment_key=attachment_key,
            title=str(entry.get("title") or str(entry.get("pdf_path") or "").split("/")[-1]
                      or attachment_key),
            reasons=tuple(reasons),
            quality=quality,
        ))
    return warnings


def _print_quality_warnings(warnings: list[QualityWarning]) -> None:
    if not warnings:
        return
    print("\n" + "=" * 80, file=sys.__stderr__)
    print(
        "⚠️  [RAG QUALITY WARNING] The following files might have poor retrieval quality:",
        file=sys.__stderr__,
    )
    for warning in warnings:
        print(f'  - [{warning.attachment_key}] "{warning.title}"', file=sys.__stderr__)
        print(f"    ↳ Issue: {' & '.join(warning.reasons)}", file=sys.__stderr__)
        if warning.quality.get("scanned_pages"):
            print(
                f"      scanned/empty pages: {warning.quality.get('scanned_pages')}",
                file=sys.__stderr__,
            )
        if warning.quality.get("corrupted_pages"):
            print(
                f"      corrupted/garbled pages: {warning.quality.get('corrupted_pages')}",
                file=sys.__stderr__,
            )
    print(
        "\nRecommendation: We highly recommend processing these files using high-fidelity",
        file=sys.__stderr__,
    )
    print(
        "AI-based document layout parsers like Docling or Marker to improve RAG accuracy.",
        file=sys.__stderr__,
    )
    print("=" * 80 + "\n", file=sys.__stderr__)


def _resolve_attachment_source(attachment: ZoteroAttachment) -> ResolvedAttachmentSource | None:
    """Resolve a Zotero attachment to one stable file and source classification."""
    file_path = Path(attachment.pdf_path).expanduser()
    if file_path.is_dir():
        preferred = [file_path / name for name in ("index.html", "index.htm")]
        selected = next((path for path in preferred if path.is_file()), None)
        if selected is None:
            html_files = sorted(
                path for path in file_path.iterdir()
                if path.is_file() and path.suffix.casefold() in {".html", ".htm"}
            )
            selected = html_files[0] if html_files else None
        if selected is None:
            return None
        file_path = selected
    if not file_path.exists():
        return None

    content_type = getattr(attachment, "contentType", None)
    source_type = getattr(attachment, "source_type", None) or "pdf"
    if content_type == "application/epub+zip" or file_path.suffix.casefold() == ".epub":
        source_type = "epub"
    elif file_path.suffix.casefold() in {".html", ".htm"}:
        source_type = "html"
    elif source_type not in {"pdf", "html", "epub"}:
        source_type = "pdf"
    stat = file_path.stat()
    return ResolvedAttachmentSource(
        path=file_path,
        source_type=source_type,
        mtime=float(stat.st_mtime),
        size=int(stat.st_size),
    )


def _reparse_decision(
    args: argparse.Namespace,
    *,
    source_type: str,
    previous: dict[str, Any] | None,
    reocr_route: dict[str, Any] | None,
) -> ReparseDecision:
    """Choose explicit parser overrides without performing extraction."""
    has_quality = bool(previous and isinstance(previous.get("quality"), dict))
    target_engine = str((reocr_route or {}).get("target_engine") or "").strip()
    force_mistral = bool(reocr_route and target_engine == "mistral_ocr")
    force_ndlocr = False
    force_docling = False
    if source_type == "pdf":
        if reocr_route:
            force_ndlocr = not force_mistral and str(reocr_route.get("lang") or "") == "ja"
            force_docling = not force_mistral and not force_ndlocr
        elif args.use_docling:
            force_docling = not (
                has_quality and previous["quality"].get("parser") == "docling"
            )
        elif args.reparse_corrupted and has_quality:
            quality = previous["quality"]
            force_docling = bool(
                (quality.get("is_scanned") or quality.get("is_corrupted"))
                and quality.get("parser") != "docling"
            )
    return ReparseDecision(
        force_docling=force_docling,
        force_ndlocr=force_ndlocr,
        force_mistral=force_mistral,
    )







def _extract_pdf_override(
    *,
    attachment_key: str,
    file_path: Path,
    meta_base: dict[str, Any],
    force_ndlocr: bool,
    use_docling: bool,
    docling_worker: Any,
    show_progress: bool,
) -> PdfExtraction | None:
    """Run an explicitly selected PDF parser, or leave normal routing alone."""
    if force_ndlocr:
        if show_progress:
            print(
                "[PROGRESS]   ↳ parsing Japanese re-OCR candidate with NDLOCR-Lite...",
                file=sys.__stderr__,
            )
        chunks, quality = extract_chunks_from_pdf_with_ndlocr(
            file_path, attachment_key, meta_base,
        )
        return PdfExtraction(chunks, quality)
    if not use_docling:
        return None
    if show_progress:
        print(
            "[PROGRESS]   ↳ parsing with high-fidelity IBM Docling...",
            file=sys.__stderr__,
        )
    try:
        chunks, quality = docling_worker.extract(
            file_path, attachment_key, meta_base,
        )
    except RuntimeError as exc:
        print(
            f"[WARN] Docling worker extraction failed: attachment={attachment_key} err={exc}",
            file=sys.__stderr__,
        )
        chunks, quality = [], {}
    # PyTorch/MPS-CUDA cache clearing happens inside the worker subprocess
    # (see docling_worker._worker_loop), which owns the torch state.
    return PdfExtraction(chunks, quality)


def _pdf_gate_plan(
    *,
    structure_recovery: bool,
    scanned_batch_defer: bool,
    chunks_present: bool,
    attempted_local_ocr: bool,
    fast_path_accepted: bool,
    queue_enabled: bool,
    total_pages: int,
    minimum_pages: int,
    initial_route_reason: str = "",
) -> PdfGatePlan:
    """Choose the post-extraction PDF action without mutating any store."""
    if not structure_recovery:
        return PdfGatePlan("disabled")
    needs_gate = (
        scanned_batch_defer
        or not chunks_present
        or (not attempted_local_ocr and not fast_path_accepted)
    )
    if not needs_gate:
        return PdfGatePlan("keep")
    local_ocr_exhausted = attempted_local_ocr and not chunks_present
    if scanned_batch_defer:
        return PdfGatePlan(
            "defer",
            local_ocr_exhausted=local_ocr_exhausted,
            policy_reason=initial_route_reason,
        )
    if queue_enabled and (
        local_ocr_exhausted
        or (not attempted_local_ocr and total_pages >= minimum_pages)
    ):
        return PdfGatePlan(
            "defer",
            local_ocr_exhausted=local_ocr_exhausted,
            policy_reason="mistral_batch_queue",
        )
    if local_ocr_exhausted:
        return PdfGatePlan("local_ocr_exhausted", local_ocr_exhausted=True)
    return PdfGatePlan("docling_escalation")


def _pdf_gate_plan_for_extraction(
    *,
    structure_recovery: bool,
    scanned_batch_defer: bool,
    chunks: list,
    attempted_local_ocr: bool,
    quality: dict[str, Any],
    total_pages: int,
    minimum_pages: int,
    initial_route_reason: str,
) -> PdfGatePlan:
    """Adapt current extractor state and feature flags to the pure gate plan."""
    fast_path_accepted = False
    if (
        structure_recovery
        and not scanned_batch_defer
        and chunks
        and not attempted_local_ocr
    ):
        fast_path_accepted = pymupdf_fast_path_passes(quality)
    needs_queue_check = (
        structure_recovery
        and not scanned_batch_defer
        and (not chunks or (not attempted_local_ocr and not fast_path_accepted))
    )
    return _pdf_gate_plan(
        structure_recovery=structure_recovery,
        scanned_batch_defer=scanned_batch_defer,
        chunks_present=bool(chunks),
        attempted_local_ocr=attempted_local_ocr,
        fast_path_accepted=fast_path_accepted,
        queue_enabled=mistral_batch_queue_enabled() if needs_queue_check else False,
        total_pages=total_pages,
        minimum_pages=minimum_pages,
        initial_route_reason=initial_route_reason,
    )


def _extract_pdf_chunks(
    *,
    a: Any,
    args: Any,
    col: Any,
    docling_worker: Any,
    file_path: Any,
    files_manifest: Any,
    force_docling: Any,
    force_ndlocr: Any,
    granite_worker: Any,
    manifest: Any,
    meta_base: Any,
    mtime: Any,
    prev: Any,
    scope_item_key: Any,
    show_progress: Any,
    size: Any,
    source_metadata: Any,
    stored_signature: Any,
    structure_recovery: Any,
    v3_pipeline_fingerprint: Any,
) -> PdfExtraction:
    """Read a PDF, choosing and escalating between extractors as it goes.

    Six hundred and sixty-five lines of main_async's per-attachment loop, which
    is most of what that loop was: PyMuPDF first, then the OCR-layer audit, then
    Docling or NDLOCR or Granite depending on what the pages turn out to be,
    then the coverage checks that decide whether what came back is enough to
    index.

    Lifted whole rather than in pieces because the interface is narrow --
    ``chunks`` and ``quality`` are the only names that escaped, and the one
    ``continue`` is now the deferred flag -- and because its own size is easier
    to see, and to split further, once it is somewhere of its own.

    The parameter list is long and deliberately unreduced. Grouping the per-run
    values apart from the per-attachment ones is the obvious next move and a
    separate one: doing both at once would leave no way to tell which change
    caused a difference.
    """
    override = _extract_pdf_override(
        attachment_key=a.attachmentKey,
        file_path=file_path,
        meta_base=meta_base,
        force_ndlocr=bool(force_ndlocr),
        use_docling=bool(force_docling or args.use_docling),
        docling_worker=docling_worker,
        show_progress=bool(show_progress),
    )
    if override is not None:
        chunks, quality_info = override.chunks, override.quality
    else:
        # Approved routing (evaluations/ocr_bakeoff_v3/results/routing_proposal.md):
        # PyMuPDF is only canonical for embedded-text PDFs with a usable
        # outline; everything else escalates to Docling, the higher-scoring
        # local default. Legacy (non-V3) ingestion keeps its prior behavior.
        chunks, quality_info = extract_chunks_from_pdf(file_path, a.attachmentKey, meta_base)
        quality_info = _attach_pdf_source_provenance(quality_info, source_metadata)
        recovered = None
        total_pages = int(quality_info.get("total_pages") or 0)
        minimum_pages = int(os.environ.get("PDF_AI_TOC_MIN_PAGES", "30"))
        attempted_local_ocr = False
        scanned_ocr_replacement_attempted = False
        scanned_ocr_batch_defer = False
        # A pre-existing OCR layer must reach stage 2 below before it
        # can be replaced. Only a scan classified as having *no* text
        # layer is safe to route immediately.
        initial_scan_route, initial_scan_route_reason = _initial_scanned_pdf_ocr_route(
            quality_info, total_pages=total_pages, item_key=scope_item_key,
        )
        if initial_scan_route == "mistral_batch":
            # The Batch queue owns cloud submission and later adoption.
            # Do not run any local OCR before preserving this non-canonical
            # deferral for a long scan without a usable OCR layer.
            scanned_ocr_batch_defer = True
            scanned_ocr_replacement_attempted = True
            chunks = []
        elif initial_scan_route in {"docling", "granite"}:
            scanned_ocr_replacement_attempted = True
            attempted_local_ocr = True
            if show_progress:
                print(
                    f"[PROGRESS]   ↳ scan OCR replacement: parsing with "
                    f"{initial_scan_route} ({initial_scan_route_reason})...",
                    file=sys.__stderr__,
                )
            try:
                chunks, quality_info = _structure_with_engine(
                    initial_scan_route, file_path, a.attachmentKey, meta_base,
                    docling_worker=docling_worker, granite_worker=granite_worker,
                )
                quality_info = _attach_pdf_source_provenance(quality_info, source_metadata)
                quality_info["ocr_layer_quality"] = "not_applicable"
                quality_info["ocr_layer_audit_reason"] = (
                    "not_applicable_no_ocr_layer"
                )
            except RuntimeError as exc:
                print(
                    f"[WARN] Docling worker extraction failed: attachment={a.attachmentKey} err={exc}",
                    file=sys.__stderr__,
                )
                chunks, quality_info = [], {}
        elif not chunks and total_pages > 0:
            # RapidOCR/NDLOCR are retained for fixed-layout EPUBs and
            # explicit re-OCR overrides, but are not an ordinary PDF
            # route. Docling is the local PDF baseline.
            attempted_local_ocr = True
            if show_progress:
                print(
                    "[PROGRESS]   ↳ no usable text layer; parsing with IBM Docling...",
                    file=sys.__stderr__,
                )
            try:
                chunks, quality_info = docling_worker.extract(
                    file_path, a.attachmentKey, meta_base,
                )
                quality_info = _attach_pdf_source_provenance(quality_info, source_metadata)
            except RuntimeError as exc:
                print(
                    f"[WARN] Docling worker extraction failed: attachment={a.attachmentKey} err={exc}",
                    file=sys.__stderr__,
                )
                chunks, quality_info = [], {}
        # Scanned/image-only pages inside an otherwise clean embedded-text
        # PDF (figure plates, a scanned dedication page, etc.) currently
        # fail-closed the PyMuPDF fast-path and AI-TOC gates entirely,
        # sending the whole document to Mistral OCR even when most pages
        # are fine. Patch those pages via a Docling sub-PDF pass first
        # (E2c, dev-notes/current/77). Gate on ``is_scanned`` (ratio>=0.8,
        # a genuine scan job where local text-page patching wouldn't help)
        # rather than a raw scanned-page count: the page-level "attempted
        # = resolved" semantics above already prevent garbage/index
        # pollution regardless of how many pages that covers, so a count
        # cap was never actually protecting anything (user decision
        # 2026-07-25, following the M5TQ4HLZ case: a 127-page catalog
        # with 60 genuine figure-plate pages was previously escalated
        # whole to Mistral OCR and rejected outright by its repeat-
        # artifact gate).
        #
        # Restricted to born-digital documents (note 79): the "no text
        # here means a figure" reading only holds when the text layer is
        # the typeset source. In a scan the same observation means OCR
        # failed on that page, and marking it ``figure`` would record a
        # gap as an illustration -- those pages go to the scan-derived
        # page repair below instead.
        scanned_pages = list(quality_info.get("scanned_pages") or [])
        source_class = str(quality_info.get("source_class") or "")
        text_layer_is_authoritative = (
            source_class == BORN_DIGITAL if source_class
            else not quality_info.get("is_scanned")
        )
        if (
            chunks and scanned_pages and not attempted_local_ocr
            and structure_recovery
            and os.environ.get("PDF_SCANNED_PAGE_PATCH_ENABLE", "1").strip() == "1"
            and text_layer_is_authoritative
        ):
            try:
                from docling_extract import patch_scanned_pages_with_docling
            except ImportError:  # pragma: no cover - direct src entrypoint
                from .docling_extract import patch_scanned_pages_with_docling
            try:
                patched, attempted_pages = patch_scanned_pages_with_docling(
                    file_path, scanned_pages, attachment_key=a.attachmentKey, meta_base=meta_base,
                    worker=docling_worker,
                )
            except Exception as exc:
                patched, attempted_pages = [], set()
                if show_progress:
                    print(
                        f"[WARN] scanned-page Docling patch failed: attachment={a.attachmentKey} err={exc}",
                        file=sys.__stderr__,
                    )
            if attempted_pages:
                # Every attempted page is resolved for the scanned-ratio
                # gate, whether Docling recovered text or concluded the
                # page has none (a figure/photo/poster plate): forcing
                # more OCR on non-text content risks index-polluting
                # garbage rather than recovering anything real.
                # P5 (note 78): splice in reading order, not by appending.
                chunks = _sort_chunks_in_reading_order(list(chunks) + patched)
                quality_info = recompute_scanned_quality_after_patch(
                    quality_info, attempted_pages, total_pages,
                )
                if show_progress:
                    print(
                        f"[PROGRESS]   ↳ scanned-page Docling patch: {len(patched)} text chunk(s) "
                        f"recovered from {len(attempted_pages)} page(s) "
                        f"({len(quality_info['scanned_pages'])} still unattempted)",
                        file=sys.__stderr__,
                    )
        # Scan-derived text: the same page-level repair, but a page
        # without usable text is an OCR failure rather than a figure
        # (note 79, U2). The document-level local-OCR gate is all-or-
        # nothing, so before this there was no way to fix a scan whose
        # OCR failed on only part of its pages -- the whole document had
        # to be re-OCRed or accepted as-is. Marker chunks for pages that
        # stay unrecovered say ``corrupted_unresolved``, never
        # ``figure``.
        if (
            chunks and str(quality_info.get("source_class") or "") in SCAN_DERIVED_CLASSES
            and structure_recovery
            and os.environ.get("PDF_SCAN_PAGE_REPAIR_ENABLE", "1").strip() == "1"
        ):
            failed_pages = _scan_pages_needing_repair(chunks, total_pages)
            if failed_pages:
                try:
                    from docling_extract import patch_corrupted_pages_with_docling
                except ImportError:  # pragma: no cover - direct src entrypoint
                    from .docling_extract import patch_corrupted_pages_with_docling
                try:
                    patched, attempted_pages = patch_corrupted_pages_with_docling(
                        file_path, failed_pages,
                        attachment_key=a.attachmentKey, meta_base=meta_base,
                        worker=docling_worker,
                        # This uses the corrupted-page marker semantics
                        # for scan-derived OCR failures, but is a
                        # separate repair provenance from the later
                        # text-corruption pass below.  Keep their IDs
                        # disjoint when both target the same page.
                        chunk_namespace="scanrepair",
                    )
                except Exception as exc:
                    patched, attempted_pages = [], set()
                    if show_progress:
                        print(
                            f"[WARN] scan-derived page repair failed: "
                            f"attachment={a.attachmentKey} err={exc}",
                            file=sys.__stderr__,
                        )
                if attempted_pages:
                    kept = [
                        row for row in chunks
                        if int((row[2] or {}).get("page") or 0) not in attempted_pages
                    ]
                    chunks = _sort_chunks_in_reading_order(kept + patched)
                    # Unlike the scanned-page patch above (which calls
                    # recompute_scanned_quality_after_patch), nothing
                    # here reconciled empty_pages/blank_pages with the
                    # pages this just recovered -- see
                    # recompute_blank_pages_after_patch's docstring
                    # (found 2026-08-02, diagnosing YX3MMS4D page 444).
                    quality_info = recompute_blank_pages_after_patch(
                        quality_info, attempted_pages,
                    )
                    if show_progress:
                        print(
                            f"[PROGRESS]   ↳ scan-derived page repair: {len(patched)} chunk(s) "
                            f"from {len(attempted_pages)} failed OCR page(s)",
                            file=sys.__stderr__,
                        )
        # Text-corrupted pages (font-encoding mismatch or OCR/linguistic
        # noise already baked into the PDF's text layer -- see
        # pdf_extract.analyze_text_quality's is_corrupted) inside an
        # otherwise clean PDF get the same "patch just the bad pages"
        # treatment as scanned pages above (E2d, dev-notes/current/77,
        # user decision 2026-07-26). Unlike scanned pages, corrupted
        # pages already produced (garbled) chunks upstream, so those
        # must be dropped before splicing in the Docling-recovered
        # replacements -- otherwise both the garbage and the fix would
        # be indexed side by side. Gated on ``is_corrupted`` (ratio
        # >=0.6, a genuinely corrupted document where per-page local
        # patching wouldn't help) rather than a raw corrupted-page
        # count, mirroring the scanned-page patch's reasoning: the
        # page-level "attempted = resolved" semantics below already
        # prevent garbage/index pollution regardless of how many pages
        # that covers.
        corrupted_pages = list(quality_info.get("corrupted_pages") or [])
        if (
            chunks and corrupted_pages and not attempted_local_ocr
            and structure_recovery
            and os.environ.get("PDF_CORRUPTED_PAGE_PATCH_ENABLE", "0").strip() == "1"
            and not quality_info.get("is_corrupted")
        ):
            try:
                from docling_extract import patch_corrupted_pages_with_docling
            except ImportError:  # pragma: no cover - direct src entrypoint
                from .docling_extract import patch_corrupted_pages_with_docling
            try:
                corrupt_patched, corrupt_attempted_pages = patch_corrupted_pages_with_docling(
                    file_path, corrupted_pages, attachment_key=a.attachmentKey, meta_base=meta_base,
                    worker=docling_worker,
                )
            except Exception as exc:
                corrupt_patched, corrupt_attempted_pages = [], set()
                if show_progress:
                    print(
                        f"[WARN] corrupted-page Docling patch failed: attachment={a.attachmentKey} err={exc}",
                        file=sys.__stderr__,
                    )
            if corrupt_attempted_pages:
                # Drop the pre-patch (garbled) chunks for every attempted
                # page first -- the patch output (recovered text or the
                # corrupted_unresolved marker) is their full replacement,
                # not an addition alongside the original mojibake.
                chunks = [
                    (cid, text, md) for cid, text, md in chunks
                    if int(md.get("page") or 0) not in corrupt_attempted_pages
                ]
                # P5 (note 78): splice in reading order, not by appending.
                chunks = _sort_chunks_in_reading_order(list(chunks) + corrupt_patched)
                quality_info = recompute_corrupted_quality_after_patch(
                    quality_info, corrupt_attempted_pages, total_pages,
                )
                if show_progress:
                    print(
                        f"[PROGRESS]   ↳ corrupted-page Docling patch: {len(corrupt_patched)} chunk(s) "
                        f"recovered from {len(corrupt_attempted_pages)} page(s) "
                        f"({len(quality_info['corrupted_pages'])} still unresolved)",
                        file=sys.__stderr__,
                    )
        # Stage 2 (note 79): measure the OCR quality of a scan-derived
        # text layer before deciding whether it can stand as canonical.
        # Runs after page repair so it audits the text we would actually
        # index, and before the fast-path gate, which consults its
        # verdict. A cached verdict for an unchanged source is reused so
        # reingestion costs no API calls.
        if (
            chunks
            and str(quality_info.get("source_class") or "") in SCAN_DERIVED_CLASSES
            and not scanned_ocr_replacement_attempted
            and not ocr_layer_audit_enabled()
        ):
            # Record that measuring was declined, so the router can tell
            # it apart from an audit that tried and failed. Without this
            # the layer reads as "unverified" and every scanned PDF gets
            # re-OCRed the moment the LLM is switched off (note 80 B).
            quality_info = dict(quality_info)
            quality_info["ocr_layer_audit_reason"] = OCR_LAYER_AUDIT_DISABLED
        elif (
            chunks
            and str(quality_info.get("source_class") or "") in SCAN_DERIVED_CLASSES
            and not scanned_ocr_replacement_attempted
        ):
            cached_audit = _cached_ocr_layer_audit(prev, mtime=mtime, size=size)
            if cached_audit is not None:
                quality_info = dict(quality_info)
                quality_info.update(cached_audit)
                quality_info["ocr_layer_audit_cached"] = True
                if show_progress:
                    print(
                        f"[PROGRESS]   ↳ OCR layer audit cache hit: "
                        f"{cached_audit.get('ocr_layer_quality')} "
                        f"rate={cached_audit.get('ocr_layer_error_rate')}",
                        file=sys.__stderr__,
                    )
            else:
                audit = audit_ocr_text_layer(file_path, scope_item_key)
                quality_info = dict(quality_info)
                quality_info.update(audit)
                if show_progress:
                    print(
                        f"[PROGRESS]   ↳ OCR layer audit: "
                        f"{audit.get('ocr_layer_quality')} "
                        f"rate={audit.get('ocr_layer_error_rate')} "
                        f"({audit.get('ocr_layer_verified_count')} verified / "
                        f"{audit.get('ocr_layer_rejected_count')} rejected) "
                        f"{audit.get('ocr_layer_audit_reason')}",
                        file=sys.__stderr__,
                    )
                # An audit that could not run says nothing about the
                # text, so the text is kept -- but silently keeping it
                # would hide the fact that it was never measured. Each
                # outcome is surfaced differently because each needs a
                # different follow-up (user decision 2026-07-27).
                if audit_was_transient_failure(quality_info):
                    # Not cached, so the next run measures it properly.
                    quality_info["ocr_layer_needs_reaudit"] = True
                    mark_artifact_status(
                        scope_item_key, "extraction", "degraded",
                        attachment_key=a.attachmentKey,
                        reason_code="ocr_layer_audit_deferred",
                        message=str(audit.get("ocr_layer_audit_reason") or "")[:500],
                        retryable=True,
                    )
                elif audit_hit_file_problem(quality_info):
                    # Re-OCR would hit the same unreadable file; this
                    # needs the source repaired, not another attempt.
                    print(
                        f"[WARN] Could not sample {a.attachmentKey} for the OCR "
                        f"audit: {audit.get('ocr_layer_audit_reason')}. The PDF "
                        "itself may be damaged -- repair or replace the file.",
                        file=sys.__stderr__,
                    )
                    mark_artifact_status(
                        scope_item_key, "extraction", "degraded",
                        attachment_key=a.attachmentKey,
                        reason_code="source_file_unreadable",
                        message=str(audit.get("ocr_layer_audit_reason") or "")[:500],
                        retryable=False,
                    )
                elif audit_sample_too_small(quality_info):
                    # Too little text to measure, and too little for a
                    # re-OCR to improve on. Index it, marked unmeasured.
                    chunks, quality_info = _adopt_with_quality_uncertain(
                        chunks, quality_info, reason="ocr_layer_sample_too_small",
                    )
        # An OCR-layer scan is canonical only after the stage-2 audit
        # explicitly accepts it.  Failed/unavailable audits follow the
        # same page-count/cloud policy as a scan with no OCR layer.
        # This is intentionally after the audit and before the generic
        # PDF structural gate, so RapidOCR/NDLOCR cannot slip in first.
        if not scanned_ocr_replacement_attempted:
            scan_route, scan_route_reason = _scanned_pdf_ocr_route(
                quality_info, total_pages=total_pages, item_key=scope_item_key,
            )
            if scan_route == "mistral_batch":
                scanned_ocr_batch_defer = True
                scanned_ocr_replacement_attempted = True
                chunks = []
            elif scan_route in {"docling", "granite"}:
                scanned_ocr_replacement_attempted = True
                attempted_local_ocr = True
                pre_replacement_quality = dict(quality_info)
                if show_progress:
                    print(
                        "[PROGRESS]   ↳ rejected/unverified scan OCR layer; "
                        f"parsing with {scan_route} ({scan_route_reason})...",
                        file=sys.__stderr__,
                    )
                try:
                    chunks, quality_info = _structure_with_engine(
                        scan_route, file_path, a.attachmentKey, meta_base,
                        docling_worker=docling_worker, granite_worker=granite_worker,
                    )
                    quality_info = _attach_pdf_source_provenance(
                        quality_info, source_metadata,
                    )
                    quality_info = _carry_ocr_layer_audit(
                        quality_info, pre_replacement_quality,
                    )
                except RuntimeError as exc:
                    print(
                        f"[WARN] Docling worker extraction failed: "
                        f"attachment={a.attachmentKey} err={exc}",
                        file=sys.__stderr__,
                    )
                    chunks, quality_info = [], {}
        # P1 (note 78): a prior run of this identical source file may
        # already have established that the document has no headings
        # (the LLM itself said so). That verdict is final for the file
        # -- skip the fast path (saving the DeepSeek call and the
        # full-document page sweep) and carry the status forward so
        # the manifest keeps confirming it for the next reingest too.
        prior_no_structure = _prior_no_structure_ai_toc_status(
            prev, mtime=mtime, size=size,
        )
        if (
            chunks and not quality_info.get("has_outline")
            and total_pages >= minimum_pages
            and prior_no_structure
        ):
            quality_info = dict(quality_info)
            quality_info["ai_toc_recovery_status"] = prior_no_structure
            quality_info["ai_toc_recovery_status_cached"] = True
            if show_progress:
                print(
                    f"[PROGRESS]   ↳ AI TOC skipped: prior run confirmed no "
                    f"structure ({prior_no_structure}); indexing unstructured",
                    file=sys.__stderr__,
                )
        elif (
            chunks and not quality_info.get("has_outline")
            and total_pages >= minimum_pages
            and structure_recovery
        ):
            recovered = try_ai_toc_fast_path(
                file_path, scope_item_key, chunks, quality_info,
                docling_worker=docling_worker,
            )
            quality_info = dict(quality_info)
            quality_info["ai_toc_recovery_status"] = (
                "accepted" if recovered.accepted else recovered.reason
            )
            if recovered.diagnostics:
                quality_info["ai_toc_body_coverage"] = recovered.diagnostics.get("body_coverage")
                quality_info["ai_toc_matched_count"] = recovered.diagnostics.get("matched_count")
                quality_info["ai_toc_diagnostics"] = recovered.diagnostics
            if recovered.accepted:
                chunks = recovered.chunks
                if show_progress:
                    print(
                        f"[PROGRESS]   ↳ AI TOC fast path accepted: "
                        f"coverage={recovered.diagnostics.get('body_coverage')} "
                        f"anchors={recovered.diagnostics.get('matched_count')}",
                        file=sys.__stderr__,
                    )
        gate_plan = _pdf_gate_plan_for_extraction(
            structure_recovery=bool(structure_recovery),
            scanned_batch_defer=scanned_ocr_batch_defer,
            chunks=chunks,
            attempted_local_ocr=attempted_local_ocr,
            quality=quality_info,
            total_pages=total_pages,
            minimum_pages=minimum_pages,
            initial_route_reason=initial_scan_route_reason,
        )
        if gate_plan.action == "disabled":
            # Plain-text indexing was requested: PyMuPDF chunks (or the
            # local OCR output above) are the final answer, so the
            # structural gate has nothing to escalate to.
            quality_info = dict(quality_info)
            quality_info["pdf_structure_recovery"] = "disabled"
        elif gate_plan.action != "keep":
            # A document whose local OCR chain (rapidocr/ndlocr →
            # Docling fallback) was tried and rejected by
            # evaluate_local_ocr_gate: local engines are exhausted.
            local_ocr_exhausted = gate_plan.local_ocr_exhausted
            # P7 (note 78): the queue is governed by its own flag
            # alone -- it is not an AI-TOC subfeature, so
            # PDF_AI_TOC_FAST_PATH_ENABLE no longer gates it.
            # P2 (note 78): scanned documents whose local OCR chain
            # was exhausted are exactly the class where Mistral OCR
            # is the strongest engine (bake-off 0.973 vs Docling
            # 0.753), so they may queue regardless of the AI-TOC
            # page minimum; the minimum still applies to the
            # structure-recovery deferrals it was designed for.
            policy_reason = gate_plan.policy_reason
            if gate_plan.action == "defer":
                diagnostics = recovered.diagnostics if recovered is not None else {}
                # P6 (note 78): record the fast-path reason and the
                # AI-TOC rejection reason separately -- the AI-TOC
                # reason used to overwrite the fast-path one even when
                # the fast path was what actually sent the document
                # here. gate_reason keeps its historical precedence
                # (and the counts key "ai_toc_reason" keeps carrying
                # it) for existing queue-listing consumers.
                fast_path_reason = (
                    None if not chunks
                    else pymupdf_fast_path_rejection_reason(quality_info)
                )
                ai_toc_rejection_reason = (
                    recovered.reason
                    if recovered is not None and not recovered.accepted else None
                )
                if scanned_ocr_batch_defer:
                    gate_reason = "scanned_pdf_ocr_replacement"
                elif local_ocr_exhausted:
                    gate_reason = "local_ocr_quality_gate_failed"
                elif not chunks:
                    gate_reason = "pymupdf_no_chunks"
                elif recovered is not None:
                    gate_reason = recovered.reason
                else:
                    gate_reason = fast_path_reason or "pymupdf_fast_path_rejected"
                mark_artifact_status(
                    scope_item_key, "extraction", "blocked",
                    attachment_key=a.attachmentKey,
                    reason_code=MISTRAL_TOC_QUEUE_REASON,
                    message=f"AI TOC/PyMuPDF gate failed: {gate_reason}",
                    retryable=False,
                    source_fingerprint=f"stat:{mtime}:{size}",
                    processor_version=MISTRAL_TOC_QUEUE_PROCESSOR_VERSION,
                    counts={
                        "source_mtime": mtime, "source_size": size,
                        "total_pages": total_pages, "ai_toc_reason": gate_reason,
                        "fast_path_reason": fast_path_reason,
                        "ai_toc_rejection_reason": ai_toc_rejection_reason,
                        "local_ocr_exhausted": local_ocr_exhausted,
                        "ai_toc_diagnostics": diagnostics,
                    },
                    fallback_kind="mistral_ocr",
                )
                inflight = {
                    str(value) for value in manifest.get("inflight_attachments", []) if value
                }
                if a.attachmentKey in inflight:
                    _delete_by_attachment_keys(col, [a.attachmentKey], strict=True)
                    manifest["inflight_attachments"] = [
                        value for value in manifest.get("inflight_attachments", [])
                        if value != a.attachmentKey
                    ]
                # A Mistral deferral is deliberately non-canonical:
                # its current chunks remain searchable until an
                # explicitly adopted Batch result passes the gate.
                # Its stage-2 audit is still final for this exact
                # source fingerprint, so persist just that cache and
                # provenance (not the transient extraction quality).
                files_manifest[a.attachmentKey] = _deferred_manifest_entry(
                    prev, mtime=mtime, size=size, source_path=file_path,
                    title=a.title, quality=quality_info,
                    content_signature_value=stored_signature,
                    pipeline_fingerprint=v3_pipeline_fingerprint,
                )
                save_manifest(paths().manifest_path, manifest)
                if show_progress:
                    print(
                        f"[PROGRESS]   ↳ deferred to Mistral OCR batch: "
                        f"reason={gate_reason}", file=sys.__stderr__,
                    )
                # The attachment is queued, not extracted. The caller counts it
                # and moves on; nothing below this point applies to it.
                return PdfExtraction([], {}, deferred=True)
            if gate_plan.action == "local_ocr_exhausted":
                # P2 (note 78): Docling already ran (and was rejected
                # by evaluate_local_ocr_gate) as run_local_ocr's own
                # fallback for this exact document -- re-running it
                # here would duplicate the work only to adopt, ungated,
                # the same output that was just rejected. With cloud
                # queueing unavailable too, this document is
                # unextractable right now: fall through to the
                # no-chunks failure handling (visible artifact status,
                # existing index left untouched).
                if show_progress:
                    print(
                        "[PROGRESS]   ↳ local OCR exhausted (Docling already "
                        "rejected); not re-running Docling ungated"
                        + (
                            f"; cloud policy requires local processing ({policy_reason})"
                            if policy_reason else "; Mistral queue disabled"
                        ),
                        file=sys.__stderr__,
                    )
            else:
                if show_progress:
                    reason = "produced no chunks" if not chunks else "fast-path gate failed"
                    route_reason = (
                        f"cloud policy requires local processing ({policy_reason})"
                        if policy_reason else "short PDF or Mistral queue disabled"
                    )
                    print(
                        f"[PROGRESS]   ↳ PyMuPDF {reason}; escalating to Docling "
                        f"({route_reason})...", file=sys.__stderr__,
                    )
                # U4 (note 79): keep what the local extractor already
                # produced. If Docling cannot improve on it -- and the
                # cloud route is unavailable, which is why we are here --
                # then discarding this would leave the document out of
                # the index entirely. Indexing it with a quality tag is
                # strictly better than not indexing it at all.
                pre_escalation = (list(chunks), dict(quality_info))
                try:
                    chunks, quality_info = docling_worker.extract(file_path, a.attachmentKey, meta_base)
                    # Docling returns a new quality object; retain the
                    # source classification and already-measured
                    # stage-2 verdict that selected this escalation.
                    quality_info = _attach_pdf_source_provenance(quality_info, source_metadata)
                    prior_quality = pre_escalation[1]
                    for key in _OCR_LAYER_AUDIT_FIELDS:
                        if key in prior_quality:
                            quality_info[key] = prior_quality[key]
                except RuntimeError as exc:
                    print(
                        f"[WARN] Docling worker extraction failed: attachment={a.attachmentKey} err={exc}",
                        file=sys.__stderr__,
                    )
                    chunks, quality_info = [], {}
                if chunks:
                    # P2 (note 78): mirror the local-OCR route's content
                    # checks so this escalation's output is no longer
                    # adopted entirely ungated.
                    acceptable, gate_counts = _docling_escalation_acceptable(chunks)
                    if not acceptable:
                        if show_progress:
                            print(
                                f"[PROGRESS]   ↳ Docling escalation output rejected "
                                f"by minimal quality gate: {gate_counts}",
                                file=sys.__stderr__,
                            )
                        quality_info = dict(quality_info)
                        quality_info["docling_escalation_gate"] = gate_counts
                        chunks = []
                if not chunks and pre_escalation[0]:
                    chunks, quality_info = _adopt_with_quality_uncertain(
                        *pre_escalation,
                        reason=(
                            "docling_escalation_unavailable"
                            if not quality_info else "docling_escalation_rejected"
                        ),
                    )
                    if show_progress:
                        print(
                            "[PROGRESS]   ↳ keeping local extraction with a "
                            f"quality-uncertain tag ({len(chunks)} chunks); "
                            "no better route is available",
                            file=sys.__stderr__,
                        )
    return PdfExtraction(chunks, quality_info)



def _extraction_status(
    attachment: Any,
    *,
    scope_item_key: str,
    source_type: str,
    chunks: list,
    quality: dict[str, Any],
    coverage_adopted: bool,
    coverage_gap: Any,
    truncated: bool,
    ai_toc_alignment_failed: bool,
    degraded_reason: Any,
    degraded_message: Any,
) -> tuple[str, str, dict[str, Any]]:
    """The ledger row for an attachment that produced chunks.

    "success" or "degraded", and the counts a later reader needs to tell which
    kind of degraded it was: a coverage shortfall the run accepted, a truncation,
    or an AI table of contents that would not line up with the text. Retryable
    only for the first two -- an alignment failure repeats.

    Assembled here rather than inline because it is one value with eleven
    inputs, and inline it read as though the surrounding code were still doing
    something.
    """
    return (
        scope_item_key,
        (
            "degraded"
            if (coverage_adopted or truncated or ai_toc_alignment_failed)
            else "success"
        ),
        {
            "attachment_key": attachment.attachmentKey,
            "reason_code": degraded_reason,
            "message": degraded_message,
            "retryable": truncated or coverage_adopted,
            "processor_version": str(quality.get("parser") or source_type),
            "counts": {
                "chunks": len(chunks), "source_type": source_type,
                "processed_pages": quality.get("processed_pages"),
                "expected_pages": quality.get("expected_pages"),
                "pages_without_chunks": _pages_without_chunks(
                    chunks, quality.get("expected_pages") or quality.get("total_pages"),
                ),
                "chars_out": sum(len(text) for _cid, text, _md in chunks),
                **(
                    {"ai_toc_reason": quality.get("ai_toc_recovery_status")}
                    if ai_toc_alignment_failed else {}
                ),
                **(
                    {"source_coverage_shortfall": coverage_gap}
                    if coverage_adopted else {}
                ),
            },
        },
    )


def _report_purge(purge_counts: dict[str, int]) -> None:
    """Say what the relations purge did, and say it differently when refused.

    ``refused`` is a report about the caller's view of the library, not a row
    that was deleted, so summing it into the total printed "Purged removed
    items ... =0, =0, =0" at exactly the moment a purge had been refused.
    """
    refused = int(purge_counts.get("refused") or 0)
    if refused:
        print(
            f"[PROGRESS] Refused to purge {refused} items from relations.db "
            "(see the error above); nothing was deleted.",
            file=sys.__stderr__,
        )
    deleted = sum(value for key, value in purge_counts.items() if key != "refused")
    if deleted > 0:
        print(
            f"[PROGRESS] Purged removed items from relations.db: "
            f"item_citation_status={purge_counts['item_citation_status']}, "
            f"global_citations={purge_counts['global_citations']}, "
            f"global_references={purge_counts['global_references']}",
            file=sys.__stderr__,
        )


def _report_empty_extraction(
    attachment: Any,
    *,
    scope_item_key: str,
    source_type: str,
    file_path: Path,
    quality: dict[str, Any],
    forced_mistral: bool,
    mtime: float,
    size: int,
) -> None:
    """Say why an attachment yielded nothing, and record it as what it is.

    A zero-chunk attachment never enters the manifest, so unless the reason is
    printed here it is unrecoverable afterwards -- the run simply moves on and
    the file looks untouched. extract_chunks_from_html_snapshot puts the reason
    in ``failure_reason`` (html_read_failed, html_size_truncated, no_dom_blocks,
    gibberish); the other extractors set no such key, so it is omitted for them
    rather than printed as a misleading ``reason=None`` (2026-08-02, 3QHTRQN7).

    A Mistral batch that came back with nothing adoptable is blocked rather than
    failed: the candidate is deliberately kept for a later batch, so calling it
    a failure would invite a retry of the one thing already known not to work.
    """
    reason = quality.get("failure_reason")
    print(
        f"[WARN] Extracted 0 chunks; leaving existing index/manifest unchanged: "
        f"attachment={attachment.attachmentKey} type={source_type} file={file_path}"
        + (f" reason={reason}" if reason else ""),
        file=sys.__stderr__,
    )
    if forced_mistral:
        mark_artifact_status(
            scope_item_key, "extraction", "blocked",
            attachment_key=attachment.attachmentKey,
            reason_code=MISTRAL_TOC_QUEUE_REASON,
            message="Mistral OCR batch did not produce adoptable chunks; candidate retained.",
            retryable=False,
            counts={"source_mtime": mtime, "source_size": size},
            fallback_kind="mistral_ocr",
        )
    else:
        mark_artifact_status(
            scope_item_key, "extraction", "failed",
            attachment_key=attachment.attachmentKey, reason_code="no_chunks",
            message=f"No chunks extracted from {source_type} attachment.", retryable=True,
        )


def _progress_line(
    attachment: Any, *, index: int, total: int, source_type: str, file_path: Path,
) -> str:
    """The one line a run prints per attachment as it reaches it.

    Naming what it shows rather than assembling it inline: an attachment with
    no title falls back to its filename and then to the file on disk, because a
    line reading ``attachment=ABCD1234 item=- type=pdf`` with nothing after it
    identifies nothing to whoever is watching the run.
    """
    title = (getattr(attachment, "title", None) or "").strip()
    if not title:
        title = (getattr(attachment, "filename", None) or "").strip()
    if not title:
        title = file_path.name
    if len(title) > 80:
        title = title[:77] + "..."
    # A top-level attachment has no parent, which is ordinary but worth saying
    # out loud: it is also what a lost parent looks like.
    parent = attachment.parentItemKey or "- (orphan?)"
    return (
        f"[PROGRESS] ({index}/{total}) attachment={attachment.attachmentKey} "
        f"item={parent} type={source_type} {title}"
    )


def _source_verdict(
    args: argparse.Namespace,
    *,
    attachment_key: str,
    previous: dict[str, Any] | None,
    file_path: Path,
    mtime: float,
    size: int,
    pipeline_fingerprint: str,
    inflight: set[str],
    reparse: ReparseDecision,
) -> SourceVerdict:
    """Decide what this run owes an attachment, without extracting anything.

    The three answers the loop acts on -- index it, read its quality only, skip
    it -- gathered here because they are one question asked of the same handful
    of facts. Whether the file changed, whether it was indexed by this pipeline,
    whether a batch still has it in flight, and whether the invocation demands a
    reparse regardless.

    The content hash is computed only once modification time and size already
    match a row that carried one: a difference in either is conclusive on its
    own and cheaper to trust, so an actually-changed file is never hashed, and
    an unchanged one is hashed once and the result handed back for the manifest
    row to reuse.
    """
    has_quality = bool(previous and "quality" in previous)
    entry_pipeline_matches = bool(
        previous and str(previous.get("pipeline_fingerprint") or "") == pipeline_fingerprint
    )

    signature = None
    stat_matches = bool(
        previous
        and float(previous.get("mtime", -1)) == mtime
        and int(previous.get("size", -1)) == size
    )
    if stat_matches and previous.get("content_signature"):
        try:
            signature = content_signature(file_path, size)
        except OSError:
            signature = None

    unchanged = (
        _source_content_unchanged(previous, mtime=mtime, size=size, signature=signature)
        and entry_pipeline_matches
        and attachment_key not in inflight
        and not args.retry_failed
        and not args.force_reparse
        and not reparse.force_docling
        and not reparse.force_ndlocr
        and not reparse.force_mistral
    )
    if not unchanged:
        return SourceVerdict("index", signature)
    if args.check_quality or not has_quality:
        return SourceVerdict("quality_only", signature)
    return SourceVerdict("skip", signature)


def _start_optional_tracing() -> None:
    """Two debug switches that report on the run rather than perform it.

    Lifted out of ``main_async`` to pay for the lines the seam parameters and
    their docstring added: the size ratchet refuses to record a function that
    grew, and this is the least entangled thing in the loop -- it reads no
    local and writes none.
    """
    if os.environ.get("TRACE_UNAWAITED") == "1":
        import tracemalloc
        tracemalloc.start(25)

    if os.environ.get("DEBUG_IMPORTS") == "1":
        import inspect
        import zotero_source_localapi as _zsl
        print(f"[DEBUG] zotero_source_localapi.__file__={_zsl.__file__}", file=sys.stderr)
        print(
            "[DEBUG] iscoroutinefunction(iter_normalized_attachments)="
            f"{inspect.iscoroutinefunction(ZoteroLocalAPI.iter_normalized_attachments)}",
            file=sys.stderr,
        )
        print(
            "[DEBUG] isasyncgenfunction(iter_normalized_attachments)="
            f"{inspect.isasyncgenfunction(ZoteroLocalAPI.iter_normalized_attachments)}",
            file=sys.stderr,
        )


async def main_async(
    args: argparse.Namespace,
    *,
    source: Any | None = None,
    open_collection: Callable[..., Any] | None = None,
) -> None:
    """Run one indexing pass, releasing the lock however it ends.

    The release used to happen on the last line of the happy path and, for
    anything else, on ``atexit``. One run per process made that survivable;
    a caller that runs an ingest and keeps going met a lock naming its own live
    process, with a message telling it to delete the file by hand.
    """
    try:
        await _index_library(args, source=source, open_collection=open_collection)
    finally:
        _release_indexing_lock()


async def _index_library(
    args: argparse.Namespace,
    *,
    source: Any | None = None,
    open_collection: Callable[..., Any] | None = None,
) -> None:
    """Run one indexing pass.

    ``source`` and ``open_collection`` exist so a caller can supply a library
    and an embedding function of its own. Both default to what the CLI has
    always used, so the production path is unchanged; what they buy is a run
    that a test can watch from inside, against a handful of synthetic documents
    and a deterministic embedder, in the time a unit test is allowed to take.
    The alternative -- the only end-to-end check before this -- is starting the
    real indexer against the real library in a child process.
    """
    paths().pdf_cache_dir.mkdir(parents=True, exist_ok=True)
    paths().data_dir.mkdir(parents=True, exist_ok=True)

    # A feature switched on without its resource is a configuration error, not
    # something to work around: silently skipping it would leave the operator
    # wondering why an enabled feature did nothing (note 80).
    feature_problems = verify_enabled_features()
    if feature_problems:
        for problem in feature_problems:
            print(f"[ERROR] {problem}", file=sys.__stderr__)
        raise SystemExit(
            "Configuration error: fix the settings above, or set the flag to 0."
        )

    if not args.dry_run:
        lock_data = _acquire_indexing_lock()

    run_code_fingerprint = code_fingerprint(RUN_CODE_PATHS)
    # Choice (A) (note 80). Resolved once per run so a mid-run env change
    # cannot make half the documents structured and half not.
    structure_recovery = pdf_structure_recovery_enabled()

    manifest = load_manifest(paths().manifest_path)
    files_any = manifest.get("files", {})
    files_manifest: dict[str, dict[str, Any]] = files_any if isinstance(files_any, dict) else {}

    notes_any = manifest.get("notes", {})
    notes_manifest: dict[str, dict[str, Any]] = notes_any if isinstance(notes_any, dict) else {}

    manifest["files"] = files_manifest
    manifest["notes"] = notes_manifest
    reocr_routes = _load_reocr_routes(args.reocr_candidates, args.reocr_limit)
    api = source if source is not None else ZoteroLocalAPI()
    show_progress = bool(args.progress) or (os.environ.get("PROGRESS") == "1")
    if show_progress and not structure_recovery:
        print(
            "[PROGRESS] PDF structure recovery is off: indexing PDFs as plain text "
            "(local OCR still runs where there is no text layer).",
            file=sys.__stderr__,
        )
    t0 = time.perf_counter()

    _start_optional_tracing()

    discovery = await _discover_attachments(
        args,
        api=api,
        manifest=manifest,
        files_manifest=files_manifest,
        reocr_routes=reocr_routes,
        show_progress=show_progress,
    )
    attachments = discovery.attachments
    preflight_notes = discovery.preflight_notes
    excluded_attachments = discovery.excluded_attachments
    inventory_attachments = discovery.inventory_attachments
    preferred_pdf_attachments = discovery.preferred_pdf_attachments

    total_attachments = len(attachments)
    if show_progress:
        if reocr_routes:
            print(
                f"[PROGRESS] Found {total_attachments} explicitly queued re-OCR attachment(s).",
                file=sys.__stderr__,
            )
        elif args.reparse_corrupted:
            print(
                f"[PROGRESS] Found {total_attachments} scanned/corrupted PDF(s) requiring high-fidelity Docling parsing.",
                file=sys.__stderr__,
            )
        else:
            print(
                f"[PROGRESS] Attachments resolved: {total_attachments} "
                f"(tag-excluded={len(excluded_attachments)}, collection={args.collection or 'ALL'})",
                file=sys.__stderr__,
            )

    if args.dump_attachments:
        dump = []
        for a in attachments:
            d = asdict(a) if hasattr(a, "__dataclass_fields__") else dict(a.__dict__)
            dump.append(
                {
                    "attachmentKey": d.get("attachmentKey"),
                    "parentItemKey": d.get("parentItemKey"),
                    "title": d.get("title"),
                    "year": d.get("year"),
                    "creators": d.get("creators"),
                    "pdf_path": d.get("pdf_path"),
                    "source_type": d.get("source_type"),
                    "contentType": d.get("contentType"),
                    "filename": d.get("filename"),
                }
            )
        print(json.dumps(dump, ensure_ascii=False, indent=2))

    if args.dry_run:
        attachments = _select_item_scope(attachments, None, args.limit)
        source_types: dict[str, int] = {}
        for attachment in attachments:
            suffix = Path(str(attachment.pdf_path)).suffix.casefold()
            source_type = "epub" if suffix == ".epub" else ("html" if suffix in {".html", ".htm"} else "pdf")
            source_types[source_type] = source_types.get(source_type, 0) + 1
        print(json.dumps({
            "dry_run": True,
            "rebuild": bool(args.rebuild),
            "structured_v3": True,
            "target_collection": paths().collection_name or CHROMA_COLLECTION_DEFAULT,
            "items": len({str(a.parentItemKey or a.attachmentKey) for a in attachments}),
            "attachments": len(attachments),
            "source_types": dict(sorted(source_types.items())),
            "legacy_ocr_reuse_candidates": 0,
            "canonical_data_modified": False,
        }, ensure_ascii=False, indent=2))
        return

    if args.sync_rag_exclusions_only:
        _sync_tag_exclusions_without_embedding_runtime(
            manifest, files_manifest, excluded_attachments,
            preferred_pdf_attachments=preferred_pdf_attachments,
            inventory_attachments=inventory_attachments,
            show_progress=show_progress,
        )
        return

    if args.rebuild:
        _reset_rebuild_target()
        manifest = {"version": 1, "files": {}, "notes": {}}
        files_manifest = manifest["files"]
        notes_manifest = manifest["notes"]

    col = (open_collection or get_collection)(
        chroma_dir=paths().chroma_dir,
        project_root=PROJECT_ROOT,
        chroma_collection_env=paths().collection_name,
        chroma_collection_default=CHROMA_COLLECTION_DEFAULT,
        persist_active_config=False,
    )
    atexit.register(_close_chroma_collection, col)

    v3_pipeline_fingerprint = ""
    collection_name = str(paths().collection_name or "zotero_paragraphs_v3")
    removed_sentinels = _remove_stale_hnsw_sentinels(
        col, collection_name=collection_name,
    )
    if removed_sentinels:
        print(
            f"[INFO] Removed {len(removed_sentinels)} stale HNSW sentinel(s).",
            file=sys.__stderr__,
        )
    runtime_embedder = dict(getattr(col, "_zotero_embedder_config", {}) or {})
    runtime = pipeline_payload(
        runtime_embedder, collection=collection_name,
        run_code_fingerprint=run_code_fingerprint,
    )
    actual_dim = _collection_embedding_dim(col)
    expected_dim = runtime_embedder.get("embedding_dim")
    if actual_dim is not None and expected_dim is not None and actual_dim != expected_dim:
        raise RuntimeError(
            f"V3 collection dimension mismatch: collection={actual_dim} runtime={expected_dim}"
        )
    stored_pipeline, created_pipeline = ensure_pipeline_config(
        paths().pipeline_config_path, runtime, existing_chunk_count=int(col.count()),
    )
    v3_pipeline_fingerprint = str(stored_pipeline["pipeline_fingerprint"])
    bind_manifest_pipeline(
        manifest, v3_pipeline_fingerprint, adopt_existing=created_pipeline,
    )
    manifest["last_run_code_fingerprint"] = run_code_fingerprint
    manifest["hnsw_validated"] = False
    if created_pipeline and int(col.count()) > 0:
        recovery_items = set(manifest.get("post_index_pending", []))
        recovery_items.update(list_item_keys(collection_name=collection_name))
        manifest["post_index_pending"] = sorted(str(value) for value in recovery_items if value)
    save_manifest(paths().manifest_path, manifest)
    pending_recovery = [
        str(value) for value in manifest.get("post_index_pending", []) if value
    ]
    if pending_recovery:
        _finalize_v3_pending(
            manifest, pending_recovery, collection_name=collection_name,
            code_paths=RUN_CODE_PATHS,
            expected_code_fingerprint=run_code_fingerprint,
        )

    # Delete stale attachment items (skipped during selective re-parsing)
    deleted_stale = 0
    partial_scope = bool(
        args.item or args.attachment or args.limit or args.collection or args.source_type
    )
    if not args.reparse_corrupted and not reocr_routes and not partial_scope:
        current_keys = {a.attachmentKey for a in attachments}
        inventory_keys = {a.attachmentKey for a in inventory_attachments}
        excluded_by_key = {a.attachmentKey: a for a in excluded_attachments}
        excluded_keys = set(files_manifest).intersection(excluded_by_key)
        stale_keys = set(files_manifest.keys()) - inventory_keys

        # A routine sync retires the few attachments actually removed from
        # Zotero. A wholesale disappearance means the *enumeration* came back
        # short, not that the library emptied -- and the enumeration has no
        # completeness check of its own: it pages until a batch looks final,
        # with no Total-Results comparison and no floor. Everything absent from
        # a truncated listing would be deleted here, unconditionally and with
        # no confirmation against Zotero. Fail closed: a deletion this large is
        # evidence about the listing, not about the library. Deliberate bulk
        # removal has its own command (scripts/purge_orphans.py), which
        # confirms every candidate against Zotero individually.
        deletion_limit = max(STALE_DELETE_MIN_KEYS, int(len(files_manifest) * STALE_DELETE_MAX_RATIO))
        if len(stale_keys) > deletion_limit:
            print(
                f"[ERROR] Refusing to delete {len(stale_keys)} attachments: that is more than "
                f"{deletion_limit} of {len(files_manifest)} tracked, which indicates the Zotero "
                f"listing returned short ({len(current_keys)} attachments), not that the library "
                f"shrank. Nothing was deleted. Re-run once Zotero responds fully, or use "
                f"scripts/purge_orphans.py, which confirms each item against Zotero before "
                f"removing it.",
                file=sys.stderr,
            )
            stale_keys = set()

        deletion_keys = stale_keys | excluded_keys
        excluded_parent_rebuilds: set[str] = set()

        for stale_key in deletion_keys:
            # Reuses the same helper the other three deletion call sites in
            # this file use, rather than a hand-rolled copy: a copy here built
            # its own lexical path from LEXICAL_DB_PATH with a data/lexical_v3
            # fallback, which the retired legacy plane would have taken --
            # that env var is only set in the V3 branch above, so a legacy run
            # would delete from the wrong file and leave the retired
            # attachment answering keyword search (2026-07-28). This helper
            # instead defers to lexical_index's own default resolution, the
            # same as every other caller.
            # strict=True so a real delete failure (transient Chroma/lexical
            # I/O error) raises instead of being swallowed inside the helper
            # -- strict=False's default fail-open meant this try/except could
            # never actually fire, so deleted_stale and the manifest pop below
            # ran unconditionally even when the underlying delete failed,
            # orphaning the stale rows with no manifest entry left to retry
            # them from (found in code review, fixed 2026-07-30).
            try:
                _delete_by_attachment_keys(col, [stale_key], strict=True)
            except Exception as exc:
                print(f"[WARN] Failed to delete stale attachment {stale_key}: {exc}", file=sys.stderr)
                continue
            deleted_stale += 1
            files_manifest.pop(stale_key, None)
            excluded_row = excluded_by_key.get(stale_key)
            if excluded_row is not None:
                excluded_parent_rebuilds.add(str(
                    excluded_row.parentItemKey or excluded_row.attachmentKey
                ))
            # This is the one place that has already confirmed stale_key is
            # gone from Zotero. Without this, an attachment deleted and
            # replaced under the same still-live parent item leaves its old
            # extraction/etc. ledger rows permanently unresolved: the item
            # itself is never a purge candidate for purge_removed_items()
            # (item-key granularity), and drop_stale_identity_rows() only
            # covers an attachment gaining a parent, not one disappearing
            # outright (2026-08-03).
            # Best-effort like the manifest pop above, not wrapped into the
            # strict=True Chroma delete above: a locked/busy relations.db
            # here must not abort the whole full-scope run over one key that
            # already had its Chroma/lexical rows and manifest entry cleaned
            # up successfully (found in code review, 2026-08-03).
            try:
                purge_artifact_status_for_attachments([stale_key])
            except Exception as exc:
                print(
                    f"[WARN] Failed to purge ledger rows for stale attachment "
                    f"{stale_key}: {exc}",
                    file=sys.stderr,
                )

        if excluded_parent_rebuilds:
            _finalize_v3_pending(
                manifest, excluded_parent_rebuilds,
                collection_name=collection_name,
                code_paths=RUN_CODE_PATHS,
                expected_code_fingerprint=run_code_fingerprint,
            )
            _delete_summary_embeddings_for_items(
                excluded_parent_rebuilds, collection_name=collection_name,
            )

        # relations.db からも削除済みアイテムのレコードをパージ。
        #
        # The live set must be derived exactly the way the pipeline derives the
        # keys it *writes*, or purging deletes live data. It previously read
        # ``{a.parentItemKey for a in attachments if a.parentItemKey}``, which
        # silently dropped two whole classes of live item: a top-level PDF
        # (tracked under its attachment key, since scope_item_key falls back to
        # it) and any item whose only content is notes (never an attachment at
        # all). Harmless while purge_removed_items only touched citation
        # tables; once it was extended to delete document structures and status
        # rows, this would have purged e.g. FSIXT5VE -- an item with 42 note
        # chunks that is present in Zotero and not deleted (2026-07-27).
        try:
            purge_notes = (
                preflight_notes
                if preflight_notes is not None
                else await api.list_notes(collection_key=args.collection)
            )
        except Exception as exc:
            purge_notes = None
            print(
                f"[WARN] Skipping purge: could not enumerate notes ({exc}). "
                "Purging on a partial view of the library risks deleting live items.",
                file=sys.__stderr__,
            )
        current_item_keys = (
            live_item_keys(inventory_attachments, purge_notes) if purge_notes is not None else None
        )
        purge_counts = (
            purge_removed_items(current_item_keys) if current_item_keys is not None
            else {"item_citation_status": 0, "global_citations": 0, "global_references": 0}
        )
        if show_progress:
            _report_purge(purge_counts)

    updated_pdf = updated_html = updated_epub = 0
    skipped_pdf = skipped_html = skipped_epub = 0
    failed_extract = 0  # zero chunks or inconsistent source coverage
    deferred_extract = 0
    coverage_adopted_extract = 0  # indexed despite a partial-coverage gap

    pending = PendingIndexBatch.empty()
    # Compatibility aliases keep the extraction loop readable while all
    # mutable flush state now has one owner and one clear/commit operation.
    pending_ids = pending.ids
    pending_docs = pending.documents
    pending_metas = pending.metadatas
    pending_manifest_updates = pending.manifest_updates
    pending_extraction_statuses = pending.extraction_statuses
    pending_delete_attachment_keys = pending.delete_attachment_keys
    pending_source_types = pending.source_types
    pending_item_keys = pending.item_keys
    processing_item_keys: set[str] = set()
    last_written_id: str | None = None

    # Docling/RapidOCR has a known semaphore leak on macOS that can crash the
    # process running it. Run it in an isolated, persistent subprocess so a
    # crash there takes down only the worker -- never this process, which
    # holds the live Chroma client mid-write. One worker per run (model
    # loading is expensive); atexit guarantees shutdown on any exit path
    # (normal completion, early return, or exception), mirroring the existing
    # atexit.register(_release_indexing_lock) pattern in this file.
    docling_worker = DoclingWorker()
    atexit.register(docling_worker.shutdown)
    # Granite runs one subprocess per document in its own virtualenv, so there
    # is no persistent state to keep alive; the object just carries settings.
    granite_worker = GraniteWorker()

    for idx, a in enumerate(attachments, start=1):
        assert_code_unchanged(RUN_CODE_PATHS, run_code_fingerprint)
        original_path = Path(a.pdf_path).expanduser()
        source = _resolve_attachment_source(a)
        if source is None:
            if original_path.is_dir():
                print(
                    f"[WARN] Web snapshot directory has no index.html: "
                    f"attachment={a.attachmentKey} dir={original_path}",
                    file=sys.__stderr__,
                )
            continue
        file_path = source.path
        stype = source.source_type
        ctype = getattr(a, "contentType", None)
        mtime = source.mtime
        size = source.size
        scope_item_key = str(a.parentItemKey or a.attachmentKey)

        # A PDF filed at the top level of Zotero is tracked under its
        # attachment key, because scope_item_key falls back to it. Filing it
        # under a parent item later -- an ordinary thing to do -- moves every
        # subsequent status write to the parent key and strands the old rows,
        # which then show as permanently unresolved (AJSX4LFZ, 2026-07-27).
        # Both keys are in hand right here, so the superseded identity can be
        # retired at the moment it becomes stale rather than accumulating until
        # someone runs a cleanup.
        for stale_key in stale_identity_keys(a.attachmentKey, a.parentItemKey):
            dropped = drop_stale_identity_rows(stale_key, scope_item_key)
            if dropped and show_progress:
                print(
                    f"[PROGRESS]   ↳ retired superseded ledger identity {stale_key} "
                    f"→ {scope_item_key} ({dropped} row(s))",
                    file=sys.__stderr__,
                )

        prev = files_manifest.get(a.attachmentKey)

        reocr_route = reocr_routes.get(a.attachmentKey)
        reparse = _reparse_decision(
            args, source_type=stype, previous=prev, reocr_route=reocr_route,
        )
        force_docling = reparse.force_docling
        force_ndlocr = reparse.force_ndlocr
        force_mistral = reparse.force_mistral

        inflight_attachments = {
            str(value) for value in manifest.get("inflight_attachments", []) if value
        }
        if (
                not args.rebuild
                and _skip_current_mistral_toc_candidate(
                    stype=stype, reocr_route=reocr_route,
                    force_reparse=args.force_reparse,
                    reparse_corrupted=args.reparse_corrupted,
                    use_docling=args.use_docling,
                )
                and mistral_batch_queue_enabled()
                and _is_current_mistral_toc_candidate(
                    scope_item_key, a.attachmentKey, mtime=mtime, size=size,
                )):
            skipped_pdf += 1
            if show_progress:
                print(
                    f"[PROGRESS]   ↳ skipped (awaiting Mistral OCR batch): "
                    f"attachment={a.attachmentKey}",
                    file=sys.__stderr__,
                )
            continue
        verdict = _source_verdict(
            args, attachment_key=a.attachmentKey, previous=prev, file_path=file_path,
            mtime=mtime, size=size, pipeline_fingerprint=v3_pipeline_fingerprint,
            inflight=inflight_attachments,
            reparse=reparse,
        )
        current_signature = verdict.signature
        quality_check_only = verdict.action == "quality_only"
        if verdict.action == "skip":
            if stype == "html":
                skipped_html += 1
            elif stype == "epub":
                skipped_epub += 1
            else:
                skipped_pdf += 1
            if show_progress:
                print(
                    f"[PROGRESS]   ↳ skipped (unchanged): attachment={a.attachmentKey}",
                    file=sys.__stderr__,
                )
            continue
        if quality_check_only and show_progress:
            print(
                f"[PROGRESS]   ↳ analyzing text quality of existing item: attachment={a.attachmentKey}",
                file=sys.__stderr__,
            )

        # We are about to write a manifest entry for this attachment -- record
        # its content signature so a *future* run can tell a same-size
        # replacement from a genuinely unchanged file. Reuses the signature
        # already computed above when mtime/size matched a prior signature-
        # bearing row, so an unchanged file is never hashed twice.
        try:
            stored_signature = current_signature or content_signature(file_path, size)
        except OSError:
            stored_signature = None

        if args.limit and scope_item_key not in processing_item_keys:
            if len(processing_item_keys) >= args.limit:
                continue
            processing_item_keys.add(scope_item_key)

        creators_str = None
        if getattr(a, "creators", None):
            creators_str = "; ".join([c for c in a.creators if isinstance(c, str) and c.strip()]) or None

        # (stype/ctype computed above)

        meta_base = {
            # Same identity the ledger, structure and status rows use
            # (scope_item_key above). Without the fallback a top-level
            # attachment -- one filed with no parent -- carries itemKey=None,
            # Chroma stores no key at all, and the chunk belongs to no item:
            # invisible to citation, to item filters, to hierarchical routing,
            # and to every audit, since those iterate over item keys. 17 such
            # attachments held 1,774 chunks that way (2026-07-28). Two
            # definitions of identity in one function is the whole defect.
            "itemKey": scope_item_key,
            "attachmentKey": a.attachmentKey,
            "title": a.title,
            "year": a.year,
            "creators": creators_str,
            "source_type": stype,
            "contentType": ctype,
            "filename": getattr(a, "filename", None),
            "path": str(file_path),
            "locator": None,
            "lang": detect_lang("", getattr(a, "language", None)),
        }
        source_metadata: dict[str, Any] = {}
        if stype == "pdf":
            try:
                source_metadata = classify_pdf_source(file_path).as_metadata()
            except Exception as exc:
                if show_progress:
                    print(
                        f"[WARN] PDF source classification failed: attachment={a.attachmentKey} err={exc}",
                        file=sys.__stderr__,
                    )

        if show_progress:
            print(
                _progress_line(
                    a, index=idx, total=total_attachments,
                    source_type=stype, file_path=file_path,
                ),
                file=sys.__stderr__,
            )

        t_pdf = time.perf_counter()
        attempted_mistral = False
        if force_mistral:
            # A staged Batch result is already local, has been through the
            # Batch quality gate, and is being explicitly adopted.  Applying
            # the *cloud-send* policy here would incorrectly block adoption
            # even though this run makes no API request or text transmission.
            staged_result_path = str(reocr_route.get("mistral_result_path") or "").strip()
            if staged_result_path:
                allowed, policy_reason = True, "staged_batch_result"
            elif stype == "pdf" and not args.refetch_ocr and has_archived_result(file_path, data_dir=paths().data_dir):
                # Same reasoning as the staged-result case just above: this
                # makes no API request and sends nothing, so the cloud-send
                # policy has nothing to gate here.
                allowed, policy_reason = True, "archived_ocr_response"
            else:
                allowed, policy_reason = mistral_ocr_available()
            if not allowed:
                chunks, quality_info = [], {}
                mark_artifact_status(
                    scope_item_key, "extraction", "blocked",
                    attachment_key=a.attachmentKey,
                    reason_code=MISTRAL_TOC_QUEUE_REASON,
                    message=f"Mistral OCR cloud policy unavailable: {policy_reason}",
                    retryable=False,
                    counts={"source_mtime": mtime, "source_size": size},
                    fallback_kind="mistral_ocr",
                )
            else:
                attempted_mistral = True
                try:
                    batch_document_path = Path(str(
                        reocr_route.get("batch_document_path") or file_path
                    )).expanduser()
                    if staged_result_path:
                        expected_size = reocr_route.get("source_size")
                        expected_mtime = reocr_route.get("source_mtime")
                        if (
                            expected_size is not None and int(expected_size) != size
                            or expected_mtime is not None and abs(float(expected_mtime) - mtime) > 0.001
                        ):
                            raise RuntimeError("source fingerprint changed after Mistral Batch submission")
                        staged_result = json.loads(Path(staged_result_path).read_text(encoding="utf-8"))
                        chunks, quality_info = extract_chunks_from_mistral_ocr_result(
                            batch_document_path, a.attachmentKey, meta_base, staged_result,
                            model=str(reocr_route.get("model") or "mistral-ocr-latest"),
                            problem_pages=reocr_route.get("mistral_problem_pages") or {},
                        )
                        if stype == "epub":
                            mapping_path = Path(str(reocr_route.get("epub_mapping_path") or ""))
                            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
                            dom_chunks, dom_quality = extract_chunks_from_epub_snapshot(
                                file_path, a.attachmentKey, meta_base,
                            )
                            epub_profile = dict(dom_quality.get("epub_profile") or {})
                            if epub_profile.get("classification") != "fixed_layout_image":
                                raise RuntimeError(
                                    "EPUB OCR adoption is limited to fixed-layout image EPUBs"
                                )
                            selected_indices = [
                                int(value)
                                for value in mapping.get("requested_spine_indices") or []
                            ]
                            profile_count = int(
                                epub_profile.get("spine_document_count") or 0
                            )
                            if selected_indices != list(range(profile_count)):
                                raise RuntimeError(
                                    "fixed-layout EPUB OCR must cover the complete OPF spine"
                                )
                            returned_pdf_pages = {
                                int(page.get("index", -1)) + 1
                                for page in staged_result.get("pages") or []
                                if isinstance(page, dict)
                            }
                            chunks, quality_info = add_fixed_layout_terminal_markers(
                                chunks, quality_info, mapping,
                                returned_pdf_pages=returned_pdf_pages,
                                attachment_key=a.attachmentKey,
                                metadata=meta_base,
                            )
                            ocr_chunks = remap_ocr_chunks_to_epub(
                                chunks, mapping, epub_path=file_path,
                            )
                            quality_info = dict(quality_info)
                            quality_info["parser"] = "mistral_ocr_epub"
                            chunks = ocr_chunks
                            quality_info = _epub_ocr_quality_from_mapping(
                                quality_info, chunks, mapping,
                                epub_profile=epub_profile,
                            )
                        quality_info["batch_job_id"] = str(reocr_route.get("batch_job_id") or "")
                    elif stype == "pdf":
                        chunks, quality_info = extract_chunks_from_pdf_with_mistral_ocr(
                            file_path, a.attachmentKey, meta_base,
                            data_dir=paths().data_dir, use_cache=not args.refetch_ocr,
                        )
                        if show_progress:
                            # extract_chunks_from_pdf_with_mistral_ocr always
                            # sets exactly one of these three.
                            note = {
                                "hit": "archived OCR reused (no API call)",
                                "stored": "fetched from API and archived",
                                "miss": "fetched from API (not archived)",
                            }[str(quality_info["ocr_cache"])]
                            print(
                                f"[PROGRESS]   ↳ Mistral OCR: {note}",
                                file=sys.__stderr__,
                            )
                    else:
                        raise RuntimeError("fixed-layout EPUB Mistral route requires a staged Batch result")
                except Exception as exc:
                    print(
                        f"[WARN] Mistral OCR batch failed: attachment={a.attachmentKey} err={exc}",
                        file=sys.__stderr__,
                    )
                    chunks, quality_info = [], {}
        elif stype == "html":
            chunks, quality_info = extract_chunks_from_html_snapshot(file_path, a.attachmentKey, meta_base)
        elif stype == "epub":
            chunks, quality_info = extract_chunks_from_epub_snapshot(file_path, a.attachmentKey, meta_base)
        else:
            extraction = _extract_pdf_chunks(
                a=a, args=args, col=col, docling_worker=docling_worker,
                file_path=file_path, files_manifest=files_manifest,
                force_docling=force_docling, force_ndlocr=force_ndlocr,
                granite_worker=granite_worker, manifest=manifest,
                meta_base=meta_base, mtime=mtime, prev=prev,
                scope_item_key=scope_item_key, show_progress=show_progress,
                size=size, source_metadata=source_metadata,
                stored_signature=stored_signature,
                structure_recovery=structure_recovery,
                v3_pipeline_fingerprint=v3_pipeline_fingerprint,
            )
            if extraction.deferred:
                skipped_pdf += 1
                deferred_extract += 1
                continue
            chunks, quality_info = extraction.chunks, extraction.quality

        dt = time.perf_counter() - t_pdf
        assert_code_unchanged(RUN_CODE_PATHS, run_code_fingerprint)
        if show_progress:
            print(
                f"[PROGRESS]   ↳ extracted {len(chunks)} chunks in {dt:.1f}s",
                file=sys.__stderr__,
            )

        if not chunks and stype == "pdf" and not attempted_mistral and not force_mistral:
            # Automatic PDF OCR is Batch-only.  The routing above has already
            # either deferred a qualifying scan without changing canonical
            # chunks or used Docling locally; never make a synchronous Mistral
            # API call as a hidden terminal fallback.
            if show_progress:
                print(
                    "[PROGRESS]   ↳ no synchronous Mistral PDF fallback; "
                    "automatic cloud OCR uses the Batch queue",
                    file=sys.__stderr__,
                )

        epub_profile = (
            dict(quality_info.get("epub_profile") or {})
            if stype == "epub"
            and isinstance(quality_info.get("epub_profile"), dict)
            else {}
        )
        if epub_profile.get("classification") == "fixed_layout_image":
            # The DOM extractor intentionally discards every wrapper in this
            # class, so every image-backed spine is an OCR target regardless
            # of incidental wrapper/alt text length.
            image_primary_indices = sorted({
                int(page["spine_index"])
                for page in epub_profile.get("pages") or []
                if isinstance(page, dict) and page.get("image_exists")
            })
        else:
            # Structured/flowable EPUBs are canonicalized from their DOM.
            # Covers and illustrations must not trigger local or cloud OCR.
            image_primary_indices = []
        failed_epub_units = {
            int(value) for value in quality_info.get("failed_spines") or []
        }
        image_primary_units = {index + 1 for index in image_primary_indices}
        ocr_target_indices = [
            index for index in image_primary_indices
            if index + 1 in failed_epub_units
        ]
        unrecoverable_epub_units = failed_epub_units - image_primary_units
        if (
            stype == "epub"
            and epub_profile.get("classification") == "fixed_layout_image"
            and not chunks
            and ocr_target_indices
            and not unrecoverable_epub_units
            and not force_mistral
        ):
            dom_chunks = list(chunks)
            dom_quality = dict(quality_info)
            try:
                derivative = save_fixed_layout_derivative(
                    file_path, paths().data_dir / "epub_ocr_cache", a.attachmentKey,
                )
                derivative_pdf = Path(str(derivative["derivative_path"]))
                expected_pages = len(derivative.get("pages") or [])
                ocr_chunks, ocr_quality, local_gate = run_local_ocr(
                    derivative_pdf, a.attachmentKey, meta_base,
                    language=meta_base.get("lang"),
                    expected_pages=expected_pages,
                    require_text_for_every_page=True,
                    extractors={
                        "rapidocr": extract_chunks_from_pdf_with_rapidocr,
                        "ndlocr_lite": extract_chunks_from_pdf_with_ndlocr,
                        "docling": docling_worker.extract,
                    },
                )
                if ocr_chunks:
                    ocr_chunks = remap_ocr_chunks_to_epub(
                        ocr_chunks, derivative, epub_path=file_path,
                    )
                    ocr_quality = dict(ocr_quality)
                    ocr_quality["parser"] = (
                        f"{ocr_quality.get('parser') or 'local_ocr'}_epub"
                    )
                    chunks = ocr_chunks
                    quality_info = _epub_ocr_quality_from_mapping(
                        ocr_quality, chunks, derivative,
                        epub_profile=epub_profile,
                    )
                else:
                    chunks, quality_info = dom_chunks, dom_quality
                    cloud_allowed, cloud_reason = mistral_ocr_available()
                    if cloud_allowed:
                        mark_artifact_status(
                            scope_item_key, "extraction", "blocked",
                            attachment_key=a.attachmentKey,
                            reason_code=MISTRAL_TOC_QUEUE_REASON,
                            message=(
                                "Local OCR and Docling quality gates failed for "
                                "image-backed EPUB spines; awaiting Mistral OCR batch."
                            ),
                            retryable=False,
                            counts={
                                "source_type": "epub",
                                "source_mtime": mtime,
                                "source_size": size,
                                "total_pages": expected_pages,
                                "ai_toc_reason": "local_ocr_quality_gate_failed",
                                "local_ocr_gate": local_gate,
                                "batch_document_path": str(derivative_pdf),
                                "epub_mapping_path": str(derivative["mapping_path"]),
                                "epub_ocr_spine_indices": ocr_target_indices,
                                "source_sha256": derivative.get("source_sha256"),
                                "derivative_sha256": derivative.get("derivative_sha256"),
                            },
                            fallback_kind="mistral_ocr",
                        )
                        files_manifest[a.attachmentKey] = _deferred_manifest_entry(
                            prev, mtime=mtime, size=size, source_path=file_path,
                            title=a.title, quality=quality_info,
                            content_signature_value=stored_signature,
                            pipeline_fingerprint=v3_pipeline_fingerprint,
                        )
                        save_manifest(paths().manifest_path, manifest)
                        deferred_extract += 1
                        if show_progress:
                            print(
                                "[PROGRESS]   ↳ local OCR gates failed; deferred "
                                "EPUB image spines to Mistral OCR batch",
                                file=sys.__stderr__,
                            )
                        continue
                    else:
                        quality_info = dict(quality_info)
                        quality_info["cloud_fallback_unavailable"] = cloud_reason
                if show_progress:
                    print(
                        f"[PROGRESS]   ↳ EPUB image-spine local OCR "
                        f"{'accepted' if ocr_chunks else 'failed'}: "
                        f"engine={ocr_quality.get('parser')} pages={expected_pages} "
                        f"reasons={local_gate.get('reasons') or []}",
                        file=sys.__stderr__,
                    )
            except Exception as exc:
                chunks, quality_info = dom_chunks, dom_quality
                print(
                    f"[WARN] EPUB image-spine derivative/OCR failed: "
                    f"attachment={a.attachmentKey} err={exc}",
                    file=sys.__stderr__,
                )

        # A: 抽出0件は失敗扱い（manifest更新しない・削除しない・警告のみ）
        if not chunks:
            failed_extract += 1
            _report_empty_extraction(
                a, scope_item_key=scope_item_key, source_type=stype,
                file_path=file_path, quality=quality_info,
                forced_mistral=force_mistral, mtime=mtime, size=size,
            )
            continue

        source_coverage = coverage_from_extraction(stype, chunks, quality_info)
        coverage_verdict = validate_source_coverage(source_coverage)
        quality_info = dict(quality_info)
        quality_info["source_coverage"] = source_coverage
        quality_info["source_coverage_verdict"] = coverage_verdict
        coverage_adopted = False
        coverage_gap = coverage_shortfall(source_coverage, coverage_verdict)
        if not coverage_verdict["passed"]:
            if not coverage_gap_is_adoptable(coverage_verdict):
                # The extractor contradicted its own unit numbering (units
                # outside the expected set, or a unit reported both blank and
                # non-blank). Its page/spine labels cannot be trusted, so
                # indexing it would attach wrong citations to real text.
                failed_extract += 1
                print(
                    f"[WARN] Inconsistent source coverage; leaving existing index/manifest unchanged: "
                    f"attachment={a.attachmentKey} type={stype} "
                    f"reasons={coverage_verdict['reasons']} "
                    f"unaccounted={coverage_verdict['unaccounted_units']}",
                    file=sys.__stderr__,
                )
                mark_artifact_status(
                    scope_item_key, "extraction", "failed",
                    attachment_key=a.attachmentKey,
                    reason_code="incomplete_source_coverage",
                    message=(
                        f"Source coverage failed: {coverage_verdict['reasons']}; "
                        f"unaccounted={coverage_verdict['unaccounted_units']}"
                    )[:1000],
                    retryable=True,
                    counts={
                        "source_type": stype,
                        "chunks": len(chunks),
                        "coverage": source_coverage,
                    },
                )
                continue
            # U5 (user decision 2026-07-30): a partially recovered document is
            # still indexed. One unreadable page used to cost the whole
            # document its embeddings; now the recovered text is embedded with
            # a quality-uncertain tag, and the shortfall is recorded so a
            # better engine can be pointed at exactly these attachments later.
            coverage_adopted = True
            coverage_adopted_extract += 1
            chunks, quality_info = _adopt_with_quality_uncertain(
                chunks, quality_info,
                reason="incomplete_source_coverage:" + "+".join(
                    coverage_verdict["reasons"]
                ),
            )
            quality_info["source_coverage_adopted"] = True
            quality_info["source_coverage_shortfall"] = coverage_gap
            quality_info["source_coverage_covered_ratio"] = coverage_gap["covered_ratio"]
            print(
                f"[WARN] Incomplete source coverage; indexing with a quality-uncertain tag: "
                f"attachment={a.attachmentKey} type={stype} "
                f"reasons={coverage_verdict['reasons']} "
                f"unaccounted={coverage_verdict['unaccounted_units']} "
                f"covered={coverage_gap['accounted_units']}/{coverage_gap['expected_units']} "
                f"{coverage_gap['unit_kind']}s",
                file=sys.__stderr__,
            )

        truncated = bool(quality_info.get("truncated_by_timeout"))
        # P1 (note 78): an AI-TOC alignment failure (headings likely exist but
        # couldn't be deterministically attached) indexes the document
        # unstructured -- record that as "degraded" so it stays distinguishable
        # from a genuinely structure-less success and remains findable as a
        # future Docling/Mistral structuring candidate. A confirmed
        # no-structure verdict (NO_STRUCTURE_AI_TOC_REASONS) stays "success":
        # there is nothing more any engine could recover for it.
        ai_toc_alignment_failed = (
            str(quality_info.get("ai_toc_recovery_status") or "")
            in AI_TOC_ALIGNMENT_FAILURE_REASONS
        )
        # An adopted coverage gap outranks the other two: it is the reason a
        # re-extraction would be worth running, and its message carries the
        # units a targeted rerun needs. Timeout truncation is one of its
        # possible causes and stays visible in the recorded reasons.
        if coverage_adopted:
            degraded_reason = "incomplete_source_coverage"
            degraded_message = (
                f"Indexed with a quality-uncertain tag: "
                f"{coverage_gap['accounted_units']}/{coverage_gap['expected_units']} "
                f"{coverage_gap['unit_kind']}s accounted for; "
                f"reasons={coverage_verdict['reasons']}; "
                f"unaccounted={coverage_verdict['unaccounted_units'][:50]}"
            )[:1000]
        elif truncated:
            degraded_reason = "docling_timeout_partial"
            degraded_message = "Docling returned partial content after its document timeout."
        elif ai_toc_alignment_failed:
            degraded_reason = "ai_toc_alignment_failed"
            degraded_message = (
                "AI TOC inferred headings but could not align them; indexed "
                f"unstructured ({quality_info.get('ai_toc_recovery_status')})."
            )
        else:
            degraded_reason = None
            degraded_message = None
        extraction_status = _extraction_status(
            a, scope_item_key=scope_item_key, source_type=stype, chunks=chunks,
            quality=quality_info, coverage_adopted=coverage_adopted,
            coverage_gap=coverage_gap, truncated=truncated,
            ai_toc_alignment_failed=ai_toc_alignment_failed,
            degraded_reason=degraded_reason, degraded_message=degraded_message,
        )

        for _cid, text, md in chunks:
            md["lang"] = detect_lang(text, getattr(a, "language", None))

        extraction_quality_json = summarize_extraction_quality(quality_info)
        for _cid, _text, md in chunks:
            md["extraction_engine"] = resolve_extraction_engine(
                quality_info, md.get("extraction_engine"),
            )
            md["extraction_version"] = md.get("extraction_version") or "3"
            md["extraction_quality"] = extraction_quality_json

        if a.parentItemKey:
            # Keep a repair-stage retry from producing duplicate writes.  This
            # removes only equivalent records; conflicting duplicate IDs remain
            # a hard failure so their producer cannot be masked.
            chunks = _deduplicate_exact_chunk_records(chunks)
            source_rows = [
                {"id": cid, "text": text, "metadata": dict(md)} for cid, text, md in chunks
            ]
            built = build_document_structure(a.parentItemKey, source_rows)
            annotated = attach_structure_metadata(source_rows, built["nodes"])
            chunks = [
                (str(row["id"]), str(row.get("text") or ""), dict(row.get("metadata") or {}))
                for row in annotated
            ]
            quality_info = dict(quality_info)
            quality_info["structure_v3"] = {
                "status": built["status"], "version": built["structure_version"],
                "nodes": len(built["nodes"]), "leaves": built["diagnostics"]["leaf_count"],
                "zone_counts": built["diagnostics"].get("zone_counts", {}),
            }

        if quality_check_only:
            # Only update manifest quality, do not write to ChromaDB
            mark_artifact_status(
                extraction_status[0], "extraction", extraction_status[1],
                **extraction_status[2],
            )
            files_manifest[a.attachmentKey] = _manifest_entry(
                mtime=mtime, size=size, source_path=file_path, title=a.title,
                quality=quality_info, content_signature_value=stored_signature,
                pipeline_fingerprint=v3_pipeline_fingerprint,
            )
            if stype == "html":
                skipped_html += 1
            elif stype == "epub":
                skipped_epub += 1
            else:
                skipped_pdf += 1
            continue

        pending_delete_attachment_keys.add(a.attachmentKey)
        pending_extraction_statuses[a.attachmentKey] = extraction_status

        for cid, text, md in chunks:
            pending_ids.append(cid)
            pending_docs.append(text)
            pending_metas.append(md)

        pending_manifest_updates[a.attachmentKey] = _manifest_entry(
            mtime=mtime, size=size, source_path=file_path, title=a.title,
            quality=quality_info, content_signature_value=stored_signature,
            pipeline_fingerprint=v3_pipeline_fingerprint,
        )
        pending_source_types[a.attachmentKey] = stype
        pending_item_keys[a.attachmentKey] = str(a.parentItemKey or a.attachmentKey)
        if len(pending_ids) >= FLUSH_SIZE:
            outcome = _flush_pending_index_batch(
                col, pending,
                manifest=manifest,
                files_manifest=files_manifest,
                run_code_fingerprint=run_code_fingerprint,
                show_progress=show_progress,
                label="upsert batch",
                context_label="periodic flush",
            )
            updated_pdf += outcome.updated_pdf
            updated_html += outcome.updated_html
            updated_epub += outcome.updated_epub
            last_written_id = outcome.last_written_id or last_written_id

    if pending_manifest_updates:
        outcome = _flush_pending_index_batch(
            col, pending,
            manifest=manifest,
            files_manifest=files_manifest,
            run_code_fingerprint=run_code_fingerprint,
            show_progress=show_progress,
            label="final upsert",
            context_label="final flush",
        )
        updated_pdf += outcome.updated_pdf
        updated_html += outcome.updated_html
        updated_epub += outcome.updated_epub
        last_written_id = outcome.last_written_id or last_written_id

    newly_ready_preferred_pdfs = _ready_preferred_pdfs(
        preferred_pdf_attachments, inventory_attachments, files_manifest,
    )
    if newly_ready_preferred_pdfs:
        retired, _affected_items = _retire_indexed_attachments(
            col, manifest, files_manifest, newly_ready_preferred_pdfs,
            collection_name=collection_name,
            code_paths=RUN_CODE_PATHS,
            expected_code_fingerprint=run_code_fingerprint,
        )
        deleted_stale += retired

    note_outcome = await _index_notes_phase(
        args,
        api=api,
        col=col,
        notes_manifest=notes_manifest,
        preflight_notes=preflight_notes,
        partial_scope=partial_scope,
        processing_item_keys=processing_item_keys,
        attachments=attachments,
        show_progress=show_progress,
    )
    notes_manifest = note_outcome.manifest
    updated_notes = note_outcome.updated
    skipped_notes = note_outcome.skipped
    deleted_stale_notes = note_outcome.deleted_stale

    last_written_id = _finalize_index_storage(
        col,
        manifest=manifest,
        files_manifest=files_manifest,
        notes_manifest=notes_manifest,
        last_written_id=last_written_id,
    )

    print(
        f"Done. Updated PDFs={updated_pdf}, Updated HTML(WebClip)={updated_html}, Updated EPUB={updated_epub}, "
        f"Skipped PDFs={skipped_pdf}, Skipped HTML(WebClip)={skipped_html}, Skipped EPUB={skipped_epub}, "
        f"Deleted stale={deleted_stale}, Failed extract/coverage={failed_extract}, "
        f"Partial coverage indexed={coverage_adopted_extract}"
        f" | Updated Notes={updated_notes}, Skipped Notes={skipped_notes}, Deleted stale Notes={deleted_stale_notes}"
    )
    print(json.dumps({
        "event": "index_batch_result",
        "processed_parent_items": len(processing_item_keys),
        "updated_pdf": updated_pdf,
        "skipped_pdf": skipped_pdf,
        "failed_extract": failed_extract,
        "deferred_extract": deferred_extract,
        "coverage_adopted_extract": coverage_adopted_extract,
        "inflight_attachments": list(manifest.get("inflight_attachments", [])),
        "hnsw_validated": bool(manifest.get("hnsw_validated")),
    }, ensure_ascii=False))

    _print_quality_warnings(_quality_warnings(files_manifest))

    if show_progress:
        print(f"[PROGRESS] Total runtime: {time.perf_counter() - t0:.1f}s", file=sys.__stderr__)

    _close_chroma_collection(col)
    _release_indexing_lock()

    if args.rebuild:
        resolved_attachment_keys = {str(row.attachmentKey) for row in attachments}
        indexed_attachment_keys = {str(value) for value in files_manifest}
        missing_after_rebuild = sorted(resolved_attachment_keys - indexed_attachment_keys)
        if failed_extract or deferred_extract or missing_after_rebuild:
            raise RuntimeError(
                "Clean rebuild is incomplete: "
                f"failed={failed_extract} deferred={deferred_extract} "
                f"missing_manifest_attachments={missing_after_rebuild[:20]}"
            )


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
