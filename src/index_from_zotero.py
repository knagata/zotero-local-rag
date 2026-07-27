#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import atexit
import json
import os
import shutil
import sys
import time
import uuid
import gc
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, List, Optional

from zotero_source_localapi import ZoteroLocalAPI, ZoteroAttachment

from embedder import get_collection
from html_extract import (
    extract_chunks_from_html_snapshot,
    extract_chunks_from_epub_snapshot,
)
from pdf_extract import (
    extract_chunks_from_pdf, recompute_scanned_quality_after_patch,
    recompute_corrupted_quality_after_patch,
)
from docling_worker import DoclingWorker
from granite_worker import GraniteWorker
from ndlocr_extract import extract_chunks_from_pdf_with_ndlocr
from rapidocr_extract import extract_chunks_from_pdf_with_rapidocr
from local_ocr_pipeline import REPEAT_ARTIFACT_RE, run_local_ocr
from mistral_ocr_extract import (
    extract_chunks_from_mistral_ocr_result,
    extract_chunks_from_pdf_with_mistral_ocr,
    mistral_ocr_available,
)
from epub_fallback import (
    remap_ocr_chunks_to_epub, save_fixed_layout_derivative,
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
    ai_toc_enabled, mistral_batch_queue_enabled, ocr_layer_audit_enabled,
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

from manifest import load_manifest, save_manifest
from db_relations import (
    drop_stale_identity_rows, get_item_processing_status, invalidate_item_summaries,
    mark_artifact_status, purge_removed_items, replace_document_structure,
)
from document_structure import attach_structure_metadata, build_document_structure
from chunk_store import get_item_chunks, list_chunk_ids, list_item_keys
from v3_migration import is_ocr_derived, reuse_ocr_chunks_for_v3
from v3_runtime import (
    assert_code_unchanged, bind_manifest_pipeline, code_fingerprint,
    ensure_pipeline_config, pipeline_payload,
)



# ----------------------------
# Paths / Env
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]


from env_utils import load_dotenv_native

load_dotenv_native(PROJECT_ROOT)

DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", str(DATA_DIR / "chroma")))
PDF_CACHE_DIR = Path(os.environ.get("PDF_CACHE_DIR", str(DATA_DIR / "pdf_cache")))
STRUCTURED_V3_ENABLE = os.environ.get("INGEST_STRUCTURED_V3_ENABLE", "0") == "1"
MANIFEST_PATH = Path(os.environ.get(
    "MANIFEST_PATH", str(DATA_DIR / ("manifest_v3.json" if STRUCTURED_V3_ENABLE else "manifest.json")),
))
INDEXING_LOCK_PATH = DATA_DIR / "indexing.lock"
V3_PIPELINE_CONFIG_PATH = CHROMA_DIR / "embedder_config_v3.json"

ZOTERO_DATA_DIR = os.environ.get("ZOTERO_DATA_DIR")  # required for local storage resolution in your pipeline
CHROMA_COLLECTION_ENV = os.environ.get("CHROMA_COLLECTION") or (
    "zotero_paragraphs_v3" if STRUCTURED_V3_ENABLE else None
)
CHROMA_COLLECTION_DEFAULT = "zotero_paragraphs"
if STRUCTURED_V3_ENABLE:
    os.environ.setdefault("LEXICAL_DB_PATH", str(DATA_DIR / "lexical_v3.sqlite3"))

#: Guards the stale-attachment deletion below. A sync that would retire more
#: than this is reporting on a failed enumeration, not on the library.
STALE_DELETE_MAX_RATIO = 0.05
STALE_DELETE_MIN_KEYS = 10


RUN_CODE_PATHS = tuple(
    PROJECT_ROOT / "src" / name for name in (
        "index_from_zotero.py", "embedder.py", "text_utils.py", "html_extract.py",
        "pdf_extract.py", "docling_extract.py", "docling_worker.py", "extraction_engine.py",
        "local_ocr_pipeline.py", "rapidocr_extract.py", "ndlocr_extract.py", "epub_fallback.py",
        "pdf_toc_recovery.py", "document_structure.py", "v3_migration.py", "v3_runtime.py",
        "pdf_provenance.py", "ocr_layer_audit.py",
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

def _acquire_indexing_lock() -> dict:
    """Create the indexing lock file.  Exits with an error if another indexer
    is currently running (lock file exists and the owning process is alive).

    Returns the lock metadata dict that should be passed to ``_release_indexing_lock``.
    """
    if INDEXING_LOCK_PATH.exists():
        # A lock file already exists — check if it is stale
        try:
            existing = json.loads(INDEXING_LOCK_PATH.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        existing_pid = existing.get("pid")
        if existing_pid is not None:
            try:
                os.kill(existing_pid, 0)  # signal 0 = existence check
                # Process is still alive → genuine conflict
                raise SystemExit(
                    f"別のインデクサーが実行中です (PID={existing_pid})。\n"
                    f"ロックファイル: {INDEXING_LOCK_PATH}\n"
                    "インデクサーの完了を待ってから再実行してください。\n"
                    "（プロセスが存在しないはずなのにロックが残っている場合は、"
                    "手動で削除してください）"
                )
            except OSError:
                # Process is dead — stale lock, remove below
                print(
                    f"[WARN] 古いロックファイルを削除します（PID={existing_pid} は存在しません）",
                    file=sys.__stderr__,
                )
        else:
            # No PID in lock file — treat as stale
            print(
                "[WARN] PID情報のない古いロックファイルを削除します",
                file=sys.__stderr__,
            )
        # Remove stale lock
        try:
            INDEXING_LOCK_PATH.unlink()
        except OSError:
            pass

    lock_data = {
        "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "operation": "indexing",
    }
    INDEXING_LOCK_PATH.write_text(json.dumps(lock_data, ensure_ascii=False), encoding="utf-8")

    # Ensure the lock is released on normal exit, SystemExit, or KeyboardInterrupt.
    atexit.register(_release_indexing_lock)

    return lock_data


def _release_indexing_lock() -> None:
    """Remove the indexing lock file (best-effort, never raises)."""
    try:
        INDEXING_LOCK_PATH.unlink()
    except FileNotFoundError:
        pass  # already gone — nothing to do
    except OSError as e:
        print(f"[WARN] ロックファイルの削除に失敗しました: {e}", file=sys.__stderr__)


def _dedupe_by_id(
    ids: list[str],
    docs: list[str],
    metas: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Dedupe records by id, keeping the last occurrence."""
    uniq: dict[str, tuple[str, dict[str, Any]]] = {}
    for cid, doc, md in zip(ids, docs, metas):
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


def _source_document_chunks(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Canonical structures cover source attachments, never Zotero notes."""
    return [
        row for row in rows
        if str((row.get("metadata") or {}).get("source_type") or "") != "note"
    ]


def _finalize_v3_item(item_key: str, *, collection_name: str) -> dict[str, Any]:
    item_chunks = _source_document_chunks(get_item_chunks(
        item_key, chroma_dir=CHROMA_DIR, collection_name=collection_name,
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
    save_manifest(MANIFEST_PATH, manifest)
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
        save_manifest(MANIFEST_PATH, manifest)


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
    p.add_argument("--retry-failed", action="store_true", help="Process only items with retryable failed artifacts.")
    p.add_argument(
        "--force-reparse", action="store_true",
        help="Re-extract selected items even if unchanged (bypasses the mtime/pipeline skip). "
             "Use after an extraction-code change. Requires --item/--attachment/--limit/--source-type to scope.",
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
    except Exception:
        pass
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
        (chunk_id, text, {**metadata, "quality_uncertain": True,
                          "quality_uncertain_reason": reason})
        for chunk_id, text, metadata in chunks
    ]
    updated = dict(quality_info)
    updated["quality_uncertain"] = True
    updated["quality_uncertain_reason"] = reason
    return tagged, updated


def _reset_rebuild_target() -> None:
    """Reset only the isolated V3 target; legacy rebuild keeps old behavior."""
    if not STRUCTURED_V3_ENABLE:
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)
        if MANIFEST_PATH.exists():
            MANIFEST_PATH.unlink()
        return
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            client.delete_collection(str(CHROMA_COLLECTION_ENV or "zotero_paragraphs_v3"))
        except Exception:
            pass
    finally:
        if MANIFEST_PATH.exists():
            MANIFEST_PATH.unlink()
        if V3_PIPELINE_CONFIG_PATH.exists():
            V3_PIPELINE_CONFIG_PATH.unlink()
        lexical_v3 = Path(os.environ.get("LEXICAL_DB_PATH", DATA_DIR / "lexical_v3.sqlite3"))
        if lexical_v3.exists():
            lexical_v3.unlink()


def _legacy_source_collection() -> str | None:
    explicit = (os.environ.get("V3_REUSE_SOURCE_COLLECTION") or "").strip()
    if explicit:
        return explicit
    try:
        payload = json.loads((CHROMA_DIR / "embedder_config.json").read_text(encoding="utf-8"))
        return str(payload.get("collection") or "").strip() or None
    except (OSError, ValueError, TypeError):
        return None


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
    if not ZOTERO_DATA_DIR:
        raise SystemExit(
            "ERROR: ZOTERO_DATA_DIR is not set.\n"
            "Set it to your Zotero data directory (must contain 'storage/' and 'zotero.sqlite').\n"
        )
    zdd = Path(ZOTERO_DATA_DIR).expanduser()
    if not (zdd.exists() and (zdd / "storage").exists() and (zdd / "zotero.sqlite").exists()):
        raise SystemExit(
            f"ERROR: ZOTERO_DATA_DIR looks invalid: {zdd}\n"
            "Expected to find 'storage/' and 'zotero.sqlite' inside it.\n"
        )


async def main_async(args: argparse.Namespace) -> None:
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

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
        _acquire_indexing_lock()

    run_code_fingerprint = code_fingerprint(RUN_CODE_PATHS)
    # Choice (A) (note 80). Resolved once per run so a mid-run env change
    # cannot make half the documents structured and half not.
    structure_recovery = pdf_structure_recovery_enabled()

    manifest = load_manifest(MANIFEST_PATH)
    files_any = manifest.get("files", {})
    files_manifest: dict[str, dict[str, Any]] = files_any if isinstance(files_any, dict) else {}

    notes_any = manifest.get("notes", {})
    notes_manifest: dict[str, dict[str, Any]] = notes_any if isinstance(notes_any, dict) else {}

    manifest["files"] = files_manifest
    manifest["notes"] = notes_manifest
    reocr_routes = _load_reocr_routes(args.reocr_candidates, args.reocr_limit)
    legacy_manifest = load_manifest(DATA_DIR / "manifest.json") if STRUCTURED_V3_ENABLE else {"files": {}}
    legacy_files = legacy_manifest.get("files") if isinstance(legacy_manifest.get("files"), dict) else {}
    legacy_collection = _legacy_source_collection() if STRUCTURED_V3_ENABLE else None

    api = ZoteroLocalAPI()
    show_progress = bool(args.progress) or (os.environ.get("PROGRESS") == "1")
    if show_progress and not structure_recovery:
        print(
            "[PROGRESS] PDF structure recovery is off: indexing PDFs as plain text "
            "(local OCR still runs where there is no text layer).",
            file=sys.__stderr__,
        )
    t0 = time.perf_counter()

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

    zotero_data_dir: Optional[str] = None
    if _zotero_data_dir_is_valid(ZOTERO_DATA_DIR):
        zotero_data_dir = ZOTERO_DATA_DIR
    else:
        if ZOTERO_DATA_DIR:
            if args.require_data_dir:
                _validate_zotero_data_dir_or_exit()
            else:
                print(
                    f"[WARN] ZOTERO_DATA_DIR looks invalid: {Path(ZOTERO_DATA_DIR).expanduser()}\n"
                    "      Falling back to Zotero Local API file download into PDF_CACHE_DIR.",
                    file=sys.__stderr__,
                )
        else:
            if args.require_data_dir:
                _validate_zotero_data_dir_or_exit()
            else:
                print(
                    "[WARN] ZOTERO_DATA_DIR is not set. Falling back to Zotero Local API file download into PDF_CACHE_DIR.",
                    file=sys.__stderr__,
                )

    if show_progress:
        print("[PROGRESS] Fetching attachment metadata from Zotero (this may take a minute)...", file=sys.__stderr__)

    attachments: List[ZoteroAttachment] = await api.list_normalized_attachments(
        zotero_data_dir=zotero_data_dir,
        pdf_cache_dir=str(PDF_CACHE_DIR),
        collection_key=args.collection,
    )
    attachments = [a for a in attachments if getattr(a, "pdf_path", None)]

    # Exact attachment selection is deliberately applied before parent-item
    # selection: --item keeps its normal "all attachments of this parent"
    # behavior, while a queue worker can constrain that parent to one file.
    attachments = _select_attachment_scope(attachments, args.attachment)
    attachments = _select_item_scope(attachments, args.item, 0)

    if args.retry_failed:
        inflight = {
            str(value) for value in manifest.get("inflight_attachments", []) if value
        }
        retryable_items = {
            str(a.parentItemKey or a.attachmentKey) for a in attachments
            if (
                a.attachmentKey in inflight
                or _retryable_failed(str(a.parentItemKey or a.attachmentKey))
            )
        }
        attachments = [
            a for a in attachments
            if str(a.parentItemKey or a.attachmentKey) in retryable_items
        ]

    if args.source_type:
        keep_type = str(args.source_type)
        attachments = [
            a for a in attachments
            if _resolve_source_type(getattr(a, "contentType", None), Path(getattr(a, "pdf_path", "")), getattr(a, "source_type", None)) == keep_type
        ]

    if reocr_routes:
        attachments = [a for a in attachments if a.attachmentKey in reocr_routes]

    # If --reparse-corrupted is set, filter attachments upfront to show correct progress
    if args.reparse_corrupted:
        corrupted_attachments = []
        for a in attachments:
            prev = files_manifest.get(a.attachmentKey)
            if prev and "quality" in prev:
                q = prev["quality"]
                is_problematic = q.get("is_scanned") or q.get("is_corrupted")
                already_docling = q.get("parser") == "docling"
                if is_problematic and not already_docling:
                    corrupted_attachments.append(a)
        attachments = corrupted_attachments

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
                f"[PROGRESS] Attachments resolved: {total_attachments} (collection={args.collection or 'ALL'})",
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
        reuse_candidates = 0
        for attachment in attachments:
            suffix = Path(str(attachment.pdf_path)).suffix.casefold()
            source_type = "epub" if suffix == ".epub" else ("html" if suffix in {".html", ".htm"} else "pdf")
            source_types[source_type] = source_types.get(source_type, 0) + 1
            legacy_entry = legacy_files.get(attachment.attachmentKey, {}) if isinstance(legacy_files, dict) else {}
            quality = legacy_entry.get("quality", {}) if isinstance(legacy_entry, dict) else {}
            if source_type == "pdf" and is_ocr_derived(quality):
                reuse_candidates += 1
        print(json.dumps({
            "dry_run": True,
            "rebuild": bool(args.rebuild),
            "structured_v3": STRUCTURED_V3_ENABLE,
            "target_collection": CHROMA_COLLECTION_ENV or CHROMA_COLLECTION_DEFAULT,
            "items": len({str(a.parentItemKey or a.attachmentKey) for a in attachments}),
            "attachments": len(attachments),
            "source_types": dict(sorted(source_types.items())),
            "legacy_ocr_reuse_candidates": reuse_candidates,
            "canonical_data_modified": False,
        }, ensure_ascii=False, indent=2))
        return

    if args.rebuild:
        _reset_rebuild_target()
        manifest = {"version": 1, "files": {}, "notes": {}}
        files_manifest = manifest["files"]
        notes_manifest = manifest["notes"]

    col = get_collection(
        chroma_dir=CHROMA_DIR,
        project_root=PROJECT_ROOT,
        chroma_collection_env=CHROMA_COLLECTION_ENV,
        chroma_collection_default=CHROMA_COLLECTION_DEFAULT,
        persist_active_config=not STRUCTURED_V3_ENABLE,
    )
    atexit.register(_close_chroma_collection, col)

    v3_pipeline_fingerprint = ""
    if STRUCTURED_V3_ENABLE:
        collection_name = str(CHROMA_COLLECTION_ENV or "zotero_paragraphs_v3")
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
            V3_PIPELINE_CONFIG_PATH, runtime, existing_chunk_count=int(col.count()),
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
        save_manifest(MANIFEST_PATH, manifest)
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
        stale_keys = set(files_manifest.keys()) - current_keys

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

        for stale_key in stale_keys:
            try:
                col.delete(where={"attachmentKey": stale_key})
                deleted_stale += 1
            except Exception:
                pass
            files_manifest.pop(stale_key, None)

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
            purge_notes = await api.list_notes(collection_key=args.collection)
        except Exception as exc:
            purge_notes = None
            print(
                f"[WARN] Skipping purge: could not enumerate notes ({exc}). "
                "Purging on a partial view of the library risks deleting live items.",
                file=sys.__stderr__,
            )
        current_item_keys = (
            live_item_keys(attachments, purge_notes) if purge_notes is not None else None
        )
        purge_counts = (
            purge_removed_items(current_item_keys) if current_item_keys is not None
            else {"item_citation_status": 0, "global_citations": 0, "global_references": 0}
        )
        purged_total = sum(purge_counts.values())
        if purged_total > 0 and show_progress:
            print(
                f"[PROGRESS] Purged removed items from relations.db: "
                f"item_citation_status={purge_counts['item_citation_status']}, "
                f"global_citations={purge_counts['global_citations']}, "
                f"global_references={purge_counts['global_references']}",
                file=sys.__stderr__,
            )

    updated_pdf = updated_html = updated_epub = 0
    skipped_pdf = skipped_html = skipped_epub = 0
    failed_extract = 0  # extracted 0 chunks (treated as failure)

    pending_ids: list[str] = []
    pending_docs: list[str] = []
    pending_metas: list[dict[str, Any]] = []

    pending_manifest_updates: dict[str, dict[str, Any]] = {}
    pending_delete_attachment_keys: set[str] = set()
    pending_source_types: dict[str, str] = {}
    pending_item_keys: dict[str, str] = {}
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
        if STRUCTURED_V3_ENABLE:
            assert_code_unchanged(RUN_CODE_PATHS, run_code_fingerprint)
        file_path = Path(a.pdf_path).expanduser()
        # Zotero Web Snapshots can be stored as a directory containing an index.html.
        if file_path.is_dir():
            for name in ("index.html", "index.htm"):
                cand = file_path / name
                if cand.exists() and cand.is_file():
                    file_path = cand
                    break
            else:
                # Try a shallow search for any html file.
                htmls = sorted([p for p in file_path.iterdir() if p.is_file() and p.suffix.lower() in {".html", ".htm"}])
                if htmls:
                    file_path = htmls[0]
                else:
                    print(
                        f"[WARN] Web snapshot directory has no index.html: attachment={a.attachmentKey} dir={file_path}",
                        file=sys.__stderr__,
                    )
                    continue
        if not file_path.exists():
            continue

        # Derive a stable source type early (used for skip counters/logging).
        ctype = getattr(a, "contentType", None)
        stype = getattr(a, "source_type", None) or "pdf"
        if (ctype == "application/epub+zip") or (file_path.suffix.lower() == ".epub"):
            stype = "epub"
        elif file_path.suffix.lower() in {".html", ".htm"}:
            stype = "html"
        elif stype not in {"pdf", "html", "epub"}:
            stype = "pdf"

        st = file_path.stat()
        mtime = float(st.st_mtime)
        size = int(st.st_size)
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
        has_quality = prev and "quality" in prev
        quality_check_only = False

        # Decide if we need to force a re-parser for this file.
        force_docling = False
        force_ndlocr = False
        force_mistral = False
        reocr_route = reocr_routes.get(a.attachmentKey)
        if reocr_route:
            target_engine = str(reocr_route.get("target_engine") or "").strip()
            force_mistral = target_engine == "mistral_ocr"
        if stype == "pdf":
            if reocr_route:
                target_engine = str(reocr_route.get("target_engine") or "").strip()
                force_ndlocr = not force_mistral and str(reocr_route.get("lang") or "") == "ja"
                force_docling = not force_mistral and not force_ndlocr
            elif args.use_docling:
                already_docling = has_quality and prev["quality"].get("parser") == "docling"
                if not already_docling:
                    force_docling = True
            elif args.reparse_corrupted and has_quality:
                q = prev["quality"]
                is_problematic = q.get("is_scanned") or q.get("is_corrupted")
                already_docling = q.get("parser") == "docling"
                if is_problematic and not already_docling:
                    force_docling = True

        inflight_attachments = {
            str(value) for value in manifest.get("inflight_attachments", []) if value
        }
        entry_pipeline_matches = bool(
            prev and str(prev.get("pipeline_fingerprint") or "") == v3_pipeline_fingerprint
        ) if STRUCTURED_V3_ENABLE else True
        if (
                _skip_current_mistral_toc_candidate(
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
        if (prev and float(prev.get("mtime", -1)) == mtime
                and int(prev.get("size", -1)) == size
                and entry_pipeline_matches
                and a.attachmentKey not in inflight_attachments
                and not args.retry_failed
                and not args.force_reparse
                and not force_docling and not force_ndlocr and not force_mistral):
            if args.check_quality or not has_quality:
                quality_check_only = True
                if show_progress:
                    print(
                        f"[PROGRESS]   ↳ analyzing text quality of existing item: attachment={a.attachmentKey}",
                        file=sys.__stderr__,
                    )
            else:
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
            short_title = (a.title or "").strip()
            if not short_title:
                short_title = (getattr(a, "filename", None) or "").strip()
            if not short_title:
                short_title = file_path.name

            if len(short_title) > 80:
                short_title = short_title[:77] + "..."

            parent_disp = a.parentItemKey or "-"
            if parent_disp == "-":
                parent_disp = "- (orphan?)"

            # stype already computed above
            print(
                f"[PROGRESS] ({idx}/{total_attachments}) attachment={a.attachmentKey} "
                f"item={parent_disp} type={stype} {short_title}",
                file=sys.__stderr__,
            )

        t_pdf = time.perf_counter()
        reused_ocr = False
        attempted_mistral = False
        legacy_entry = legacy_files.get(a.attachmentKey, {}) if isinstance(legacy_files, dict) else {}
        legacy_quality = legacy_entry.get("quality", {}) if isinstance(legacy_entry, dict) else {}
        if (
            stype == "pdf" and STRUCTURED_V3_ENABLE and not reocr_route
            and not args.use_docling and not args.reparse_corrupted and not args.force_reparse
            and legacy_collection and is_ocr_derived(legacy_quality)
        ):
            legacy_chunks = get_item_chunks(
                a.parentItemKey or a.attachmentKey,
                chroma_dir=CHROMA_DIR, collection_name=legacy_collection,
            )
            chunks, quality_info = reuse_ocr_chunks_for_v3(
                legacy_chunks, a.attachmentKey, meta_base, original_quality=legacy_quality,
            )
            reused_ocr = bool(chunks)
            if reused_ocr:
                # Reused text is OCR output by definition, so it is exactly what
                # the stage-1/stage-2 checks exist for -- and it would otherwise
                # skip them entirely, since this branch bypasses the V3 PDF
                # routing block below (note 79). Reusing text that measures
                # degraded would carry the damage forward into V3 for free,
                # which is the opposite of the point of reuse.
                chunks, quality_info, reused_ocr = _audit_reused_ocr_chunks(
                    chunks, quality_info, file_path,
                    item_key=scope_item_key, prev=prev, mtime=mtime, size=size,
                    show_progress=show_progress,
                )
        if reused_ocr:
            if show_progress:
                print("[PROGRESS]   ↳ reusing legacy OCR text under V3 boundaries...", file=sys.__stderr__)
        elif force_mistral:
            # A staged Batch result is already local, has been through the
            # Batch quality gate, and is being explicitly adopted.  Applying
            # the *cloud-send* policy here would incorrectly block adoption
            # even though this run makes no API request or text transmission.
            staged_result_path = str(reocr_route.get("mistral_result_path") or "").strip()
            allowed, policy_reason = (
                (True, "staged_batch_result") if staged_result_path
                else mistral_ocr_available()
            )
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
                            chunks = remap_ocr_chunks_to_epub(
                                chunks, mapping, epub_path=file_path,
                            )
                            quality_info["parser"] = "mistral_ocr_epub_fixed_layout"
                        quality_info["batch_job_id"] = str(reocr_route.get("batch_job_id") or "")
                    elif stype == "pdf":
                        chunks, quality_info = extract_chunks_from_pdf_with_mistral_ocr(
                            file_path, a.attachmentKey, meta_base,
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
            use_docling_for_this_file = force_docling or args.use_docling
            if force_ndlocr:
                if show_progress:
                    print(
                        f"[PROGRESS]   ↳ parsing Japanese re-OCR candidate with NDLOCR-Lite...",
                        file=sys.__stderr__,
                    )
                chunks, quality_info = extract_chunks_from_pdf_with_ndlocr(
                    file_path, a.attachmentKey, meta_base,
                )
            elif use_docling_for_this_file:
                if show_progress:
                    print(
                        f"[PROGRESS]   ↳ parsing with high-fidelity IBM Docling...",
                        file=sys.__stderr__,
                    )
                try:
                    chunks, quality_info = docling_worker.extract(
                        file_path, a.attachmentKey, meta_base,
                    )
                except RuntimeError as exc:
                    print(
                        f"[WARN] Docling worker extraction failed: attachment={a.attachmentKey} err={exc}",
                        file=sys.__stderr__,
                    )
                    chunks, quality_info = [], {}
                # Note: PyTorch/MPS-CUDA cache clearing now happens inside the
                # worker subprocess itself (see docling_worker._worker_loop),
                # since the worker -- not this process -- holds the torch state.
            elif STRUCTURED_V3_ENABLE:
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
                    and os.environ.get("PDF_SCANNED_PAGE_PATCH_ENABLE", "0").strip() == "1"
                    and text_layer_is_authoritative
                ):
                    try:
                        from docling_extract import patch_scanned_pages_with_docling
                    except ImportError:  # pragma: no cover - direct src entrypoint
                        from .docling_extract import patch_scanned_pages_with_docling
                    try:
                        patched, attempted_pages = patch_scanned_pages_with_docling(
                            file_path, scanned_pages, attachment_key=a.attachmentKey, meta_base=meta_base,
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
                    recovered = try_ai_toc_fast_path(file_path, scope_item_key, chunks, quality_info)
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
                if not structure_recovery:
                    # Plain-text indexing was requested: PyMuPDF chunks (or the
                    # local OCR output above) are the final answer, so the
                    # structural gate has nothing to escalate to.
                    quality_info = dict(quality_info)
                    quality_info["pdf_structure_recovery"] = "disabled"
                elif scanned_ocr_batch_defer or not chunks or (
                    not attempted_local_ocr
                    and not pymupdf_fast_path_passes(quality_info)
                ):
                    use_mistral_queue = False
                    policy_reason = ""
                    # A document whose local OCR chain (rapidocr/ndlocr →
                    # Docling fallback) was tried and rejected by
                    # evaluate_local_ocr_gate: local engines are exhausted.
                    local_ocr_exhausted = attempted_local_ocr and not chunks
                    # P7 (note 78): the queue is governed by its own flag
                    # alone -- it is not an AI-TOC subfeature, so
                    # PDF_AI_TOC_FAST_PATH_ENABLE no longer gates it.
                    # P2 (note 78): scanned documents whose local OCR chain
                    # was exhausted are exactly the class where Mistral OCR
                    # is the strongest engine (bake-off 0.973 vs Docling
                    # 0.753), so they may queue regardless of the AI-TOC
                    # page minimum; the minimum still applies to the
                    # structure-recovery deferrals it was designed for.
                    if scanned_ocr_batch_defer:
                        use_mistral_queue = True
                        policy_reason = initial_scan_route_reason
                    elif (
                        mistral_batch_queue_enabled()
                        and (
                            local_ocr_exhausted
                            or (not attempted_local_ocr and total_pages >= minimum_pages)
                        )
                    ):
                        use_mistral_queue = True
                        policy_reason = "mistral_batch_queue"
                    if use_mistral_queue:
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
                        deferred_entry = dict(prev) if isinstance(prev, dict) else {}
                        deferred_entry.update({
                            "mtime": mtime, "size": size, "pdf_path": str(file_path),
                            "title": a.title,
                            "quality": _merge_deferred_ocr_audit_quality(prev, quality_info),
                        })
                        if STRUCTURED_V3_ENABLE:
                            deferred_entry["pipeline_fingerprint"] = v3_pipeline_fingerprint
                        files_manifest[a.attachmentKey] = deferred_entry
                        save_manifest(MANIFEST_PATH, manifest)
                        skipped_pdf += 1
                        if show_progress:
                            print(
                                f"[PROGRESS]   ↳ deferred to Mistral OCR batch: "
                                f"reason={gate_reason}", file=sys.__stderr__,
                            )
                        continue
                    if local_ocr_exhausted:
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
                                f"[PROGRESS]   ↳ local OCR exhausted (Docling already "
                                f"rejected); not re-running Docling ungated"
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
            else:
                chunks, quality_info = extract_chunks_from_pdf(file_path, a.attachmentKey, meta_base)

        dt = time.perf_counter() - t_pdf
        if STRUCTURED_V3_ENABLE:
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

        if (
            not chunks and stype == "epub"
            and isinstance(quality_info.get("epub_profile"), dict)
            and quality_info["epub_profile"].get("classification") == "fixed_layout_image"
            and not force_mistral
        ):
            try:
                derivative = save_fixed_layout_derivative(
                    file_path, DATA_DIR / "epub_ocr_cache", a.attachmentKey,
                )
                derivative_pdf = Path(str(derivative["derivative_path"]))
                expected_pages = len(derivative.get("pages") or [])
                chunks, quality_info, local_gate = run_local_ocr(
                    derivative_pdf, a.attachmentKey, meta_base,
                    language=meta_base.get("lang"),
                    expected_pages=expected_pages,
                    extractors={
                        "rapidocr": extract_chunks_from_pdf_with_rapidocr,
                        "ndlocr_lite": extract_chunks_from_pdf_with_ndlocr,
                        "docling": docling_worker.extract,
                    },
                )
                if chunks:
                    chunks = remap_ocr_chunks_to_epub(
                        chunks, derivative, epub_path=file_path,
                    )
                    quality_info = dict(quality_info)
                    quality_info["parser"] = (
                        f"{quality_info.get('parser') or 'local_ocr'}_epub_fixed_layout"
                    )
                    quality_info["epub_profile"] = {
                        **quality_info.get("epub_profile", {}),
                        "classification": "fixed_layout_image",
                    }
                else:
                    cloud_allowed, cloud_reason = mistral_ocr_available()
                    if cloud_allowed:
                        mark_artifact_status(
                            scope_item_key, "extraction", "blocked",
                            attachment_key=a.attachmentKey,
                            reason_code=MISTRAL_TOC_QUEUE_REASON,
                            message=(
                                "Local OCR and Docling quality gates failed for "
                                "fixed-layout image EPUB; awaiting Mistral OCR batch."
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
                                "source_sha256": derivative.get("source_sha256"),
                                "derivative_sha256": derivative.get("derivative_sha256"),
                            },
                            fallback_kind="mistral_ocr",
                        )
                        if show_progress:
                            print(
                                f"[PROGRESS]   ↳ local OCR gates failed; deferred "
                                f"fixed-layout EPUB to Mistral OCR batch",
                                file=sys.__stderr__,
                            )
                        continue
                    else:
                        quality_info = dict(quality_info)
                        quality_info["cloud_fallback_unavailable"] = cloud_reason
                if show_progress:
                    print(
                        f"[PROGRESS]   ↳ fixed-layout EPUB local OCR "
                        f"{'accepted' if chunks else 'failed'}: "
                        f"engine={quality_info.get('parser')} pages={expected_pages} "
                        f"reasons={local_gate.get('reasons') or []}",
                        file=sys.__stderr__,
                    )
            except Exception as exc:
                print(
                    f"[WARN] Fixed-layout EPUB derivative failed: "
                    f"attachment={a.attachmentKey} err={exc}",
                    file=sys.__stderr__,
                )

        # A: 抽出0件は失敗扱い（manifest更新しない・削除しない・警告のみ）
        if not chunks:
            failed_extract += 1
            print(
                f"[WARN] Extracted 0 chunks; leaving existing index/manifest unchanged: "
                f"attachment={a.attachmentKey} type={stype} file={file_path}",
                file=sys.__stderr__,
            )
            if force_mistral:
                mark_artifact_status(
                    scope_item_key, "extraction", "blocked",
                    attachment_key=a.attachmentKey,
                    reason_code=MISTRAL_TOC_QUEUE_REASON,
                    message="Mistral OCR batch did not produce adoptable chunks; candidate retained.",
                    retryable=False,
                    counts={"source_mtime": mtime, "source_size": size},
                    fallback_kind="mistral_ocr",
                )
            else:
                mark_artifact_status(
                    scope_item_key, "extraction", "failed",
                    attachment_key=a.attachmentKey, reason_code="no_chunks",
                    message=f"No chunks extracted from {stype} attachment.", retryable=True,
                )
            continue

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
        if truncated:
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
        mark_artifact_status(
            a.parentItemKey or a.attachmentKey, "extraction",
            "degraded" if (truncated or ai_toc_alignment_failed) else "success",
            attachment_key=a.attachmentKey,
            reason_code=degraded_reason,
            message=degraded_message,
            retryable=truncated,
            processor_version=str(quality_info.get("parser") or stype),
            counts={
                "chunks": len(chunks), "source_type": stype,
                "processed_pages": quality_info.get("processed_pages"),
                "expected_pages": quality_info.get("expected_pages"),
                **(
                    {"ai_toc_reason": quality_info.get("ai_toc_recovery_status")}
                    if ai_toc_alignment_failed else {}
                ),
            },
        )

        if reocr_route and a.parentItemKey:
            counts = invalidate_item_summaries(a.parentItemKey)
            if show_progress:
                print(
                    f"[PROGRESS]   ↳ invalidated derived summaries for item={a.parentItemKey}: "
                    f"sections={counts['section_summaries']}",
                    file=sys.__stderr__,
                )

        for _cid, text, md in chunks:
            md["lang"] = detect_lang(text, getattr(a, "language", None))

        if STRUCTURED_V3_ENABLE:
            # Stamp extraction provenance + a compact quality summary on every V3
            # chunk so a later, better structuring/OCR algorithm can query for
            # reprocessing candidates (e.g. all chunks from scanned PDFs, or all
            # chunks extracted by a given engine). Extractors that already set
            # extraction_engine/extraction_version (EPUB/HTML DOM paths, adapter
            # engines) are preserved as-is.
            extraction_quality_json = summarize_extraction_quality(quality_info)
            for _cid, text, md in chunks:
                md["extraction_engine"] = resolve_extraction_engine(
                    quality_info, md.get("extraction_engine"),
                )
                md["extraction_version"] = md.get("extraction_version") or "3"
                md["extraction_quality"] = extraction_quality_json

        if STRUCTURED_V3_ENABLE and a.parentItemKey:
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
            files_manifest[a.attachmentKey] = {
                "mtime": mtime,
                "size": size,
                "pdf_path": str(file_path),
                "title": a.title,
                "quality": quality_info,
                **({"pipeline_fingerprint": v3_pipeline_fingerprint} if STRUCTURED_V3_ENABLE else {}),
            }
            if stype == "html":
                skipped_html += 1
            elif stype == "epub":
                skipped_epub += 1
            else:
                skipped_pdf += 1
            continue

        pending_delete_attachment_keys.add(a.attachmentKey)

        for cid, text, md in chunks:
            pending_ids.append(cid)
            pending_docs.append(text)
            pending_metas.append(md)

        pending_manifest_updates[a.attachmentKey] = {
            "mtime": mtime,
            "size": size,
            "pdf_path": str(file_path),
            "title": a.title,
            "quality": quality_info,
            **({"pipeline_fingerprint": v3_pipeline_fingerprint} if STRUCTURED_V3_ENABLE else {}),
        }
        pending_source_types[a.attachmentKey] = stype
        pending_item_keys[a.attachmentKey] = str(a.parentItemKey or a.attachmentKey)
        if len(pending_ids) >= FLUSH_SIZE:
            ids, docs, metas = _dedupe_by_id(pending_ids, pending_docs, pending_metas)
            committed_item_keys = set(pending_item_keys.values())
            expected_ids: dict[str, set[str]] = {
                key: set() for key in pending_delete_attachment_keys
            }
            for chunk_id, metadata in zip(ids, metas):
                attachment_key = str(metadata.get("attachmentKey") or "")
                if attachment_key in expected_ids:
                    expected_ids[attachment_key].add(str(chunk_id))

            if STRUCTURED_V3_ENABLE:
                assert_code_unchanged(RUN_CODE_PATHS, run_code_fingerprint)
                inflight = {
                    str(value) for value in manifest.get("inflight_attachments", []) if value
                }
                inflight.update(pending_delete_attachment_keys)
                manifest["inflight_attachments"] = sorted(inflight)
                save_manifest(MANIFEST_PATH, manifest)

            _delete_by_attachment_keys(
                col, pending_delete_attachment_keys, strict=STRUCTURED_V3_ENABLE,
            )

            _upsert_in_subbatches(
                col,
                ids,
                docs,
                metas,
                subbatch_size=UPSERT_BATCH_SIZE,
                show_progress=show_progress,
                label="upsert batch",
                strict_lexical=STRUCTURED_V3_ENABLE,
            )

            _fail_flush_on_unhealthy_collection(
                col, dict(pending_item_keys), context_label="periodic flush",
            )
            if STRUCTURED_V3_ENABLE:
                _verify_written_attachments(col, expected_ids)

            for ak, entry in pending_manifest_updates.items():
                files_manifest[ak] = entry
            if STRUCTURED_V3_ENABLE:
                manifest["inflight_attachments"] = [
                    value for value in manifest.get("inflight_attachments", [])
                    if value not in pending_delete_attachment_keys
                ]
            for t in pending_source_types.values():
                if t == "html":
                    updated_html += 1
                elif t == "epub":
                    updated_epub += 1
                else:
                    updated_pdf += 1

            if ids:
                last_written_id = str(ids[-1])
            save_manifest(MANIFEST_PATH, manifest)
            if STRUCTURED_V3_ENABLE:
                _finalize_v3_pending(
                    manifest, committed_item_keys,
                    collection_name=str(CHROMA_COLLECTION_ENV or "zotero_paragraphs_v3"),
                    code_paths=RUN_CODE_PATHS,
                    expected_code_fingerprint=run_code_fingerprint,
                )

            pending_manifest_updates.clear()
            pending_delete_attachment_keys.clear()
            pending_source_types.clear()
            pending_item_keys.clear()

            pending_ids.clear()
            pending_docs.clear()
            pending_metas.clear()

    if pending_ids:
        ids, docs, metas = _dedupe_by_id(pending_ids, pending_docs, pending_metas)
        committed_item_keys = set(pending_item_keys.values())
        expected_ids = {key: set() for key in pending_delete_attachment_keys}
        for chunk_id, metadata in zip(ids, metas):
            attachment_key = str(metadata.get("attachmentKey") or "")
            if attachment_key in expected_ids:
                expected_ids[attachment_key].add(str(chunk_id))
        if STRUCTURED_V3_ENABLE:
            assert_code_unchanged(RUN_CODE_PATHS, run_code_fingerprint)
            inflight = {
                str(value) for value in manifest.get("inflight_attachments", []) if value
            }
            inflight.update(pending_delete_attachment_keys)
            manifest["inflight_attachments"] = sorted(inflight)
            save_manifest(MANIFEST_PATH, manifest)
        _delete_by_attachment_keys(
            col, pending_delete_attachment_keys, strict=STRUCTURED_V3_ENABLE,
        )
        _upsert_in_subbatches(
            col,
            ids,
            docs,
            metas,
            subbatch_size=UPSERT_BATCH_SIZE,
            show_progress=show_progress,
            label="final upsert",
            strict_lexical=STRUCTURED_V3_ENABLE,
        )

        _fail_flush_on_unhealthy_collection(
            col, dict(pending_item_keys), context_label="final flush",
        )
        if STRUCTURED_V3_ENABLE:
            _verify_written_attachments(col, expected_ids)

    if pending_manifest_updates:
        for ak, entry in pending_manifest_updates.items():
            files_manifest[ak] = entry
        for t in pending_source_types.values():
            if t == "html":
                updated_html += 1
            elif t == "epub":
                updated_epub += 1
            else:
                updated_pdf += 1
        if STRUCTURED_V3_ENABLE:
            manifest["inflight_attachments"] = [
                value for value in manifest.get("inflight_attachments", [])
                if value not in pending_delete_attachment_keys
            ]
        if pending_ids:
            last_written_id = str(ids[-1])
        save_manifest(MANIFEST_PATH, manifest)
        if STRUCTURED_V3_ENABLE:
            _finalize_v3_pending(
                manifest, committed_item_keys,
                collection_name=str(CHROMA_COLLECTION_ENV or "zotero_paragraphs_v3"),
                code_paths=RUN_CODE_PATHS,
                expected_code_fingerprint=run_code_fingerprint,
            )
        pending_manifest_updates.clear()
        pending_delete_attachment_keys.clear()
        pending_source_types.clear()
        pending_item_keys.clear()

    # ------------------------------------------------------------------
    # Notes -> chunks (indexed, but excluded from rag_search by default)
    # ----------------------------
    try:
        notes = await api.list_notes(collection_key=args.collection)
    except Exception as e:
        notes = []
        print(f"[WARN] Failed to list notes via Zotero Local API: err={e}", file=sys.__stderr__)

    if partial_scope:
        scoped_item_keys = set(processing_item_keys) if args.limit else {
            str(attachment.parentItemKey or attachment.attachmentKey)
            for attachment in attachments
        }
        notes = [
            note for note in notes
            if str(note.get("parentItemKey") or "") in scoped_item_keys
        ]

    notes_manifest, note_stats = index_notes(
        notes,
        col=col,
        notes_manifest=notes_manifest,
        batch_size=UPSERT_BATCH_SIZE,
        show_progress=show_progress,
        dedupe_fn=_dedupe_by_id,
        upsert_fn=_upsert_in_subbatches,
        lexical_delete_fn=delete_lexical_note,
        delete_stale=not partial_scope,
        strict_lexical=STRUCTURED_V3_ENABLE,
    )

    updated_notes = int(note_stats.get("updated_notes", 0))
    skipped_notes = int(note_stats.get("skipped_notes", 0))
    deleted_stale_notes = int(note_stats.get("deleted_stale_notes", 0))

    # ------------------------------------------------------------------
    # Force a final HNSW flush so the pickle label-map is fully in sync
    # with all records added during this run.
    #
    # Background: ChromaDB only writes index_metadata.pickle every
    # sync_threshold records. If the total added this run is not a
    # multiple of sync_threshold, the last N < sync_threshold records
    # end up in SQLite but NOT in the pickle. On the next query the
    # Rust HNSW backend tries to look up those IDs and throws
    # "Error finding id".
    #
    # Fix: temporarily lower sync_threshold to 1, upsert multiple
    # sentinel documents (which triggers an immediate flush of all
    # pending entries), then delete them.
    # ------------------------------------------------------------------
    try:
        if last_written_id is None:
            sample = col.get(limit=1, include=[])
            sample_ids = sample.get("ids") or []
            last_written_id = str(sample_ids[0]) if sample_ids else None
        _flush_and_verify_hnsw(col, last_written_id)
        manifest["hnsw_validated"] = True
        print("[INFO] HNSW final flush and query smoke test complete.", file=sys.__stderr__)
    except Exception as e:
        manifest["hnsw_validated"] = False
        save_manifest(MANIFEST_PATH, manifest)
        if STRUCTURED_V3_ENABLE:
            raise RuntimeError(f"HNSW final validation failed: {e}") from e
        print(f"[WARN] HNSW final flush failed (non-fatal): {e}", file=sys.__stderr__)

    manifest["notes"] = notes_manifest
    manifest["files"] = files_manifest
    save_manifest(MANIFEST_PATH, manifest)

    print(
        f"Done. Updated PDFs={updated_pdf}, Updated HTML(WebClip)={updated_html}, Updated EPUB={updated_epub}, "
        f"Skipped PDFs={skipped_pdf}, Skipped HTML(WebClip)={skipped_html}, Skipped EPUB={skipped_epub}, "
        f"Deleted stale={deleted_stale}, Failed extract(0 chunks)={failed_extract}"
        f" | Updated Notes={updated_notes}, Skipped Notes={skipped_notes}, Deleted stale Notes={deleted_stale_notes}"
    )
    print(json.dumps({
        "event": "index_batch_result",
        "processed_parent_items": len(processing_item_keys),
        "updated_pdf": updated_pdf,
        "skipped_pdf": skipped_pdf,
        "failed_extract": failed_extract,
        "inflight_attachments": list(manifest.get("inflight_attachments", [])),
        "hnsw_validated": bool(manifest.get("hnsw_validated")),
    }, ensure_ascii=False))

    # Compile and print warnings for scanned or corrupted files. Also surface
    # documents that stayed *under* the is_scanned/is_corrupted thresholds
    # (including the native-outline tolerance in
    # extraction_engine.pymupdf_fast_path_rejection_reason) but still have a
    # nonzero residual scanned_pages/corrupted_pages list -- e.g. a corrupted
    # page the E2d Docling patch (PDF_CORRUPTED_PAGE_PATCH_ENABLE) couldn't
    # recover clean text from, left as a corrupted_unresolved marker chunk
    # rather than silently indexed or dropped. Repair is tried first (see
    # pymupdf_fast_path_rejection_reason's docstring); this report is where
    # whatever repair didn't resolve stays visible even when it was small
    # enough to pass the fast-path tolerance (user decision 2026-07-26).
    problematic_files = []
    for k, entry in files_manifest.items():
        q = entry.get("quality")
        if q and isinstance(q, dict):
            is_scanned = q.get("is_scanned", False)
            is_corrupted = q.get("is_corrupted", False)
            residual_scanned_pages = q.get("scanned_pages") or []
            residual_corrupted_pages = q.get("corrupted_pages") or []
            if is_scanned or is_corrupted or residual_scanned_pages or residual_corrupted_pages:
                title = entry.get("title") or entry.get("pdf_path", "").split("/")[-1] or k
                reasons = []
                if is_scanned:
                    reasons.append("scanned/empty pages")
                elif residual_scanned_pages:
                    reasons.append(f"{len(residual_scanned_pages)} unresolved scanned page(s) (within tolerance)")
                if is_corrupted:
                    reasons.append("text layout/character encoding corruption")
                elif residual_corrupted_pages:
                    reasons.append(f"{len(residual_corrupted_pages)} unresolved corrupted page(s) (within tolerance)")
                problematic_files.append((k, title, reasons, q))

    if problematic_files:
        print("\n" + "=" * 80, file=sys.__stderr__)
        print("⚠️  [RAG QUALITY WARNING] The following files might have poor retrieval quality:", file=sys.__stderr__)
        for k, title, reasons, q in problematic_files:
            reasons_str = " & ".join(reasons)
            print(f"  - [{k}] \"{title}\"", file=sys.__stderr__)
            print(f"    ↳ Issue: {reasons_str}", file=sys.__stderr__)
            if q.get("scanned_pages"):
                print(f"      scanned/empty pages: {q.get('scanned_pages')}", file=sys.__stderr__)
            if q.get("corrupted_pages"):
                print(f"      corrupted/garbled pages: {q.get('corrupted_pages')}", file=sys.__stderr__)
        print("\nRecommendation: We highly recommend processing these files using high-fidelity", file=sys.__stderr__)
        print("AI-based document layout parsers like Docling or Marker to improve RAG accuracy.", file=sys.__stderr__)
        print("=" * 80 + "\n", file=sys.__stderr__)

    if show_progress:
        print(f"[PROGRESS] Total runtime: {time.perf_counter() - t0:.1f}s", file=sys.__stderr__)

    _close_chroma_collection(col)
    _release_indexing_lock()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
