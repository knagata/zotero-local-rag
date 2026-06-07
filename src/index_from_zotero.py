#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
import gc
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, List, Optional

from zotero_source_localapi import ZoteroLocalAPI, ZoteroAttachment

from embedder import get_collection
from html_extract import (
    extract_chunks_from_html_snapshot,
    extract_chunks_from_epub_snapshot,
)
from pdf_extract import extract_chunks_from_pdf
from docling_extract import extract_chunks_from_pdf_with_docling
from note_extract import index_notes

from manifest import load_manifest, save_manifest



# ----------------------------
# Paths / Env
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv_native() -> None:
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip()
                        if len(v) >= 2 and (
                            (v.startswith('"') and v.endswith('"'))
                            or (v.startswith("'") and v.endswith("'"))
                        ):
                            v = v[1:-1]
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass


load_dotenv_native()

DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", str(DATA_DIR / "chroma")))
PDF_CACHE_DIR = Path(os.environ.get("PDF_CACHE_DIR", str(DATA_DIR / "pdf_cache")))
MANIFEST_PATH = Path(os.environ.get("MANIFEST_PATH", str(DATA_DIR / "manifest.json")))

ZOTERO_DATA_DIR = os.environ.get("ZOTERO_DATA_DIR")  # required for local storage resolution in your pipeline
CHROMA_COLLECTION_ENV = os.environ.get("CHROMA_COLLECTION")
CHROMA_COLLECTION_DEFAULT = "zotero_paragraphs"


# Batch sizing
# One knob: BATCH_SIZE
# - When pending chunks reach this size, we flush (delete+upsert) to Chroma.
# - We also use the same value as the sub-batch size for `col.upsert(...)` to reduce memory spikes.
BATCH_SIZE = int((os.environ.get("BATCH_SIZE") or "128").strip())


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


def _delete_by_attachment_keys(col: Any, attachment_keys: Iterable[str]) -> None:
    """Best-effort delete all chunks for each attachmentKey."""
    for dk in attachment_keys:
        try:
            col.delete(where={"attachmentKey": dk})
        except Exception:
            pass


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
) -> None:
    """Upsert in smaller sub-batches to reduce memory spikes."""
    total = len(ids)
    if total == 0:
        return
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
        relieve_memory_pressure()




# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Index Zotero local PDFs/HTML snapshots into Chroma (paragraph-level).")
    p.add_argument("--collection", help="Zotero collection key to restrict by (optional).", default=None)
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
    return p.parse_args()


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

    if args.rebuild:
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)
        if MANIFEST_PATH.exists():
            MANIFEST_PATH.unlink()

    manifest = load_manifest(MANIFEST_PATH)
    files_any = manifest.get("files", {})
    files_manifest: dict[str, dict[str, Any]] = files_any if isinstance(files_any, dict) else {}

    notes_any = manifest.get("notes", {})
    notes_manifest: dict[str, dict[str, Any]] = notes_any if isinstance(notes_any, dict) else {}

    manifest["files"] = files_manifest
    manifest["notes"] = notes_manifest

    api = ZoteroLocalAPI()
    show_progress = bool(args.progress) or (os.environ.get("PROGRESS") == "1")
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
        if args.reparse_corrupted:
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

    col = get_collection(
        chroma_dir=CHROMA_DIR,
        project_root=PROJECT_ROOT,
        chroma_collection_env=CHROMA_COLLECTION_ENV,
        chroma_collection_default=CHROMA_COLLECTION_DEFAULT,
    )

    # Delete stale attachment items (skipped during selective re-parsing)
    deleted_stale = 0
    if not args.reparse_corrupted:
        current_keys = {a.attachmentKey for a in attachments}
        stale_keys = set(files_manifest.keys()) - current_keys

        for stale_key in stale_keys:
            try:
                col.delete(where={"attachmentKey": stale_key})
                deleted_stale += 1
            except Exception:
                pass
            files_manifest.pop(stale_key, None)

    updated_pdf = updated_html = updated_epub = 0
    skipped_pdf = skipped_html = skipped_epub = 0
    failed_extract = 0  # extracted 0 chunks (treated as failure)

    pending_ids: list[str] = []
    pending_docs: list[str] = []
    pending_metas: list[dict[str, Any]] = []

    pending_manifest_updates: dict[str, dict[str, Any]] = {}
    pending_delete_attachment_keys: set[str] = set()
    pending_source_types: dict[str, str] = {}

    for idx, a in enumerate(attachments, start=1):
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

        prev = files_manifest.get(a.attachmentKey)
        has_quality = prev and "quality" in prev
        quality_check_only = False

        # Decide if we need to force Docling for this file
        force_docling = False
        if stype == "pdf":
            if args.use_docling:
                already_docling = has_quality and prev["quality"].get("parser") == "docling"
                if not already_docling:
                    force_docling = True
            elif args.reparse_corrupted and has_quality:
                q = prev["quality"]
                is_problematic = q.get("is_scanned") or q.get("is_corrupted")
                already_docling = q.get("parser") == "docling"
                if is_problematic and not already_docling:
                    force_docling = True

        if prev and float(prev.get("mtime", -1)) == mtime and int(prev.get("size", -1)) == size and not force_docling:
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

        creators_str = None
        if getattr(a, "creators", None):
            creators_str = "; ".join([c for c in a.creators if isinstance(c, str) and c.strip()]) or None

        # (stype/ctype computed above)

        meta_base = {
            "itemKey": a.parentItemKey,
            "attachmentKey": a.attachmentKey,
            "title": a.title,
            "year": a.year,
            "creators": creators_str,
            "source_type": stype,
            "contentType": ctype,
            "filename": getattr(a, "filename", None),
            "path": str(file_path),
            "locator": None,
        }

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
        if stype == "html":
            chunks, quality_info = extract_chunks_from_html_snapshot(file_path, a.attachmentKey, meta_base)
        elif stype == "epub":
            chunks, quality_info = extract_chunks_from_epub_snapshot(file_path, a.attachmentKey, meta_base)
        else:
            use_docling_for_this_file = force_docling or args.use_docling
            if use_docling_for_this_file:
                if show_progress:
                    print(
                        f"[PROGRESS]   ↳ parsing with high-fidelity IBM Docling...",
                        file=sys.__stderr__,
                    )
                chunks, quality_info = extract_chunks_from_pdf_with_docling(file_path, a.attachmentKey, meta_base)
                # Force immediate PyTorch garbage collection and MPS/CUDA cache clearing
                relieve_memory_pressure()
            else:
                chunks, quality_info = extract_chunks_from_pdf(file_path, a.attachmentKey, meta_base)

        dt = time.perf_counter() - t_pdf
        if show_progress:
            print(
                f"[PROGRESS]   ↳ extracted {len(chunks)} chunks in {dt:.1f}s",
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
            continue

        if quality_check_only:
            # Only update manifest quality, do not write to ChromaDB
            files_manifest[a.attachmentKey] = {
                "mtime": mtime,
                "size": size,
                "pdf_path": str(file_path),
                "title": a.title,
                "quality": quality_info,
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
        }
        pending_source_types[a.attachmentKey] = stype

        if len(pending_ids) >= BATCH_SIZE:
            ids, docs, metas = _dedupe_by_id(pending_ids, pending_docs, pending_metas)

            _delete_by_attachment_keys(col, pending_delete_attachment_keys)

            _upsert_in_subbatches(
                col,
                ids,
                docs,
                metas,
                subbatch_size=BATCH_SIZE,
                show_progress=show_progress,
                label="upsert batch",
            )

            for ak, entry in pending_manifest_updates.items():
                files_manifest[ak] = entry
            for t in pending_source_types.values():
                if t == "html":
                    updated_html += 1
                elif t == "epub":
                    updated_epub += 1
                else:
                    updated_pdf += 1

            pending_manifest_updates.clear()
            pending_delete_attachment_keys.clear()
            pending_source_types.clear()

            pending_ids.clear()
            pending_docs.clear()
            pending_metas.clear()

    if pending_delete_attachment_keys:
        for dk in list(pending_delete_attachment_keys):
            try:
                col.delete(where={"attachmentKey": dk})
            except Exception:
                pass

    if pending_ids:
        ids, docs, metas = _dedupe_by_id(pending_ids, pending_docs, pending_metas)
        _upsert_in_subbatches(
            col,
            ids,
            docs,
            metas,
            subbatch_size=BATCH_SIZE,
            show_progress=show_progress,
            label="final upsert",
        )

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
        pending_manifest_updates.clear()
        pending_delete_attachment_keys.clear()
        pending_source_types.clear()

    # ----------------------------
    # Notes -> chunks (indexed, but excluded from rag_search by default)
    # ----------------------------
    try:
        notes = await api.list_notes(collection_key=args.collection)
    except Exception as e:
        notes = []
        print(f"[WARN] Failed to list notes via Zotero Local API: err={e}", file=sys.__stderr__)

    notes_manifest, note_stats = index_notes(
        notes,
        col=col,
        notes_manifest=notes_manifest,
        batch_size=BATCH_SIZE,
        show_progress=show_progress,
        dedupe_fn=_dedupe_by_id,
        upsert_fn=_upsert_in_subbatches,
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
        # Try both 'configuration' (newer Chroma) and 'metadata' (older/compat)
        # and use both int and string for the value just in case.
        for val in [1, "1"]:
            try:
                col.modify(configuration={"hnsw": {"sync_threshold": val}})
            except Exception:
                pass
            try:
                col.modify(metadata={"hnsw:sync_threshold": val})
            except Exception:
                pass

        # Use multiple sentinels to ensure the threshold is crossed and a flush is triggered.
        _flush_ids = [f"__hnsw_flush_sentinel_{i}__" for i in range(3)]
        col.upsert(
            ids=_flush_ids,
            documents=["hnsw flush sentinel — safe to ignore"] * len(_flush_ids)
        )
        col.delete(ids=_flush_ids)
        
        # Small sleep to give the OS a chance to sync files before process exit.
        time.sleep(0.5)
        print("[INFO] HNSW final flush complete.", file=sys.__stderr__)
    except Exception as e:
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

    # Compile and print warnings for scanned or corrupted files
    problematic_files = []
    for k, entry in files_manifest.items():
        q = entry.get("quality")
        if q and isinstance(q, dict):
            is_scanned = q.get("is_scanned", False)
            is_corrupted = q.get("is_corrupted", False)
            if is_scanned or is_corrupted:
                title = entry.get("title") or entry.get("pdf_path", "").split("/")[-1] or k
                reasons = []
                if is_scanned:
                    reasons.append("scanned/empty pages")
                if is_corrupted:
                    reasons.append("text layout/character encoding corruption")
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


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))