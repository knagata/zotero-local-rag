#!/usr/bin/env python3
"""Rebuild the V3 Chroma collection by re-embedding from the chunk backup
recovered after the 2026-07-22 HNSW-corruption incident (see
dev-notes/current/65_review_findings_v3_batch.md and TASKS.md).

The corrupted zotero_paragraphs_v3 collection was already deleted in a
separate process (avoids ChromaDB's SharedSystemClient per-path caching
gotcha -- see src/embedder.py's open_chroma_collection docstring). This
script only recreates and repopulates it; it does not delete anything.

No re-extraction/OCR is needed: chunk text + metadata were already
recovered via direct SQLite reads into a JSONL backup. This script only
re-runs the (local, free) embedding step and upserts with the identical
chunk ids/documents/metadatas, so the resulting collection is bit-for-bit
equivalent in content to what existed before the crash.

Resumable: progress is checkpointed after every batch, so re-running this
script after an interruption skips already-upserted lines instead of
starting over.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from env_utils import load_dotenv_native  # noqa: E402

load_dotenv_native(ROOT)

from embedder import create_embedding_function, open_chroma_collection, resolve_embedder_settings  # noqa: E402

BACKUP_PATH = ROOT / "data" / "backups" / "v3_chunks_recovered_20260722.jsonl"
CHECKPOINT_PATH = ROOT / "data" / "backups" / "v3_recovery_checkpoint.json"
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION_NAME = "zotero_paragraphs_v3"
BATCH_SIZE = 256


def _load_checkpoint() -> int:
    """Returns the raw file line index to resume from (NOT a chunk count --
    sentinel rows mean these can differ)."""
    if CHECKPOINT_PATH.exists():
        try:
            return int(json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))["next_line_index"])
        except Exception:
            return 0
    return 0


def _save_checkpoint(next_line_index: int) -> None:
    tmp = CHECKPOINT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"next_line_index": next_line_index}), encoding="utf-8")
    tmp.replace(CHECKPOINT_PATH)


def main() -> int:
    if not BACKUP_PATH.exists():
        print(f"[ERROR] backup file not found: {BACKUP_PATH}", file=sys.stderr, flush=True)
        return 1

    start_line = _load_checkpoint()

    # Pre-scan once: separate real chunks from Chroma's own internal
    # HNSW-flush bookkeeping rows that got swept into the same SQLite table
    # during recovery (e.g. id="__hnsw_flush_sentinel_0__",
    # document="hnsw flush sentinel -- safe to ignore", metadata={}). A
    # genuine chunk always carries at least itemKey etc., so "empty metadata"
    # reliably identifies these -- they are not real content and must not be
    # upserted as chunks. This also gives an accurate expected total up front
    # and correctly accounts for any such rows before the resume point.
    total_raw_lines = 0
    expected_chunk_count = 0
    already_done = 0
    with BACKUP_PATH.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            total_raw_lines += 1
            if json.loads(line).get("metadata"):
                expected_chunk_count += 1
                if i < start_line:
                    already_done += 1

    print(
        f"[INFO] backup has {total_raw_lines} rows ({expected_chunk_count} real chunks, "
        f"{total_raw_lines - expected_chunk_count} non-chunk sentinel rows); "
        f"resuming from line {start_line} ({already_done} real chunks already done)",
        flush=True,
    )

    cfg = resolve_embedder_settings(ROOT)
    print(f"[INFO] embedder: model={cfg.model_name} device={cfg.device}", flush=True)
    ef = create_embedding_function(cfg)
    col = open_chroma_collection(CHROMA_DIR, COLLECTION_NAME, ef)

    batch_ids: list[str] = []
    batch_docs: list[str] = []
    batch_metas: list[dict] = []
    processed = already_done
    next_line_index = start_line
    t_start = time.monotonic()

    def flush_batch(line_index_after_batch: int) -> None:
        nonlocal batch_ids, batch_docs, batch_metas, processed
        if not batch_ids:
            return
        embeddings = ef(batch_docs)
        col.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas, embeddings=embeddings)
        processed += len(batch_ids)
        _save_checkpoint(line_index_after_batch)
        elapsed = time.monotonic() - t_start
        done_this_run = processed - already_done
        rate = done_this_run / elapsed if elapsed > 0 else 0.0
        remaining_min = ((expected_chunk_count - processed) / rate / 60.0) if rate > 0 else float("inf")
        print(
            f"[PROGRESS] {processed}/{expected_chunk_count} "
            f"({rate:.1f} chunks/s, ~{remaining_min:.0f} min remaining)",
            flush=True,
        )
        batch_ids, batch_docs, batch_metas = [], [], []

    skipped_sentinels = 0
    with BACKUP_PATH.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            next_line_index = i + 1
            if i < start_line:
                continue
            row = json.loads(line)
            if not row.get("metadata"):
                skipped_sentinels += 1
                continue
            batch_ids.append(row["id"])
            batch_docs.append(row["document"] or "")
            batch_metas.append(row["metadata"])
            if len(batch_ids) >= BATCH_SIZE:
                flush_batch(next_line_index)

    if skipped_sentinels:
        print(f"[INFO] skipped {skipped_sentinels} non-chunk sentinel rows (empty metadata)", flush=True)

    flush_batch(next_line_index)

    final_count = col.count()
    print(f"[DONE] final collection count: {final_count} (expected {expected_chunk_count})", flush=True)
    if final_count != expected_chunk_count:
        print("[ERROR] count mismatch after recovery -- do not proceed to cutover", flush=True)
        return 1

    client = getattr(col, "_chroma_client", None)
    if client is not None:
        client.close()
    print("[SUCCESS] V3 collection fully recovered.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
