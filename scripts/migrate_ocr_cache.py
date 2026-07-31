#!/usr/bin/env python3
"""Move already-fetched raw OCR responses into the content-addressed archive.

The Batch collector writes each gated response to ``<work_dir>/results/<key>.json``
and records its path in that run's adoption queue. That copy is reachable only
by passing the same queue back via ``--reocr-candidates``, and it sits next to
the multi-gigabyte base64 request payloads that make work directories the
obvious thing to delete -- so responses that were already paid for become
unreachable, and a later ``--force-reparse`` buys them again.

This re-files them under ``data/ocr_cache/`` keyed by the source PDF's sha256,
where an ingest can find them from the file alone. Nothing is deleted: the
original ``results/`` files stay put, so this is safe to re-run and safe to
abandon.

The response is stored raw and unmodified. Chunking, zone assignment and
quality gates are all recomputed by the ingest on current code, so migrating an
old response never pins old parsing behaviour.

Usage:
    python scripts/migrate_ocr_cache.py --dry-run
    python scripts/migrate_ocr_cache.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.manifest import load_manifest
from src.ocr_cache import (
    MISTRAL_REQUEST_CONTRACT, entry_path, source_digest, store_result,
)

#: Batch runs did not all land under one root: pilots and per-collection runs
#: got their own trees, and some nest one level deeper (``<root>/<name>/<run>/
#: results/``). Searching recursively for ``results/`` rather than assuming a
#: fixed depth is what keeps those from being silently left behind -- a
#: fixed ``*/results/*.json`` glob missed two documents (2026-08-01).
DEFAULT_WORK_DIRS = (
    ROOT / "data" / "mistral_ocr_batches",
    ROOT / "data" / "mistral_ocr_pilots",
)
DEFAULT_MANIFEST = ROOT / "data" / "manifest_v3.json"


def _pdf_paths_by_attachment(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """attachment key -> {path, title} from the ingest manifest."""
    try:
        manifest = load_manifest(manifest_path)
    except (OSError, ValueError):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, entry in (manifest.get("files") or {}).items():
        if not isinstance(entry, dict):
            continue
        out[str(key)] = {
            "path": str(entry.get("pdf_path") or ""),
            "title": str(entry.get("title") or ""),
            "model": str(((entry.get("quality") or {}).get("model")) or ""),
            "batch_job_id": str(((entry.get("quality") or {}).get("batch_job_id")) or ""),
        }
    return out


def _run_job_id(result_file: Path) -> str:
    """Batch job id of the run this result file belongs to."""
    queue = result_file.parent.parent / "adoption_queue.json"
    try:
        payload = json.loads(queue.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return str(payload.get("batch_job_id") or "")


def _adoption_model(result_file: Path) -> str:
    """Recover the model from the batch run's state/adoption queue, if present."""
    queue = result_file.parent.parent / "adoption_queue.json"
    try:
        payload = json.loads(queue.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    key = result_file.stem
    for row in payload.get("candidates") or []:
        if isinstance(row, dict) and str(row.get("attachment_key")) == key:
            return str(row.get("model") or "")
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir", type=Path, action="append", dest="work_dirs",
        help="Root to search for results/ directories; repeatable. "
             "Defaults to every known batch root.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    index = _pdf_paths_by_attachment(args.manifest)
    roots = [Path(p) for p in (args.work_dirs or DEFAULT_WORK_DIRS)]
    files = sorted({
        path
        for root in roots if root.exists()
        for path in root.rglob("results/*.json")
    })
    if not files:
        print(f"no batch results found under {', '.join(str(r) for r in roots)}")
        return 0

    # A document re-submitted across batch runs has several stored responses,
    # and OCR is not bit-reproducible, so they differ. Taking whichever was
    # encountered first would archive a revision that was never adopted: the
    # index would then change on the next --force-reparse for no reason the
    # operator asked for. Group by cache key and pick the authoritative one.
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}
    skipped = missing_source = failed = 0
    for result_file in files:
        attachment_key = result_file.stem
        info = index.get(attachment_key) or {}
        source = str(info.get("path") or "")
        if not source or not Path(source).exists():
            # The attachment is gone from the manifest or from disk, so the
            # source bytes that would key this entry cannot be recovered.
            missing_source += 1
            continue
        try:
            body = json.loads(result_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            failed += 1
            continue
        if not isinstance(body, dict) or not body.get("pages"):
            # Not a usable response; archiving it would replay an empty result.
            skipped += 1
            continue

        model = (
            str(body.get("model") or "")
            or _adoption_model(result_file)
            or str(info.get("model") or "")
        )
        if not model:
            skipped += 1
            continue

        try:
            digest = source_digest(Path(source))
        except OSError:
            missing_source += 1
            continue

        candidates.setdefault((digest, model), []).append({
            "path": result_file, "attachment_key": attachment_key,
            "info": info, "body": body, "source": source,
            "job_id": _run_job_id(result_file),
            "run": result_file.parent.parent.name,
        })

    migrated = already = contested = 0
    for (digest, model), rows in sorted(candidates.items()):
        adopted_job = str((rows[0]["info"] or {}).get("batch_job_id") or "")
        if len(rows) > 1:
            contested += 1
            # The manifest records which batch job produced the chunks that are
            # actually indexed; that job's response is the one the archive must
            # be able to reproduce. Fall back to the most recent run only when
            # the manifest cannot say.
            rows.sort(key=lambda r: (r["job_id"] == adopted_job, r["run"]))
        chosen = rows[-1]
        target = entry_path(
            args.data_dir, engine="mistral_ocr", model=model,
            contract_version=MISTRAL_REQUEST_CONTRACT, digest=digest,
        )
        if target.exists():
            try:
                existing = json.loads(target.read_text(encoding="utf-8")).get("result")
            except (OSError, ValueError):
                existing = None
            if existing == chosen["body"]:
                already += 1
                continue
        if args.dry_run:
            note = f" (chose run {chosen['run']} of {len(rows)})" if len(rows) > 1 else ""
            print(
                f"would archive {chosen['attachment_key']} -> "
                f"{target.relative_to(args.data_dir)}{note}"
            )
            migrated += 1
            continue
        try:
            store_result(
                args.data_dir, engine="mistral_ocr", model=model,
                contract_version=MISTRAL_REQUEST_CONTRACT, digest=digest,
                result=chosen["body"], attachment_key=chosen["attachment_key"],
                title=str((chosen["info"] or {}).get("title") or ""),
                source_size=Path(chosen["source"]).stat().st_size,
                source_path=chosen["source"],
                batch_job_id=chosen["job_id"] or adopted_job,
            )
        except OSError as exc:
            print(f"[WARN] {chosen['attachment_key']}: {exc}")
            failed += 1
            continue
        migrated += 1

    verb = "would archive" if args.dry_run else "archived"
    print(
        f"{verb}={migrated} already_present={already} "
        f"source_missing={missing_source} unusable={skipped} failed={failed} "
        f"(scanned {len(files)} result files -> {len(candidates)} documents, "
        f"{contested} with more than one stored response)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
