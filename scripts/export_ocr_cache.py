#!/usr/bin/env python3
"""Build a clean, portable copy of the OCR archive for another machine.

``data/ocr_cache/`` is the live archive: an ingest writes into it as it fetches,
so at any moment it can hold a half-written entry, an editor's stray file, or
(on macOS) ``.DS_Store``. This produces a separate directory containing only
entries that were checked to be complete and self-consistent, so what gets
copied is known-good rather than whatever happened to be on disk.

Portability comes from the archive's key being the sha256 of the source file's
bytes: the same Zotero library synced to another machine has the same bytes
under a different path and a different mtime, so a copied archive keeps hitting
there. Nothing in an entry refers to this machine's paths for lookup --
``source_path`` is recorded for humans only.

Usage:
    python scripts/export_ocr_cache.py                 # -> data/ocr_cache_export/
    python scripts/export_ocr_cache.py --dest /Volumes/USB/rag-cache
    python scripts/export_ocr_cache.py --verify --dest /Volumes/USB/rag-cache

The export contains a top-level ``ocr_cache/`` directory: copy that whole
folder into ``data/`` on the other machine and it is in place.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ocr_cache import SCHEMA_VERSION, cache_root

DEFAULT_DEST = ROOT / "data" / "ocr_cache_export"

README = """# OCR response archive

Raw OCR engine responses, keyed by the sha256 of each source file's bytes.

## What this is for

Re-ingesting a document after a chunking or structure-parsing fix normally
means paying for its OCR again. With this archive present, the ingest replays
the stored response and re-derives chunks with current code instead. Only raw
engine output is stored -- chunking, zone assignment and quality gates all
re-run every time, so a stored response never pins old parsing behaviour.

## Installing on another machine

Copy the `ocr_cache/` directory here into `data/` of the checkout:

    cp -R ocr_cache /path/to/zotero-local-rag/data/

Nothing else is needed. Lookups key on file content, not paths or timestamps,
so the same Zotero library synced to that machine will hit these entries even
though every path and mtime differs.

## Layout

    ocr_cache/<engine>/<model>__<request-contract>/<sha256-of-source>.json

The model and request contract are part of the path on purpose. A response
fetched under one request shape must never be read as if it were another --
for example, entries fetched without `include_blocks` carry no bounding
boxes, so if that parameter is ever enabled the contract changes and these
entries miss rather than being served as though they had the extra fields.

## Entry fields

Each file wraps the untouched engine response in `result`, alongside
`engine`, `model`, `request_contract_version`, `source_sha256`, `source_size`,
`attachment_key`, `title`, `batch_job_id` and `fetched_at` -- enough to audit
the archive on its own, without the ingest manifest or the batch job's work
directory.

## Verifying after a copy

    python scripts/export_ocr_cache.py --verify --dest <this directory>
"""


def _iter_entries(root: Path):
    for path in sorted(root.rglob("*.json")):
        if path.name.startswith("."):
            continue
        yield path


def _check(path: Path) -> tuple[dict[str, Any] | None, str]:
    """Return (entry, reason). ``entry`` is None when it must not be exported."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"unreadable ({type(exc).__name__})"
    if not isinstance(payload, dict):
        return None, "not an object"
    if payload.get("schema_version") != SCHEMA_VERSION:
        return None, f"unknown schema_version {payload.get('schema_version')!r}"
    digest = str(payload.get("source_sha256") or "")
    if not digest:
        return None, "no source_sha256"
    if path.stem != digest:
        # The filename is the lookup key; a mismatch means this entry would be
        # found under an identity it does not claim.
        return None, "filename does not match source_sha256"
    result = payload.get("result")
    if not isinstance(result, dict) or not result:
        return None, "no raw result"
    return payload, "ok"


def _summarise(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("result") or {}
    return {
        "source_sha256": payload.get("source_sha256"),
        "attachment_key": payload.get("attachment_key"),
        "title": payload.get("title"),
        "engine": payload.get("engine"),
        "model": payload.get("model"),
        "request_contract_version": payload.get("request_contract_version"),
        "pages": len(result.get("pages") or []),
        "fetched_at": payload.get("fetched_at"),
    }


def export(source_root: Path, dest: Path, *, dry_run: bool) -> int:
    if not source_root.exists():
        print(f"no archive at {source_root}")
        return 1
    out_root = dest / "ocr_cache"
    entries: list[dict[str, Any]] = []
    rejected: list[tuple[str, str]] = []
    copied = 0

    for path in _iter_entries(source_root):
        payload, reason = _check(path)
        if payload is None:
            rejected.append((str(path.relative_to(source_root)), reason))
            continue
        entries.append(_summarise(payload))
        if dry_run:
            copied += 1
            continue
        target = out_root / path.relative_to(source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1

    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "README.md").write_text(README, encoding="utf-8")
        index = {
            "schema_version": "ocr-cache-export-1",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "entry_count": len(entries),
            "entries": sorted(entries, key=lambda row: str(row.get("title") or "")),
        }
        (dest / "INDEX.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    total_pages = sum(int(row["pages"]) for row in entries)
    verb = "would export" if dry_run else "exported"
    print(f"{verb} {copied} entries ({total_pages:,} pages) -> {out_root}")
    if rejected:
        print(f"excluded {len(rejected)} file(s):")
        for name, reason in rejected[:20]:
            print(f"  {name}: {reason}")
    return 0


def verify(dest: Path) -> int:
    out_root = dest / "ocr_cache"
    if not out_root.exists():
        print(f"no ocr_cache/ directory under {dest}")
        return 1
    index_path = dest / "INDEX.json"
    expected = None
    if index_path.exists():
        try:
            expected = int(json.loads(index_path.read_text(encoding="utf-8"))["entry_count"])
        except (OSError, ValueError, KeyError, TypeError):
            expected = None

    ok = 0
    problems: list[tuple[str, str]] = []
    for path in _iter_entries(out_root):
        payload, reason = _check(path)
        if payload is None:
            problems.append((str(path.relative_to(out_root)), reason))
        else:
            ok += 1

    print(f"verified {ok} entries under {out_root}")
    if expected is not None and expected != ok:
        print(f"[WARN] INDEX.json expected {expected} entries, found {ok}")
    for name, reason in problems:
        print(f"  [BAD] {name}: {reason}")
    return 1 if problems or (expected is not None and expected != ok) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--verify", action="store_true",
        help="Check an existing export instead of writing one (run this after "
             "copying it to the other machine).",
    )
    args = parser.parse_args(argv)

    if args.verify:
        return verify(args.dest)
    return export(cache_root(args.data_dir), args.dest, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
