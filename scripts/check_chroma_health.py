#!/usr/bin/env python3
"""Chroma sqlite health check: FTS5 index integrity + orphaned segment dirs.

Two failure patterns were found by hand on 2026-08-03, both traced to past
abnormal terminations (a Ctrl+C during Setup's embedding step, and a Docling
SIGSEGV during extraction -- the latter is now isolated in a subprocess, but
the damage it already did was still sitting undetected):

- An interrupted write can desync ``embedding_fulltext_search`` (an
  external-content FTS5 index, kept in sync via triggers as a step separate
  from the row write itself) from the ``embeddings``/``embedding_metadata``
  tables it mirrors, without touching those tables' own row counts. Nothing
  routine ever ran ``PRAGMA integrity_check``, so this can sit undetected
  indefinitely. FTS5-only corruption is repaired in place here (a rebuild of
  a derived search index, never the source rows) -- but anything else
  integrity_check reports is left alone and only surfaced: that could mean
  real data damage, which needs a human decision, not a script guessing.
- An interrupted Setup run leaves its half-built Chroma segment directory on
  disk with no ``segments`` table row pointing at it -- harmless disk
  clutter, but only visible by cross-referencing the two by hand until now.
  Reported only; nothing here deletes anything under ``--chroma-dir``.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_utils import load_dotenv_native  # noqa: E402

load_dotenv_native(ROOT)

from src.v3_data_plane import chroma_dir as resolve_chroma_dir  # noqa: E402

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE,
)


def check_integrity(db_path: Path) -> list[str]:
    """``PRAGMA integrity_check`` results; empty when the database is clean."""
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        rows = connection.execute("PRAGMA integrity_check").fetchall()
    finally:
        connection.close()
    issues = [str(row[0]) for row in rows]
    return [] if issues == ["ok"] else issues


def is_fts_only(issues: list[str]) -> bool:
    return bool(issues) and all("embedding_fulltext_search" in issue for issue in issues)


def repair_fts_index(db_path: Path) -> None:
    """Rebuild the FTS5 shadow tables from the (undamaged) content table.

    ``INSERT ... VALUES('rebuild')`` is FTS5's own repair command for exactly
    this shadow-table desync; it only touches the derived index, never
    ``embeddings``/``embedding_metadata``.
    """
    connection = sqlite3.connect(str(db_path), timeout=30)
    try:
        connection.execute(
            "INSERT INTO embedding_fulltext_search(embedding_fulltext_search) VALUES('rebuild')"
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def orphaned_segment_dirs(chroma_directory: Path, db_path: Path) -> list[str]:
    """Segment directories on disk with no matching row in ``segments``."""
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        referenced = {str(row[0]) for row in connection.execute("SELECT id FROM segments")}
    finally:
        connection.close()
    orphans = []
    for entry in sorted(chroma_directory.iterdir()):
        if entry.is_dir() and _UUID_RE.match(entry.name) and entry.name not in referenced:
            orphans.append(entry.name)
    return orphans


def run_check(chroma_directory: Path, *, repair_fts: bool) -> dict[str, Any]:
    db_path = chroma_directory / "chroma.sqlite3"
    if not db_path.exists():
        return {
            "chroma_sqlite_path": str(db_path), "checked": False,
            "reason": "chroma.sqlite3 not found", "passed": False,
        }

    issues = check_integrity(db_path)
    repaired = False
    if issues and repair_fts and is_fts_only(issues):
        repair_fts_index(db_path)
        repaired = True
        issues = check_integrity(db_path)

    orphans = orphaned_segment_dirs(chroma_directory, db_path)

    return {
        "chroma_sqlite_path": str(db_path),
        "checked": True,
        "integrity_issues": issues,
        "fts_repair_attempted": repaired,
        "orphaned_segment_dirs": orphans,
        # Orphaned segments are cosmetic disk clutter (reported, never
        # deleted here) and never fail this check; a remaining integrity
        # issue after an FTS-only repair attempt -- or one that was never
        # FTS-only to begin with -- means something real is still wrong.
        "passed": not issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chroma-dir", type=Path, default=None,
        help="Directory holding chroma.sqlite3 (defaults to CHROMA_DIR / data/chroma).",
    )
    parser.add_argument("--output", type=Path, help="Write the full report as JSON.")
    parser.add_argument(
        "--no-repair-fts", action="store_true",
        help="Report FTS5-only corruption instead of rebuilding it in place.",
    )
    args = parser.parse_args()

    chroma_directory = args.chroma_dir or resolve_chroma_dir(ROOT)
    report = run_check(chroma_directory, repair_fts=not args.no_repair_fts)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
