#!/usr/bin/env python3
"""Back up and remove canonical works that have no bibliographic identity."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
IDENTITY_COLUMNS = (
    "s2_paper_id", "doi", "isbn", "openalex_id", "cinii_crid", "ndl_bibid",
    "zotero_item_key", "title", "authors", "year",
)


def _placeholders(values: list[int]) -> str:
    return ",".join("?" for _ in values)


def inspect_targets(connection: sqlite3.Connection, work_ids: list[int]) -> dict[str, Any]:
    if not work_ids:
        raise ValueError("at least one work ID is required")
    placeholders = _placeholders(work_ids)
    rows = connection.execute(
        f"SELECT * FROM works WHERE work_id IN ({placeholders}) ORDER BY work_id", work_ids
    ).fetchall()
    found = {int(row["work_id"]): row for row in rows}
    missing = sorted(set(work_ids) - set(found))
    unsafe = []
    for work_id, row in found.items():
        populated = [column for column in IDENTITY_COLUMNS if row[column] not in (None, "")]
        if populated:
            unsafe.append({"work_id": work_id, "populated_identity_columns": populated})
    counts = {
        "incoming_edges": connection.execute(
            f"SELECT COUNT(*) FROM work_edges WHERE cited_work_id IN ({placeholders})", work_ids
        ).fetchone()[0],
        "outgoing_edges": connection.execute(
            f"SELECT COUNT(*) FROM work_edges WHERE citing_work_id IN ({placeholders})", work_ids
        ).fetchone()[0],
        "work_links": connection.execute(
            f"SELECT COUNT(*) FROM work_links WHERE work_id_a IN ({placeholders}) "
            f"OR work_id_b IN ({placeholders})", [*work_ids, *work_ids]
        ).fetchone()[0],
        "child_works": connection.execute(
            f"SELECT COUNT(*) FROM works WHERE container_work_id IN ({placeholders})", work_ids
        ).fetchone()[0],
    }
    return {
        "requested_work_ids": work_ids, "found_work_ids": sorted(found),
        "missing_work_ids": missing, "unsafe_targets": unsafe, **counts,
    }


def _backup_database(connection: sqlite3.Connection, backup_path: Path) -> None:
    if backup_path.exists():
        raise FileExistsError(f"backup already exists: {backup_path}")
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    destination = sqlite3.connect(backup_path)
    try:
        connection.backup(destination)
    finally:
        destination.close()


def repair(
    db_path: Path, work_ids: list[int], *, commit: bool = False,
    backup_path: Path | None = None,
) -> dict[str, Any]:
    uri = f"file:{db_path}?mode={'rw' if commit else 'ro'}"
    connection = sqlite3.connect(uri, uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        before = inspect_targets(connection, work_ids)
        blockers = (
            before["missing_work_ids"] or before["unsafe_targets"]
            or before["outgoing_edges"] or before["work_links"] or before["child_works"]
        )
        result: dict[str, Any] = {"status": "dry_run", "before": before}
        if not commit:
            return result
        if blockers:
            raise RuntimeError(f"refusing unsafe repair: {json.dumps(before, ensure_ascii=False)}")
        if backup_path is None:
            raise ValueError("--backup is required with --commit")
        _backup_database(connection, backup_path)
        placeholders = _placeholders(work_ids)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        edge_cursor = connection.execute(
            f"DELETE FROM work_edges WHERE cited_work_id IN ({placeholders})", work_ids
        )
        work_cursor = connection.execute(
            f"DELETE FROM works WHERE work_id IN ({placeholders})", work_ids
        )
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            connection.rollback()
            raise RuntimeError(f"integrity check failed: {integrity}")
        connection.commit()
        result.update({
            "status": "repaired", "backup": str(backup_path),
            "deleted_edges": edge_cursor.rowcount, "deleted_works": work_cursor.rowcount,
            "integrity_check": integrity,
        })
        return result
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_ids", nargs="+", type=int)
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "relations.db")
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--backup", type=Path)
    args = parser.parse_args()
    print(json.dumps(
        repair(args.db, args.work_ids, commit=args.commit, backup_path=args.backup),
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
