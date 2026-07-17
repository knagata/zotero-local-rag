#!/usr/bin/env python3
"""Remove false mappings created by resolving Ibid./同書 references independently."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reference_text import is_short_form_reference


DEFAULT_DB = ROOT / "data" / "relations.db"


def repair(
    db_path: Path, *, commit: bool = False, backup_path: Path | None = None,
) -> dict[str, object]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT id, citing_item_key, citing_chunk_id, source, raw_reference_text, "
            "s2_status, cited_paper_id, cited_doi, cited_title "
            "FROM global_references ORDER BY id"
        ).fetchall()
        targets = [
            row for row in rows
            if is_short_form_reference(row["raw_reference_text"])
            and (
                row["s2_status"] == "mapped" or row["cited_paper_id"]
                or row["cited_doi"] or row["cited_title"]
            )
        ]
        edge_ids: set[int] = set()
        for row in targets:
            matches = connection.execute('''
                SELECT e.id
                FROM work_edges e
                JOIN works citing ON citing.work_id = e.citing_work_id
                WHERE citing.zotero_item_key = ?
                  AND e.source = ?
                  AND e.raw_reference = ?
            ''', (
                row["citing_item_key"], row["source"] or "legacy-reference",
                row["raw_reference_text"] or "",
            )).fetchall()
            edge_ids.update(int(match["id"]) for match in matches)
        result: dict[str, object] = {
            "short_form_references": len(targets),
            "false_edges": len(edge_ids),
            "reference_ids": [int(row["id"]) for row in targets],
            "committed": False,
            "backup": None,
        }
        if not commit:
            return result
        if backup_path is None:
            raise ValueError("backup_path is required with commit=True")
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        destination = sqlite3.connect(backup_path)
        try:
            connection.backup(destination)
        finally:
            destination.close()
        reference_ids = [int(row["id"]) for row in targets]
        if edge_ids:
            placeholders = ",".join("?" for _ in edge_ids)
            connection.execute(
                f"DELETE FROM work_edges WHERE id IN ({placeholders})", sorted(edge_ids)
            )
        if reference_ids:
            placeholders = ",".join("?" for _ in reference_ids)
            connection.execute(f'''
                UPDATE global_references
                SET cited_paper_id=NULL, cited_title=NULL, cited_year=NULL,
                    cited_citation_count=0, cited_influential_count=0,
                    cited_doi=NULL, cited_authors=NULL, s2_status='short_form'
                WHERE id IN ({placeholders})
            ''', reference_ids)
        connection.commit()
        result["committed"] = True
        result["backup"] = str(backup_path)
        return result
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--backup", type=Path)
    args = parser.parse_args()
    backup = args.backup
    if args.commit and backup is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = ROOT / "data" / "backups" / f"relations-before-short-form-repair-{stamp}.db"
    print(json.dumps(
        repair(args.db, commit=args.commit, backup_path=backup), ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
