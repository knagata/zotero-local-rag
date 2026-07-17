#!/usr/bin/env python3
"""Remove EPUB-to-S2 mappings that lack decisive evidence in the raw citation."""
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

from src.reference_text import s2_candidate_is_supported

DEFAULT_DB = ROOT / "data" / "relations.db"


def repair(db_path: Path, *, commit: bool = False, backup_path: Path | None = None) -> dict[str, object]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute('''
            SELECT id, citing_item_key, source, raw_reference_text, s2_status,
                   cited_paper_id, cited_title, cited_year, cited_doi, cited_authors
            FROM global_references
            WHERE source='epub' AND s2_status='mapped'
            ORDER BY id
        ''').fetchall()
        rejected = []
        verified = []
        for row in rows:
            candidate = {
                "paperId": row["cited_paper_id"], "title": row["cited_title"],
                "year": row["cited_year"], "authors": row["cited_authors"],
                "externalIds": {"DOI": row["cited_doi"]} if row["cited_doi"] else {},
            }
            (verified if s2_candidate_is_supported(row["raw_reference_text"], candidate) else rejected).append(row)

        edge_ids: set[int] = set()
        for row in rejected:
            matches = connection.execute('''
                SELECT e.id FROM work_edges e
                JOIN works citing ON citing.work_id=e.citing_work_id
                WHERE citing.zotero_item_key=? AND e.source='epub' AND e.raw_reference=?
            ''', (row["citing_item_key"], row["raw_reference_text"] or "")).fetchall()
            edge_ids.update(int(match["id"]) for match in matches)

        result: dict[str, object] = {
            "mapped_examined": len(rows), "verified_retained": len(verified),
            "unverified_references": len(rejected), "false_or_unverified_edges": len(edge_ids),
            "unverified_reference_ids": [int(row["id"]) for row in rejected],
            "committed": False, "backup": None,
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
        if edge_ids:
            placeholders = ",".join("?" for _ in edge_ids)
            connection.execute(f"DELETE FROM work_edges WHERE id IN ({placeholders})", sorted(edge_ids))
        rejected_ids = [int(row["id"]) for row in rejected]
        if rejected_ids:
            placeholders = ",".join("?" for _ in rejected_ids)
            connection.execute(f'''
                UPDATE global_references
                SET cited_paper_id=NULL, cited_title=NULL, cited_year=NULL,
                    cited_citation_count=0, cited_influential_count=0,
                    cited_doi=NULL, cited_authors=NULL, s2_status='unverified'
                WHERE id IN ({placeholders})
            ''', rejected_ids)
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
        backup = ROOT / "data" / "backups" / f"relations-before-epub-mapping-repair-{stamp}.db"
    print(json.dumps(repair(args.db, commit=args.commit, backup_path=backup), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
