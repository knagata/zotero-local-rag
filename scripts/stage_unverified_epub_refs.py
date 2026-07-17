#!/usr/bin/env python3
"""Stage unresolved EPUB citations for evidence-based review without graph writes."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import db_relations

DEFAULT_DB = ROOT / "data" / "relations.db"
YEAR_RE = re.compile(r"(?<!\d)(?:18|19|20)\d{2}(?!\d)")


def _single_year(raw: str) -> int | None:
    years = set(YEAR_RE.findall(unicodedata.normalize("NFKC", raw)))
    return int(next(iter(years))) if len(years) == 1 else None


def stage(db_path: Path, *, commit: bool = False, limit: int | None = None) -> dict[str, int | bool]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        sql = '''
            SELECT id, citing_item_key, raw_reference_text, context_snippet
            FROM global_references
            WHERE source='epub' AND s2_status='unverified'
            ORDER BY id
        '''
        params: tuple[int, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (max(limit, 0),)
        rows = connection.execute(sql, params).fetchall()
        columns = {row[1] for row in connection.execute("PRAGMA table_info(reference_review_queue)")}
        existing: set[tuple[str, str]] = set()
        if columns:
            existing = {
                (str(row[0]), str(row[1])) for row in connection.execute(
                    "SELECT item_key, raw_hash FROM reference_review_queue"
                ).fetchall()
            }
    finally:
        connection.close()

    unique: dict[tuple[str, str], sqlite3.Row] = {}
    for row in rows:
        raw = str(row["raw_reference_text"] or "").strip()
        if not raw:
            continue
        raw_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        unique.setdefault((str(row["citing_item_key"]), raw_hash), row)
    new_rows = [row for key, row in unique.items() if key not in existing]
    result: dict[str, int | bool] = {
        "examined": len(rows), "unique": len(unique), "already_staged": len(unique) - len(new_rows),
        "to_stage": len(new_rows), "staged": 0, "committed": False,
    }
    if not commit:
        return result

    db_relations.DB_PATH = str(db_path)
    db_relations._db_initialized = False
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in new_rows:
        raw = str(row["raw_reference_text"]).strip()
        grouped[str(row["citing_item_key"])].append({
            "raw": raw, "title": None, "authors": [], "year": _single_year(raw),
            "doi": None, "isbn": None, "container": None, "lang": None,
            "type": "epub-reference", "contributors": [],
            "source_reference_id": int(row["id"]),
            "source_context": row["context_snippet"], "source_kind": "epub-unverified",
        })
    for item_key, references in grouped.items():
        counts = db_relations.stage_reference_candidates(item_key, "epub-unverified", references)
        result["staged"] += counts["staged"]
    result["committed"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--commit", action="store_true")
    args = parser.parse_args()
    print(json.dumps(stage(args.db, commit=args.commit, limit=args.limit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
