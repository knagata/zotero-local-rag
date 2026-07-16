#!/usr/bin/env python3
"""Read-only audit of title-only works that may combine distinct references."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def audit(db_path: Path, *, sample_size: int = 5) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        invalid_rows = connection.execute('''
            SELECT w.work_id, w.title, COUNT(e.id) AS edge_count,
                   COUNT(DISTINCT e.raw_reference) AS distinct_raw_references,
                   COUNT(DISTINCT e.citing_work_id) AS distinct_citing_works
            FROM works w JOIN work_edges e ON e.cited_work_id = w.work_id
            WHERE (w.title IS NULL OR TRIM(w.title) = '')
              AND w.s2_paper_id IS NULL AND w.doi IS NULL AND w.isbn IS NULL
              AND w.openalex_id IS NULL AND w.cinii_crid IS NULL AND w.ndl_bibid IS NULL
              AND w.zotero_item_key IS NULL
            GROUP BY w.work_id ORDER BY edge_count DESC, w.work_id
        ''').fetchall()
        ambiguous_rows = connection.execute('''
            SELECT w.work_id, w.title, COUNT(e.id) AS edge_count,
                   COUNT(DISTINCT e.raw_reference) AS distinct_raw_references,
                   COUNT(DISTINCT e.citing_work_id) AS distinct_citing_works
            FROM works w JOIN work_edges e ON e.cited_work_id = w.work_id
            WHERE w.s2_paper_id IS NULL AND w.doi IS NULL AND w.isbn IS NULL
              AND w.openalex_id IS NULL AND w.cinii_crid IS NULL AND w.ndl_bibid IS NULL
              AND w.authors IS NULL AND w.year IS NULL
              AND w.title IS NOT NULL AND TRIM(w.title) <> ''
            GROUP BY w.work_id
            HAVING COUNT(DISTINCT e.raw_reference) > 1
            ORDER BY distinct_raw_references DESC, w.work_id
        ''').fetchall()

        def expand(rows: list[sqlite3.Row], classification: str) -> list[dict[str, Any]]:
            output = []
            for row in rows:
                samples = connection.execute('''
                    SELECT e.id AS edge_id, e.raw_reference, e.source,
                           citing.zotero_item_key AS citing_item_key
                    FROM work_edges e
                    LEFT JOIN works citing ON citing.work_id = e.citing_work_id
                    WHERE e.cited_work_id = ? ORDER BY e.id LIMIT ?
                ''', (row["work_id"], max(sample_size, 0))).fetchall()
                output.append({
                    **dict(row), "classification": classification,
                    "samples": [dict(sample) for sample in samples],
                })
            return output

        invalid = expand(invalid_rows, "invalid_missing_identity")
        ambiguous = []
        for candidate in expand(ambiguous_rows, "needs_manual_identity_review"):
            raw_values = [sample.get("raw_reference") or "" for sample in candidate["samples"]]
            if raw_values and all(value.startswith("legacy-reference:") for value in raw_values):
                candidate["classification"] = "duplicate_legacy_rows_review"
            ambiguous.append(candidate)
        return {
            "database": str(db_path),
            "invalid_missing_identity_count": len(invalid),
            "ambiguous_title_only_count": len(ambiguous),
            "candidate_count": len(invalid) + len(ambiguous),
            "policy": "Keep separate unless stable ID or corroborating author/year proves identity.",
            "invalid_missing_identity": invalid,
            "ambiguous_title_only": ambiguous,
        }
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "relations.db")
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.db, sample_size=args.sample_size)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
