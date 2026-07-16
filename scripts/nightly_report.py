#!/usr/bin/env python3
"""Summarize recent nightly outputs and pending reference-review work."""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build_report(db_path: Path, *, since_hours: float) -> dict:
    since = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        summaries = [dict(row) for row in connection.execute('''
            SELECT COALESCE(model, 'unknown') AS model, COUNT(*) AS items
            FROM item_summaries WHERE updated_at >= ? GROUP BY model ORDER BY items DESC
        ''', (since.strftime("%Y-%m-%d %H:%M:%S"),)).fetchall()]
        sections = connection.execute(
            "SELECT COUNT(*) FROM section_summaries WHERE updated_at >= ?",
            (since.strftime("%Y-%m-%d %H:%M:%S"),),
        ).fetchone()[0]
        cases = connection.execute(
            "SELECT COUNT(*) FROM case_annotations WHERE updated_at >= ?",
            (since.strftime("%Y-%m-%d %H:%M:%S"),),
        ).fetchone()[0]
        queue = {
            row["status"]: row["n"] for row in connection.execute(
                "SELECT status, COUNT(*) AS n FROM reference_review_queue GROUP BY status"
            ).fetchall()
        }
        committed = connection.execute(
            "SELECT COUNT(*) FROM reference_review_queue WHERE committed_edge_id IS NOT NULL"
        ).fetchone()[0]
        return {
            "generated_at": datetime.now().astimezone().isoformat(),
            "since_hours": since_hours, "item_summaries": summaries,
            "section_summaries": sections, "case_annotations": cases,
            "reference_queue": queue, "committed_reference_candidates": committed,
        }
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "relations.db")
    parser.add_argument("--since-hours", type=float, default=24)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.db, since_hours=args.since_hours)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
