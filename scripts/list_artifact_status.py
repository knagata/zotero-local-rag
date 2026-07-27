#!/usr/bin/env python3
"""List artifact processing status grouped by type / status / reason_code.

Use this to track V3 batch progress and collect unresolved items for retry.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db_relations import DB_PATH


def _build_report(conn: Any, *, item_key: str | None = None) -> dict[str, Any]:
    where = "WHERE item_key = ?" if item_key else "WHERE 1=1"
    params = (item_key,) if item_key else ()
    rows = conn.execute(f"""
        SELECT item_key, attachment_key, artifact_type, status,
               reason_code, retryable, started_at,
               substr(COALESCE(message, ''), 1, 200) AS message,
               counts_json
        FROM artifact_processing_status
        {where}
        ORDER BY artifact_type, status, item_key
    """, params).fetchall()

    # Group by type→status→reason_code and collect item counts
    by_type: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"count": 0, "item_keys": []}))
    )
    # unresolved = genuine hard-failure states needing action (retry/investigation).
    # empty/degraded are informational only: empty may be a correct zero-result
    # (e.g. a paper with no reference section), and degraded means a fallback
    # artifact exists at lower quality — neither is a failure to fix.
    unresolved: list[dict[str, Any]] = []
    informational: list[dict[str, Any]] = []

    for row in rows:
        r = dict(row)
        group = by_type[r["artifact_type"]][r["status"]][r["reason_code"] or "unspecified"]
        group["count"] += 1
        if r["item_key"] not in group["item_keys"]:
            group["item_keys"].append(r["item_key"])

        if r["status"] in ("failed", "blocked", "degraded", "empty"):
            try:
                counts = json.loads(r.pop("counts_json") or "{}")
            except (TypeError, ValueError):
                counts = {}
            r["counts"] = counts
            if r["status"] in ("failed", "blocked"):
                unresolved.append(r)
            else:
                informational.append(r)

    # Collapse into sorted structure
    summary: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for artifact_type, statuses in sorted(by_type.items()):
        summary[artifact_type] = {}
        for status, reasons in sorted(statuses.items()):
            summary[artifact_type][status] = [
                {"reason_code": reason, "count": info["count"], "item_keys": sorted(info["item_keys"])}
                for reason, info in sorted(reasons.items())
            ]

    return {
        "schema_version": "artifact-status-2",
        "item_key": item_key,
        "summary": summary,
        "unresolved_count": len(unresolved),
        "unresolved": [
            {
                "item_key": r["item_key"], "attachment_key": r.get("attachment_key"),
                "artifact_type": r["artifact_type"], "status": r["status"],
                "reason_code": r.get("reason_code"), "retryable": bool(r.get("retryable")),
                "message": r.get("message"), "counts": r.get("counts"),
            }
            for r in unresolved
        ],
        "informational_count": len(informational),
        "informational": [
            {
                "item_key": r["item_key"], "attachment_key": r.get("attachment_key"),
                "artifact_type": r["artifact_type"], "status": r["status"],
                "reason_code": r.get("reason_code"), "message": r.get("message"),
                "counts": r.get("counts"),
            }
            for r in informational
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item", help="Restrict to a single item key")
    parser.add_argument("--unresolved-only", action="store_true",
                        help="Only print items that need attention (failed/blocked). "
                             "Excludes empty/degraded, which are informational, not failures.")
    parser.add_argument("--retryable-only", action="store_true",
                        help="Only print retryable failed items (for --retry-failed)")
    args = parser.parse_args()

    conn = sqlite3.connect(f"file:{Path(DB_PATH)}?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        report = _build_report(conn, item_key=args.item)
    finally:
        conn.close()

    if args.retryable_only:
        report["unresolved"] = [
            r for r in report["unresolved"]
            if r["status"] == "failed" and r["retryable"]
        ]
        report["unresolved_count"] = len(report["unresolved"])

    if args.unresolved_only:
        del report["summary"]
        del report["informational"]
        del report["informational_count"]

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
