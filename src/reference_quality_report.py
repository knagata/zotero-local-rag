"""Report automatic preflight signals for staged reference-extraction candidates."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .db_relations import get_reference_review_candidates
except ImportError:  # pragma: no cover
    from db_relations import get_reference_review_candidates


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    flags = Counter()
    models = Counter()
    statuses = Counter()
    items = set()
    field_counts = Counter()
    flagged_candidates = []
    for row in rows:
        items.add(row["item_key"])
        models[str(row.get("extraction_model") or "unknown")] += 1
        statuses[str(row.get("status") or "unknown")] += 1
        candidate_flags = []
        raw = str(row.get("raw_reference") or "").strip()
        title = str(row.get("title") or "").strip()
        if len(raw) < 15:
            candidate_flags.append("short_raw")
        if len(title) < 3:
            candidate_flags.append("missing_or_short_title")
        if title and title.casefold() == raw.casefold():
            candidate_flags.append("title_equals_raw")
        if not row.get("authors") and not row.get("year") and not row.get("doi") and not row.get("isbn"):
            candidate_flags.append("title_only_unverified")
        for field in ("title", "authors", "year", "doi", "isbn"):
            if row.get(field):
                field_counts[field] += 1
        for flag in candidate_flags:
            flags[flag] += 1
        if candidate_flags:
            flagged_candidates.append({
                "review_id": row["review_id"], "item_key": row["item_key"],
                "flags": candidate_flags,
            })
    total = len(rows)
    return {
        "items": len(items), "candidates": total,
        "manual_review_required": True,
        "note": "Automatic flags measure completeness, not extraction precision or recall.",
        "models": dict(models), "statuses": dict(statuses), "flags": dict(flags),
        "field_coverage": {
            field: round(count / total, 4) if total else 0.0
            for field, count in sorted(field_counts.items())
        },
        "flagged_candidates": flagged_candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", choices=["pending", "approved", "rejected"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(get_reference_review_candidates(args.status))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
