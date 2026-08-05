#!/usr/bin/env python3
"""Classify flat-fallback document structures without modifying any index."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chunk_store import get_item_chunks
from src.db_relations import get_db_connection
from src.flat_structure_diagnostics import diagnose_flat_item
from src.v3_data_plane import V3_COLLECTION


def _flat_item_keys(selected: list[str] | None) -> list[str]:
    connection = get_db_connection()
    try:
        rows = connection.execute(
            "SELECT item_key FROM document_structures WHERE status = 'flat_fallback' ORDER BY item_key"
        ).fetchall()
    finally:
        connection.close()
    available = [str(row[0]) for row in rows]
    if not selected:
        return available
    wanted = set(selected)
    return [key for key in available if key in wanted]


def _markdown(payload: dict) -> str:
    lines = [
        "# Flat structure diagnostics", "",
        f"Items: {payload['summary']['items']}", "",
        "## Reason counts", "",
    ]
    for reason, count in payload["summary"]["reason_counts"].items():
        lines.append(f"- `{reason}`: {count}")
    lines.extend(["", "## Ranked candidates", ""])
    for item in payload["items"]:
        marker = " gold-candidate" if item["gold_recommended"] else ""
        title = f" — {item['title']}" if item.get("title") else ""
        lines.append(f"### {item['item_key']}{title} (priority {item['priority']}{marker})")
        lines.append("")
        for row in item["attachments"]:
            lines.append(
                f"- `{row['attachment_key']}` {row['source_type']}: "
                f"`{row['reason_code']}`; chunks={row['chunk_count']}, "
                f"chars={row['text_chars']}, toc={row['toc_entries']}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item", action="append", help="Restrict to a flat item key; repeatable")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--collection", default=V3_COLLECTION)
    args = parser.parse_args()
    keys = _flat_item_keys(args.item)
    if args.limit > 0:
        keys = keys[:args.limit]
    items = [
        diagnose_flat_item(
            key, get_item_chunks(key, collection_name=args.collection),
        )
        for key in keys
    ]
    items.sort(key=lambda row: (-int(row["priority"]), str(row["item_key"])))
    reasons = Counter(
        attachment["reason_code"] for item in items for attachment in item["attachments"]
    )
    payload = {
        "read_only": True, "collection": args.collection,
        "summary": {
            "items": len(items),
            "attachments": sum(len(item["attachments"]) for item in items),
            "gold_candidates": sum(bool(item["gold_recommended"]) for item in items),
            "reason_counts": dict(sorted(reasons.items())),
        },
        "items": items,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n" if args.format == "json" else _markdown(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(args.output)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
