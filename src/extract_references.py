"""CLI for PDF/HTML/EPUB reference extraction into the canonical works graph."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .chunk_store import list_item_keys
    from .reference_agent import extract_references_for_item
    from .db_relations import stage_reference_candidates
except ImportError:  # pragma: no cover
    from chunk_store import list_item_keys
    from reference_agent import extract_references_for_item
    from db_relations import stage_reference_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--item", action="append", help="Item key; repeat to process a sample set.")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--commit", action="store_true", help="Persist edges; default is dry-run.")
    action.add_argument(
        "--stage", action="store_true",
        help="Store dry-run candidates in the review queue without changing the works graph.",
    )
    parser.add_argument("--heuristic", action="store_true", help="Do not send reference text to an LLM.")
    parser.add_argument("--output", type=Path, help="Write one complete result JSON object per line.")
    args = parser.parse_args()
    keys = list(args.item) if args.item else list_item_keys()
    if args.limit is not None:
        keys = keys[: max(args.limit, 0)]
    totals = {"items": 0, "references": 0, "excluded": 0, "staged": 0, "updated": 0}
    output_handle = args.output.open("w", encoding="utf-8") if args.output else None
    try:
        for index, item_key in enumerate(keys, start=1):
            result = extract_references_for_item(
                item_key, dry_run=not args.commit, use_llm=not args.heuristic,
            )
            if args.stage and result.get("status") == "dry_run":
                staged = stage_reference_candidates(
                    item_key, str(result.get("model") or "unknown"), result.get("references") or [],
                )
                totals["staged"] += staged["staged"]
                totals["updated"] += staged["updated"]
            totals["items"] += 1
            totals["references"] += len(result.get("references") or [])
            totals["excluded"] += result.get("status") == "excluded"
            print(
                f"[{index}/{len(keys)}] {item_key}: {result['status']} "
                f"({len(result.get('references') or [])} references)", flush=True,
            )
            if output_handle:
                output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            if len(keys) == 1:
                print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        if output_handle:
            output_handle.close()
    print(json.dumps(totals, ensure_ascii=False))


if __name__ == "__main__":
    main()
