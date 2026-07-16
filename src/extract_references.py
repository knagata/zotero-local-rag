"""CLI for PDF/HTML/EPUB reference extraction into the canonical works graph."""
from __future__ import annotations

import argparse
import json

try:
    from .chunk_store import list_item_keys
    from .reference_agent import extract_references_for_item
except ImportError:  # pragma: no cover
    from chunk_store import list_item_keys
    from reference_agent import extract_references_for_item


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--item")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--commit", action="store_true", help="Persist edges; default is dry-run.")
    parser.add_argument("--heuristic", action="store_true", help="Do not send reference text to an LLM.")
    args = parser.parse_args()
    keys = [args.item] if args.item else list_item_keys()
    if args.limit is not None:
        keys = keys[: max(args.limit, 0)]
    totals = {"items": 0, "references": 0, "excluded": 0}
    for index, item_key in enumerate(keys, start=1):
        result = extract_references_for_item(
            item_key, dry_run=not args.commit, use_llm=not args.heuristic,
        )
        totals["items"] += 1
        totals["references"] += len(result.get("references") or [])
        totals["excluded"] += result.get("status") == "excluded"
        print(
            f"[{index}/{len(keys)}] {item_key}: {result['status']} "
            f"({len(result.get('references') or [])} references)", flush=True,
        )
        if len(keys) == 1:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(totals, ensure_ascii=False))


if __name__ == "__main__":
    main()
