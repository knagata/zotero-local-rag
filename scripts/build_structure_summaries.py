#!/usr/bin/env python3
"""Generate v2 document-node summaries for one item or the local library."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.build_structure_summaries import build_structure_summaries, embed_structure_summaries
from src.chunk_store import list_item_keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--item", action="append")
    selector.add_argument("--all", action="store_true")
    parser.add_argument("--mode", choices=("extractive", "llm"), default="extractive")
    parser.add_argument("--embed", action="store_true", help="Rebuild the searchable __sum_node collection after summaries")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    keys = list(dict.fromkeys(args.item or list_item_keys()))
    if args.limit > 0:
        keys = keys[:args.limit]
    output = []
    failures = 0
    for key in keys:
        try:
            output.append(build_structure_summaries(key, mode=args.mode))
        except Exception as exc:
            failures += 1
            output.append({"item_key": key, "status": "failed", "error": str(exc)})
    embedding = None
    if args.embed and not failures:
        embedding = embed_structure_summaries(item_keys=set(keys))
    print(json.dumps({"items": output, "embedding": embedding, "failed": failures}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
