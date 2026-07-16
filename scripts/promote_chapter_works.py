#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chunk_store import list_item_keys
from src.work_identity import promote_chapters


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote clear multi-author sections to child works.")
    parser.add_argument("--item")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--commit", action="store_true")
    args = parser.parse_args()
    if not args.item and not args.all:
        parser.error("--item or --all is required")
    keys = [args.item] if args.item else list_item_keys()
    for key in keys:
        result = promote_chapters(key, dry_run=not args.commit)
        if result["eligible"] or args.item:
            print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
