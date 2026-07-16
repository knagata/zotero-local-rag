#!/usr/bin/env python3
"""Inspect or adjudicate staged reference candidates without changing the works graph."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db_relations import get_reference_review_candidates, set_reference_review_status
from src.reference_agent import commit_approved_reference_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--status", choices=["pending", "approved", "rejected"], default="pending")
    list_parser.add_argument("--limit", type=int, default=20)
    set_parser = subparsers.add_parser("set")
    set_parser.add_argument("review_id", type=int)
    set_parser.add_argument("status", choices=["pending", "approved", "rejected"])
    set_parser.add_argument("--note")
    commit_parser = subparsers.add_parser("commit-approved")
    commit_parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    if args.command == "list":
        rows = get_reference_review_candidates(args.status)[: max(args.limit, 0)]
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    if args.command == "commit-approved":
        print(json.dumps(commit_approved_reference_candidates(limit=args.limit), ensure_ascii=False, indent=2))
        return
    changed = set_reference_review_status(args.review_id, args.status, args.note)
    print(json.dumps({
        "review_id": args.review_id, "status": args.status,
        "result": "updated" if changed else "not_found",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
