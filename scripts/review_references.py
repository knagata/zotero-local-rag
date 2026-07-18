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

from src.db_relations import (
    apply_reference_review_decisions, get_reference_review_candidates,
    set_reference_review_status,
)
from src.reference_agent import commit_approved_reference_candidates


def validate_decision_coverage(decisions: list[dict], batch_path: Path) -> None:
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    expected = {int(row["review_id"]) for row in batch.get("candidates") or []}
    actual = [int(row["review_id"]) for row in decisions]
    if len(actual) != len(set(actual)):
        raise ValueError("duplicate review_id in response")
    if set(actual) != expected:
        missing = sorted(expected - set(actual))
        extra = sorted(set(actual) - expected)
        raise ValueError(f"response coverage mismatch: missing={missing}, extra={extra}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--status", choices=["pending", "approved", "rejected"], default="pending")
    list_parser.add_argument("--limit", type=int, default=20)
    list_parser.add_argument("--source-kind")
    set_parser = subparsers.add_parser("set")
    set_parser.add_argument("review_id", type=int)
    set_parser.add_argument("status", choices=["pending", "approved", "rejected"])
    set_parser.add_argument("--note")
    commit_parser = subparsers.add_parser("commit-approved")
    commit_parser.add_argument("--limit", type=int, default=100)
    apply_parser = subparsers.add_parser("apply-decisions")
    apply_parser.add_argument("path", type=Path)
    apply_parser.add_argument("--expected-batch", type=Path)
    args = parser.parse_args()

    if args.command == "list":
        rows = get_reference_review_candidates(
            args.status, source_kind=args.source_kind,
        )[: max(args.limit, 0)]
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    if args.command == "commit-approved":
        print(json.dumps(commit_approved_reference_candidates(limit=args.limit), ensure_ascii=False, indent=2))
        return
    if args.command == "apply-decisions":
        payload = json.loads(args.path.read_text(encoding="utf-8"))
        decisions = payload.get("decisions") if isinstance(payload, dict) else payload
        if not isinstance(decisions, list):
            raise ValueError("decision file must be an array or an object containing decisions")
        if args.expected_batch:
            validate_decision_coverage(decisions, args.expected_batch)
        applied = apply_reference_review_decisions(decisions)
        print(json.dumps({"examined": len(decisions), "applied": applied}, ensure_ascii=False, indent=2))
        return
    changed = set_reference_review_status(args.review_id, args.status, args.note)
    print(json.dumps({
        "review_id": args.review_id, "status": args.status,
        "result": "updated" if changed else "not_found",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
