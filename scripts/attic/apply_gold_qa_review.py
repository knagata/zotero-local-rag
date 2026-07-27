#!/usr/bin/env python3
"""Validate Claude's Gold QA decisions and write evaluator-ready JSONL."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def apply_review(package_path: Path, response_path: Path, output_path: Path, *, minimum: int = 20) -> dict[str, Any]:
    package = json.loads(package_path.read_text(encoding="utf-8"))
    response = json.loads(response_path.read_text(encoding="utf-8"))
    candidates = {row["candidate_id"]: row for row in package.get("candidates") or []}
    decisions = response.get("gold_qa") if isinstance(response, dict) else None
    if not isinstance(decisions, list):
        raise ValueError("response must contain a gold_qa array")
    output_rows = []
    seen_ids = set()
    seen_queries = set()
    for decision in decisions:
        candidate_id = str(decision.get("candidate_id") or "")
        if candidate_id in seen_ids or candidate_id not in candidates:
            raise ValueError(f"invalid or duplicate candidate_id: {candidate_id}")
        seen_ids.add(candidate_id)
        if decision.get("decision") == "exclude":
            continue
        if decision.get("decision") != "include":
            raise ValueError(f"{candidate_id}: invalid decision")
        candidate = candidates[candidate_id]
        query = " ".join(str(decision.get("query") or "").split())
        if len(query) < 6:
            raise ValueError(f"{candidate_id}: query is too short")
        query_key = re.sub(r"\W+", "", query).casefold()
        if query_key in seen_queries:
            raise ValueError(f"{candidate_id}: duplicate query")
        seen_queries.add(query_key)
        if decision.get("expected_item_keys") != [candidate["item_key"]]:
            raise ValueError(f"{candidate_id}: expected_item_keys changed")
        evidence_ids = decision.get("evidence_chunk_ids") or []
        if candidate["chunk_id"] not in evidence_ids:
            raise ValueError(f"{candidate_id}: source chunk is missing")
        output_rows.append({
            "id": candidate_id, "query": query,
            "expected_item_keys": [candidate["item_key"]],
            "evidence_chunk_ids": evidence_ids,
            "language": candidate.get("language"), "note": decision.get("note"),
        })
    if seen_ids != set(candidates):
        missing = sorted(set(candidates) - seen_ids)
        raise ValueError(f"response does not cover every candidate: missing={missing}")
    if len(output_rows) < minimum:
        raise ValueError(f"only {len(output_rows)} included questions; minimum is {minimum}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows), encoding="utf-8",
    )
    return {"reviewed": len(decisions), "included": len(output_rows), "output": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("response", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/quality/gold_qa.jsonl"))
    parser.add_argument("--minimum", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(
        apply_review(args.package, args.response, args.output, minimum=args.minimum),
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
