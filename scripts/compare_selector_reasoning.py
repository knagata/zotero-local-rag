#!/usr/bin/env python3
"""Compare DeepSeek case judging with and without reasoning; never write the DB."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import build_summaries
from src.env_utils import load_dotenv_native
from src.llm_client import DeepSeekClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--min-votes", type=int, default=2)
    args = parser.parse_args()
    if args.samples < 1 or not 1 <= args.min_votes <= args.samples:
        parser.error("require positive --samples and 1 <= --min-votes <= --samples")
    load_dotenv_native(ROOT)
    report = json.loads(args.input.read_text(encoding="utf-8"))
    units: list[dict[str, str]] = []
    provenance: dict[str, dict[str, str]] = {}
    seen_quotes: set[str] = set()
    excluded_items: list[dict[str, str]] = []
    for item in report.get("items") or []:
        item_key = str(item.get("item_key") or "")
        excluded, reason = build_summaries._excluded_from_llm(item_key)
        if excluded:
            excluded_items.append({"item_key": item_key, "reason": str(reason or "excluded")})
            continue
        for section in item.get("sections") or []:
            for case in section.get("cases") or []:
                quote = str(case.get("evidence_quote") or "").strip()
                if not quote or quote in seen_quotes:
                    continue
                seen_quotes.add(quote)
                unit_id = f"u{len(units) + 1:04d}"
                units.append({"unit_id": unit_id, "chunk_id": "report", "text": quote})
                provenance[unit_id] = {
                    "item_key": item_key, "section_id": str(section.get("section_id") or ""),
                }
    candidate_ids = {unit["unit_id"] for unit in units}
    comparisons: dict[str, dict] = {}
    for thinking in ("disabled", "enabled"):
        client = DeepSeekClient(args.model, thinking=thinking)
        accepted, stats = build_summaries._judge_selector_case_ids(
            client, candidate_ids, units, samples=args.samples, min_votes=args.min_votes,
        )
        comparisons[thinking] = {
            **stats, "accepted_ids": sorted(accepted),
            "accepted": [
                {**provenance[unit["unit_id"]], "evidence_quote": unit["text"]}
                for unit in units if unit["unit_id"] in accepted
            ],
        }
    output = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source": str(args.input), "writes_database": False, "model": args.model,
        "samples": args.samples, "min_votes": args.min_votes,
        "candidate_count": len(units), "excluded_items": excluded_items,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output), "candidates": len(units),
        "disabled_accepted": len(comparisons["disabled"]["accepted"]),
        "enabled_accepted": len(comparisons["enabled"]["accepted"]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
