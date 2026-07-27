#!/usr/bin/env python3
"""Generate evidence-grounded v2 cases from canonical document leaves."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.build_structure_cases import build_structure_cases
from src.build_summaries import embed_summaries
from src.chunk_store import list_item_keys
from src.db_relations import mark_artifact_status
from src.env_utils import load_dotenv_native
from src.llm_client import LLMError, RateLimitReached


def main() -> None:
    load_dotenv_native(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--item", action="append")
    selector.add_argument("--all", action="store_true")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--max-cases-per-leaf", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-embed", action="store_true", help="Leave case-vector rebuilding to a later command.")
    args = parser.parse_args()
    if args.samples < 1 or args.max_cases_per_leaf < 1:
        parser.error("--samples and --max-cases-per-leaf must be positive")
    keys = list(dict.fromkeys(args.item or list_item_keys()))
    if args.limit > 0:
        keys = keys[:args.limit]
    report = {"items": [], "stop_reason": "completed"}
    for key in keys:
        try:
            report["items"].append(build_structure_cases(key, samples=args.samples, max_cases_per_leaf=args.max_cases_per_leaf))
        except RateLimitReached as exc:
            report["items"].append({"item_key": key, "status": "rate_limited", "error": str(exc)})
            report["stop_reason"] = "rate_limited"
            break
        except LLMError as exc:
            report["items"].append({"item_key": key, "status": "failed", "error": str(exc)})
    updated = {
        str(row["item_key"]) for row in report["items"]
        if row.get("status") in {"success", "empty"}
    }
    if updated and not args.no_embed and report["stop_reason"] == "completed":
        try:
            report["embedded"] = embed_summaries(item_keys=updated)
            for key in updated:
                mark_artifact_status(key, "embeddings", "success", processor_version="structure-case-v2-1", counts=report["embedded"])
        except Exception as exc:
            report["embedding_error"] = str(exc)
            for key in updated:
                mark_artifact_status(key, "embeddings", "failed", reason_code="embedding_failed", message=str(exc)[:1000], retryable=True)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
