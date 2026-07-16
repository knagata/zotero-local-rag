#!/usr/bin/env python3
"""Generate a no-write extractive/Codex section-summary quality comparison."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import build_summaries
from src.chunk_store import get_item_chunks, list_item_keys
from src.env_utils import load_dotenv_native
from src.llm_client import LLMError, RateLimitReached


def compare_item(item_key: str, *, max_sections: int) -> dict:
    excluded, reason = build_summaries._excluded_from_llm(item_key)
    if excluded:
        return {"item_key": item_key, "status": "excluded", "reason": reason, "sections": []}
    chunks = get_item_chunks(item_key)
    sections = build_summaries.split_sections(chunks)[:max(max_sections, 0)]
    output = []
    for section in sections:
        extractive = build_summaries._extractive_section(section)
        generated, model = build_summaries._llm_section(section)
        output.append({
            "section_id": section["section_id"], "chapter": section["chapter"],
            "model": model, "extractive_summary": extractive["summary"],
            "llm_summary": generated.get("summary"), "cases": generated.get("cases") or [],
            "chapter_authors": generated.get("chapter_authors") or [],
            "first_publication_note": generated.get("first_publication_note"),
        })
    return {"item_key": item_key, "status": "compared", "sections": output}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item", action="append")
    parser.add_argument("--max-items", type=int, default=20)
    parser.add_argument("--max-sections", type=int, default=3)
    parser.add_argument("--llm", default="codex_cli:auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    load_dotenv_native(ROOT)
    os.environ["LLM_SUMMARY"] = args.llm
    keys = (args.item or list_item_keys())[:max(args.max_items, 0)]
    report = {
        "created_at": datetime.now().astimezone().isoformat(), "llm": args.llm,
        "writes_database": False, "items": [], "stop_reason": "completed",
    }
    for key in keys:
        try:
            report["items"].append(compare_item(key, max_sections=args.max_sections))
        except RateLimitReached as exc:
            report["stop_reason"] = "rate_limit"
            report["error"] = str(exc)
            break
        except LLMError as exc:
            report["items"].append({"item_key": key, "status": "error", "error": str(exc)})
    output = args.output or ROOT / "data" / "quality" / "summary-comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "items": len(report["items"]),
        "stop_reason": report["stop_reason"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
