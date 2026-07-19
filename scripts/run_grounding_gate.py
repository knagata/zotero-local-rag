#!/usr/bin/env python3
"""Run a named-item grounding gate with DB writes and a complete audit JSON."""
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

from src.build_summaries import build_item
from src.env_utils import load_dotenv_native
from src.llm_client import LLMError, RateLimitReached


def _save(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _safe_error(exc: Exception) -> str:
    """Keep provider diagnostics without copying the submitted source text."""
    lines = [line.strip() for line in str(exc).splitlines()]
    diagnostics = [
        line for line in lines
        if line.startswith("ERROR:") or "usage limit" in line.casefold()
        or "rate limit" in line.casefold()
    ]
    return "\n".join(dict.fromkeys(diagnostics[-3:])) or f"{type(exc).__name__}: generation failed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llm", default="codex_cli:gpt-5.6-luna")
    parser.add_argument(
        "--write-database", action="store_true",
        help="Required acknowledgment: section/item summaries and cases will be replaced.",
    )
    args = parser.parse_args()
    if not args.write_database:
        parser.error("--write-database is required for this gate runner")
    load_dotenv_native(ROOT)
    os.environ["LLM_CHEAP"] = args.llm
    report = {
        "created_at": datetime.now().astimezone().isoformat(), "llm": args.llm,
        "writes_database": True, "items": [], "stop_reason": "completed",
    }
    for item_key in args.item:
        sections: list[dict] = []
        try:
            result = build_item(
                item_key, mode="llm", force=True, audit_sections=sections,
            )
            report["items"].append({
                "item_key": item_key, "status": result.get("status"),
                "result": result, "sections": sections,
            })
            print(json.dumps({
                "item_key": item_key, "status": result.get("status"),
                "sections": len(sections),
            }, ensure_ascii=False), flush=True)
        except RateLimitReached as exc:
            report["stop_reason"] = "rate_limit"
            report["error"] = _safe_error(exc)
            report["items"].append({
                "item_key": item_key, "status": "rate_limit_partial",
                "sections": sections,
            })
            _save(args.output, report)
            break
        except LLMError as exc:
            report["items"].append({
                "item_key": item_key, "status": "error", "error": _safe_error(exc),
                "sections": sections,
            })
        _save(args.output, report)
    _save(args.output, report)
    print(json.dumps({
        "output": str(args.output), "items": len(report["items"]),
        "stop_reason": report["stop_reason"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
