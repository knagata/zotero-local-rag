#!/usr/bin/env python3
"""Generate a resumable, no-database-write section-summary quality comparison."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import signal
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import build_summaries
from src.chunk_store import get_item_chunks, list_item_keys
from src.env_utils import load_dotenv_native
from src.llm_client import InvalidLLMResponse, LLMError, RateLimitReached


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    temporary.replace(path)


def _section_hash(section: dict) -> str:
    return hashlib.sha256(
        build_summaries._section_source_text(section).encode("utf-8")
    ).hexdigest()


def _classify(result: dict) -> str:
    if result["status"] == "skipped_non_content":
        return "non_content"
    if build_summaries.is_meta_summary(str(result.get("llm_summary") or "")):
        return "model_quality_failure"
    verification = result.get("verification") or {}
    # A section that generated no structured evidence is valid; only generated
    # fields that were actually discarded can make the section suspicious.
    if verification.get("total_generated", 0) and verification.get("discard_rate", 0) > 0.5:
        return "model_quality_failure"
    return "accepted"


def _compare_section(
    section: dict, *, strategy: str = "quote", samples: int = 3, min_votes: int = 2,
    judge_samples: int = 3, judge_min_votes: int = 2, judge_reasoning: str = "disabled",
) -> dict:
    extractive = build_summaries._extractive_section(section)
    if build_summaries.classify_section_content(section) == "non_content":
        result = {
            "section_id": section["section_id"], "chapter": section["chapter"],
            "status": "skipped_non_content", "extractive_summary": extractive["summary"],
            "llm_summary": None, "chapter_authors": [],
            "first_publication_note": None, "verification": None,
        }
        result["classification"] = _classify(result)
        return result
    try:
        generated, model = build_summaries._llm_section(section)
    except InvalidLLMResponse as exc:
        return {
            "section_id": section["section_id"], "chapter": section["chapter"],
            "status": "model_quality_failure", "classification": "model_quality_failure",
            "error": str(exc), "extractive_summary": extractive["summary"],
            "llm_summary": None, "chapter_authors": [],
            "first_publication_note": None, "verification": None,
        }
    result = {
        "section_id": section["section_id"], "chapter": section["chapter"],
        "status": "generated", "model": model, "extractive_summary": extractive["summary"],
        "llm_summary": generated.get("summary"),
        "chapter_authors": generated.get("chapter_authors") or [],
        "first_publication_note": generated.get("first_publication_note"),
        "verification": generated.get("_verification"),
    }
    result["classification"] = _classify(result)
    return result


def compare_item(
    item_key: str, *, max_sections: int, workers: int = 1,
    checkpoint_dir: Path | None = None, llm_spec: str = "", strategy: str = "quote",
    samples: int = 3, min_votes: int = 2,
    judge_samples: int = 3, judge_min_votes: int = 2, judge_reasoning: str = "disabled",
) -> dict:
    chunks = get_item_chunks(item_key)
    sections = build_summaries.split_sections(chunks)[:max(max_sections, 0)]
    checkpoint_path = checkpoint_dir / f"{item_key}.json" if checkpoint_dir else None
    checkpoint: dict = {"item_key": item_key, "llm": llm_spec, "sections": {}}
    if checkpoint_path and checkpoint_path.exists():
        try:
            loaded = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if loaded.get("item_key") == item_key and loaded.get("llm") == llm_spec:
                checkpoint = loaded
        except (OSError, json.JSONDecodeError):
            pass

    output_by_id: dict[str, dict] = {}
    pending: list[dict] = []
    for section in sections:
        section_id = section["section_id"]
        input_hash = _section_hash(section)
        cached = checkpoint["sections"].get(section_id)
        if cached and cached.get("input_hash") == input_hash and isinstance(cached.get("result"), dict):
            output_by_id[section_id] = cached["result"]
        else:
            checkpoint["sections"].pop(section_id, None)
            pending.append(section)

    def save_result(section: dict, result: dict) -> None:
        section_id = section["section_id"]
        output_by_id[section_id] = result
        checkpoint["sections"][section_id] = {
            "input_hash": _section_hash(section),
            "classification": result["classification"],
            "result": result,
        }
        if checkpoint_path:
            _write_json_atomic(checkpoint_path, checkpoint)

    if workers == 1:
        for section in pending:
            save_result(section, _compare_section(
                section, strategy=strategy, samples=samples, min_votes=min_votes,
                judge_samples=judge_samples, judge_min_votes=judge_min_votes,
                judge_reasoning=judge_reasoning,
            ))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _compare_section, section, strategy=strategy,
                    samples=samples, min_votes=min_votes,
                    judge_samples=judge_samples, judge_min_votes=judge_min_votes,
                    judge_reasoning=judge_reasoning,
                ): section for section in pending
            }
            for future in as_completed(futures):
                save_result(futures[future], future.result())

    output = [output_by_id[section["section_id"]] for section in sections]
    statuses = {section["status"] for section in output}
    status = "skipped_non_content" if statuses == {"skipped_non_content"} else "compared"
    return {"item_key": item_key, "status": status, "sections": output}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item", action="append")
    parser.add_argument("--max-items", type=int, default=20)
    parser.add_argument("--max-sections", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--llm", default="deepseek:deepseek-v4-pro")
    parser.add_argument("--strategy", choices=["summary-only"], default="summary-only")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--min-votes", type=int, default=2)
    parser.add_argument("--judge-samples", type=int, default=3)
    parser.add_argument("--judge-min-votes", type=int, default=2)
    parser.add_argument(
        "--judge-reasoning", choices=["enabled", "disabled"], default="disabled",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=ROOT / "data" / "summary_checkpoints" / "deepseek-pilot",
    )
    parser.add_argument(
        "--stop-file", type=Path,
        help="Stop before starting the next item while this file exists.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.min_votes < 1 or args.min_votes > args.samples:
        parser.error("--min-votes must be between 1 and --samples")
    if args.judge_samples < 1:
        parser.error("--judge-samples must be positive")
    if args.judge_min_votes < 1 or args.judge_min_votes > args.judge_samples:
        parser.error("--judge-min-votes must be between 1 and --judge-samples")
    load_dotenv_native(ROOT)
    os.environ["LLM_CHEAP"] = args.llm
    keys = (args.item or list_item_keys())[:max(args.max_items, 0)]
    checkpoint_spec = (
        f"{args.llm}|strategy={args.strategy}"
        f"|samples={args.samples}|votes={args.min_votes}"
        f"|judge_samples={args.judge_samples}|judge_votes={args.judge_min_votes}"
        f"|judge_reasoning={args.judge_reasoning}"
    )
    report = {
        "created_at": datetime.now().astimezone().isoformat(), "llm": args.llm,
        "strategy": args.strategy, "samples": args.samples, "min_votes": args.min_votes,
        "judge_samples": args.judge_samples, "judge_min_votes": args.judge_min_votes,
        "judge_reasoning": args.judge_reasoning,
        "strategy_version": None,
        "writes_database": False, "items": [], "stop_reason": "completed",
    }
    output = args.output or ROOT / "data" / "quality" / "summary-comparison.json"
    signal_state = {"requested": False}

    def request_stop(_signum: int, _frame: object) -> None:
        signal_state["requested"] = True

    previous_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    for key in keys:
        if signal_state["requested"] or (
            args.stop_file is not None and args.stop_file.exists()
        ):
            report["stop_reason"] = "stop_requested"
            break
        try:
            report["items"].append(compare_item(
                key, max_sections=args.max_sections, workers=args.workers,
                checkpoint_dir=args.checkpoint_dir, llm_spec=checkpoint_spec,
                strategy=args.strategy, samples=args.samples, min_votes=args.min_votes,
                judge_samples=args.judge_samples, judge_min_votes=args.judge_min_votes,
                judge_reasoning=args.judge_reasoning,
            ))
        except RateLimitReached as exc:
            report["stop_reason"] = "rate_limit"
            report["error"] = str(exc)
            break
        except LLMError as exc:
            report["items"].append({
                "item_key": key, "status": "provider_unavailable", "error": str(exc),
            })
            report["stop_reason"] = "provider_unavailable"
            break
        _write_json_atomic(output, report)
    for signum, handler in previous_handlers.items():
        signal.signal(signum, handler)
    _write_json_atomic(output, report)
    print(json.dumps({
        "output": str(output), "items": len(report["items"]),
        "stop_reason": report["stop_reason"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
