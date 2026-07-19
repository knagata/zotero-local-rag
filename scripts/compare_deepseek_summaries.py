#!/usr/bin/env python3
"""Compare DeepSeek summary-only models/reasoning modes without writing the DB."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import build_summaries
from src.chunk_store import get_item_chunks
from src.env_utils import load_dotenv_native
from src.llm_client import InvalidLLMResponse, LLMError, RateLimitReached


VERSION = 3


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


def _extractive_keys(db_path: Path) -> list[str]:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return [row[0] for row in connection.execute(
            "SELECT item_key FROM item_summaries WHERE model='extractive' ORDER BY item_key"
        )]
    finally:
        connection.close()


def _merge_usage(rows: list[dict]) -> dict:
    keys = (
        "prompt_tokens", "completion_tokens", "total_tokens",
        "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
    )
    return {key: sum(int(row.get(key) or 0) for row in rows) for key in keys}


def _generate_mode(
    section: dict, model: str, reasoning: str, attempts: int = 1,
) -> dict:
    last_error: Exception | None = None
    attempt_rows: list[dict] = []
    for attempt in range(1, attempts + 1):
        try:
            generated, model_name = build_summaries._llm_summary_only_section(
                section, model=model, reasoning=reasoning,
            )
            verification = generated.get("_verification") or {}
            row = {
                "status": "accepted" if verification.get("accepted") else "quality_failure",
                "model": model_name, "summary": generated.get("summary"),
                "sentences": generated.get("sentences") or [],
                "verification": verification, "usage": generated.get("_usage"),
            }
            attempt_rows.append(row)
            if row["status"] == "accepted":
                row["attempt_count"] = attempt
                row["usage"] = _merge_usage([
                    value.get("usage") or {} for value in attempt_rows
                ])
                return row
        except RateLimitReached:
            raise
        except (InvalidLLMResponse, LLMError) as exc:
            last_error = exc
            attempt_rows.append({"status": "provider_failure", "error": str(exc)})
    if attempt_rows and any(row["status"] == "quality_failure" for row in attempt_rows):
        result = next(
            row for row in reversed(attempt_rows) if row["status"] == "quality_failure"
        )
        result["attempt_count"] = len(attempt_rows)
        result["usage"] = _merge_usage([
            value.get("usage") or {} for value in attempt_rows
        ])
        return result
    return {
        "status": "provider_failure", "attempt_count": len(attempt_rows),
        "error": str(last_error or "invalid output"),
    }


def _compare_section(
    section: dict, model: str,
    reasoning_modes: tuple[str, ...] = ("disabled", "enabled"),
    attempts: int = 1,
) -> dict:
    if build_summaries.classify_section_content(section) == "non_content":
        return {
            "section_id": section["section_id"], "chapter": section["chapter"],
            "status": "non_content", "extractive_summary": None, "modes": {},
        }
    extractive = build_summaries._extractive_section(section)["summary"]
    modes = {
        reasoning: _generate_mode(section, model, reasoning, attempts)
        for reasoning in reasoning_modes
    }
    return {
        "section_id": section["section_id"], "chapter": section["chapter"],
        "status": "compared", "extractive_summary": extractive, "modes": modes,
    }


def _compare_item(
    item_key: str, *, model: str, max_sections: int, workers: int,
    checkpoint_dir: Path, reasoning_modes: tuple[str, ...] = ("disabled", "enabled"),
    attempts: int = 1,
) -> dict:
    excluded, reason = build_summaries._excluded_from_llm(item_key)
    if excluded:
        return {"item_key": item_key, "status": "excluded", "reason": reason, "sections": []}
    sections = build_summaries.split_sections(get_item_chunks(item_key))[:max_sections]
    checkpoint_path = checkpoint_dir / f"{item_key}.json"
    spec = (
        f"v={VERSION}|model={model}|modes={','.join(reasoning_modes)}"
        f"|attempts={attempts}"
    )
    checkpoint = {"item_key": item_key, "spec": spec, "sections": {}}
    if checkpoint_path.exists():
        try:
            loaded = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if loaded.get("item_key") == item_key and loaded.get("spec") == spec:
                checkpoint = loaded
        except (OSError, json.JSONDecodeError):
            pass
    output_by_id: dict[str, dict] = {}
    pending: list[dict] = []
    for section in sections:
        section_id = section["section_id"]
        cached = checkpoint["sections"].get(section_id)
        if cached and cached.get("input_hash") == _section_hash(section):
            output_by_id[section_id] = cached["result"]
        else:
            pending.append(section)

    def save(section: dict, result: dict) -> None:
        section_id = section["section_id"]
        output_by_id[section_id] = result
        checkpoint["sections"][section_id] = {
            "input_hash": _section_hash(section), "result": result,
        }
        _write_json_atomic(checkpoint_path, checkpoint)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_compare_section, section, model, reasoning_modes, attempts): section
            for section in pending
        }
        for future in as_completed(futures):
            save(futures[future], future.result())
    return {
        "item_key": item_key, "status": "compared",
        "sections": [output_by_id[section["section_id"]] for section in sections],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item", action="append")
    parser.add_argument("--max-items", type=int, default=3)
    parser.add_argument("--max-sections", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument(
        "--reasoning", action="append", choices=["disabled", "enabled"],
        help="Reasoning mode to test; repeat to test both (default: both).",
    )
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=ROOT / "data" / "summary_checkpoints" / "deepseek-summary-only",
    )
    parser.add_argument("--stop-file", type=Path)
    args = parser.parse_args()
    if args.max_items < 0 or args.max_sections < 1 or args.workers < 1 or args.attempts < 1:
        parser.error("require non-negative --max-items and positive section/worker/attempt limits")
    load_dotenv_native(ROOT)
    db_path = Path(os.environ.get("RELATIONS_DB_PATH", ROOT / "data" / "relations.db"))
    extractive = set(_extractive_keys(db_path))
    requested = args.item or sorted(extractive)
    keys = [key for key in requested if key in extractive][:args.max_items]
    skipped_non_extractives = [key for key in requested if key not in extractive]
    reasoning_modes = tuple(dict.fromkeys(args.reasoning or ("disabled", "enabled")))
    report = {
        "created_at": datetime.now().astimezone().isoformat(), "version": VERSION,
        "model": args.model, "reasoning_modes": list(reasoning_modes),
        "attempts": args.attempts,
        "writes_database": False,
        "items": [], "skipped_non_extractive_items": skipped_non_extractives,
        "stop_reason": "completed",
    }
    stop = {"requested": False}

    def request_stop(_signum: int, _frame: object) -> None:
        stop["requested"] = True

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for key in keys:
            if stop["requested"] or (args.stop_file and args.stop_file.exists()):
                report["stop_reason"] = "stop_requested"
                break
            try:
                report["items"].append(_compare_item(
                    key, model=args.model, max_sections=args.max_sections,
                    workers=args.workers, checkpoint_dir=args.checkpoint_dir,
                    reasoning_modes=reasoning_modes,
                    attempts=args.attempts,
                ))
            except RateLimitReached as exc:
                report["stop_reason"] = "rate_limit"
                report["error"] = str(exc)
                break
            except LLMError as exc:
                report["stop_reason"] = "provider_unavailable"
                report["error"] = str(exc)
                break
            _write_json_atomic(args.output, report)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    _write_json_atomic(args.output, report)
    print(json.dumps({
        "output": str(args.output), "items": len(report["items"]),
        "stop_reason": report["stop_reason"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
