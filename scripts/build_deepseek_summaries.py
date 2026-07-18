#!/usr/bin/env python3
"""Build verified DeepSeek summary hierarchies for extractive-only items."""
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
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import build_summaries, db_relations
from src.chunk_store import get_item_chunks
from src.env_utils import load_dotenv_native
from src.llm_client import InvalidLLMResponse, LLMError, RateLimitReached
from src.manifest import load_manifest


VERSION = 1


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _eligible_keys(db_path: Path) -> list[str]:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return [row[0] for row in connection.execute(
            "SELECT item_key FROM item_summaries WHERE model='extractive' ORDER BY item_key"
        )]
    finally:
        connection.close()


def _backup_database(db_path: Path, backup_path: Path) -> None:
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(str(db_path))
    destination = sqlite3.connect(str(backup_path))
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()


def _section_hash(section: dict) -> str:
    return hashlib.sha256(
        build_summaries._section_source_text(section).encode("utf-8")
    ).hexdigest()


def _retry_summary(call) -> dict:
    last_error: Exception | None = None
    for _ in range(3):
        try:
            generated, model = call()
            verification = generated.get("_verification") or {}
            if verification.get("accepted"):
                return {
                    "status": "accepted", "model": model,
                    "summary": generated["summary"], "verification": verification,
                }
            last_error = InvalidLLMResponse(
                f"grounding gate rejected output: {verification}"
            )
        except RateLimitReached:
            raise
        except (InvalidLLMResponse, LLMError) as exc:
            last_error = exc
    kind = "quality_failure" if isinstance(last_error, InvalidLLMResponse) else "provider_failure"
    return {"status": kind, "error": str(last_error or "invalid output")}


def _generate_section(section: dict, model: str) -> dict:
    if build_summaries.classify_section_content(section) == "non_content":
        return {"status": "non_content"}
    return _retry_summary(lambda: build_summaries._llm_summary_only_section(
        section, model=model, reasoning="disabled",
    ))


def _process_item(
    item_key: str, *, model: str, workers: int, checkpoint_dir: Path,
) -> dict:
    excluded, reason = build_summaries._excluded_from_llm(item_key)
    if excluded:
        return {"item_key": item_key, "status": "excluded", "reason": reason}
    chunks = get_item_chunks(item_key)
    if not chunks:
        return {"item_key": item_key, "status": "no_chunks"}
    sections = build_summaries.split_sections(chunks)
    checkpoint_path = checkpoint_dir / f"{item_key}.json"
    spec = f"v={VERSION}|model={model}|reasoning=disabled"
    checkpoint = {"item_key": item_key, "spec": spec, "sections": {}}
    if checkpoint_path.exists():
        try:
            loaded = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if loaded.get("item_key") == item_key and loaded.get("spec") == spec:
                checkpoint = loaded
        except (OSError, json.JSONDecodeError):
            pass
    results: dict[str, dict] = {}
    pending: list[dict] = []
    for section in sections:
        section_id = section["section_id"]
        cached = checkpoint["sections"].get(section_id)
        if (
            cached and cached.get("input_hash") == _section_hash(section)
            and cached.get("result", {}).get("status") in {"accepted", "non_content"}
        ):
            results[section_id] = cached["result"]
        else:
            pending.append(section)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_generate_section, section, model): section for section in pending}
        for future in as_completed(futures):
            section = futures[future]
            section_id = section["section_id"]
            result = future.result()
            results[section_id] = result
            if result.get("status") in {"accepted", "non_content"}:
                checkpoint["sections"][section_id] = {
                    "input_hash": _section_hash(section), "result": result,
                }
                _write_json_atomic(checkpoint_path, checkpoint)

    failures = [
        {"section_id": section["section_id"], **results[section["section_id"]]}
        for section in sections
        if results[section["section_id"]].get("status") not in {"accepted", "non_content"}
    ]
    if failures:
        return {"item_key": item_key, "status": "section_failure", "failures": failures}
    section_rows = [
        {
            "section_id": section["section_id"], "chapter": section.get("chapter"),
            "summary": results[section["section_id"]]["summary"],
            "chunk_count": len(section["chunks"]),
        }
        for section in sections if results[section["section_id"]]["status"] == "accepted"
    ]
    if not section_rows:
        return {"item_key": item_key, "status": "no_content"}
    title = str(chunks[0].get("metadata", {}).get("title") or item_key)
    item_result = _retry_summary(lambda: build_summaries._llm_summary_only_item(
        title, section_rows, model=model, reasoning="disabled",
    ))
    if item_result["status"] != "accepted":
        return {"item_key": item_key, "status": "item_failure", "failure": item_result}
    existing = db_relations.get_item_summary(item_key) or {}
    model_name = item_result["model"]
    replaced = db_relations.replace_extractive_summary_bundle(
        item_key, item_result["summary"], section_rows, model=model_name,
        chunk_count=len(chunks),
        source_mtime=build_summaries._source_mtime(
            chunks, load_manifest(build_summaries.MANIFEST_PATH),
        ),
    )
    if not replaced:
        return {"item_key": item_key, "status": "protected_existing"}
    return {
        "item_key": item_key, "status": "updated", "model": model_name,
        "sections": len(section_rows), "previous_model": existing.get("model"),
        "item_verification": item_result["verification"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item", action="append")
    parser.add_argument("--max-items", type=int, default=10)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--no-embed", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=ROOT / "data" / "nightly_checkpoint" / "deepseek-summary-only-batch",
    )
    parser.add_argument("--backup", type=Path)
    args = parser.parse_args()
    if args.max_items < 0 or args.workers < 1 or (args.max_hours is not None and args.max_hours <= 0):
        parser.error("invalid item, worker, or time limit")
    load_dotenv_native(ROOT)
    db_path = Path(os.environ.get("RELATIONS_DB_PATH", ROOT / "data" / "relations.db"))
    eligible = _eligible_keys(db_path)
    requested = args.item or eligible
    keys = [key for key in requested if key in set(eligible)][:args.max_items]
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    backup_path = args.backup or ROOT / "data" / "backups" / f"relations-before-deepseek-{stamp}.db"
    report = {
        "created_at": datetime.now().astimezone().isoformat(), "version": VERSION,
        "model": args.model, "items": [], "backup": None, "stop_reason": "completed",
    }
    stop = {"requested": False}

    def request_stop(_signum: int, _frame: object) -> None:
        stop["requested"] = True

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    updated: set[str] = set()
    started = time.monotonic()
    try:
        for key in keys:
            if stop["requested"] or (args.stop_file and args.stop_file.exists()):
                report["stop_reason"] = "stop_requested"
                break
            if args.max_hours is not None and time.monotonic() - started >= args.max_hours * 3600:
                report["stop_reason"] = "max_hours"
                break
            if report["backup"] is None:
                _backup_database(db_path, backup_path)
                report["backup"] = str(backup_path)
            try:
                result = _process_item(
                    key, model=args.model, workers=args.workers,
                    checkpoint_dir=args.checkpoint_dir,
                )
            except RateLimitReached as exc:
                report["stop_reason"] = "rate_limit"
                report["error"] = str(exc)
                break
            report["items"].append(result)
            if result["status"] == "updated":
                updated.add(key)
            _write_json_atomic(args.output, report)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if updated and not args.no_embed:
        report["embedded"] = build_summaries.embed_summaries(item_keys=updated)
    report["updated"] = len(updated)
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    _write_json_atomic(args.output, report)
    print(json.dumps({
        "output": str(args.output), "updated": len(updated),
        "stop_reason": report["stop_reason"], "embedded": report.get("embedded"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
