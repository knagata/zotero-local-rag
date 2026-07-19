#!/usr/bin/env python3
"""Collect high-recall grounded cases independently from summary generation."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import build_summaries, db_relations
from src.chunk_store import get_item_chunks, list_item_keys
from src.env_utils import load_dotenv_native
from src.llm_client import InvalidLLMResponse, LLMError, RateLimitReached, get_llm


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("version") == 1 and isinstance(value.get("items"), dict):
            return value
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    return {"version": 1, "items": {}}


def _source_hash(chunks: list[dict]) -> str:
    return hashlib.sha256("\n".join(
        f"{row.get('id')}\0{row.get('text')}" for row in chunks
    ).encode("utf-8")).hexdigest()


def _extend_boundary_evidence(unit_id: str, units: list[dict], quote: str) -> tuple[str, list[dict]]:
    """Join only an obvious lower-case continuation and retain both exact source pieces."""
    lookup = {row["unit_id"]: index for index, row in enumerate(units)}
    index = lookup[unit_id]
    current = units[index]
    evidence = [{"field_name": "description", "chunk_id": current["chunk_id"],
                 "evidence_quote": quote}]
    if quote.rstrip()[-1:] in ".!?。！？』」”’)]" or index + 1 >= len(units):
        return quote, evidence
    following = str(units[index + 1]["text"] or "").strip()
    first = re.search(r"[A-Za-z]", following[:80])
    if not first or not first.group(0).islower():
        return quote, evidence
    end = re.search(r"[.!?。！？]", following)
    continuation = following[: end.end() if end else min(len(following), 500)].strip()
    if not continuation:
        return quote, evidence
    evidence.append({"field_name": "description", "chunk_id": units[index + 1]["chunk_id"],
                     "evidence_quote": continuation})
    return f"{quote.rstrip()} {continuation}", evidence


def _extract_section(section: dict, *, samples: int, max_cases: int = 5) -> tuple[list[dict], dict]:
    units = build_summaries._section_evidence_units(section)
    if not units:
        return [], {"candidates": 0, "accepted": 0}
    cheap = get_llm("cheap")
    source = "\n\n".join(f"[{row['unit_id']}]\n{row['text']}" for row in units)
    prompt = (
        "次の学術資料の原文単位から、経験的・民族誌的・歴史的な具体事例の候補を"
        "最大8件選んでください。実在する人物・集団・組織・場所・作品についての具体的な"
        "行為、実践、経験、観察、事件、測定を含む単位を優先します。抽象理論、章の方針、"
        "単なる人名・文献列挙は選びません。引用を転記せず、必ずevidence_unit_idを返して"
        "ください。region等は同じ単位に文字どおり存在する値だけを記入し、不明ならnullまたは"
        "空配列にしてください。\n\n" + source
    )
    generated = []
    last_error = None
    for _ in range(samples + 2):
        try:
            generated.append(cheap.generate_json(
                prompt, schema=build_summaries.SELECTOR_SECTION_SCHEMA, timeout=300,
            ))
        except InvalidLLMResponse as exc:
            last_error = exc
        if len(generated) >= samples:
            break
    if len(generated) < samples:
        raise last_error or InvalidLLMResponse("Too few valid case-selector samples.")
    votes: Counter[str] = Counter()
    first: dict[str, dict] = {}
    for sample in generated:
        seen = set()
        for row in sample.get("cases") or []:
            unit_id = str(row.get("evidence_unit_id") or "") if isinstance(row, dict) else ""
            if unit_id and unit_id not in seen:
                votes[unit_id] += 1
                first.setdefault(unit_id, row)
                seen.add(unit_id)
    lookup = {row["unit_id"]: row for row in units}
    candidate_ids = {
        unit_id for unit_id in votes
        if unit_id in lookup and build_summaries._is_self_contained_evidence(lookup[unit_id]["text"])
    }
    standard = get_llm("standard")
    accepted, judge = build_summaries._judge_selector_case_ids(
        standard, candidate_ids, units, samples=1, min_votes=1, max_cases=3,
    )
    ranked = sorted(candidate_ids, key=lambda value: (value not in accepted, -votes[value], value))[:max_cases]
    cases = []
    verification_rows = []
    for unit_id in ranked:
        selected = {"summary": "", "cases": [first[unit_id]],
                    "chapter_authors": [], "first_publication_note": None}
        hydrated, verification = build_summaries._hydrate_selector_result(selected, units, section)
        verification_rows.append(verification)
        if not hydrated.get("cases"):
            continue
        row = hydrated["cases"][0]
        optional = sum(bool(row.get(field)) for field in ("region", "group", "period", "practices", "phenomena"))
        is_accepted = unit_id in accepted
        row["quality_status"] = "confirmed" if is_accepted and optional >= 2 else "partial" if is_accepted else "candidate"
        row["confidence"] = round((0.7 + 0.1 * min(votes[unit_id], 3)) if is_accepted else 0.35, 2)
        row["chunk_id"] = lookup[unit_id]["chunk_id"]
        row["description"], row["evidence"] = _extend_boundary_evidence(
            unit_id, units, row["evidence_quote"],
        )
        if len(row["evidence"]) > 1:
            row["evidence_quote"] = row["description"]
        cases.append(row)
    return cases, {"candidates": len(candidate_ids), "accepted": len(accepted),
                   "saved": len(cases), "votes": dict(votes), "judge": judge,
                   "verification": verification_rows}


def main() -> int:
    load_dotenv_native(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item", action="append")
    parser.add_argument("--force", action="store_true", help="Rebuild explicitly selected/protected items.")
    parser.add_argument("--max-items", type=int, default=int(os.environ.get("CASE_BATCH_MAX_ITEMS", "10")))
    parser.add_argument("--samples", type=int, default=int(os.environ.get("CASE_SELECTOR_SAMPLES", "2")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CASE_BATCH_WORKERS", "10")))
    parser.add_argument("--max-cases-per-section", type=int, default=int(os.environ.get("CASE_MAX_PER_SECTION", "5")))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, default=ROOT / "data" / "summary_checkpoints" / "deepseek-cases" / "ledger.json")
    parser.add_argument("--no-embed", action="store_true")
    args = parser.parse_args()
    if args.max_items < 1 or args.samples < 1 or args.workers < 1 or args.max_cases_per_section < 1:
        parser.error("limits must be positive")
    if args.force and not args.item:
        parser.error("--force requires at least one explicit --item")
    ledger = _load(args.ledger)
    protected = {row["item_key"] for row in db_relations.get_case_annotations()}
    report = {"created_at": time.time(), "items": [], "stop_reason": "completed"}
    updated = set()
    attempted = 0
    for item_key in (args.item or list_item_keys()):
        if attempted >= args.max_items:
            break
        chunks = get_item_chunks(item_key)
        digest = _source_hash(chunks)
        if not args.force and (item_key in protected or ledger["items"].get(item_key) == digest):
            continue
        excluded, reason = build_summaries._excluded_from_llm(item_key)
        if excluded:
            report["items"].append({"item_key": item_key, "status": "excluded", "reason": reason})
            db_relations.mark_artifact_status(
                item_key, "cases", "excluded", reason_code="no_cloud", message=str(reason or ""),
            )
            continue
        attempted += 1
        try:
            sections = [
                section for section in build_summaries.split_sections(chunks)
                if build_summaries.classify_section_content(section) != "non_content"
            ]
            section_results = []
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(
                        _extract_section, section, samples=args.samples,
                        max_cases=args.max_cases_per_section,
                    ): section
                    for section in sections
                }
                for future in as_completed(futures):
                    section = futures[future]
                    cases, stats = future.result()
                    section_results.append((section["section_id"], cases, stats))
            db_relations.replace_item_case_annotations(
                item_key,
                [{"section_id": section_id, "cases": cases}
                 for section_id, cases, _stats in section_results],
                model="deepseek:case-selector-v1",
            )
        except RateLimitReached:
            report["stop_reason"] = "rate_limit"
            db_relations.mark_artifact_status(item_key, "cases", "failed", reason_code="rate_limited", retryable=True)
            break
        except InvalidLLMResponse as exc:
            report["items"].append({"item_key": item_key, "status": "quality_failure", "error": str(exc)})
            db_relations.mark_artifact_status(item_key, "cases", "failed", reason_code="validation_failed", message=str(exc), retryable=True)
            _write(args.output, report)
            continue
        except LLMError as exc:
            report["items"].append({"item_key": item_key, "status": "provider_failure", "error": str(exc)})
            db_relations.mark_artifact_status(item_key, "cases", "failed", reason_code="provider_error", message=str(exc), retryable=True)
            report["stop_reason"] = "provider_unavailable"
            break
        total = sum(len(cases) for _section, cases, _stats in section_results)
        counts = Counter(case["quality_status"] for _section, cases, _stats in section_results for case in cases)
        report["items"].append({"item_key": item_key, "status": "updated", "cases": total,
                                "quality_statuses": dict(counts)})
        db_relations.mark_artifact_status(
            item_key, "cases", "success" if total else "empty",
            reason_code=None if total else "no_case_found", counts={"cases": total},
            model="deepseek:case-selector-v1",
        )
        ledger["items"][item_key] = digest
        _write(args.ledger, ledger)
        updated.add(item_key)
        _write(args.output, report)
    if updated and not args.no_embed:
        report["embedded"] = build_summaries.embed_summaries(item_keys=updated)
    report["updated"] = len(updated)
    _write(args.output, report)
    print(json.dumps({"updated": len(updated), "stop_reason": report["stop_reason"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
