"""Archived grounded case extraction over canonical document-structure v2 leaves.

The legacy case worker first inferred its own sections.  This worker instead
uses the persisted source-order tree, so each saved case can name the exact
leaf node that supplied its evidence.  The model only selects evidence units;
all evidence text and duplicate handling remain deterministic.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import re
from typing import Any, Dict, List

try:
    from . import build_summaries
    from .chunk_store import get_item_chunks
    from .db_relations import (
        get_document_nodes, get_document_structure, mark_artifact_status,
        replace_item_case_annotations,
    )
    from .llm_client import InvalidLLMResponse, LLMError, RateLimitReached, get_llm
except ImportError:  # pragma: no cover
    import build_summaries
    from chunk_store import get_item_chunks
    from db_relations import get_document_nodes, get_document_structure, mark_artifact_status, replace_item_case_annotations
    from llm_client import InvalidLLMResponse, LLMError, RateLimitReached, get_llm


PROMPT_VERSION = "structure-case-v2-1"
CASE_TYPES = ("observation", "practice", "event", "experience", "measurement", "historical", "other")

STRUCTURE_CASE_SELECTOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "cases": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "title": {"type": "string"}, "case_type": {"type": "string", "enum": list(CASE_TYPES)},
                "description": {"type": "string"}, "region": {"type": ["string", "null"]},
                "group": {"type": ["string", "null"]},
                "practices": {"type": "array", "items": {"type": "string"}},
                "phenomena": {"type": "array", "items": {"type": "string"}},
                "period": {"type": ["string", "null"]}, "locator_hint": {"type": ["string", "null"]},
                "source_kind": {"type": ["string", "null"]}, "evidence_unit_id": {"type": "string"},
            },
            "required": ["title", "case_type", "description", "region", "group", "practices", "phenomena", "period", "locator_hint", "source_kind", "evidence_unit_id"],
            "additionalProperties": False,
        }},
    },
    "required": ["cases"], "additionalProperties": False,
}


def _case_title(value: Any, evidence: str) -> str:
    """Normalize display labels without inventing factual content."""
    title = " ".join(str(value or "").split())
    if title:
        return title[:120]
    first = re.split(r"(?<=[。.!?])\s*", " ".join(evidence.split()), maxsplit=1)[0].strip()
    return first[:80].rstrip("、,;: ")


def _extend_boundary_evidence(unit_id: str, units: List[Dict[str, str]], quote: str) -> tuple[str, List[Dict[str, str]]]:
    lookup = {row["unit_id"]: index for index, row in enumerate(units)}
    index = lookup[unit_id]
    current = units[index]
    evidence = [{"field_name": "description", "chunk_id": current["chunk_id"], "evidence_quote": quote}]
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
    evidence.append({"field_name": "description", "chunk_id": units[index + 1]["chunk_id"], "evidence_quote": continuation})
    return f"{quote.rstrip()} {continuation}", evidence


def extract_leaf_cases(leaf: Dict[str, Any], source_chunks: List[Dict[str, Any]], *, samples: int = 2, max_cases: int = 3) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select and judge cases from one canonical leaf; never fabricate quotes."""
    section = {"section_id": leaf["node_id"], "chapter": leaf.get("title") or "資料", "chunks": source_chunks}
    units = build_summaries._section_evidence_units(section)
    if not units:
        return [], {"candidates": 0, "accepted": 0}
    source = "\n\n".join(f"[{row['unit_id']}]\n{row['text']}" for row in units)
    prompt = (
        "次の学術資料の原文単位から、経験的・民族誌的・歴史的な具体事例を最大8件選んでください。"
        "具体的な人物・集団・組織・場所・作品についての行為、実践、経験、観察、事件、測定を優先し、"
        "抽象理論、章の方針、単なる人名・文献列挙は選びません。titleは事例を区別する短い表示名、"
        "case_typeは observation/practice/event/experience/measurement/historical/other の一つです。"
        "本文にない事実を補わず、evidence_unit_idだけで根拠を指定してください。\n\n" + source
    )
    generated = []
    last_error: Exception | None = None
    for _ in range(samples + 2):
        try:
            generated.append(get_llm("cheap").generate_json(prompt, schema=STRUCTURE_CASE_SELECTOR_SCHEMA, timeout=300))
        except InvalidLLMResponse as exc:
            last_error = exc
        if len(generated) >= samples:
            break
    if len(generated) < samples:
        raise last_error or InvalidLLMResponse("Too few valid v2 case-selector samples.")
    votes: Counter[str] = Counter()
    first: Dict[str, Dict[str, Any]] = {}
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
    accepted, judge = build_summaries._judge_selector_case_ids(
        get_llm("standard"), candidate_ids, units, samples=1, min_votes=1, max_cases=max_cases,
    )
    rows: List[Dict[str, Any]] = []
    for unit_id in sorted(candidate_ids, key=lambda value: (value not in accepted, -votes[value], value))[:max_cases]:
        selected = {"summary": "", "cases": [first[unit_id]], "chapter_authors": [], "first_publication_note": None}
        hydrated, _verification = build_summaries._hydrate_selector_result(selected, units, section)
        if not hydrated.get("cases"):
            continue
        row = hydrated["cases"][0]
        row["description"], row["evidence"] = _extend_boundary_evidence(unit_id, units, row["evidence_quote"])
        if len(row["evidence"]) > 1:
            row["evidence_quote"] = row["description"]
        row["title"] = _case_title(first[unit_id].get("title"), row["description"])
        row["case_type"] = first[unit_id].get("case_type") if first[unit_id].get("case_type") in CASE_TYPES else "other"
        optional = sum(bool(row.get(field)) for field in ("region", "group", "period", "practices", "phenomena"))
        row["quality_status"] = "confirmed" if unit_id in accepted and optional >= 2 else "partial" if unit_id in accepted else "candidate"
        row["confidence"] = round((0.7 + 0.1 * min(votes[unit_id], 3)) if unit_id in accepted else 0.35, 2)
        row["chunk_id"] = lookup[unit_id]["chunk_id"]
        rows.append(row)
    return rows, {"candidates": len(candidate_ids), "accepted": len(accepted), "saved": len(rows), "judge": judge}


def _deduplicate(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept: Dict[str, Dict[str, Any]] = {}
    rank = {"confirmed": 3, "partial": 2, "candidate": 1}
    for row in cases:
        key = hashlib.sha256("\0".join([
            str(row.get("chunk_id") or ""), str(row.get("evidence_quote") or ""), str(row.get("case_type") or ""),
        ]).encode("utf-8")).hexdigest()
        previous = kept.get(key)
        if previous is None or rank.get(str(row.get("quality_status")), 0) > rank.get(str(previous.get("quality_status")), 0):
            kept[key] = row
    return list(kept.values())


def build_structure_cases(item_key: str, *, samples: int = 2, max_cases_per_leaf: int = 3) -> Dict[str, Any]:
    structure = get_document_structure(item_key)
    nodes = get_document_nodes(item_key, include_chunks=True)
    if not structure or not nodes:
        mark_artifact_status(item_key, "cases", "blocked", reason_code="structure_missing")
        return {"item_key": item_key, "status": "blocked", "reason": "structure_missing"}
    excluded, reason = build_summaries._excluded_from_llm(item_key)
    if excluded:
        mark_artifact_status(item_key, "cases", "excluded", reason_code=reason, source_fingerprint=structure["source_fingerprint"], processor_version=PROMPT_VERSION)
        return {"item_key": item_key, "status": "excluded", "reason": reason}
    chunks = {str(row.get("id") or ""): row for row in get_item_chunks(item_key)}
    leaves = [node for node in nodes if node.get("chunks")]
    mark_artifact_status(item_key, "cases", "running", source_fingerprint=structure["source_fingerprint"], processor_version=PROMPT_VERSION)
    collected: List[Dict[str, Any]] = []
    leaf_stats = []
    try:
        for leaf in leaves:
            source = [chunks[row["chunk_id"]] for row in leaf["chunks"] if row["chunk_id"] in chunks]
            if not source or build_summaries.classify_section_content({"section_id": leaf["node_id"], "chapter": leaf.get("title"), "chunks": source}) == "non_content":
                continue
            rows, stats = extract_leaf_cases(leaf, source, samples=samples, max_cases=max_cases_per_leaf)
            for row in rows:
                row["node_id"] = leaf["node_id"]
                row["source_fingerprint"] = structure["source_fingerprint"]
                row["normalization_version"] = PROMPT_VERSION
            collected.extend(rows)
            leaf_stats.append({"node_id": leaf["node_id"], **stats})
        collected = _deduplicate(collected)
        by_leaf: Dict[str, List[Dict[str, Any]]] = {}
        for row in collected:
            by_leaf.setdefault(str(row["node_id"]), []).append(row)
        replace_item_case_annotations(item_key, [
            {"section_id": node_id, "cases": rows} for node_id, rows in by_leaf.items()
        ], model="deepseek:structure-case-v2")
        status = "success" if collected else "empty"
        mark_artifact_status(item_key, "cases", status, reason_code=None if collected else "no_case_found", source_fingerprint=structure["source_fingerprint"], processor_version=PROMPT_VERSION, model="deepseek:structure-case-v2", counts={"cases": len(collected), "leaves": len(leaves)})
        return {"item_key": item_key, "status": status, "cases": len(collected), "leaves": len(leaves), "leaf_stats": leaf_stats}
    except RateLimitReached:
        mark_artifact_status(item_key, "cases", "failed", reason_code="rate_limited", retryable=True)
        raise
    except (LLMError, Exception) as exc:
        mark_artifact_status(item_key, "cases", "failed", reason_code="case_build_failed", message=str(exc)[:1000], retryable=True)
        raise


__all__ = ["PROMPT_VERSION", "CASE_TYPES", "build_structure_cases", "extract_leaf_cases"]
