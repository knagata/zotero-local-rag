"""Deterministic, engine-neutral re-OCR candidate and adoption checks.

This module is deliberately free of database, Chroma, filesystem, and OCR
engine calls.  It is shared by the queue/report scripts so candidate selection
and the adoption gate can be tested without touching canonical data.
"""
from __future__ import annotations

from collections import Counter
import re
from typing import Any, Mapping, Sequence

from .text_utils import looks_like_gibberish


REPEAT_ARTIFACT_RE = re.compile(r"(.)\1{19,}")
TEXT_DISORDER_TERMS = (
    "文字化け", "原文乱れ", "読み順", "ocr", "mojibake", "garbled",
    "source disorder", "reading order", "corrupt text", "corrupted text",
)
DEFAULT_MIN_CHARS_PER_PAGE = {"ja": 80, "en": 100, "unknown": 60}


def normalize_language(value: Any) -> str:
    language = str(value or "").strip().casefold().replace("_", "-")
    if language.startswith("ja"):
        return "ja"
    if language.startswith("en"):
        return "en"
    return language or "unknown"


def majority_language(chunks: Sequence[Mapping[str, Any]]) -> str:
    values = []
    for chunk in chunks:
        metadata = chunk.get("metadata")
        md = metadata if isinstance(metadata, Mapping) else {}
        language = normalize_language(md.get("lang"))
        if language != "unknown":
            values.append(language)
    return Counter(values).most_common(1)[0][0] if values else "unknown"


def _block_text(block: Mapping[str, Any]) -> str:
    return str(block.get("text") or "")


def _block_metadata(block: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = block.get("metadata")
    return metadata if isinstance(metadata, Mapping) else block


def _page_number(block: Mapping[str, Any]) -> int | None:
    value = _block_metadata(block).get("page")
    try:
        page = int(value)
        return page if page > 0 else None
    except (TypeError, ValueError):
        return None


def text_metrics(
    blocks: Sequence[Mapping[str, Any]], *, total_pages: int | None = None,
) -> dict[str, Any]:
    """Return stable text/page metrics for chunks or normalized V3 blocks."""
    texts = [_block_text(block) for block in blocks]
    nonempty = [text for text in texts if text.strip()]
    pages = sorted({page for block in blocks if (page := _page_number(block)) is not None})
    declared_pages = max(0, int(total_pages or 0))
    page_count = declared_pages or len(pages)
    characters = sum(len(text) for text in texts)
    gibberish = sum(looks_like_gibberish(text) for text in nonempty)
    repeat_artifacts = sum(len(REPEAT_ARTIFACT_RE.findall(text)) for text in texts)
    return {
        "blocks": len(blocks),
        "characters": characters,
        "pages": pages,
        "total_pages": page_count,
        "covered_pages": len(pages),
        "page_coverage": round(len(pages) / page_count, 6) if page_count else 0.0,
        "characters_per_page": round(characters / page_count, 3) if page_count else 0.0,
        "gibberish_blocks": gibberish,
        "gibberish_rate": round(gibberish / len(nonempty), 6) if nonempty else 0.0,
        "repeat_artifacts": repeat_artifacts,
    }


def summary_report_flags_text_disorder(report: Mapping[str, Any]) -> bool:
    """Identify only reports that explicitly describe source/OCR disorder."""
    haystack = " ".join((
        str(report.get("reason") or ""), str(report.get("details") or ""),
        str(report.get("reviewer_note") or ""),
    )).casefold()
    return any(term in haystack for term in TEXT_DISORDER_TERMS)


def same_engine_version(
    current_engine: Any, current_version: Any, target_engine: Any, target_version: Any,
) -> bool:
    """True only when both engine and non-empty version are exactly the same."""
    current = str(current_engine or "").strip().casefold()
    target = str(target_engine or "").strip().casefold()
    current_v = str(current_version or "").strip()
    target_v = str(target_version or "").strip()
    return bool(current and target and current == target and current_v and target_v and current_v == target_v)


def candidate_assessment(
    *, quality: Mapping[str, Any], chunks: Sequence[Mapping[str, Any]],
    structure_status: str | None, summary_reports: Sequence[Mapping[str, Any]],
    current_engine: str | None, current_version: str | None,
    target_engine: str, target_version: str,
    language: str | None = None,
    min_chars_per_page: Mapping[str, int] | None = None,
    chunk_reports: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Assess one attachment using the V3, case-independent candidate rules."""
    lang = normalize_language(language) if language else majority_language(chunks)
    total_pages = int(quality.get("total_pages") or 0)
    metrics = text_metrics(chunks, total_pages=total_pages)
    thresholds = dict(DEFAULT_MIN_CHARS_PER_PAGE)
    if min_chars_per_page:
        thresholds.update({normalize_language(k): int(v) for k, v in min_chars_per_page.items()})
    min_chars = thresholds.get(lang, thresholds["unknown"])
    scanned_ratio = float(quality.get("scanned_ratio") or 0.0)
    scan_source_without_native_text = (
        str(quality.get("source_class") or "") == "scanned_no_text"
    )
    # A classification that could not be computed is not evidence that the
    # document is born-digital, but absent it reads exactly like one: the test
    # above is False either way, and the item is never offered again. So the
    # failure is carried as its own reason rather than being scored as a pass.
    source_class_unknown = bool(quality.get("source_class_error"))
    reported_gibberish = quality.get("gibberish_rate")
    gibberish_rate = (
        float(reported_gibberish) if reported_gibberish is not None
        else float(metrics["gibberish_rate"])
    )
    report_count = sum(summary_report_flags_text_disorder(report) for report in summary_reports)
    reasons: list[str] = []
    if scanned_ratio > 0:
        reasons.append("scanned_pages")
    if scan_source_without_native_text:
        reasons.append("scan_source_without_native_text")
    if source_class_unknown:
        reasons.append("source_class_unknown")
    if gibberish_rate > 0:
        reasons.append("gibberish")
    if metrics["characters_per_page"] < min_chars:
        reasons.append("low_characters_per_page")
    if metrics["repeat_artifacts"] > 0:
        reasons.append("repeat_artifact")
    if structure_status == "flat_fallback":
        reasons.append("flat_fallback")
    if report_count:
        reasons.append("summary_quality_text_disorder")
    # Reader-reported damage in the source text itself (note 79, U0-b). This
    # signal reaches places the deterministic metrics cannot: degraded OCR that
    # leaves character counts and gibberish ratios looking healthy still stops a
    # passage being quotable, and only someone reading it notices. Weighted
    # highest per report because it is direct evidence rather than a proxy.
    chunk_report_weight = sum(
        int(report.get("report_count") or 1) for report in chunk_reports
    )
    if chunk_report_weight:
        reasons.append("reported_source_text_damage")

    excluded_same_engine = same_engine_version(
        current_engine, current_version, target_engine, target_version,
    )
    score = (
        5 * int(scanned_ratio > 0)
        + 5 * int(scan_source_without_native_text)
        # Lower than a measured scan: this says "unknown", not "known bad".
        + 2 * int(source_class_unknown)
        + 5 * int(gibberish_rate > 0)
        + 3 * int(metrics["characters_per_page"] < min_chars)
        + 4 * int(metrics["repeat_artifacts"] > 0)
        + 2 * int(structure_status == "flat_fallback")
        + 3 * report_count
        + 6 * min(chunk_report_weight, 3)
    )
    return {
        "candidate": bool(reasons) and not excluded_same_engine,
        "excluded_same_engine_version": excluded_same_engine,
        "reasons": reasons,
        "score": score,
        "language": lang,
        "current_engine": current_engine or "unknown",
        "current_version": current_version or "unknown",
        "target_engine": target_engine,
        "target_version": target_version,
        "structure_status": structure_status or "unavailable",
        "summary_quality_text_disorder_reports": report_count,
        "quality": {
            "scanned_ratio": round(scanned_ratio, 6),
            "scan_source_without_native_text": scan_source_without_native_text,
            "gibberish_rate": round(gibberish_rate, 6),
            "min_characters_per_page": min_chars,
            **metrics,
        },
    }


def evaluate_adoption_gate(
    before: Mapping[str, Any], after: Mapping[str, Any], *,
    min_character_ratio: float = 0.8, max_character_ratio: float = 1.5,
) -> dict[str, Any]:
    """Apply the §4.4.3 deterministic gate to two metric snapshots."""
    previous_chars = max(1, int(before.get("characters") or 0))
    character_ratio = float(after.get("characters") or 0) / previous_chars
    failures: list[dict[str, Any]] = []
    coverage = float(after.get("page_coverage") or 0.0)
    repeats = int(after.get("repeat_artifacts") or 0)
    old_gibberish = float(before.get("gibberish_rate") or 0.0)
    new_gibberish = float(after.get("gibberish_rate") or 0.0)
    if coverage < 1.0:
        failures.append({"code": "page_coverage", "actual": coverage, "required": 1.0})
    if repeats != 0:
        failures.append({"code": "repeat_artifacts", "actual": repeats, "required": 0})
    if new_gibberish > old_gibberish:
        failures.append({"code": "gibberish_worsened", "before": old_gibberish, "after": new_gibberish})
    if character_ratio < min_character_ratio or character_ratio > max_character_ratio:
        failures.append({
            "code": "character_ratio", "actual": round(character_ratio, 6),
            "minimum": min_character_ratio, "maximum": max_character_ratio,
        })
    return {
        "passed": not failures,
        "failures": failures,
        "metrics": {
            "page_coverage": coverage,
            "repeat_artifacts": repeats,
            "gibberish_before": old_gibberish,
            "gibberish_after": new_gibberish,
            "character_ratio": round(character_ratio, 6),
        },
        "thresholds": {
            "page_coverage": 1.0, "repeat_artifacts": 0,
            "gibberish": "non_worsening", "min_character_ratio": min_character_ratio,
            "max_character_ratio": max_character_ratio,
        },
    }


__all__ = [
    "DEFAULT_MIN_CHARS_PER_PAGE", "REPEAT_ARTIFACT_RE", "candidate_assessment",
    "evaluate_adoption_gate", "majority_language", "normalize_language",
    "same_engine_version", "summary_report_flags_text_disorder", "text_metrics",
]
