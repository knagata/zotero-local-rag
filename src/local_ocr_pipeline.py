"""Shared local-OCR routing and deterministic adoption gates.

The source container owns structure (PDF outline or EPUB navigation/spine).
OCR engines only recover page text and layout evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable

try:
    from .text_utils import looks_like_gibberish
except ImportError:  # direct src entrypoint
    from text_utils import looks_like_gibberish


REPEAT_ARTIFACT_RE = re.compile(r"(.)\1{19,}")
CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _missing_latin_word_boundaries(text: str) -> bool:
    """Detect OCR that fused a Latin page into implausibly long word runs."""
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(normalized) < 80:
        return False
    nonspace = normalized.replace(" ", "")
    if not nonspace:
        return False
    if len(CJK_RE.findall(nonspace)) / len(nonspace) >= 0.3:
        return False
    latin = sum(character.isalpha() for character in nonspace)
    if latin / len(nonspace) < 0.5:
        return False
    return len(normalized) / max(1, normalized.count(" ")) > 11


@dataclass(frozen=True)
class LocalOcrDecision:
    engine: str
    reason: str
    fallbacks: tuple[str, ...]


def normalize_language(value: Any) -> str:
    language = str(value or "").strip().casefold().replace("_", "-")
    if language.startswith(("ja", "jpn")) or "japanese" in language or "日本" in language:
        return "ja"
    if language.startswith(("en", "eng")) or "english" in language or "英語" in language:
        return "en"
    return language or "unknown"


def choose_local_ocr_engine(
    language: Any,
    *,
    vertical: bool = False,
    dense_notes_or_references: bool = False,
) -> LocalOcrDecision:
    """Choose from engines already measured on this machine.

    NDLOCR-Lite is the lightweight Japanese baseline. RapidOCR is the
    lightweight English/unknown baseline. Layout-heavy Docling is only the
    local fallback; cloud fallback is deliberately outside this module.
    """
    lang = normalize_language(language)
    if lang == "ja":
        return LocalOcrDecision(
            "ndlocr_lite",
            "Japanese image text; lightweight Japanese OCR baseline",
            ("docling",),
        )
    return LocalOcrDecision(
        "rapidocr",
        "English/other image text; lightweight RapidOCR baseline",
        ("docling",),
    )


def evaluate_local_ocr_gate(
    chunks: list[tuple[str, str, dict[str, Any]]],
    quality: dict[str, Any],
    *,
    expected_pages: int,
    require_text_for_every_page: bool = False,
) -> dict[str, Any]:
    pages: set[int] = set()
    for _, _, metadata in chunks:
        try:
            page = int(metadata.get("page") or 0)
        except (TypeError, ValueError):
            continue
        if page > 0:
            pages.add(page)
    texts = [str(text or "").strip() for _, text, _ in chunks if str(text or "").strip()]
    characters = sum(len(text) for text in texts)
    gibberish_blocks = sum(int(looks_like_gibberish(text)) for text in texts)
    repeat_artifacts = sum(len(REPEAT_ARTIFACT_RE.findall(text)) for text in texts)
    missing_word_boundaries = sum(
        int(_missing_latin_word_boundaries(text)) for text in texts
    )
    # ``ocr_pages`` lists which pages were attempted; Docling's
    # ``processed_pages`` is a *count* of them. Iterating the count raised
    # TypeError, so every Docling escalation in this gate failed outright --
    # the local fallback stage was dead rather than merely unhelpful
    # (2026-07-28). A count cannot say which pages were covered, so it is read
    # as the leading run it implies rather than silently treated as page ids.
    attempted_source = quality.get("ocr_pages") or quality.get("processed_pages") or []
    if isinstance(attempted_source, int):
        attempted_source = range(1, max(0, attempted_source) + 1)
    elif isinstance(attempted_source, (str, bytes)):
        attempted_source = []
    attempted: set[int] = set()
    for value in attempted_source:
        try:
            page = int(value)
        except (TypeError, ValueError):
            continue
        if page > 0:
            attempted.add(page)
    expected = set(range(1, max(0, int(expected_pages)) + 1))
    reported_pages_with_text: set[int] = set()
    pages_with_text_source = quality.get("pages_with_text")
    if isinstance(pages_with_text_source, (list, tuple, set, range)):
        for value in pages_with_text_source:
            try:
                page = int(value)
            except (TypeError, ValueError):
                continue
            if page > 0:
                reported_pages_with_text.add(page)
    reported_missing_pages: set[int] = set()
    missing_pages_source = quality.get("missing_pages")
    if isinstance(missing_pages_source, (list, tuple, set, range)):
        for value in missing_pages_source:
            try:
                page = int(value)
            except (TypeError, ValueError):
                continue
            if page > 0:
                reported_missing_pages.add(page)
    reasons: list[str] = []
    if not chunks:
        reasons.append("no_chunks")
    if expected and attempted and attempted != expected:
        reasons.append("incomplete_ocr_attempt_coverage")
    if expected and not attempted and not pages:
        reasons.append("no_page_coverage")
    if require_text_for_every_page and expected and pages != expected:
        reasons.append("incomplete_ocr_text_coverage")
    # NDLOCR writes one JSON file per rendered input page.  An existing JSON
    # file with no usable OCR text is still a coverage failure; accepting it
    # based solely on ``ocr_pages`` used to hide silently empty output.  Honor
    # the explicit missing-pages contract for every adapter that supplies it,
    # without imposing this stricter interpretation on legacy Docling/RapidOCR
    # quality records that do not report page-level text coverage.
    if expected and reported_missing_pages:
        reasons.append("missing_ocr_output_pages")
    if (
        expected
        and quality.get("parser") in {"ndlocr-lite", "ndlocr_lite"}
        and isinstance(pages_with_text_source, (list, tuple, set, range))
    ):
        if reported_pages_with_text != expected:
            reasons.append("incomplete_ocr_text_coverage")
    character_floor = max(80, expected_pages * 20)
    if characters < character_floor:
        reasons.append("insufficient_text")
    if gibberish_blocks:
        reasons.append("gibberish_detected")
    if repeat_artifacts:
        reasons.append("repeat_artifacts")
    if missing_word_boundaries:
        reasons.append("missing_word_boundaries")
    if quality.get("truncated_by_timeout"):
        reasons.append("truncated_by_timeout")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "engine": quality.get("parser") or quality.get("extraction_engine"),
        "expected_pages": int(expected_pages),
        "attempted_pages": len(attempted),
        "pages_with_text": len(pages),
        "reported_pages_with_text": len(reported_pages_with_text),
        "missing_pages": sorted(reported_missing_pages),
        "characters": characters,
        "chunks": len(chunks),
        "gibberish_blocks": gibberish_blocks,
        "repeat_artifacts": repeat_artifacts,
        "missing_word_boundaries": missing_word_boundaries,
    }


def run_local_ocr(
    pdf_path: Path,
    attachment_key: str,
    metadata: dict[str, Any],
    *,
    language: Any,
    expected_pages: int,
    vertical: bool = False,
    dense_notes_or_references: bool = False,
    require_text_for_every_page: bool = False,
    extractors: dict[
        str,
        Callable[
            [Path, str, dict[str, Any]],
            tuple[list[tuple[str, str, dict[str, Any]]], dict[str, Any]],
        ],
    ],
) -> tuple[
    list[tuple[str, str, dict[str, Any]]],
    dict[str, Any],
    dict[str, Any],
]:
    """Run the selected local engine and deterministic fallbacks.

    Callers inject isolated engine adapters (notably DoclingWorker.extract), so
    this router does not import or initialize heavyweight models itself.
    """
    decision = choose_local_ocr_engine(
        language,
        vertical=vertical,
        dense_notes_or_references=dense_notes_or_references,
    )
    attempts: list[dict[str, Any]] = []
    for engine in (decision.engine, *decision.fallbacks):
        extractor = extractors.get(engine)
        if extractor is None:
            attempts.append({"engine": engine, "status": "unavailable"})
            continue
        try:
            chunks, quality = extractor(pdf_path, attachment_key, metadata)
            quality = dict(quality)
            quality.setdefault("parser", engine)
            gate = evaluate_local_ocr_gate(
                chunks, quality, expected_pages=expected_pages,
                require_text_for_every_page=require_text_for_every_page,
            )
            attempts.append({"engine": engine, "status": "passed" if gate["passed"] else "rejected", "gate": gate})
            if gate["passed"]:
                quality["local_ocr"] = {
                    "decision": decision.__dict__, "attempts": attempts,
                }
                return chunks, quality, gate
        except Exception as exc:
            attempts.append({
                "engine": engine, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
    return [], {
        "parser": "local_ocr_failed",
        "local_ocr": {"decision": decision.__dict__, "attempts": attempts},
    }, {
        "passed": False, "reasons": ["all_local_ocr_engines_failed"],
        "expected_pages": int(expected_pages),
    }


__all__ = [
    "LocalOcrDecision", "choose_local_ocr_engine", "evaluate_local_ocr_gate",
    "normalize_language", "run_local_ocr",
]
