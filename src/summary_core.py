"""Shared prompt and verification primitives for V3 hierarchical summaries.

The read-only ``build_summaries.py`` compatibility module also re-exports a
subset for offline model comparisons. Canonical persistence is owned solely by
``build_structure_summaries.py``.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from typing import Any, Callable


try:
    from .llm_client import DeepSeekClient, InvalidLLMResponse, LLMError
    from .summary_prompts import cited_item_summary_prompt, cited_section_summary_prompt
except ImportError:  # pragma: no cover
    from llm_client import DeepSeekClient, InvalidLLMResponse, LLMError
    from summary_prompts import cited_item_summary_prompt, cited_section_summary_prompt


NON_CONTENT_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"目次|contents?|table\s+of\s+contents|図版一覧|表一覧|口絵|凡例|奥付|索引|index|"
    r"copyright|著作権(?:について|表示|注意)?|colophon|標題紙|title\s+page|half\s+title|"
    r"cover|本著作物について|list\s+of\s+(?:illustrations|figures|tables)|"
    r"series(?:\s+page|\s+list)?|シリーズ一覧|叢書一覧|acknowledg(?:e)?ments?|謝辞|"
    r"about\s+(?:the\s+)?author|about\s+[\w .'-]+|著者紹介|contributors?|執筆者紹介|"
    r"事項一覧|人名一覧|用語一覧|略語一覧|年表|glossary|chronology|timeline|"
    r"list\s+of\s+(?:terms|names|abbreviations)|"
    r"references?|bibliography|参考文献|引用文献"
    r")\s*$",
    re.I,
)
META_SUMMARY_RE = re.compile(
    r"(?:"
    r"要約(?:すること)?(?:が|は)?できません|要約できません|ご提示ください|"
    r"(?:入力|本文|テキスト|内容|原文)[^。\n]{0,80}(?:含(?:まれて|んで)い(?:ません|ない)|"
    r"提示されていません|見当たりません|ありません)|"
    r"裏付けること(?:が|は)?できません|"
    r"(?:抽出結果|結果|配列|フィールド)[^。\n]{0,40}空(?:と|に)しました|"
    r"cannot\s+(?:provide|produce|generate|write)\s+(?:a\s+)?summary|"
    r"please\s+(?:provide|share)\s+(?:the\s+)?(?:text|content|passage)|"
    r"(?:input|provided\s+(?:text|content))[^.\n]{0,50}(?:does\s+not|doesn't)\s+contain"
    r")",
    re.I,
)
TOC_ENTRY_RE = re.compile(r"(?:\.{3,}|…{2,}|・{3,})\s*\d+\s*$")
CHAPTER_MARKER_RE = re.compile(
    r"(?:第[一二三四五六七八九十百0-9]+章|序章|終章|chapter\s+\d+)", re.I,
)
SPECIFIC_VALUE_RE = re.compile(
    r"https?://\S+|10\.\d{4,9}/\S+|"
    r"(?:ISBN(?:-1[03])?[:\s]*)?[0-9Xx][0-9Xx\- ]{8,}|"
    r"(?:18|19|20)\d{2}|\d+(?:\.\d+)?",
    re.I,
)
DEFINITIVE_IDENTIFIER_RE = re.compile(
    r"https?://\S+|10\.\d{4,9}/\S+|"
    r"(?:ISBN(?:-1[03])?[:\s]*)?[0-9Xx][0-9Xx\- ]{8,}",
    re.I,
)
QUOTED_VALUE_RE = re.compile(r"[『「“\"]([^』」”\"]{2,})[』」”\"]")

SUMMARY_ONLY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sentences": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "evidence_unit_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["text", "evidence_unit_ids"],
            "additionalProperties": False,
        }},
    },
    "required": ["sentences"],
    "additionalProperties": False,
}

SummaryProgressCallback = Callable[[str, dict[str, Any]], None]
_summary_progress_callback: SummaryProgressCallback | None = None


def set_summary_progress_callback(callback: SummaryProgressCallback | None) -> None:
    """Install process-local progress reporting for paid summary requests."""
    global _summary_progress_callback
    _summary_progress_callback = callback


def _emit_summary_progress(event: str, **details: Any) -> None:
    if _summary_progress_callback is not None:
        _summary_progress_callback(event, details)


def _generate_summary_json(
    client: DeepSeekClient, prompt: str, *, kind: str, model: str,
) -> dict[str, Any]:
    _emit_summary_progress("request", kind=kind, model=model)
    try:
        return client.generate_json(
            prompt, schema=SUMMARY_ONLY_SCHEMA, timeout=_summary_llm_timeout_seconds(),
        )
    except LLMError as exc:
        _emit_summary_progress("error", kind=kind, model=model, error=type(exc).__name__)
        raise


def _summary_llm_timeout_seconds() -> float:
    """Return a bounded, operator-tunable timeout for summary API calls.

    The conservative default preserves the previous behaviour.  Backfills can
    choose a lower bound so one stalled provider request cannot monopolise a
    worker for several minutes.
    """
    try:
        configured = float(os.environ.get("SUMMARY_LLM_TIMEOUT_SECONDS", "300"))
    except ValueError:
        configured = 300.0
    return min(300.0, max(5.0, configured))


def _section_source_text(section: dict[str, Any]) -> str:
    return "\n\n".join(
        str(chunk.get("text") or "") for chunk in section["chunks"] if chunk.get("text")
    )[:30000]


def classify_section_content(section: dict[str, Any]) -> str:
    """Conservatively reject front matter and table-of-contents-like sections."""
    text = _section_source_text(section).strip()
    minimum = max(0, int(os.environ.get("SUMMARY_MIN_SECTION_CHARS", "400")))
    if len(text) < minimum:
        return "non_content"
    chapter = str(section.get("chapter") or "").strip()
    first_line = text.splitlines()[0].strip() if text else ""
    for heading in (chapter, first_line[:160]):
        if heading and NON_CONTENT_HEADING_RE.fullmatch(heading):
            return "non_content"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    early_lines = lines[:8]
    if any(re.match(r"^index\s+(?:a\s+note\s+about|[a-z].*\b\d+)", line, re.I) for line in early_lines):
        return "non_content"
    if any(line.casefold().startswith("contents ") for line in early_lines):
        return "non_content"
    page_entries = sum(bool(TOC_ENTRY_RE.search(line)) for line in lines)
    if len(lines) >= 5 and page_entries >= 3 and page_entries / len(lines) >= 0.3:
        return "non_content"
    # OCR often collapses a table of contents into long lines. Multiple chapter
    # markers in a short section are safer to skip than to let the model fill in.
    if len(text) < 5000 and len(CHAPTER_MARKER_RE.findall(text)) >= 3:
        return "non_content"
    return "content"


def is_meta_summary(summary: str) -> bool:
    """Detect model refusals/meta commentary that must never become an index entry."""
    return bool(META_SUMMARY_RE.search(str(summary or "")[:500]))


def _normalize_evidence(value: Any) -> str:
    return " ".join(str(value or "").split())


def _note_values_supported(note: str, evidence_quote: str) -> bool:
    normalized_quote = _normalize_evidence(evidence_quote).casefold()
    specific = [match.group(0).rstrip(".,;:。、）)") for match in SPECIFIC_VALUE_RE.finditer(note)]
    quoted = QUOTED_VALUE_RE.findall(note)
    return all(_normalize_evidence(value).casefold() in normalized_quote for value in specific + quoted)


def _split_exact_units(text: str, max_chars: int = 900) -> list[str]:
    """Split source into exact, bounded substrings without normalizing OCR text."""
    candidates = re.split(r"(?<=[。！？.!?])(?:[ \t]+|\n+)|\n{2,}", text)
    units: list[str] = []
    for candidate in candidates:
        value = candidate.strip()
        while len(value) > max_chars:
            boundary = max(
                value.rfind("\n", 0, max_chars + 1),
                value.rfind(" ", 0, max_chars + 1),
            )
            if boundary < max_chars // 2:
                boundary = max_chars
            units.append(value[:boundary].strip())
            value = value[boundary:].strip()
        if value:
            units.append(value)
    return units


def _section_evidence_units(section: dict[str, Any]) -> list[dict[str, str]]:
    units: list[dict[str, str]] = []
    remaining = 30000
    has_text = False
    for chunk_index, chunk in enumerate(section["chunks"]):
        raw_text = str(chunk.get("text") or "")
        if not raw_text:
            continue
        if has_text:
            remaining -= 2  # _section_source_text joins non-empty chunks with two newlines.
        if remaining <= 0:
            break
        raw_text = raw_text[:remaining]
        remaining -= len(raw_text)
        has_text = True
        chunk_id = str(chunk.get("id") or f"chunk-{chunk_index}")
        for text in _split_exact_units(raw_text):
            units.append({
                "unit_id": f"u{len(units) + 1:04d}", "chunk_id": chunk_id, "text": text,
            })
    return units


def _verify_summary_only_result(
    result: dict[str, Any], units: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate sentence citations and source-bound concrete values."""
    lookup = {unit["unit_id"]: unit for unit in units}
    verified_sentences: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    warnings: Counter[str] = Counter()
    generated = 0
    for sentence in result.get("sentences") or []:
        generated += 1
        if not isinstance(sentence, dict):
            reasons["invalid_shape"] += 1
            continue
        text = _normalize_evidence(sentence.get("text"))
        ids = list(dict.fromkeys(
            str(value or "") for value in sentence.get("evidence_unit_ids") or []
        ))
        if not text:
            reasons["empty_sentence"] += 1
            continue
        if not ids:
            reasons["missing_evidence_ids"] += 1
            continue
        if any(unit_id not in lookup for unit_id in ids):
            reasons["invalid_evidence_id"] += 1
            continue
        evidence = " ".join(lookup[unit_id]["text"] for unit_id in ids)
        normalized_evidence = _normalize_evidence(evidence).casefold()
        identifiers = [
            match.group(0).rstrip(".,;:。、）)")
            for match in DEFINITIVE_IDENTIFIER_RE.finditer(text)
        ]
        if any(
            _normalize_evidence(value).casefold() not in normalized_evidence
            for value in identifiers
        ):
            reasons["identifier_not_in_evidence"] += 1
            continue
        if not _note_values_supported(text, evidence):
            # Ordinary numbers, years, and translated quoted phrases are soft
            # signals in summary prose. Runtime RAG verifies claims against source
            # chunks and reports concrete discrepancies for reversible triage.
            warnings["value_not_in_evidence"] += 1
        verified_sentences.append({
            "text": text, "evidence_unit_ids": ids,
            "evidence": [lookup[unit_id]["text"] for unit_id in ids],
        })
    summary = "".join(row["text"] for row in verified_sentences)
    if is_meta_summary(summary):
        reasons["meta_response"] += len(verified_sentences)
        verified_sentences = []
        summary = ""
    kept = len(verified_sentences)
    discarded = max(0, generated - kept)
    discard_rate = round(discarded / generated, 4) if generated else 0.0
    stats = {
        "generated_sentences": generated, "kept_sentences": kept,
        "discarded_sentences": discarded,
        "discard_rate": discard_rate,
        "reasons": dict(reasons), "warnings": dict(warnings),
        "accepted": bool(kept >= 2 and discard_rate <= 0.5),
    }
    return {"summary": summary, "sentences": verified_sentences}, stats


def _llm_summary_only_section(
    section: dict[str, Any], *, model: str = "deepseek-v4-pro",
    reasoning: str = "disabled",
) -> tuple[dict[str, Any], str]:
    """Generate only a cited section summary."""
    units = _section_evidence_units(section)
    if not units:
        raise InvalidLLMResponse("Summary-only input has no evidence units.")
    client = DeepSeekClient(model, thinking=reasoning)
    source = "\n\n".join(f"[{unit['unit_id']}]\n{unit['text']}" for unit in units)
    prompt = cited_section_summary_prompt(source)
    generated = _generate_summary_json(client, prompt, kind="section", model=model)
    verified, stats = _verify_summary_only_result(generated, units)
    _emit_summary_progress("response", kind="section", model=model, verification=stats)
    verified["_verification"] = stats
    verified["_usage"] = client.last_usage
    return verified, f"deepseek:{model}:summary-only:{reasoning}"


def _llm_summary_only_item(
    title: str, section_summaries: list[dict[str, Any]], *,
    model: str = "deepseek-v4-pro", reasoning: str = "disabled",
) -> tuple[dict[str, Any], str]:
    """Synthesize a cited item summary from already verified section summaries."""
    units = [
        {
            "unit_id": f"s{index:04d}",
            "chunk_id": str(section.get("section_id") or ""),
            "text": str(section.get("summary") or "").strip(),
        }
        for index, section in enumerate(section_summaries, start=1)
        if str(section.get("summary") or "").strip()
    ]
    if not units:
        raise InvalidLLMResponse("No verified section summaries are available.")
    client = DeepSeekClient(model, thinking=reasoning)
    source = "\n\n".join(f"[{unit['unit_id']}]\n{unit['text']}" for unit in units)
    prompt = cited_item_summary_prompt(title, source)
    generated = _generate_summary_json(client, prompt, kind="reduction", model=model)
    verified, stats = _verify_summary_only_result(generated, units)
    _emit_summary_progress("response", kind="reduction", model=model, verification=stats)
    verified["_verification"] = stats
    verified["_usage"] = client.last_usage
    return verified, f"deepseek:{model}:summary-only:{reasoning}"


def _extractive_section(section: dict[str, Any]) -> dict[str, Any]:
    chunks = section["chunks"]
    sampled = chunks[:2] + (chunks[-1:] if len(chunks) > 2 else [])
    text = " ".join(str(chunk.get("text") or "").strip() for chunk in sampled)
    return {
        "summary": text[:1200], "chapter_authors": [],
        "first_publication_note": None,
    }


__all__ = [
    "NON_CONTENT_HEADING_RE", "META_SUMMARY_RE", "TOC_ENTRY_RE", "CHAPTER_MARKER_RE",
    "SPECIFIC_VALUE_RE", "DEFINITIVE_IDENTIFIER_RE", "QUOTED_VALUE_RE", "SUMMARY_ONLY_SCHEMA",
    "_section_source_text", "classify_section_content", "is_meta_summary",
    "_normalize_evidence", "_note_values_supported", "_split_exact_units",
    "_section_evidence_units", "_verify_summary_only_result",
    "set_summary_progress_callback",
    "_llm_summary_only_section", "_llm_summary_only_item", "_extractive_section",
]
