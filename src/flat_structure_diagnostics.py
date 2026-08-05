"""Read-only triage for documents whose canonical tree uses flat fallback."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    from .chapter_detect import get_epub_href_to_toc_entries, get_pdf_toc
    from .document_structure import _attachment_key
    from .source_structure_refresh import _epub_source_path, _pdf_source_path, _refresh_epub_rows
except ImportError:  # pragma: no cover
    from chapter_detect import get_epub_href_to_toc_entries, get_pdf_toc
    from document_structure import _attachment_key
    from source_structure_refresh import _epub_source_path, _pdf_source_path, _refresh_epub_rows


SHORT_DOCUMENT_CHARS = 20_000
LONG_PDF_PAGES = 30
_HEADING_NUMBER_RE = re.compile(
    r"^(?:第[0-9一二三四五六七八九十百]+[章節部編]|"
    r"(?:chapter|part|section)\s+[0-9ivxlcdm]+\b|"
    r"[0-9]{1,2}(?:\.[0-9]{1,2}){1,3}[.)]?\s+\S|"
    r"[ivxlcdm]+[.)]\s+\S)",
    re.IGNORECASE,
)
_HEADING_BLOCK_TYPES = {
    "heading", "section_header", "title", "chapter", "chapter_title",
}


def _source_type(rows: Sequence[Mapping[str, Any]]) -> str:
    metadata = rows[0].get("metadata") or {}
    value = str(metadata.get("source_type") or "").casefold()
    if value in {"pdf", "epub", "html"}:
        return value
    filename = str(metadata.get("filename") or metadata.get("path") or "").casefold()
    return "epub" if filename.endswith(".epub") else ("pdf" if filename.endswith(".pdf") else value or "unknown")


def _heading_evidence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    explicit_blocks = 0
    numbered_short_blocks = 0
    evidence_blocks = 0
    examples: list[str] = []
    for row in rows:
        metadata = row.get("metadata") or {}
        text = " ".join(str(row.get("text") or "").split())
        block_type = str(metadata.get("block_type") or "").casefold()
        explicit = block_type in _HEADING_BLOCK_TYPES or bool(
            metadata.get("heading_level") or metadata.get("header_level")
        )
        numbered = bool(text and len(text) <= 180 and _HEADING_NUMBER_RE.match(text))
        explicit_blocks += int(explicit)
        numbered_short_blocks += int(numbered)
        evidence_blocks += int(explicit or numbered)
        if (explicit or numbered) and len(examples) < 5:
            examples.append(text[:180])
    return {
        "explicit_heading_blocks": explicit_blocks,
        "numbered_short_blocks": numbered_short_blocks,
        "heading_evidence_blocks": evidence_blocks,
        "heading_examples": examples,
    }


def inspect_source_structure(
    source_type: str, rows: Sequence[Mapping[str, Any]], attachment_key: str,
) -> dict[str, Any]:
    """Inspect only local source metadata; never invoke OCR or an LLM."""
    try:
        if source_type == "pdf":
            path = _pdf_source_path(rows, attachment_key)
            toc = get_pdf_toc(str(path))
            return {"source_path": str(path), "source_available": True, "toc_entries": len(toc)}
        if source_type == "epub":
            path = _epub_source_path(rows, attachment_key)
            entries = get_epub_href_to_toc_entries(str(path))
            unique = {
                (tuple(row.get("path") or []), str(row.get("href") or ""))
                for values in entries.values() for row in values
            }
            result = {
                "source_path": str(path), "source_available": True,
                "toc_entries": len(unique), "refresh_mappable": False,
            }
            try:
                _refresh_epub_rows(rows, attachment_key)
                result["refresh_mappable"] = True
            except Exception as exc:
                result["refresh_error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
            return result
    except Exception as exc:
        return {
            "source_available": False, "toc_entries": 0,
            "source_error": f"{type(exc).__name__}: {str(exc)[:240]}",
        }
    return {"source_available": False, "toc_entries": 0}


def classify_flat_attachment(
    attachment_key: str, rows: Sequence[Mapping[str, Any]], *,
    source_inspector: Callable[[str, Sequence[Mapping[str, Any]], str], Mapping[str, Any]] = inspect_source_structure,
) -> dict[str, Any]:
    """Classify one flat attachment into an actionable, stable reason code."""
    source_type = _source_type(rows)
    first_metadata = rows[0].get("metadata") or {}
    chars = sum(len(str(row.get("text") or "")) for row in rows)
    pages = {
        int(value) for row in rows
        if (value := (row.get("metadata") or {}).get("page")) is not None
        and str(value).isdigit()
    }
    locator_count = sum(bool((row.get("metadata") or {}).get("locator")) for row in rows)
    headings = _heading_evidence(rows)
    source = dict(source_inspector(source_type, rows, attachment_key))
    toc_entries = int(source.get("toc_entries") or 0)
    heading_count = headings["heading_evidence_blocks"]

    if source_type == "pdf" and toc_entries >= 2:
        reason, priority = "pdf_outline_refresh_candidate", 100
    elif source_type == "epub" and toc_entries >= 2 and source.get("refresh_mappable"):
        reason, priority = "epub_toc_refresh_candidate", 100
    elif source_type == "epub" and toc_entries >= 2:
        reason, priority = "epub_toc_mapping_repair_candidate", 90
    elif heading_count >= 2:
        reason, priority = f"{source_type}_body_heading_recovery_candidate", 80
    elif source_type == "pdf" and len(pages) >= LONG_PDF_PAGES:
        reason, priority = "pdf_printed_toc_or_layout_candidate", 65
    elif not source.get("source_available"):
        reason, priority = "source_unavailable_for_diagnosis", 45
    elif chars <= SHORT_DOCUMENT_CHARS:
        reason, priority = "flat_likely_appropriate_short_document", 20
    elif source_type == "epub" and locator_count:
        reason, priority = "epub_spine_only_recovery_candidate", 55
    elif source_type in {"pdf", "epub"}:
        reason, priority = f"{source_type}_no_structure_evidence", 35
    else:
        reason, priority = "unsupported_source_type", 10

    return {
        "attachment_key": attachment_key,
        "title": str(first_metadata.get("title") or first_metadata.get("filename") or ""),
        "source_type": source_type,
        "reason_code": reason,
        "priority": priority,
        "gold_recommended": priority >= 55 and (
            not toc_entries or reason == "epub_toc_mapping_repair_candidate"
        ),
        "chunk_count": len(rows), "text_chars": chars,
        "page_count_observed": len(pages),
        "locator_coverage": round(locator_count / len(rows), 4) if rows else 0.0,
        **headings, **source,
    }


def diagnose_flat_item(
    item_key: str, chunks: Iterable[Mapping[str, Any]], *,
    source_inspector: Callable[[str, Sequence[Mapping[str, Any]], str], Mapping[str, Any]] = inspect_source_structure,
) -> dict[str, Any]:
    source_rows = [
        row for row in chunks
        if str((row.get("metadata") or {}).get("source_type") or "") != "note"
    ]
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in source_rows:
        groups[_attachment_key(row)].append(row)
    attachments = [
        classify_flat_attachment(key, rows, source_inspector=source_inspector)
        for key, rows in groups.items()
    ]
    attachments.sort(key=lambda row: (-int(row["priority"]), str(row["attachment_key"])))
    return {
        "item_key": item_key,
        "title": str(
            ((source_rows[0].get("metadata") or {}).get("title") if source_rows else "") or ""
        ),
        "priority": max((int(row["priority"]) for row in attachments), default=0),
        "gold_recommended": any(bool(row["gold_recommended"]) for row in attachments),
        "attachments": attachments,
    }


__all__ = [
    "classify_flat_attachment", "diagnose_flat_item", "inspect_source_structure",
]
