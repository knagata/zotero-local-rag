"""AI TOC recovery plus deterministic body-heading alignment for PDF fast-path routing."""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import re
import math
from typing import Any, Mapping, Sequence

import fitz

try:
    from .env_utils import load_dotenv_native
    from .feature_gates import ai_toc_enabled
    from .llm_client import LLMClient, LLMError, get_llm
    from .pdf_extract import extract_layout_blocks_from_pdf_page
except ImportError:  # direct execution with src/ on sys.path
    from env_utils import load_dotenv_native
    from feature_gates import ai_toc_enabled
    from llm_client import LLMClient, LLMError, get_llm
    from pdf_extract import extract_layout_blocks_from_pdf_page


TOC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "document_type": {"type": "string"},
        "headings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "level": {"type": "integer"},
                    "kind": {"type": "string"},
                    "printed_page": {"type": ["string", "null"]},
                },
                "required": ["title", "level", "kind", "printed_page"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["document_type", "headings"],
    "additionalProperties": False,
}

FUNCTIONAL_KINDS = {
    "title", "contents", "front_matter", "acknowledgements", "abstract",
    "references", "notes", "appendix", "index", "back_matter",
}
NON_BOUNDARY_KINDS = {"title", "contents"}


@dataclass(frozen=True)
class HeadingAnchor:
    title: str
    level: int
    kind: str
    page: int
    reading_order: int
    score: float


@dataclass
class TocRecoveryResult:
    accepted: bool
    reason: str
    chunks: list[tuple[str, str, dict[str, Any]]]
    diagnostics: dict[str, Any]


def normalize_heading(value: str) -> str:
    value = str(value or "").casefold().replace("\u3000", " ")
    # Soft hyphens are discretionary glyph hints, not word boundaries
    # (``Ele\u00adments`` must compare equal to ``Elements``).
    value = value.replace("\u00ad", "")
    value = re.sub(r"(?<=\w)-\s+(?=\w)", "", value)
    value = value.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    value = re.sub(r"[^\w\u3040-\u30ff\u3400-\u9fff]+", " ", value)
    return " ".join(value.split())


def _heading_score(title: str, block: str) -> float:
    expected, actual = normalize_heading(title), normalize_heading(block)
    if not expected or not actual:
        return 0.0
    if expected == actual:
        return 1.0
    if actual.startswith(expected) or expected in actual:
        # A heading block is sometimes merged with the first body sentence.
        return max(0.9, len(expected) / max(len(actual), len(expected)))
    return SequenceMatcher(None, expected, actual[: max(len(expected) * 2, 40)]).ratio()


def _printed_page_number(value: str) -> int | None:
    normalized = str(value or "").strip().casefold()
    match = re.search(r"\d+", normalized)
    if match:
        return int(match.group())
    if not normalized or not re.fullmatch(r"[ivxlcdm]+", normalized):
        return None
    values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = previous = 0
    for char in reversed(normalized):
        current = values[char]
        total += -current if current < previous else current
        previous = max(previous, current)
    return total or None


def _page_records(pdf_path: Path) -> dict[int, list[dict[str, Any]]]:
    records: dict[int, list[dict[str, Any]]] = {}
    with fitz.open(pdf_path) as document:
        for index in range(document.page_count):
            page = document[index]
            try:
                rows = extract_layout_blocks_from_pdf_page(page)
            except Exception:
                rows = []
            if not rows:
                text = page.get_text("text") or ""
                rows = [
                    {"text": line.strip(), "reading_order": order}
                    for order, line in enumerate(text.splitlines()) if line.strip()
                ]
            try:
                page_label = str(page.get_label() or "").strip()
            except Exception:
                page_label = ""
            if page_label:
                rows = [{**row, "page_label": page_label} for row in rows]
            records[index + 1] = rows
    return records


def _toc_excluded_orders(
    headings: Sequence[Mapping[str, Any]],
    page_records: Mapping[int, Sequence[Mapping[str, Any]]],
) -> tuple[set[int], dict[int, set[int]]]:
    """Identify TOC blocks without discarding body text sharing the same page."""
    normalized = [normalize_heading(str(row.get("title") or "")) for row in headings]
    toc_density_threshold = max(5, math.ceil(len(normalized) * 0.25))
    toc_pages: set[int] = set()
    excluded: dict[int, set[int]] = {}
    for page, records in page_records.items():
        if page > 20:
            continue
        blocks = [normalize_heading(str(row.get("text") or "")) for row in records]
        matches_by_order = {
            order: sum(bool(title and title in block) for title in normalized)
            for order, block in enumerate(blocks)
        }
        density = sum(
            bool(title and title in " ".join(blocks)) for title in normalized
        )
        contents_orders = {
            order for order, block in enumerate(blocks)
            if block in {"contents", "table of contents", "目次"}
            or block.startswith("contents ") or block.startswith("目次 ")
        }
        if density < toc_density_threshold and not (
            contents_orders and density >= 2
        ):
            continue
        toc_pages.add(page)
        matched_orders = {order for order, count in matches_by_order.items() if count}
        multi_heading_blocks = {
            order for order, count in matches_by_order.items() if count >= 2
        }
        if contents_orders:
            # A labelled contents section may use one block per entry.
            last = max(contents_orders | matched_orders)
            excluded[page] = set(range(0, last + 1))
        elif multi_heading_blocks:
            # Compact journal contents often occupies one block followed by the
            # real first heading on the same page. Exclude only the dense block.
            excluded[page] = multi_heading_blocks
        else:
            excluded[page] = matched_orders
    return toc_pages, excluded


def _candidate_spans(
    title: str, page: int, records: Sequence[Mapping[str, Any]],
    excluded_orders: set[int], threshold: float, printed_page: str = "",
) -> list[tuple[float, int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for width in (1, 2, 3):
        for index in range(0, len(records) - width + 1):
            orders = [
                int(records[offset].get("reading_order", offset))
                for offset in range(index, index + width)
            ]
            if any(order in excluded_orders for order in orders):
                continue
            text = " ".join(
                str(records[offset].get("text") or "")
                for offset in range(index, index + width)
            )
            score = _heading_score(title, text)
            if score >= threshold:
                page_labels = {
                    normalize_heading(str(records[offset].get("page_label") or ""))
                    for offset in range(index, index + width)
                }
                if printed_page and normalize_heading(printed_page) in page_labels:
                    score = min(1.0, score + 0.04)
                candidates.append((score, page, min(orders)))
    # Some theses place a running subtitle and the numbered chapter title in
    # separate, non-adjacent blocks on the same divider page. Try sparse pairs
    # within a small reading-order window, in either textual order.
    expected_normalized = normalize_heading(title)
    expected_tokens = set(expected_normalized.split())

    def related_fragment(text: str) -> bool:
        normalized = normalize_heading(text)
        if len(normalized) >= 4 and normalized in expected_normalized:
            return True
        compact = normalized.replace(" ", "")
        expected_compact = expected_normalized.replace(" ", "")
        if len(compact) >= 4 and compact in expected_compact:
            return True
        tokens = set(normalized.split())
        return bool(
            expected_tokens and tokens
            and len(expected_tokens & tokens) / max(1, min(len(expected_tokens), len(tokens))) >= 0.5
        )

    for first in range(len(records)):
        first_order = int(records[first].get("reading_order", first))
        if first_order in excluded_orders:
            continue
        first_text = str(records[first].get("text") or "")
        if not related_fragment(first_text):
            continue
        for second in range(first + 1, min(len(records), first + 13)):
            second_order = int(records[second].get("reading_order", second))
            if second_order in excluded_orders:
                continue
            second_text = str(records[second].get("text") or "")
            if not related_fragment(second_text):
                continue
            score = max(
                _heading_score(title, f"{first_text} {second_text}"),
                _heading_score(title, f"{second_text} {first_text}"),
            )
            if score >= threshold:
                candidates.append((score, page, min(first_order, second_order)))
    return candidates


def _best_monotonic_alignment(
    rows: Sequence[tuple[str, int, str, Sequence[tuple[float, int, int]]]],
) -> list[HeadingAnchor]:
    """Globally maximize matched headings, then text score, in document order."""
    # State: last position -> (match count, score sum, anchors).
    states: dict[tuple[int, int], tuple[int, float, list[HeadingAnchor]]] = {
        (0, -1): (0, 0.0, [])
    }
    for title, level, kind, candidates in rows:
        updated = dict(states)  # unmatched is always a legal fail-closed option
        for last, (count, total, anchors) in states.items():
            for score, page, order in candidates:
                position = (page, order)
                if last != (0, -1) and position <= last:
                    continue
                value = (
                    count + 1, total + score,
                    [*anchors, HeadingAnchor(
                        title, level, kind, page, order, round(score, 4),
                    )],
                )
                previous = updated.get(position)
                if previous is None or (value[0], value[1]) > (
                    previous[0], previous[1]
                ):
                    updated[position] = value
        # Bound pathological repeated-title documents while retaining the
        # states with the most matches and best aggregate score.
        if len(updated) > 2000:
            best = sorted(
                updated.items(),
                key=lambda row: (row[1][0], row[1][1], -row[0][0], -row[0][1]),
                reverse=True,
            )[:2000]
            updated = dict(best)
        states = updated
    return max(states.values(), key=lambda row: (row[0], row[1]))[2]


def align_headings(
    headings: Sequence[Mapping[str, Any]], page_records: Mapping[int, Sequence[Mapping[str, Any]]],
    *, threshold: float = 0.78,
) -> tuple[list[HeadingAnchor], dict[str, Any]]:
    """Align inferred headings monotonically, excluding TOC-dense pages."""
    toc_pages, excluded_orders = _toc_excluded_orders(headings, page_records)
    pending_rows: list[
        tuple[str, int, str, str, list[tuple[float, int, int]]]
    ] = []
    for heading in headings:
        title = str(heading.get("title") or "").strip()
        level = max(1, int(heading.get("level") or 1))
        kind = str(heading.get("kind") or "section").strip().casefold()
        printed_page = str(heading.get("printed_page") or "").strip()
        if kind in NON_BOUNDARY_KINDS:
            continue
        candidates: list[tuple[float, int, int]] = []
        for page in sorted(page_records):
            candidates.extend(_candidate_spans(
                title, page, page_records[page],
                excluded_orders.get(page, set()), threshold, printed_page,
            ))
        # Keep one score per position; adjacent span widths often duplicate it.
        by_position: dict[tuple[int, int], float] = {}
        for score, page, order in candidates:
            key = (page, order)
            by_position[key] = max(score, by_position.get(key, 0.0))
        pending_rows.append((
            title, level, kind, printed_page,
            [(score, page, order) for (page, order), score in by_position.items()],
        ))
    # Infer the printed-page to PDF-page offset from repeated high-confidence
    # matches. Each heading contributes at most one vote per offset so repeated
    # mentions of a title cannot dominate the estimate.
    offset_votes: Counter[int] = Counter()
    for _title, _level, _kind, printed_page, candidates in pending_rows:
        printed_number = _printed_page_number(printed_page)
        if printed_number is None:
            continue
        offsets = {
            page - printed_number
            for score, page, _order in candidates if score >= 0.9
        }
        offset_votes.update(offsets)
    estimated_offset: int | None = None
    if offset_votes:
        candidate_offset, votes = offset_votes.most_common(1)[0]
        if votes >= 2:
            estimated_offset = candidate_offset

    alignment_rows: list[
        tuple[str, int, str, Sequence[tuple[float, int, int]]]
    ] = []
    for title, level, kind, printed_page, candidates in pending_rows:
        printed_number = _printed_page_number(printed_page)
        if printed_number is not None and estimated_offset is not None:
            expected_page = printed_number + estimated_offset
            candidates = [
                (min(1.0, score + 0.08) if abs(page - expected_page) <= 1 else score,
                 page, order)
                for score, page, order in candidates
            ]
        alignment_rows.append((title, level, kind, candidates))
    anchors = _best_monotonic_alignment(alignment_rows)
    matched_titles = {normalize_heading(anchor.title) for anchor in anchors}
    unmatched = [
        title for title, _level, _kind, _candidates in alignment_rows
        if normalize_heading(title) not in matched_titles
    ]

    body = [row for row in headings if str(row.get("kind") or "").casefold() not in FUNCTIONAL_KINDS]
    body_titles = {normalize_heading(str(row.get("title") or "")) for row in body}
    matched_body = sum(normalize_heading(anchor.title) in body_titles for anchor in anchors)
    coverage = matched_body / max(1, len(body))
    return anchors, {
        "heading_count": len(headings), "body_heading_count": len(body),
        "matched_count": len(anchors), "matched_body_count": matched_body,
        "body_coverage": round(coverage, 4), "toc_pages": sorted(toc_pages),
        "unmatched": unmatched, "estimated_page_offset": estimated_offset,
    }


# Functional-section heading kinds map to canonical zones so the AI-TOC fast
# path tags reference/note sections just like the Docling and EPUB paths do
# (dev-notes/current/77 Part E).  Without this, references parsed by the AI-TOC
# path stay zone=body and are invisible to zone-aware reference extraction.
_KIND_ZONE = {
    "references": "bibliography", "notes": "endnote",
    "index": "index", "appendix": "back_matter", "back_matter": "back_matter",
}
# Reference/endnote sections the AI-TOC path zoned by heading; their pages can be
# re-parsed with Docling for one-entry-per-chunk structure (E2b, note 77).
_REFERENCE_ZONES = {"bibliography", "endnote"}


def _reference_section_pages(
    chunks: Sequence[tuple[str, str, Mapping[str, Any]]],
) -> list[int]:
    """Return the 1-based pages that carry reference/endnote-zoned chunks."""
    return sorted({
        int(md.get("page") or 0)
        for _cid, _text, md in chunks
        if md.get("zone") in _REFERENCE_ZONES and int(md.get("page") or 0) > 0
    })


#: The splice must not lose text. Docling re-parses the reference pages to get
#: one entry per chunk -- a *boundary* improvement -- so the replacement should
#: carry substantially the same characters. When it carries far fewer, the
#: re-parse failed rather than improved, and swapping it in deletes the
#: section. Four documents lost their entire endnotes this way, up to 40 pages
#: each, because the only guard was that the replacement be non-empty: a single
#: surviving chunk authorised the removal of everything (2026-07-28).
MIN_REFERENCE_SPLICE_RATIO = 0.5


def _splice_reference_chunks(
    structured: Sequence[tuple[str, str, dict[str, Any]]],
    reference_chunks: Sequence[tuple[str, str, dict[str, Any]]],
    reference_pages: Sequence[int],
) -> list[tuple[str, str, dict[str, Any]]]:
    """Replace coarse AI-TOC reference chunks with the Docling-structured ones.

    Body chunks on the same pages are kept; only the reference/endnote-zoned
    AI-TOC chunks are removed, then the Docling reference chunks are inserted and
    everything is re-sorted into reading order.

    The swap is abandoned when the replacement holds materially less text than
    what it would remove: better boundaries are worth having, a missing section
    is not, and the original chunks are already searchable.
    """
    pages = set(int(page) for page in reference_pages)
    kept, removed = [], []
    for row in structured:
        if int(row[2].get("page") or 0) in pages and row[2].get("zone") in _REFERENCE_ZONES:
            removed.append(row)
        else:
            kept.append(row)
    removed_chars = sum(len(str(row[1] or "")) for row in removed)
    incoming_chars = sum(len(str(row[1] or "")) for row in reference_chunks)
    if removed_chars and incoming_chars < removed_chars * MIN_REFERENCE_SPLICE_RATIO:
        return list(structured)
    combined = list(kept) + list(reference_chunks)
    combined.sort(key=lambda row: (int(row[2].get("page") or 0), int(row[2].get("reading_order") or 0)))
    return combined


def apply_anchors(
    chunks: Sequence[tuple[str, str, Mapping[str, Any]]], anchors: Sequence[HeadingAnchor],
) -> list[tuple[str, str, dict[str, Any]]]:
    ordered = sorted(anchors, key=lambda row: (row.page, row.reading_order, row.level))
    output: list[tuple[str, str, dict[str, Any]]] = []
    for chunk_id, text, metadata in chunks:
        md = dict(metadata)
        key = (int(md.get("page") or 0), int(md.get("reading_order") or 0))
        stack: list[str] = []
        active: list[HeadingAnchor] = []
        for anchor in ordered:
            if (anchor.page, anchor.reading_order) > key:
                break
            active = active[: anchor.level - 1]
            active.append(anchor)
        for anchor in active:
            stack.append(anchor.title)
        if stack:
            md["structure_path"] = stack
            md["structure_source"] = "ai_toc_text_alignment"
        # Tag the innermost functional section's zone (references/notes/...), so
        # a chunk under a "References"/"Notes" heading is discoverable as such.
        for anchor in reversed(active):
            zone = _KIND_ZONE.get(anchor.kind)
            if zone:
                if not md.get("zone") or md.get("zone") == "body":
                    md["zone"] = zone
                break
        output.append((chunk_id, text, md))
    return output


def _sample_first_pages(pdf_path: Path, pages: int, max_chars_per_page: int = 6000) -> str:
    sections: list[str] = []
    with fitz.open(pdf_path) as document:
        for index in range(min(pages, document.page_count)):
            text = (document[index].get_text("text") or "").strip()
            sections.append(f"\n--- PDF PAGE {index + 1} ---\n{text[:max_chars_per_page]}")
    return "".join(sections)


def infer_toc(pdf_path: Path, *, client: LLMClient | None = None, pages: int = 20) -> dict[str, Any]:
    prompt = """Infer the canonical document outline from the sampled opening pages.
For a small paper, keep major chapter-level headings only. For a book, use the
printed table-of-contents entries and do not add finer body subheadings. Keep
references, notes, appendices, acknowledgements, abstract, contents, and index
as functional regions when explicit. Preserve source wording and order. The
document title is level 1 metadata and does not make major headings level 2.
Always label the document title as kind "title"; never repeat it as a chapter
or section boundary.
Do not invent headings. printed_page is the printed label, never a PDF index.

SAMPLED PAGES:
""" + _sample_first_pages(pdf_path, pages)
    llm = client or get_llm("extract")
    return llm.generate_json(prompt, schema=TOC_SCHEMA, timeout=180.0)


def try_ai_toc_fast_path(
    pdf_path: Path, item_key: str,
    chunks: Sequence[tuple[str, str, Mapping[str, Any]]],
    quality: Mapping[str, Any], *, client: LLMClient | None = None,
) -> TocRecoveryResult:
    """Return accepted structured chunks or a fail-closed Docling fallback reason."""
    original = [(cid, text, dict(md)) for cid, text, md in chunks]
    load_dotenv_native(Path(__file__).resolve().parents[1])
    if not ai_toc_enabled():
        return TocRecoveryResult(False, "feature_disabled", original, {})
    minimum_pages = int(os.environ.get("PDF_AI_TOC_MIN_PAGES", "30"))
    total_pages = int(quality.get("total_pages") or 0)
    if total_pages < minimum_pages:
        return TocRecoveryResult(False, "below_minimum_pages", original, {
            "total_pages": total_pages, "minimum_pages": minimum_pages,
        })
    if quality.get("is_scanned") or quality.get("is_corrupted"):
        return TocRecoveryResult(False, "text_quality_gate_failed", original, {})
    # Residual-anomaly tolerance shared with the PyMuPDF fast path (P4,
    # note 78, user approval 2026-07-26): these ratios are post-repair
    # residuals (the E2c/E2d patches already removed every attempted page),
    # and accepting the *text* at <=2% residual (fast path) while refusing it
    # *structure* here was an unexplainable middle state. The old 0-tolerance
    # check was also near-dead code -- with the patches enabled it only ever
    # fired on patch exceptions or unattempted residuals.
    ratio_tolerance = float(
        os.environ.get("PYMUPDF_NATIVE_OUTLINE_ANOMALY_RATIO_MAX", "0.02")
    )
    for key in ("scanned_ratio", "corrupted_ratio", "extraction_failure_ratio"):
        if float(quality.get(key) or 0.0) > ratio_tolerance:
            return TocRecoveryResult(False, f"{key}_above_tolerance", original, {})
    try:
        inferred = infer_toc(pdf_path, client=client, pages=int(os.environ.get("PDF_AI_TOC_SAMPLE_PAGES", "20")))
        headings = list(inferred.get("headings") or [])
        if len(headings) < 2:
            return TocRecoveryResult(False, "insufficient_inferred_headings", original, {"heading_count": len(headings)})
        anchors, diagnostics = align_headings(headings, _page_records(pdf_path))
    except (LLMError, OSError, ValueError, TypeError, fitz.FileDataError) as exc:
        return TocRecoveryResult(False, f"recovery_error:{exc}", original, {})
    minimum = float(os.environ.get("PDF_AI_TOC_MIN_COVERAGE", "0.9"))
    if diagnostics["body_heading_count"] < 2:
        return TocRecoveryResult(False, "insufficient_body_headings", original, diagnostics)
    if float(diagnostics["body_coverage"]) < minimum:
        return TocRecoveryResult(False, "body_coverage_below_threshold", original, diagnostics)
    structured = apply_anchors(original, anchors)
    structured_count = sum(bool(md.get("structure_path")) for _, _, md in structured)
    structured_ratio = structured_count / max(1, len(structured))
    diagnostics.update({
        "accepted": True, "document_type": inferred.get("document_type"),
        "structured_chunk_count": structured_count,
        "structured_chunk_ratio": round(structured_ratio, 4),
        "anchor_payload": json.dumps([anchor.__dict__ for anchor in anchors], ensure_ascii=False),
    })
    minimum_structured = float(os.environ.get("PDF_AI_TOC_MIN_STRUCTURED_CHUNK_RATIO", "0.8"))
    if structured_ratio < minimum_structured:
        diagnostics["accepted"] = False
        return TocRecoveryResult(False, "structured_chunk_ratio_below_threshold", original, diagnostics)
    # E2b (note 77): re-parse just the reference/endnote section pages with
    # Docling for one-entry-per-chunk structure.  Opt-in and fail-closed -- any
    # error keeps the coarse AI-TOC chunks so ingestion is never blocked.
    if os.environ.get("PDF_AI_TOC_DOCLING_REFERENCES_ENABLE", "0").strip() == "1":
        reference_pages = _reference_section_pages(structured)
        if reference_pages:
            try:
                from .docling_extract import extract_reference_sections_with_docling
            except ImportError:  # pragma: no cover - direct src entrypoint
                from docling_extract import extract_reference_sections_with_docling
            try:
                attachment_key = str((structured[0][2] if structured else {}).get("attachmentKey") or item_key)
                # attachmentKey was missing here, so every chunk this path
                # produced carried itemKey but no attachmentKey. Deletion and
                # the source-truth check both key on attachmentKey, so these
                # chunks were reachable by neither: not purged when the
                # attachment was removed, and invisible to
                # chunks_without_item's per-attachment accounting even though
                # they are exactly that (2026-07-28, found while auditing P1).
                meta_base = {"itemKey": item_key, "attachmentKey": attachment_key}
                reference_chunks = extract_reference_sections_with_docling(
                    pdf_path, reference_pages, attachment_key=attachment_key, meta_base=meta_base,
                )
                if reference_chunks:
                    structured = _splice_reference_chunks(structured, reference_chunks, reference_pages)
                    diagnostics["docling_reference_pages"] = reference_pages
                    diagnostics["docling_reference_chunks"] = len(reference_chunks)
            except Exception as exc:  # noqa: BLE001 - enrichment must never block ingest
                diagnostics["docling_reference_enrichment_error"] = str(exc)[:200]
    return TocRecoveryResult(True, "accepted", structured, diagnostics)


__all__ = [
    "HeadingAnchor", "TocRecoveryResult", "align_headings", "apply_anchors",
    "infer_toc", "normalize_heading", "try_ai_toc_fast_path",
]
