# src/pdf_extract.py
from __future__ import annotations

import os
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import signal

import fitz  # PyMuPDF

try:
    from .text_utils import (
        HARD_MIN_CHARS, HARD_MIN_CHARS_CJK, MAX_CHARS, TARGET_CHARS, MAX_CHARS_CJK,
        TARGET_CHARS_CJK, MIN_CHUNK_CHARS, MIN_CHUNK_CHARS_NO_SPACE,
        clean_extracted_text, is_no_space_language_document, joiner_for_text,
        looks_like_gibberish, merge_short_chunk_records, normalize_paragraphs,
        split_long_paragraph, analyze_text_quality,
    )
    from .chapter_detect import (
        get_pdf_toc, build_pdf_page_chapter_lookup, build_pdf_page_structure_path_lookup,
        infer_structure_roles,
    )
    from .pdf_provenance import classify_pdf_source, detect_text_defects
    from .heading_zone import classify_heading_path
except ImportError:  # direct execution with src/ on sys.path
    from text_utils import (
        HARD_MIN_CHARS, HARD_MIN_CHARS_CJK, MAX_CHARS, TARGET_CHARS, MAX_CHARS_CJK,
        TARGET_CHARS_CJK, MIN_CHUNK_CHARS, MIN_CHUNK_CHARS_NO_SPACE,
        clean_extracted_text, is_no_space_language_document, joiner_for_text,
        looks_like_gibberish, merge_short_chunk_records, normalize_paragraphs,
        split_long_paragraph, analyze_text_quality,
    )
    from chapter_detect import (
        get_pdf_toc, build_pdf_page_chapter_lookup, build_pdf_page_structure_path_lookup,
        infer_structure_roles,
    )
    from pdf_provenance import classify_pdf_source, detect_text_defects
    from heading_zone import classify_heading_path

PDF_DROP_REPEATED_LINES = (os.environ.get("PDF_DROP_REPEATED_LINES") or "1") == "1"
# Retain unusable page text as zone="corrupted" instead of dropping it silently
# (note 79, U3). Excluded from ordinary retrieval and from summary input by
# ZONE_POLICIES; reachable via rag_search(include_corrupted=True). Applies to
# newly ingested material only -- previously discarded text is not recoverable
# without a reingest, and no retroactive pass is planned.
PDF_RETAIN_CORRUPTED_TEXT = (os.environ.get("PDF_RETAIN_CORRUPTED_TEXT") or "1") == "1"
PDF_STRIP_REPEATED_PREFIX = (os.environ.get("PDF_STRIP_REPEATED_PREFIX") or "1") == "1"

# Per-page timeout (seconds). Set PAGE_TIMEOUT_SEC=0 to disable.
PAGE_TIMEOUT_SEC = int((os.environ.get("PAGE_TIMEOUT_SEC") or "30").strip())

# Tesseract OCR fallback for font-encoding mismatch pages.
# Requires tesseract binary on PATH. Set PDF_OCR_FALLBACK=0 to disable.
PDF_OCR_FALLBACK = (os.environ.get("PDF_OCR_FALLBACK") or "1") == "1"
PDF_OCR_DPI = int((os.environ.get("PDF_OCR_DPI") or "300").strip())
# Poppler can recover text from some custom-encoded Type 1 fonts for which
# PyMuPDF cannot synthesize a Unicode mapping. It is local, deterministic, and
# much cheaper than OCR, so it precedes the Tesseract/Docling fallbacks.
PDF_POPPLER_TEXT_FALLBACK = (
    os.environ.get("PDF_POPPLER_TEXT_FALLBACK") or "1"
) == "1"


def _find_tesseract() -> Optional[str]:
    """Return the path to the tesseract binary, or None if not found.

    Checks PATH first, then common Homebrew locations which may not be
    on PATH when invoked via ``uv run``.
    """
    path = shutil.which("tesseract")
    if path:
        return path
    for candidate in ["/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"]:
        if Path(candidate).exists():
            return candidate
    return None


def _find_pdftotext() -> Optional[str]:
    path = shutil.which("pdftotext")
    if path:
        return path
    for candidate in ["/opt/homebrew/bin/pdftotext", "/usr/local/bin/pdftotext"]:
        if Path(candidate).exists():
            return candidate
    return None


def _extract_pages_with_poppler(pdf_path: Path) -> list[str]:
    """Extract every PDF page with Poppler, preserving form-feed boundaries."""
    binary = _find_pdftotext()
    if not binary:
        return []
    try:
        import subprocess
        result = subprocess.run(
            [binary, "-layout", str(pdf_path), "-"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0 or not result.stdout:
            return []
        pages = result.stdout.split("\f")
        if pages and not pages[-1].strip():
            pages.pop()
        return pages
    except (OSError, subprocess.SubprocessError):
        return []


def _resolve_ocr_lang() -> str:
    """Return the best available Tesseract language string.

    If PDF_OCR_LANG is explicitly set, use it verbatim.
    Otherwise, probe the installed Tesseract languages and prefer
    ``jpn+eng`` when Japanese data is available, falling back to
    ``eng`` with a one-time diagnostic message.

    To install Japanese Tesseract data:
      macOS:  brew install tesseract-lang
      Linux:  apt install tesseract-ocr-jpn
    """
    explicit = os.environ.get("PDF_OCR_LANG", "").strip()
    if explicit:
        return explicit

    tesseract_bin = _find_tesseract()
    if not tesseract_bin:
        return "eng"

    # Probe available languages once
    try:
        import subprocess
        result = subprocess.run(
            [tesseract_bin, "--list-langs"],
            capture_output=True, text=True, timeout=5,
        )
        installed = set(
            line.strip() for line in result.stdout.splitlines()
            if line.strip() and not line.startswith("List")
        )
    except Exception:
        return "eng"

    if "jpn" in installed:
        return "jpn+eng"
    else:
        # Warn once per process about missing Japanese OCR support
        if not getattr(_resolve_ocr_lang, "_warned", False):
            _resolve_ocr_lang._warned = True  # type: ignore[attr-defined]
            print(
                "[NOTE] Tesseract Japanese language data not found.\n"
                "       Extraction failures in Japanese PDFs will use English OCR,\n"
                "       which may produce garbled results.\n"
                "       Install Japanese support:\n"
                "         macOS:  brew install tesseract-lang\n"
                "         Linux:  apt install tesseract-ocr-jpn\n"
                "       Then set: PDF_OCR_LANG=jpn+eng",
                file=os.sys.__stderr__,
            )
        return "eng"


PDF_OCR_LANG = _resolve_ocr_lang()


def scanned_ratio_threshold() -> float:
    """Document-level scanned-ratio threshold for ``is_scanned`` (P3, note 78).

    ``TEXT_QUALITY_SCAN_THRESHOLD`` used to serve two different roles with one
    variable: (a) the *page*-level ``scan_score`` cutoff in
    ``text_utils.analyze_text_quality`` and (b) the *document*-level
    ``scanned_ratio`` cutoff here. Those are different-dimensional quantities
    -- tuning one silently moved the other (e.g. lowering the document cutoff
    to 0.6 would also reclassify every 40-79-character page as scanned and
    balloon the E2c patch workload). This dedicated variable now owns role (b);
    the page-level variable keeps its name and role (a). For backward
    compatibility, an unset ``TEXT_QUALITY_SCANNED_RATIO_THRESHOLD`` inherits
    the legacy variable's value (existing .env files keep behaving
    identically), and the default stays 0.8 (audit note 78 P3, user approval
    2026-07-26; the 0.8-vs-0.6 asymmetry with corruption is deliberate --
    scanned pages are individually patchable figure plates, so the whole-
    document escalation bar is high, while text-layer corruption at 60%
    already makes the document untrustworthy).
    """
    inherited = os.environ.get("TEXT_QUALITY_SCAN_THRESHOLD", "0.8")
    return float(os.environ.get("TEXT_QUALITY_SCANNED_RATIO_THRESHOLD", inherited))


def corrupted_ratio_threshold() -> float:
    """Document-level corrupted-ratio threshold for ``is_corrupted`` (P3, note 78).

    Same one-variable-two-roles split as ``scanned_ratio_threshold``:
    ``TEXT_QUALITY_CORRUPTION_THRESHOLD`` keeps the page-level
    ``corruption_score`` role, this variable owns the document-level
    ``corrupted_ratio`` role, inheriting the legacy value when unset.
    Default stays 0.6.
    """
    inherited = os.environ.get("TEXT_QUALITY_CORRUPTION_THRESHOLD", "0.6")
    return float(os.environ.get("TEXT_QUALITY_CORRUPTED_RATIO_THRESHOLD", inherited))


def recompute_scanned_quality_after_patch(
    quality_info: Dict[str, Any], patched_pages: set[int], total_pages: int,
) -> Dict[str, Any]:
    """Return ``quality_info`` updated after Docling-patching some scanned pages.

    Removes ``patched_pages`` from ``scanned_pages``/recomputes ``scanned_ratio``
    and ``is_scanned`` so the PyMuPDF fast-path and AI-TOC gates
    (``pymupdf_fast_path_passes`` / ``try_ai_toc_fast_path``) see the reduced
    scan footprint instead of the pre-patch counts (E2c, dev-notes/current/77).
    """
    remaining = [p for p in (quality_info.get("scanned_pages") or []) if p not in patched_pages]
    updated = dict(quality_info)
    updated["scanned_pages"] = remaining
    updated["scanned_ratio"] = round(len(remaining) / max(1, total_pages), 3)
    updated["is_scanned"] = updated["scanned_ratio"] >= scanned_ratio_threshold()
    return updated


def recompute_blank_pages_after_patch(
    quality_info: Dict[str, Any], patched_pages: set[int],
) -> Dict[str, Any]:
    """Return ``quality_info`` with ``patched_pages`` cleared from empty/blank lists.

    ``coverage_from_extraction`` (source_coverage.py) builds a document's
    ``text_units`` from the final chunk list and its ``blank_units`` from
    ``quality_info["empty_pages"]``/``["blank_pages"]``. A page the scan-derived
    repair (``_scan_pages_needing_repair`` / ``patch_corrupted_pages_with_docling``
    in index_from_zotero.py) successfully attempts gets a real chunk there --
    or, if OCR still found nothing, an explicit ``corrupted_unresolved`` marker
    chunk whose synthetic placeholder text is still non-empty. Either way the
    page now has a chunk, so it lands in ``text_units``. If it is still also
    listed in ``empty_pages``/``blank_pages`` from before the patch ran, the two
    sets disagree about the same page and coverage_from_extraction fails the
    whole document closed with ``source_unit_text_blank_conflict`` -- silently,
    since that rejection leaves the manifest untouched exactly like a 0-chunk
    extraction (found 2026-08-02, diagnosing YX3MMS4D page 444: scan-derived
    repair recovered it, but ``empty_pages`` still listed it because nothing
    called this). Sibling to ``recompute_scanned_quality_after_patch``, which
    reconciles ``scanned_pages`` for the born-digital scan-patch path; this one
    covers the blank/empty fields the scan-derived path actually leaves behind.
    """
    updated = dict(quality_info)
    for field in ("empty_pages", "blank_pages"):
        if updated.get(field):
            updated[field] = [p for p in updated[field] if p not in patched_pages]
    return updated


def recompute_corrupted_quality_after_patch(
    quality_info: Dict[str, Any], patched_pages: set[int], total_pages: int,
) -> Dict[str, Any]:
    """Return ``quality_info`` updated after Docling-patching some corrupted pages.

    Sister function to ``recompute_scanned_quality_after_patch`` (E2d, note
    77): removes ``patched_pages`` from ``corrupted_pages``/recomputes
    ``corrupted_ratio`` and ``is_corrupted`` so
    ``extraction_engine.pymupdf_fast_path_rejection_reason`` sees the residual
    unresolved-corruption footprint instead of the pre-patch counts. As with
    the scanned-page patch, every page ``patch_corrupted_pages_with_docling``
    attempted (whether or not it recovered clean text) is removed here --
    a page Docling also couldn't clean up is conclusive evidence about that
    page, not an unresolved gap, and the caller has already spliced in a
    ``corrupted_unresolved`` marker chunk for it so it stays visible rather
    than silently vanishing.
    """
    remaining = [p for p in (quality_info.get("corrupted_pages") or []) if p not in patched_pages]
    updated = dict(quality_info)
    updated["corrupted_pages"] = remaining
    updated["corrupted_ratio"] = round(len(remaining) / max(1, total_pages), 3)
    updated["is_corrupted"] = updated["corrupted_ratio"] >= corrupted_ratio_threshold()
    # corrupted_pages is the union of its per-cause breakdowns
    # (extraction_failure_pages + content_corruption_pages, see
    # extract_chunks_from_pdf), so patching a page out of the union must also
    # patch it out of the breakdowns. extraction_failure_ratio in particular
    # feeds pymupdf_fast_path_rejection_reason's unconditional early-reject
    # check -- leaving the stale pre-patch value there would veto the fast
    # path even after a successful repair (review finding, fixed 2026-07-26).
    remaining_extraction_failure = [
        p for p in (quality_info.get("extraction_failure_pages") or []) if p not in patched_pages
    ]
    remaining_content_corruption = [
        p for p in (quality_info.get("content_corruption_pages") or []) if p not in patched_pages
    ]
    updated["extraction_failure_pages"] = remaining_extraction_failure
    updated["extraction_failure_ratio"] = round(
        len(remaining_extraction_failure) / max(1, total_pages), 3,
    )
    updated["content_corruption_pages"] = remaining_content_corruption
    return updated


def normalize_block_text_to_paragraph(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return ""

    parts: List[str] = []
    for ln in lines:
        if parts and parts[-1].endswith("-"):
            parts[-1] = parts[-1][:-1] + ln
        else:
            parts.append(ln)

    joiner = joiner_for_text("".join(parts))
    merged = joiner.join(parts)
    merged = re.sub(r"\s+", " ", merged).strip()
    return merged


def _bbox_union(values: List[List[float]]) -> List[float]:
    return [
        round(min(value[0] for value in values), 3),
        round(min(value[1] for value in values), 3),
        round(max(value[2] for value in values), 3),
        round(max(value[3] for value in values), 3),
    ]


def _column_boundary(blocks: List[Dict[str, Any]], page_width: float) -> Optional[float]:
    """Return a stable gutter midpoint when two populated columns are present."""
    if page_width <= 0:
        return None
    candidates = [
        block for block in blocks
        if (block["bbox"][2] - block["bbox"][0]) <= page_width * 0.58
    ]
    best: Optional[Tuple[float, float, float]] = None
    # Candidate gutters come from non-overlapping block edges. Score by the
    # smaller side population, then gap width, and use x as deterministic tie-break.
    for left in candidates:
        for right in candidates:
            gap = right["bbox"][0] - left["bbox"][2]
            # Journal layouts commonly use a narrow gutter (about 14 pt on a
            # Letter/A4 page).  Requiring 2.5% of the page width misclassified
            # those pages as single-column and interleaved the two columns by
            # y-coordinate.  The populated-side checks below are the stronger
            # guard against treating ordinary paragraph indents as a gutter.
            if gap < max(8.0, page_width * 0.015):
                continue
            boundary = (left["bbox"][2] + right["bbox"][0]) / 2
            if not page_width * 0.28 <= boundary <= page_width * 0.72:
                continue
            left_count = sum(block["bbox"][2] <= boundary for block in candidates)
            right_count = sum(block["bbox"][0] >= boundary for block in candidates)
            if min(left_count, right_count) < 2:
                left_side = [block for block in candidates if block["bbox"][2] <= boundary]
                right_side = [block for block in candidates if block["bbox"][0] >= boundary]
                # A transition page can contain one long continuation block in
                # the left column and several short terminal-section blocks in
                # the right. Requiring two blocks on both sides made such pages
                # single-column and put right-column headings before the left
                # continuation. Accept the singleton only when it is clearly a
                # substantive column, not an ordinary indent or page label.
                singleton_side = left_side if len(left_side) == 1 else right_side
                other_side = right_side if len(left_side) == 1 else left_side
                singleton_height = (
                    singleton_side[0]["bbox"][3] - singleton_side[0]["bbox"][1]
                    if len(singleton_side) == 1 else 0.0
                )
                if len(other_side) < 2 or singleton_height < page_width * 0.35:
                    continue
            score = (float(min(left_count, right_count)), gap, -boundary)
            if best is None or score > best:
                best = score
    return -best[2] if best is not None else None


def _order_layout_blocks(
    blocks: List[Dict[str, Any]], page_width: float,
) -> List[Dict[str, Any]]:
    """Order text blocks top-to-bottom, or column-by-column within vertical bands."""
    boundary = _column_boundary(blocks, page_width)
    if boundary is None:
        ordered = sorted(blocks, key=lambda block: (
            block["bbox"][1], block["bbox"][0], block["source_block_index"],
        ))
        for reading_order, block in enumerate(ordered):
            block["reading_order"] = reading_order
            block["column"] = "single"
        return ordered

    # Full-width titles and section heads divide a page into bands. Within each
    # band the left column is read completely before the right column.
    spanning = [
        block for block in blocks
        if block["bbox"][0] < boundary < block["bbox"][2]
        or (block["bbox"][2] - block["bbox"][0]) >= page_width * 0.68
    ]
    spanning.sort(key=lambda block: (
        block["bbox"][1], block["bbox"][0], block["source_block_index"],
    ))
    remaining = [block for block in blocks if block not in spanning]
    ordered: List[Dict[str, Any]] = []
    lower = float("-inf")
    for separator in spanning:
        upper = (separator["bbox"][1] + separator["bbox"][3]) / 2
        band = [
            block for block in remaining
            if lower <= (block["bbox"][1] + block["bbox"][3]) / 2 < upper
        ]
        ordered.extend(sorted(
            (block for block in band if (block["bbox"][0] + block["bbox"][2]) / 2 < boundary),
            key=lambda block: (block["bbox"][1], block["bbox"][0], block["source_block_index"]),
        ))
        ordered.extend(sorted(
            (block for block in band if (block["bbox"][0] + block["bbox"][2]) / 2 >= boundary),
            key=lambda block: (block["bbox"][1], block["bbox"][0], block["source_block_index"]),
        ))
        remaining = [block for block in remaining if block not in band]
        ordered.append(separator)
        lower = upper
    ordered.extend(sorted(
        (block for block in remaining if (block["bbox"][0] + block["bbox"][2]) / 2 < boundary),
        key=lambda block: (block["bbox"][1], block["bbox"][0], block["source_block_index"]),
    ))
    ordered.extend(sorted(
        (block for block in remaining if (block["bbox"][0] + block["bbox"][2]) / 2 >= boundary),
        key=lambda block: (block["bbox"][1], block["bbox"][0], block["source_block_index"]),
    ))
    for reading_order, block in enumerate(ordered):
        block["reading_order"] = reading_order
        center = (block["bbox"][0] + block["bbox"][2]) / 2
        block["column"] = "full" if block in spanning else "left" if center < boundary else "right"
    return ordered


def _merge_layout_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge only adjacent blocks in the same detected column and vertical flow."""
    merged: List[Dict[str, Any]] = []
    for block in blocks:
        if merged:
            previous = merged[-1]
            gap = block["bbox"][1] - previous["bbox"][3]
            if (
                block.get("column") == previous.get("column")
                and block.get("column") != "full"
                and 0 <= gap <= 12.0
            ):
                joiner = joiner_for_text(previous["text"] + block["text"])
                previous["text"] = previous["text"] + joiner + block["text"]
                previous["bbox"] = _bbox_union([previous["bbox"], block["bbox"]])
                previous["source_block_indices"].extend(block["source_block_indices"])
                continue
        merged.append({**block, "source_block_indices": list(block["source_block_indices"])})
    for reading_order, block in enumerate(merged):
        block["reading_order"] = reading_order
    return merged


def _outline_events_by_page(
    toc: List[Tuple[int, str, int]],
) -> Dict[int, List[Dict[str, Any]]]:
    """Expand same-page TOC entries into full-path transition events."""
    events: Dict[int, List[Dict[str, Any]]] = {}
    active: List[str] = []
    for level, title, page in toc:
        level = max(1, int(level))
        active = active[: level - 1]
        active.append(str(title).strip())
        events.setdefault(max(1, int(page)), []).append({
            "title": str(title).strip(),
            "path": list(active),
        })
    return events


def _record_matches_outline_title(text: str, title: str) -> bool:
    record = " ".join(clean_extracted_text(str(text or "")).split()).casefold()
    heading = " ".join(clean_extracted_text(str(title or "")).split()).casefold()
    if not record or not heading:
        return False
    if record == heading or re.match(rf"^{re.escape(heading)}(?:\s|[.:：—–-])", record):
        return True
    # PDF generators sometimes prepend "Appendix A." to an outline title.
    # Restrict infix matching to genuinely short heading records so a mention
    # in ordinary prose cannot move the document structure.
    return len(record) <= 160 and bool(
        re.search(rf"(?:^|\s){re.escape(heading)}(?:$|\s|[.:：—–-])", record)
    )


def _resolve_record_structure_paths(
    records: List[Dict[str, Any]], *, previous_path: List[str],
    page_events: List[Dict[str, Any]],
) -> List[List[str]]:
    """Resolve TOC transitions at the matching record, not page start."""
    if not page_events:
        return []
    anchors: List[Tuple[int, List[str]]] = []
    search_from = 0
    for event in page_events:
        for index in range(search_from, len(records)):
            if _record_matches_outline_title(records[index].get("text", ""), event["title"]):
                anchors.append((index, list(event["path"])))
                search_from = index + 1
                break
    if not anchors:
        # Some outlines point to a page whose visible heading is absent from
        # the text layer. Preserve the established page-level behavior in that
        # case; the record-level correction is only justified by an observed
        # same-page heading anchor.
        return [list(page_events[-1]["path"]) for _record in records]
    # Failing to locate a heading must not retroactively relabel the page.
    active = list(previous_path)
    by_index = {index: path for index, path in anchors}
    output: List[List[str]] = []
    for index in range(len(records)):
        if index in by_index:
            active = list(by_index[index])
        output.append(list(active))
    return output


def extract_layout_blocks_from_pdf_page(page: Any) -> List[Dict[str, Any]]:
    """Extract text blocks with real PyMuPDF geometry and deterministic reading order."""
    raw_blocks = page.get_text("blocks", sort=False) or []
    # Keep the page extent with each record.  Header/footer filtering must not
    # infer position from reading order: a repeated phrase in the middle of a
    # page is body text, even when it happens to be a short paragraph.
    page_rect = getattr(page, "rect", None)
    try:
        page_y0 = float(page_rect.y0)
        page_y1 = float(page_rect.y1)
    except (AttributeError, TypeError, ValueError):
        page_y0 = page_y1 = 0.0
    blocks: List[Dict[str, Any]] = []
    for source_index, raw in enumerate(raw_blocks):
        if not raw or len(raw) < 5:
            continue
        block_type = int(raw[6]) if len(raw) >= 7 else 0
        if block_type != 0 or not isinstance(raw[4], str):
            continue
        text = normalize_block_text_to_paragraph(clean_extracted_text(raw[4]))
        if not text:
            continue
        bbox = [float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])]
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        record = {
            "text": text, "bbox": [round(value, 3) for value in bbox],
            "block_type": "text", "source_block_index": source_index,
            "source_block_indices": [source_index],
        }
        if page_y1 > page_y0:
            record["page_y0"] = round(page_y0, 3)
            record["page_y1"] = round(page_y1, 3)
        blocks.append(record)
    return _merge_layout_blocks(_order_layout_blocks(blocks, float(page.rect.width)))


def extract_paragraphs_from_pdf_page(
    page: Any, *, raise_on_text_failure: bool = False,
) -> List[str]:
    """Extract paragraphs from one PDF page.

    ``raise_on_text_failure`` is used by the document-level extractor, where
    silently treating an unreadable page as a genuine blank would hide a
    coverage gap.  The default preserves this helper's historic best-effort
    behavior for callers that only need a paragraph list.
    """
    try:
        records = extract_layout_blocks_from_pdf_page(page)
        if records:
            return [record["text"] for record in records]

    except TimeoutError:
        raise
    except Exception:
        pass

    # fallback
    try:
        raw = page.get_text("text") or ""
        raw = clean_extracted_text(raw)
        joiner = joiner_for_text(raw[:20000])
        return normalize_paragraphs(raw, joiner=joiner)
    except TimeoutError:
        raise
    except Exception:
        if raise_on_text_failure:
            raise
        return []


def _rendered_page_is_visually_blank(page: Any) -> bool:
    """Confirm textless input is blank from pixels, not extractor absence."""
    try:
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(0.5, 0.5), colorspace=fitz.csGRAY, alpha=False,
        )
        samples = pixmap.samples
        # Text/vector/Form content produces at least some non-white pixels.
        return bool(samples) and min(samples) >= 250
    except Exception:
        # Inability to render is not positive blank-page evidence.
        return False


_PAGE_MARGIN_FRACTION = 0.12
_MAX_MARGIN_BLOCK_FRACTION = 0.20


def _record_margin_position(record: Any) -> Optional[str]:
    """Return ``top``/``bottom`` only when real geometry proves a margin.

    Text-only fallback records deliberately have no usable margin position.
    Treating a repeated string as a header without that proof was the source
    of body-text loss in documents with refrains and boilerplate paragraphs.
    """
    if not isinstance(record, dict):
        return None
    bbox = record.get("bbox")
    try:
        y0, y1 = float(bbox[1]), float(bbox[3])
        page_y0 = float(record["page_y0"])
        page_y1 = float(record["page_y1"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    page_height = page_y1 - page_y0
    if page_height <= 0 or y1 <= y0:
        return None
    # A full-page fallback paragraph may touch both edges but is never a
    # header/footer.  Margin artefacts are small relative to their page.
    if y1 - y0 > page_height * _MAX_MARGIN_BLOCK_FRACTION:
        return None
    margin = page_height * _PAGE_MARGIN_FRACTION
    if y0 - page_y0 <= margin:
        return "top"
    if page_y1 - y1 <= margin:
        return "bottom"
    return None


def _record_text(record: Any) -> str:
    return str(record.get("text") or "").strip() if isinstance(record, dict) else ""


def _repeated_margin_key(record: Any) -> str:
    """Canonical text used to compare geometrically proven margin records."""
    text = re.sub(r"\s+", " ", _record_text(record)).strip()
    if not text:
        return ""
    # Page labels are often emitted in the same PDF block as a running footer,
    # making the otherwise identical block differ on every page.
    if _record_margin_position(record) == "bottom":
        without_page_label = re.sub(
            r"^(?:(?:\d+)|(?:[ivxlcdm]+))[.)]?\s+",
            "",
            text,
            count=1,
            flags=re.IGNORECASE,
        )
        if without_page_label:
            text = without_page_label
    return text


def detect_repeated_lines(records_by_page: List[List[Any]]) -> set[str]:
    """
    Detect repeated margin lines across many pages (typical headers/footers).

    A string is a candidate only if its bbox places it in a narrow page margin.
    Legacy text-only inputs have no position evidence and therefore produce no
    candidates; failing closed is preferable to deleting real body prose.
    """
    from collections import Counter

    cnt: Counter[str] = Counter()
    pages_with_any = 0
    for records in records_by_page:
        if not records:
            continue
        pages_with_any += 1
        uniq = {
            text
            for record in records
            if _record_margin_position(record)
            for text in [_repeated_margin_key(record)]
            if text and len(text) <= 180
        }
        for p in uniq:
            cnt[p] += 1

    if pages_with_any <= 3:
        return set()

    threshold = max(4, int(pages_with_any * 0.25))
    repeated = {s for s, c in cnt.items() if c >= threshold and len(s) <= 180}
    return repeated


def drop_repeated_lines_from_paras(paras: List[str], repeated_lines: set[str]) -> List[str]:
    """Legacy text-only helper: retain text when no bbox evidence is present."""
    return [p for p in paras if p]


def detect_repeated_prefixes(records_by_page: List[List[Any]]) -> set[str]:
    """
    Detect repeated prefixes on the first paragraph (e.g., journal title + page number).
    Returns small prefix strings to strip if they reoccur frequently.
    """
    from collections import Counter

    cnt: Counter[str] = Counter()
    pages_with_any = 0
    for records in records_by_page:
        if not records:
            continue
        pages_with_any += 1
        # Prefix stripping is even more destructive than dropping a complete
        # block, so require both first-block status and top-margin geometry.
        first_record = records[0]
        if _record_margin_position(first_record) != "top":
            continue
        first = _record_text(first_record)
        if not first:
            continue
        # Take first 40 chars as a candidate prefix (after collapsing spaces).
        cand = re.sub(r"\s+", " ", first)[:40].strip()
        if cand:
            cnt[cand] += 1

    if pages_with_any <= 3:
        return set()

    threshold = max(4, int(pages_with_any * 0.25))
    return {s for s, c in cnt.items() if c >= threshold and 5 <= len(s) <= 40}


def strip_repeated_prefix_from_first_para(paras: List[str], repeated_prefixes: set[str]) -> List[str]:
    """Legacy text-only helper: do not mutate content without bbox evidence."""
    return list(paras)


def _strip_repeated_prefix_from_first_record(
    records: List[Dict[str, Any]], repeated_prefixes: set[str],
) -> List[Dict[str, Any]]:
    """Strip a proven running prefix from the geometrically top first block."""
    if not records or _record_margin_position(records[0]) != "top":
        return records
    first = records[0]
    norm_first = re.sub(r"\s+", " ", _record_text(first)).strip()
    if not norm_first:
        return records
    for pref in sorted(repeated_prefixes, key=len, reverse=True):
        if norm_first.startswith(pref):
            new_first = norm_first[len(pref):].lstrip(" -–—:：\t")
            if new_first:
                return [{**first, "text": new_first}, *records[1:]]
            return records[1:]
    return records


@contextmanager
def _capture_os_stderr():
    import os as _os

    saved_fd = _os.dup(2)
    r_fd, w_fd = _os.pipe()
    _os.dup2(w_fd, 2)
    _os.close(w_fd)
    try:
        yield r_fd
    finally:
        _os.dup2(saved_fd, 2)
        _os.close(saved_fd)


def _read_fd_text(fd: int) -> str:
    import os as _os

    chunks: List[bytes] = []
    try:
        while True:
            b = _os.read(fd, 8192)
            if not b:
                break
            chunks.append(b)
    finally:
        try:
            _os.close(fd)
        except Exception:
            pass
    return b"".join(chunks).decode("utf-8", errors="replace")


@contextmanager
def _page_timeout(seconds: int):
    """Context manager that raises TimeoutError if the block exceeds `seconds`.

    Uses SIGALRM (Unix/macOS only). If SIGALRM is unavailable or seconds <= 0,
    the block runs without a timeout.
    """
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"PyMuPDF page.get_text() timed out after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _ocr_page_with_tesseract(page: Any) -> List[str]:
    """Render a PDF page to an image and OCR it with Tesseract.

    Used as a fallback when PyMuPDF text extraction fails due to
    custom font encodings without ToUnicode CMaps. The rendered
    image contains correct glyphs that Tesseract can read.

    Returns a list of paragraph strings, or an empty list on failure.
    """
    import subprocess
    import tempfile

    try:
        mat = fitz.Matrix(PDF_OCR_DPI / 72, PDF_OCR_DPI / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
    except Exception:
        return []

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(img_bytes)
            tmp_path = f.name

        tesseract_bin = _find_tesseract()
        if not tesseract_bin:
            return []

        result = subprocess.run(
            [tesseract_bin, tmp_path, "stdout", "-l", PDF_OCR_LANG, "--psm", "6"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            stderr_msg = (result.stderr or "").strip()
            if stderr_msg:
                # Only log the first error per process to avoid spam
                if not getattr(_ocr_page_with_tesseract, "_logged_error", False):
                    _ocr_page_with_tesseract._logged_error = True  # type: ignore[attr-defined]
                    print(
                        f"[WARN] Tesseract OCR failed (lang={PDF_OCR_LANG}): {stderr_msg[:200]}\n"
                        f"       Install the language data or set PDF_OCR_LANG to an available language.",
                        file=os.sys.__stderr__,
                    )
            return []

        text = result.stdout.strip()
        if not text:
            return []

        # Parse OCR output into paragraphs
        joiner = joiner_for_text(text[:20000])
        paras = normalize_paragraphs(text, joiner=joiner)
        return [p for p in paras if p.strip()]
    except FileNotFoundError:
        # tesseract not installed — not an error, just unavailable
        if not getattr(_ocr_page_with_tesseract, "_logged_missing", False):
            _ocr_page_with_tesseract._logged_missing = True  # type: ignore[attr-defined]
            print(
                "[NOTE] Tesseract not found on PATH. OCR fallback unavailable.\n"
                "       Install: brew install tesseract (macOS) / apt install tesseract-ocr (Linux)",
                file=os.sys.__stderr__,
            )
        return []
    except Exception:
        return []
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def extract_chunks_from_pdf(
    pdf_path: Path,
    attachment_key: str,
    meta_base: Dict[str, Any],
) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], Dict[str, Any]]:
    chunks: List[Tuple[str, str, Dict[str, Any]]] = []
    
    # Safe default quality info in case of early exit/failure
    quality_info = {
        "is_scanned": False,
        "is_corrupted": False,
        "scanned_pages": [],
        "corrupted_pages": [],
        "extraction_failure_pages": [],
        "content_corruption_pages": [],
        "poppler_fallback_pages": [],
        "ocr_fallback_pages": [],
        "low_text_pages": [],
        "empty_pages": [],
        "total_pages": 1,
        "scanned_ratio": 0.0,
        "corrupted_ratio": 0.0,
        "extraction_failure_ratio": 0.0,
        "ocr_fallback_ratio": 0.0,
        "avg_corruption_score": 0.0,
        "max_corruption_score": 0.0,
        "avg_extraction_failure_score": 0.0,
        "page_scores": {},
    }

    # 章構造を事前に取得（失敗しても処理続行）
    try:
        _toc = get_pdf_toc(str(pdf_path))
        _chapter_lookup = build_pdf_page_chapter_lookup(_toc) if _toc else None
        _structure_path_lookup = build_pdf_page_structure_path_lookup(_toc) if _toc else None
    except Exception:
        _toc = None
        _chapter_lookup = None
        _structure_path_lookup = None
    quality_info["has_outline"] = bool(_toc)
    _outline_events = _outline_events_by_page(_toc or [])

    # Text-layer provenance.  A scan that has been OCRed also has a text layer,
    # so this is not "does it have text?" but "can the text be trusted?" -- and
    # it decides whether a page without text is a figure (born-digital) or an
    # OCR failure (scan).  Both the E2c figure patch and the fast-path gate
    # read it, so it has to be resolved before any page-level repair.
    try:
        _source_class = classify_pdf_source(pdf_path)
    except Exception as exc:
        _source_class = None
        print(
            f"[WARN] PDF source classification failed: attachment={attachment_key} err={exc}",
            file=os.sys.__stderr__,
        )
    if _source_class is not None:
        quality_info.update(_source_class.as_metadata())

    captured_text = ""
    r_fd: Optional[int] = None
    try:
        with _capture_os_stderr() as _r_fd:
            r_fd = _r_fd
            doc = fitz.open(str(pdf_path))
            try:
                paras_by_page: List[List[str]] = []
                layout_by_page: List[List[Dict[str, Any]]] = []
                page_labels: Dict[int, str] = {}
                scanned_pages = []          # image-only pages (needs Docling)
                corrupted_pages = []        # above corruption threshold
                # Pages whose content could not be read at all (for example a
                # load_page/get_text failure) as well as text-layer decoding
                # failures.  These are neither genuine blanks nor scans: a
                # later engine must explicitly retry them.
                extraction_failure_pages = []
                content_corruption_pages = []  # OCR / linguistic corruption
                poppler_fallback_pages = []  # pages recovered through pdftotext
                ocr_fallback_pages = []     # pages where Tesseract fallback was used
                poppler_pages: Optional[list[str]] = None
                poppler_attempted = False
                ocr_notified = False        # one-time "OCR started" message
                low_text_pages = []         # very little text but not image-only
                empty_pages = []            # raster-confirmed white pages
                unresolved_nontext_pages = []  # rendered/vector content with no text layer
                gibberish_pages = []        # unusable text, retained as zone=corrupted (U3)
                page_scores: Dict[int, Dict[str, Any]] = {}  # per-page scores
                total_pages = doc.page_count

                # Document-level ratio thresholds (P3, note 78): distinct from
                # the same-named page-level score thresholds analyze_text_quality
                # reads inside the loop below.
                SCAN_THRESHOLD = scanned_ratio_threshold()
                CORRUPTION_THRESHOLD = corrupted_ratio_threshold()

                for pi in range(doc.page_count):
                    # Keep a failed load from reusing the previous iteration's
                    # page object in the exception handler.  In particular, an
                    # unreadable page after an image-bearing page used to be
                    # misreported as scanned rather than as an extraction gap.
                    page = None
                    try:
                        with _page_timeout(PAGE_TIMEOUT_SEC):
                            page = doc.load_page(pi)
                            # Capture PDF page label (book page number, e.g. "xii", "15")
                            try:
                                lbl = page.get_label()
                                if lbl:
                                    page_labels[pi] = lbl
                            except Exception:
                                pass
                            try:
                                layout_records = extract_layout_blocks_from_pdf_page(page)
                            except TimeoutError:
                                raise
                            except Exception:
                                layout_records = []
                            if layout_records:
                                paras = [record["text"] for record in layout_records]
                            else:
                                paras = extract_paragraphs_from_pdf_page(
                                    page, raise_on_text_failure=True,
                                )
                                page_bbox = [
                                    round(float(page.rect.x0), 3), round(float(page.rect.y0), 3),
                                    round(float(page.rect.x1), 3), round(float(page.rect.y1), 3),
                                ]
                                layout_records = [{
                                    "text": text, "bbox": page_bbox,
                                    "block_type": "text_fallback", "reading_order": index,
                                    "column": "single", "source_block_indices": [],
                                    "page_y0": page_bbox[1], "page_y1": page_bbox[3],
                                } for index, text in enumerate(paras)]
                    except Exception as e:
                        print(
                            f"[WARN] Failed to extract page paragraphs: attachment={attachment_key} file={pdf_path} page={pi+1} err={e}",
                            file=os.sys.__stderr__,
                        )
                        paras_by_page.append([])
                        layout_by_page.append([])
                        extraction_failure_pages.append(pi + 1)
                        continue

                    if not paras:
                        paras_by_page.append([])
                        layout_by_page.append([])
                        try:
                            has_images = len(page.get_images()) > 0
                        except Exception:
                            has_images = False
                        if has_images:
                            scanned_pages.append(pi + 1)
                        elif _rendered_page_is_visually_blank(page):
                            empty_pages.append(pi + 1)
                        else:
                            unresolved_nontext_pages.append(pi + 1)
                            scanned_pages.append(pi + 1)
                        continue

                    joined = "\n\n".join(paras)

                    # Analyze text quality per page (score-based)
                    quality = analyze_text_quality(joined)

                    # First try Poppler for font-encoding failures. Its glyph-name
                    # fallback can recover custom Type 1 fonts without OCR.
                    poppler_used = False
                    if (
                        PDF_POPPLER_TEXT_FALLBACK
                        and quality["corruption_type"] == "extraction_failure"
                        and quality["extraction_failure_score"] >= 0.5
                    ):
                        if not poppler_attempted:
                            poppler_attempted = True
                            poppler_pages = _extract_pages_with_poppler(pdf_path)
                        if poppler_pages is not None and pi < len(poppler_pages):
                            poppler_text = poppler_pages[pi].strip()
                            if poppler_text:
                                poppler_quality = analyze_text_quality(poppler_text)
                                if (
                                    poppler_quality["extraction_failure_score"]
                                    < quality["extraction_failure_score"]
                                    and not looks_like_gibberish(poppler_text)
                                ):
                                    joiner = joiner_for_text(poppler_text[:20000])
                                    poppler_paras = [
                                        value for value in normalize_paragraphs(
                                            poppler_text, joiner=joiner,
                                        ) if value.strip()
                                    ]
                                    if poppler_paras:
                                        paras = poppler_paras
                                        page_bbox = [
                                            round(float(page.rect.x0), 3),
                                            round(float(page.rect.y0), 3),
                                            round(float(page.rect.x1), 3),
                                            round(float(page.rect.y1), 3),
                                        ]
                                        layout_records = [{
                                            "text": text, "bbox": page_bbox,
                                            "block_type": "poppler_text",
                                            "reading_order": index,
                                            "column": "unknown",
                                            "source_block_indices": [],
                                            "page_y0": page_bbox[1], "page_y1": page_bbox[3],
                                        } for index, text in enumerate(paras)]
                                        joined = "\n\n".join(paras)
                                        quality = analyze_text_quality(joined)
                                        poppler_used = True
                                        poppler_fallback_pages.append(pi + 1)

                    # Tesseract OCR fallback for font-encoding mismatch.
                    # When PyMuPDF can't decode custom fonts, we render the
                    # page to an image and OCR it — much lighter than Docling.
                    ocr_used = False
                    if (
                        PDF_OCR_FALLBACK
                        and quality["corruption_type"] == "extraction_failure"
                        and quality["extraction_failure_score"] >= 0.5
                    ):
                        if not ocr_notified:
                            ocr_notified = True
                            print(
                                f"[PROGRESS]   OCR fallback activated for "
                                f"attachment={attachment_key} "
                                f"(font-encoding mismatch detected). "
                                f"This may take ~1 sec per affected page...",
                                file=os.sys.__stderr__,
                            )
                        ocr_paras = _ocr_page_with_tesseract(page)
                        if ocr_paras:
                            paras = ocr_paras
                            page_bbox = [
                                round(float(page.rect.x0), 3), round(float(page.rect.y0), 3),
                                round(float(page.rect.x1), 3), round(float(page.rect.y1), 3),
                            ]
                            layout_records = [{
                                "text": text, "bbox": page_bbox, "block_type": "ocr_text",
                                "reading_order": index, "column": "unknown",
                                "source_block_indices": [],
                                "page_y0": page_bbox[1], "page_y1": page_bbox[3],
                            } for index, text in enumerate(paras)]
                            joined = "\n\n".join(paras)
                            quality = analyze_text_quality(joined)
                            ocr_used = True

                    page_scores[pi + 1] = {
                        "scan_score": quality["scan_score"],
                        "extraction_failure_score": quality["extraction_failure_score"],
                        "content_corruption_score": quality["content_corruption_score"],
                        "corruption_score": quality["corruption_score"],
                    }
                    if ocr_used:
                        page_scores[pi + 1]["ocr_fallback"] = True
                        ocr_fallback_pages.append(pi + 1)
                        # Progress every 10 pages (after append so count is current)
                        if len(ocr_fallback_pages) % 10 == 0:
                            print(
                                f"[PROGRESS]   ↳ OCR fallback: {len(ocr_fallback_pages)} pages done "
                                f"(page {pi+1}/{total_pages})",
                                file=os.sys.__stderr__,
                            )
                    if poppler_used:
                        page_scores[pi + 1]["poppler_fallback"] = True

                    if quality["is_scanned"]:
                        try:
                            has_images = len(page.get_images()) > 0
                        except Exception:
                            has_images = False
                        # A readable title on an image-bearing cover is not an
                        # image-only scan and must not reject the whole PDF.
                        readable_cover = pi < 2 and len(joined.strip()) >= 40
                        if has_images and not readable_cover:
                            scanned_pages.append(pi + 1)
                        else:
                            low_text_pages.append(pi + 1)
                    elif quality["is_corrupted"]:
                        corrupted_pages.append(pi + 1)
                        if quality["corruption_type"] == "extraction_failure":
                            extraction_failure_pages.append(pi + 1)
                        elif quality["corruption_type"] == "content_corruption":
                            content_corruption_pages.append(pi + 1)

                    if looks_like_gibberish(joined):
                        # U3 (note 79): retain rather than discard. The text is
                        # unusable as prose, but dropping it silently loses the
                        # only record that this page had *something* on it, and
                        # makes "why is page 88 missing?" unanswerable. Marking
                        # it zone="corrupted" keeps it out of ordinary retrieval
                        # and out of summary input (see document_structure's
                        # ZONE_POLICIES) while leaving it reachable through
                        # rag_search(include_corrupted=True).
                        gibberish_pages.append(pi + 1)
                        paras_by_page.append([])
                        layout_by_page.append([
                            {
                                "text": joined,
                                "zone": "corrupted",
                                "block_type": "corrupted_text",
                                "reading_order": 0,
                                "page": pi + 1,
                            }
                        ] if PDF_RETAIN_CORRUPTED_TEXT else [])
                        continue

                    paras_by_page.append(paras)
                    layout_by_page.append(layout_records)

                # Compute aggregate scores
                scanned_ratio = len(scanned_pages) / max(1, total_pages)
                corrupted_ratio = len(corrupted_pages) / max(1, total_pages)
                extraction_failure_ratio = len(extraction_failure_pages) / max(1, total_pages)

                # Aggregate per-page scores
                avg_corruption_score = 0.0
                max_corruption_score = 0.0
                avg_extraction_failure_score = 0.0
                if page_scores:
                    avg_corruption_score = sum(s["corruption_score"] for s in page_scores.values()) / len(page_scores)
                    max_corruption_score = max(s["corruption_score"] for s in page_scores.values())
                    avg_extraction_failure_score = sum(s["extraction_failure_score"] for s in page_scores.values()) / len(page_scores)

                if ocr_fallback_pages:
                    print(
                        f"[PROGRESS]   OCR fallback summary: "
                        f"{len(ocr_fallback_pages)}/{total_pages} pages processed with OCR",
                        file=os.sys.__stderr__,
                    )

                quality_info = {
                    "is_scanned": scanned_ratio >= SCAN_THRESHOLD,
                    "is_corrupted": corrupted_ratio >= CORRUPTION_THRESHOLD,
                    "scanned_pages": scanned_pages,
                    "corrupted_pages": corrupted_pages,
                    "extraction_failure_pages": extraction_failure_pages,
                    "content_corruption_pages": content_corruption_pages,
                    "poppler_fallback_pages": poppler_fallback_pages,
                    "ocr_fallback_pages": ocr_fallback_pages,
                    "low_text_pages": low_text_pages,
                    "empty_pages": empty_pages,
                    "unresolved_nontext_pages": unresolved_nontext_pages,
                    "gibberish_pages": gibberish_pages,
                    "total_pages": total_pages,
                    "scanned_ratio": round(scanned_ratio, 3),
                    "corrupted_ratio": round(corrupted_ratio, 3),
                    "extraction_failure_ratio": round(extraction_failure_ratio, 3),
                    "ocr_fallback_ratio": round(len(ocr_fallback_pages) / max(1, total_pages), 3),
                    "poppler_fallback_ratio": round(
                        len(poppler_fallback_pages) / max(1, total_pages), 3
                    ),
                    "avg_corruption_score": round(avg_corruption_score, 3),
                    "max_corruption_score": round(max_corruption_score, 3),
                    "avg_extraction_failure_score": round(avg_extraction_failure_score, 3),
                    "page_scores": page_scores,
                    "has_outline": bool(_toc),
                    # A repeat filter must never erase an entire page.  These
                    # pages retain their source blocks and remain auditable.
                    "repeated_header_dropped_pages": [],
                    "repeated_margin_lines_removed_pages": [],
                    "repeated_header_restored_pages": [],
                }
                # This dict replaces the safe-default one wholesale, so the
                # source-class fields have to be re-applied here too.
                if _source_class is not None:
                    quality_info.update(_source_class.as_metadata())

                repeated_lines: set[str] = set()
                if PDF_DROP_REPEATED_LINES:
                    repeated_lines = detect_repeated_lines(layout_by_page)
                    if repeated_lines and os.environ.get("DEBUG_PDF_REPEAT") == "1":
                        ex = list(sorted(repeated_lines))[:10]
                        print(
                            f"[DEBUG] repeated header/footer lines detected: attachment={attachment_key} file={pdf_path} n={len(repeated_lines)} ex={ex}",
                            file=os.sys.__stderr__,
                        )

                repeated_prefixes: set[str] = set()
                if PDF_STRIP_REPEATED_PREFIX:
                    repeated_prefixes = detect_repeated_prefixes(layout_by_page)
                    if repeated_prefixes and os.environ.get("DEBUG_PDF_REPEAT") == "1":
                        ex = list(sorted(repeated_prefixes))[:10]
                        print(
                            f"[DEBUG] repeated header/footer prefixes detected: attachment={attachment_key} file={pdf_path} n={len(repeated_prefixes)} ex={ex}",
                            file=os.sys.__stderr__,
                        )

                for pi, layout_records in enumerate(layout_by_page):
                    if not layout_records:
                        continue

                    original_layout_records = layout_records
                    if repeated_lines:
                        before_repeated_line_count = len(layout_records)
                        layout_records = [
                            record for record in layout_records
                            if not (
                                _repeated_margin_key(record) in repeated_lines
                                and _record_margin_position(record) is not None
                            )
                        ]
                        if not layout_records:
                            # A page consisting only of a proven running line
                            # is unusual but still source text.  Preserve it
                            # rather than silently converting it into a gap.
                            quality_info["repeated_header_restored_pages"].append(pi + 1)
                            layout_records = original_layout_records
                        elif len(layout_records) < before_repeated_line_count:
                            # ``repeated_header_dropped_pages`` is part of the
                            # source-coverage failure contract and means the
                            # page itself was lost. Here only a proven running
                            # margin line was removed; record that separately.
                            quality_info["repeated_margin_lines_removed_pages"].append(pi + 1)

                    if repeated_prefixes:
                        stripped_records = _strip_repeated_prefix_from_first_record(
                            layout_records, repeated_prefixes,
                        )
                        if stripped_records:
                            layout_records = stripped_records
                        if not layout_records:
                            quality_info["repeated_header_restored_pages"].append(pi + 1)
                            layout_records = original_layout_records

                    record_structure_paths: List[List[str]] = []
                    page_events = _outline_events.get(pi + 1) or []
                    if page_events and _structure_path_lookup is not None:
                        previous_path = (
                            _structure_path_lookup(pi) if pi > 0 else []
                        )
                        record_structure_paths = _resolve_record_structure_paths(
                            layout_records,
                            previous_path=previous_path,
                            page_events=page_events,
                        )

                    paras = [record["text"] for record in layout_records]
                    joined = "\n\n".join(paras)
                    is_cjk = is_no_space_language_document(joined)
                    local_min_chunk = MIN_CHUNK_CHARS_NO_SPACE if is_cjk else MIN_CHUNK_CHARS
                    local_max_chars = MAX_CHARS_CJK if is_cjk else MAX_CHARS
                    local_target_chars = TARGET_CHARS_CJK if is_cjk else TARGET_CHARS

                    page_chunks: List[Tuple[str, str, Dict[str, Any]]] = []
                    for para_index, record in enumerate(layout_records):
                        para_text = str(record["text"]).strip()
                        if not para_text:
                            continue

                        parts = split_long_paragraph(para_text, max_chars=local_max_chars, target_chars=local_target_chars)
                        for part_index, part in enumerate(parts):
                            part = part.strip()
                            if not part:
                                continue

                            chunk_id = f"{attachment_key}:p{pi+1}:para{para_index}:part{part_index}"
                            md = dict(meta_base)
                            chapter_info: Dict[str, Any] = {}
                            resolved_path = (
                                record_structure_paths[para_index]
                                if para_index < len(record_structure_paths)
                                else None
                            )
                            if resolved_path:
                                chapter_info["structure_path"] = resolved_path
                                if resolved_path:
                                    chapter_info["chapter"] = resolved_path[0]
                                if len(resolved_path) > 1:
                                    chapter_info["section"] = resolved_path[1]
                            elif _chapter_lookup is not None:
                                try:
                                    _ch, _sec = _chapter_lookup(pi + 1)
                                    if _ch:
                                        chapter_info["chapter"] = _ch
                                    if _sec:
                                        chapter_info["section"] = _sec
                                except Exception:
                                    pass
                            if resolved_path is None and _structure_path_lookup is not None:
                                try:
                                    structure_path = _structure_path_lookup(pi + 1)
                                    if structure_path:
                                        chapter_info["structure_path"] = structure_path
                                except Exception:
                                    pass
                            # This was the one PDF path that assigned no zone at
                            # all beyond "corrupted": paratext classification
                            # existed only for AI-TOC and Docling documents,
                            # which most PDFs never route through. Measured
                            # impact of that gap: ~10.6 million characters
                            # across 217 items sitting under headings like
                            # "Sources" or "Bibliography" while indexed as
                            # zone=body (2026-07-28). The heading path already
                            # computed above for chapter/section metadata is
                            # exactly the input classify_heading_path needs;
                            # corrupted text still wins, since reliability is a
                            # different axis from structure.
                            heading_path = (
                                chapter_info.get("structure_path")
                                or [value for value in (chapter_info.get("chapter"), chapter_info.get("section")) if value]
                            )
                            heading_zone = classify_heading_path(heading_path) if heading_path else "body"
                            if heading_path:
                                chapter_info["structure_roles"] = infer_structure_roles(heading_path)
                            md.update(
                                {
                                    "source_type": "pdf",
                                    "locator": f"p{pi+1}:para{para_index}",
                                    "page": int(pi + 1),
                                    "page_label": page_labels.get(pi, ""),
                                    "pdf_path": str(pdf_path),
                                    "path": str(pdf_path),
                                    "para_index": int(para_index),
                                    "part_index": int(part_index),
                                    "bbox": record.get("bbox"),
                                    "block_type": record.get("block_type") or "text",
                                    # Propagated so build_document_structure can
                                    # put retained corrupted text (U3) in its own
                                    # zone=corrupted leaf, which ZONE_POLICIES
                                    # then excludes from retrieval and summaries.
                                    **({"zone": record["zone"]} if record.get("zone")
                                       else ({"zone": heading_zone} if heading_zone != "body" else {})),
                                    "reading_order": int(record.get("reading_order", para_index)),
                                    "column": record.get("column") or "unknown",
                                    "source_block_indices": list(record.get("source_block_indices") or []),
                                    **chapter_info,
                                }
                            )
                            page_chunks.append((chunk_id, part, md))

                    before_merge = list(page_chunks)
                    local_hard_min = HARD_MIN_CHARS_CJK if is_cjk else HARD_MIN_CHARS
                    page_chunks = merge_short_chunk_records(
                        page_chunks, min_chars=local_min_chunk, max_chars=local_max_chars,
                        boundary_key=lambda _cid, _text, metadata: (
                            metadata.get("page"), metadata.get("reading_order"),
                        ),
                        hard_min_chars=local_hard_min,
                    )
                    # A page that had text and now has none was not filtered,
                    # it was erased. The boundary above is unique per record, so
                    # nothing on a page can merge; when _merge_layout_blocks
                    # falls back to one line per block (~34 characters) every
                    # line sits under HARD_MIN_CHARS and the merge drops them
                    # all. One document lost 13 pages of body text this way
                    # (4CY8EIIB, 2026-07-28). Retry those pages as a single
                    # boundary so the lines can combine into real chunks --
                    # only for the degenerate case, leaving ordinary pages, and
                    # therefore existing chunk granularity, untouched.
                    if before_merge and not page_chunks:
                        page_chunks = merge_short_chunk_records(
                            before_merge, min_chars=local_min_chunk, max_chars=local_max_chars,
                            boundary_key=lambda _cid, _text, metadata: (
                                metadata.get("page"), metadata.get("chapter"),
                            ),
                            hard_min_chars=local_hard_min,
                        )
                    if before_merge and not page_chunks:
                        # A short title, dedication, running line, or other
                        # standalone source fragment is still source content.
                        # Search can suppress it through MIN_RETURN_CHARS, but
                        # canonical ingestion must retain it so page coverage
                        # never depends on a length threshold.
                        page_chunks = before_merge
                    chunks.extend(page_chunks)

            finally:
                try:
                    doc.close()
                except Exception:
                    pass

        captured_text = _read_fd_text(r_fd)
        r_fd = None

    except Exception as e:
        try:
            if r_fd is not None:
                captured_text = _read_fd_text(r_fd)
        except Exception:
            pass
        print(
            f"[WARN] Failed to open/extract PDF: attachment={attachment_key} file={pdf_path} err={e}",
            file=os.sys.__stderr__,
        )
        return [], quality_info

    finally:
        if captured_text and "MuPDF error" in captured_text:
            # Deduplicate: group identical messages, show each once with count
            from collections import Counter
            error_lines = [line.strip() for line in captured_text.splitlines()
                          if "MuPDF error" in line]
            counts = Counter(error_lines)
            for msg, count in counts.most_common():
                suffix = f" (x{count})" if count > 1 else ""
                print(
                    f"[WARN] PyMuPDF: {msg}{suffix} "
                    f"[attachment={attachment_key}]",
                    file=os.sys.__stderr__,
                )

    ids = [cid for (cid, _, _) in chunks]
    if len(ids) != len(set(ids)):
        dup = len(ids) - len(set(ids))
        raise RuntimeError(f"Duplicate chunk ids generated ({dup}). This should not happen.")

    # Deterministic extraction-defect detection on the text we are actually
    # about to index.  Measured on the real chunk bodies rather than on raw
    # page text, because clean_extracted_text's NFKC pass has already resolved
    # preserved ligature codepoints by this point -- what survives here is what
    # would reach the index.
    quality_info.update(detect_text_defects("\n".join(text for (_, text, _) in chunks)))

    return chunks, quality_info
