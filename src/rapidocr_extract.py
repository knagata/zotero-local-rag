"""Lightweight English/other-language OCR adapter using RapidOCR directly."""
from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import median
from typing import Any

import fitz

try:
    from .chapter_detect import (
        build_pdf_page_chapter_lookup, build_pdf_page_structure_path_lookup,
        get_pdf_toc,
        infer_structure_roles,
    )
    from .text_utils import (
        MAX_CHARS, MAX_CHARS_CJK, TARGET_CHARS, TARGET_CHARS_CJK,
        HARD_MIN_CHARS, HARD_MIN_CHARS_CJK,
        is_no_space_language_document, merge_short_chunk_records,
        split_long_paragraph,
    )
except ImportError:  # direct src entrypoint
    from chapter_detect import (
        build_pdf_page_chapter_lookup, build_pdf_page_structure_path_lookup,
        get_pdf_toc,
        infer_structure_roles,
    )
    from text_utils import (
        MAX_CHARS, MAX_CHARS_CJK, TARGET_CHARS, TARGET_CHARS_CJK,
        HARD_MIN_CHARS, HARD_MIN_CHARS_CJK,
        is_no_space_language_document, merge_short_chunk_records,
        split_long_paragraph,
    )


def _bbox(box: Any) -> dict[str, float] | None:
    try:
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
    except (TypeError, ValueError, IndexError):
        return None
    return {"l": min(xs), "t": min(ys), "r": max(xs), "b": max(ys)}


def _output_rows(output: Any, *, minimum_confidence: float) -> list[dict[str, Any]]:
    # RapidOCR returns NumPy arrays for boxes/scores.  Do not use ``or []``
    # here: NumPy deliberately rejects boolean coercion for multi-value arrays.
    texts_value = getattr(output, "txts", None)
    boxes_value = getattr(output, "boxes", None)
    scores_value = getattr(output, "scores", None)
    texts = list(texts_value) if texts_value is not None else []
    boxes = list(boxes_value) if boxes_value is not None else []
    scores = list(scores_value) if scores_value is not None else []
    rows: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        value = str(text or "").strip()
        try:
            confidence = float(scores[index])
        except (IndexError, TypeError, ValueError):
            confidence = 0.0
        if not value or confidence < minimum_confidence:
            continue
        box = _bbox(boxes[index] if index < len(boxes) else None)
        rows.append({"text": value, "confidence": confidence, "bbox": box})
    rows.sort(key=lambda row: (
        row["bbox"]["t"] if row["bbox"] else float("inf"),
        row["bbox"]["l"] if row["bbox"] else float("inf"),
    ))
    return rows


def _paragraphs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    heights = [
        row["bbox"]["b"] - row["bbox"]["t"]
        for row in rows if row.get("bbox")
    ]
    typical_height = median(heights) if heights else 12.0
    groups: list[list[dict[str, Any]]] = []
    pending: list[dict[str, Any]] = []
    for row in rows:
        join = False
        if pending and row.get("bbox") and pending[-1].get("bbox"):
            previous = pending[-1]["bbox"]
            current = row["bbox"]
            vertical_gap = current["t"] - previous["b"]
            left_delta = abs(current["l"] - previous["l"])
            join = vertical_gap <= typical_height * 1.5 and left_delta <= typical_height * 4
        if pending and not join:
            groups.append(pending)
            pending = []
        pending.append(row)
    if pending:
        groups.append(pending)
    paragraphs: list[dict[str, Any]] = []
    for group in groups:
        boxes = [row["bbox"] for row in group if row.get("bbox")]
        paragraphs.append({
            "text": " ".join(row["text"] for row in group),
            "confidence": sum(row["confidence"] for row in group) / len(group),
            "bbox": ({
                "l": min(box["l"] for box in boxes), "t": min(box["t"] for box in boxes),
                "r": max(box["r"] for box in boxes), "b": max(box["b"] for box in boxes),
            } if boxes else None),
        })
    return paragraphs


def _merge_page_chunks(
    chunks: list[tuple[str, str, dict[str, Any]]],
) -> list[tuple[str, str, dict[str, Any]]]:
    """Coalesce RapidOCR visual-line records into retrieval-sized page chunks.

    RapidOCR commonly detects a separate record for every visual line.  Its
    simple paragraph grouping above is deliberately conservative, and on a
    multi-column page the top-to-bottom ordering interleaves columns so that
    grouping cannot join those lines.  Indexing those records directly creates
    thousands of 1--30 character vectors.  Merge within one page only: page is
    the stable source boundary available to this lightweight OCR adapter.
    """
    if not chunks:
        return []
    sample = "\n".join(text for _chunk_id, text, _metadata in chunks)
    no_space = is_no_space_language_document(sample)
    target = TARGET_CHARS_CJK if no_space else TARGET_CHARS
    maximum = MAX_CHARS_CJK if no_space else MAX_CHARS
    return merge_short_chunk_records(
        chunks,
        # ``merge_short_chunk_records`` emits once its buffer reaches
        # ``min_chars``.  Here the desired retrieval unit, rather than merely
        # a noise floor, is the correct threshold.
        min_chars=target,
        max_chars=maximum,
        boundary_key=lambda _chunk_id, _text, metadata: metadata.get("page"),
        hard_min_chars=HARD_MIN_CHARS_CJK if no_space else HARD_MIN_CHARS,
    )


def extract_chunks_from_pdf_with_rapidocr(
    pdf_path: Path,
    attachment_key: str,
    meta_base: dict[str, Any],
) -> tuple[list[tuple[str, str, dict[str, Any]]], dict[str, Any]]:
    from rapidocr import RapidOCR

    dpi = max(72, int(os.environ.get("RAPIDOCR_DPI", "200")))
    minimum_confidence = min(1.0, max(0.0, float(
        os.environ.get("RAPIDOCR_MIN_CONFIDENCE", "0.45")
    )))
    engine = RapidOCR()
    toc = get_pdf_toc(str(pdf_path))
    toc_lookup = build_pdf_page_chapter_lookup(toc)
    path_lookup = build_pdf_page_structure_path_lookup(toc)
    chunks: list[tuple[str, str, dict[str, Any]]] = []
    confidences: list[float] = []
    attempted_pages: list[int] = []
    pages_with_text: list[int] = []
    with fitz.open(str(pdf_path)) as document:
        total_pages = document.page_count
        labels = {index + 1: str(page.get_label() or "") for index, page in enumerate(document)}
        for page_index, page in enumerate(document):
            page_no = page_index + 1
            attempted_pages.append(page_no)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
            output = engine(pixmap.tobytes("png"))
            rows = _output_rows(output, minimum_confidence=minimum_confidence)
            paragraphs = _paragraphs(rows)
            if paragraphs:
                pages_with_text.append(page_no)
            chapter_info: dict[str, str] = {}
            if toc_lookup is not None:
                chapter, section = toc_lookup(page_no)
                if chapter:
                    chapter_info["chapter"] = chapter
                if section:
                    chapter_info["section"] = section
            structure_path = path_lookup(page_no) if path_lookup is not None else []
            page_chunks: list[tuple[str, str, dict[str, Any]]] = []
            for paragraph_index, paragraph in enumerate(paragraphs):
                for part_index, part in enumerate(
                    split_long_paragraph(
                        paragraph["text"], max_chars=MAX_CHARS,
                        target_chars=TARGET_CHARS,
                    ) or [paragraph["text"]]
                ):
                    text = part.strip()
                    if not text:
                        continue
                    chunk_id = (
                        f"{attachment_key}:p{page_no}:para{paragraph_index}:part{part_index}"
                    )
                    metadata = dict(meta_base)
                    metadata.update({
                        "source_type": "pdf", "page": page_no,
                        "page_label": labels.get(page_no, ""),
                        "locator": f"p{page_no}:para{paragraph_index}",
                        "path": str(pdf_path), "pdf_path": str(pdf_path),
                        "para_index": paragraph_index, "part_index": part_index,
                        "block_type": "text", "zone": "body",
                        "reading_order": paragraph_index,
                        "ocr_confidence": float(paragraph["confidence"]),
                        **chapter_info,
                    })
                    if structure_path:
                        metadata["structure_path"] = list(structure_path)
                        metadata["structure_roles"] = infer_structure_roles(structure_path)
                    if paragraph.get("bbox"):
                        metadata["bbox"] = json.dumps(
                            paragraph["bbox"], separators=(",", ":"),
                        )
                        for axis, value in paragraph["bbox"].items():
                            metadata[f"bbox_{axis}"] = float(value)
                    page_chunks.append((chunk_id, text, metadata))
                    confidences.append(float(paragraph["confidence"]))
            chunks.extend(_merge_page_chunks(page_chunks))
    return chunks, {
        "parser": "rapidocr", "is_scanned": True, "is_corrupted": False,
        "total_pages": total_pages, "ocr_pages": attempted_pages,
        "pages_with_text": pages_with_text,
        # An attempted raster page with no accepted OCR text is unresolved,
        # not a confirmed blank page. Reporting it explicitly makes the local
        # gate reject partial fixed-layout EPUB output and try the layout-aware
        # fallback instead of postponing the failure until final adoption.
        "missing_pages": sorted(
            set(range(1, total_pages + 1)) - set(pages_with_text)
        ),
        "scanned_pages": list(range(1, total_pages + 1)),
        "corrupted_pages": [],
        "mean_confidence": (
            sum(confidences) / len(confidences) if confidences else 0.0
        ),
        "dpi": dpi, "minimum_confidence": minimum_confidence,
    }


__all__ = ["extract_chunks_from_pdf_with_rapidocr"]
