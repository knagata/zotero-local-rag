from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fitz

try:
    from .chapter_detect import (
        build_pdf_page_chapter_lookup, build_pdf_page_structure_path_lookup, get_pdf_toc,
    )
    from .text_utils import (
        MAX_CHARS, MAX_CHARS_CJK, TARGET_CHARS, TARGET_CHARS_CJK,
        is_no_space_language_document, split_long_paragraph,
    )
except ImportError:  # direct `python src/index_from_zotero.py` execution
    from chapter_detect import (
        build_pdf_page_chapter_lookup, build_pdf_page_structure_path_lookup, get_pdf_toc,
    )
    from text_utils import (
        MAX_CHARS, MAX_CHARS_CJK, TARGET_CHARS, TARGET_CHARS_CJK,
        is_no_space_language_document, split_long_paragraph,
    )


_HEADING_ROLE = "section_headings"
_PAGE_FURNITURE_ROLES = {"page_header", "page_footer"}
_CAPTION_ROLE = "caption"

_FOOTNOTE_LABEL_RE = re.compile(r"footnote|脚注|註", re.I)
_ENDNOTE_LABEL_RE = re.compile(r"endnote|巻末注|後注", re.I)
_BIBLIOGRAPHY_LABEL_RE = re.compile(r"bibliography|references?|参考文献|引用文献|文献一覧", re.I)
_TOC_LABEL_RE = re.compile(r"table\s+of\s+contents|contents?|目次|索引|index", re.I)


def _zone_from_heading_text(text: str) -> str | None:
    """Infer a zone from heading vocabulary; None means no explicit signal."""
    if _BIBLIOGRAPHY_LABEL_RE.search(text):
        return "bibliography"
    if _ENDNOTE_LABEL_RE.search(text):
        return "endnote"
    if _FOOTNOTE_LABEL_RE.search(text):
        return "footnote"
    if _TOC_LABEL_RE.search(text):
        return "toc"
    return None


def _configure_lite(configs: Dict[str, Any], *, device: str) -> None:
    """Mirror the CLI's ``--lite`` preset so CPU runs stay fast (see yomitoku.cli.main)."""
    configs["ocr"]["text_recognizer"]["model_name"] = "parseq-tiny-dynw-v4"
    if device == "cpu":
        configs["ocr"]["text_detector"]["infer_onnx"] = True
        configs["ocr"]["text_recognizer"]["num_parallel_batches"] = 4
    configs["ocr"]["text_recognizer"]["dynamic_width"] = True
    configs["ocr"]["text_recognizer"]["batch_bucketing"] = True
    configs["ocr"]["text_recognizer"]["source_downscale"] = True


def _build_analyzer(*, device: str, lite: bool):
    from yomitoku import DocumentAnalyzer

    configs: Dict[str, Any] = {"ocr": {"text_detector": {}, "text_recognizer": {}}}
    if lite:
        _configure_lite(configs, device=device)
    return DocumentAnalyzer(configs=configs, device=device, visualize=False, reading_order="auto")


def _box_to_bbox(box: Any) -> Dict[str, float] | None:
    try:
        x1, y1, x2, y2 = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    return {"l": min(x1, x2), "t": min(y1, y2), "r": max(x1, x2), "b": max(y1, y2)}


def _word_center_in_box(points: Any, box: Dict[str, float]) -> bool:
    try:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
    except (TypeError, ValueError, IndexError):
        return False
    if not xs or not ys:
        return False
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    return box["l"] <= cx <= box["r"] and box["t"] <= cy <= box["b"]


def _confidence_for_box(box: Dict[str, float] | None, words: List[Dict[str, Any]]) -> float:
    if not box or not words:
        return 0.0
    scores = [
        float(word.get("rec_score") or 0.0)
        for word in words if _word_center_in_box(word.get("points"), box)
    ]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def _table_text(table: Dict[str, Any]) -> str:
    cells = sorted(
        (cell for cell in table.get("cells", []) if isinstance(cell, dict)),
        key=lambda cell: (int(cell.get("row") or 0), int(cell.get("col") or 0)),
    )
    rows: Dict[int, List[str]] = {}
    for cell in cells:
        rows.setdefault(int(cell.get("row") or 0), []).append(str(cell.get("contents") or "").strip())
    return "\n".join("\t".join(values) for _, values in sorted(rows.items()))


def _page_blocks(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize one Yomitoku page result into ordered, zone-tagged blocks.

    Reading order and paragraph role come directly from Yomitoku's own layout
    and reading-order estimation; this function only maps that vocabulary onto
    the shared zone/block_type contract and keeps the source's own order.
    """
    words = [word for word in payload.get("words", []) if isinstance(word, dict)]
    items: List[Dict[str, Any]] = []
    for paragraph in payload.get("paragraphs", []):
        if not isinstance(paragraph, dict):
            continue
        text = str(paragraph.get("contents") or "").strip()
        if not text:
            continue
        role = str(paragraph.get("role") or "")
        box = _box_to_bbox(paragraph.get("box"))
        if role in _PAGE_FURNITURE_ROLES:
            block_type, zone = "page_furniture", "other_paratext"
        elif role == _CAPTION_ROLE:
            block_type, zone = "caption", "body"
        elif role == _HEADING_ROLE:
            block_type, zone = "heading", "body"
        else:
            block_type, zone = "text", "body"
        items.append({
            "order": paragraph.get("order"), "text": text, "role": role,
            "block_type": block_type, "zone": zone, "bbox": box,
            "direction": str(paragraph.get("direction") or "horizontal"),
            "confidence": _confidence_for_box(box, words),
        })
    for table in payload.get("tables", []):
        if not isinstance(table, dict):
            continue
        text = _table_text(table)
        if not text:
            continue
        box = _box_to_bbox(table.get("box"))
        items.append({
            "order": table.get("order"), "text": text, "role": "table",
            "block_type": "table", "zone": "body", "bbox": box,
            "direction": "horizontal", "confidence": _confidence_for_box(box, words),
        })
    items.sort(key=lambda item: (item["order"] if isinstance(item["order"], int) else 10**9))

    # Propagate a heading's zone vocabulary (bibliography/footnote/toc) to the
    # ordinary body paragraphs that follow it, until the next heading. Roles
    # alone do not distinguish these zones (see `DocumentAnalyzerSchema`), so
    # this is the only source-level signal available for them.
    active_zone = "body"
    for item in items:
        if item["block_type"] == "heading":
            active_zone = _zone_from_heading_text(item["text"]) or "body"
            continue
        if item["zone"] == "body":
            item["zone"] = active_zone
    return items


def extract_chunks_from_pdf_with_yomitoku(
    pdf_path: Path,
    attachment_key: str,
    meta_base: Dict[str, Any],
) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], Dict[str, Any]]:
    """OCR a PDF with Yomitoku and emit the standard chunk contract.

    Defaults to the lite/CPU preset validated in the 2026-07-17 bake-off
    (``dev-notes/archive/28_ocr_bakeoff.md``); override with
    ``YOMITOKU_DEVICE``/``YOMITOKU_LITE`` if a faster local accelerator is
    confirmed safe for this machine.
    """
    from yomitoku.data.functions import load_pdf

    device = os.environ.get("YOMITOKU_DEVICE", "cpu").strip() or "cpu"
    lite = os.environ.get("YOMITOKU_LITE", "1").strip() != "0"
    dpi = max(72, int(os.environ.get("YOMITOKU_DPI", "200")))
    analyzer = _build_analyzer(device=device, lite=lite)
    toc = get_pdf_toc(str(pdf_path))
    toc_lookup = build_pdf_page_chapter_lookup(toc)
    structure_path_lookup = build_pdf_page_structure_path_lookup(toc)

    doc = fitz.open(str(pdf_path))
    try:
        page_labels = {index + 1: str(page.get_label() or "") for index, page in enumerate(doc)}
        page_count = doc.page_count
    finally:
        doc.close()

    print(f"[INFO] Yomitoku: {page_count} pages at {dpi} DPI, device={device}, lite={lite} ({pdf_path.name})", file=sys.stderr)
    images = load_pdf(str(pdf_path), dpi=dpi)

    chunks: List[Tuple[str, str, Dict[str, Any]]] = []
    page_confidences: Dict[str, float] = {}
    total_blocks = 0
    heading_blocks = 0
    inferred_path: List[str] = []

    for page_index, image in enumerate(images):
        page_no = page_index + 1
        result, _ocr, _layout = analyzer(image)
        payload = result.model_dump()
        blocks = _page_blocks(payload)
        confidences = [block["confidence"] for block in blocks if block["confidence"]]
        if confidences:
            page_confidences[str(page_no)] = round(sum(confidences) / len(confidences), 4)
        if not blocks:
            continue

        chapter_info: Dict[str, str] = {}
        if toc_lookup is not None:
            chapter, section = toc_lookup(page_no)
            if chapter:
                chapter_info["chapter"] = chapter
            if section:
                chapter_info["section"] = section
        outline_path = structure_path_lookup(page_no) if structure_path_lookup is not None else []
        active_path = list(outline_path or inferred_path)
        is_cjk = is_no_space_language_document("\n".join(block["text"] for block in blocks)[:5000])
        max_chars = MAX_CHARS_CJK if is_cjk else MAX_CHARS
        target_chars = TARGET_CHARS_CJK if is_cjk else TARGET_CHARS

        for block_index, block in enumerate(blocks):
            block_text = block["text"]
            if block["block_type"] == "heading":
                normalized_heading = " ".join(block_text.split())
                if not active_path or active_path[-1] != normalized_heading:
                    active_path = list(outline_path) + [normalized_heading]
                heading_blocks += 1
            parts = split_long_paragraph(block_text, max_chars=max_chars, target_chars=target_chars) or [block_text]
            for part_index, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue
                chunk_id = f"{attachment_key}:p{page_no}:para{block_index}:part{part_index}"
                metadata = dict(meta_base)
                metadata.update({
                    "source_type": "pdf", "locator": f"p{page_no}:para{block_index}",
                    "page": page_no, "page_label": page_labels.get(page_no, ""),
                    "pdf_path": str(pdf_path), "path": str(pdf_path),
                    "para_index": block_index, "part_index": part_index,
                    "ocr_confidence": block["confidence"],
                    "block_type": block["block_type"], "zone": block["zone"],
                    "reading_order": block_index,
                    "writing_mode": block["direction"],
                    "yomitoku_role": block["role"],
                    **chapter_info,
                })
                if active_path:
                    metadata["structure_path"] = list(active_path)
                if block["bbox"]:
                    metadata["bbox"] = json.dumps(block["bbox"], ensure_ascii=False, separators=(",", ":"))
                    for axis in ("l", "t", "r", "b"):
                        metadata[f"bbox_{axis}"] = float(block["bbox"][axis])
                chunks.append((chunk_id, part, metadata))
                total_blocks += 1
        if not outline_path and active_path:
            inferred_path = list(active_path)

    return chunks, {
        "is_scanned": False, "is_corrupted": False,
        "scanned_pages": [], "corrupted_pages": [],
        "total_pages": page_count, "parser": "yomitoku",
        "device": device, "lite": lite,
        "ocr_pages": list(range(1, page_count + 1)),
        "page_confidences": page_confidences,
        "blocks": total_blocks, "heading_blocks": heading_blocks,
        "dpi": dpi,
    }


__all__ = ["extract_chunks_from_pdf_with_yomitoku"]
