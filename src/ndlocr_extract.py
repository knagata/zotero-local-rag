from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from statistics import median
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fitz

try:
    from .chapter_detect import (
        build_pdf_page_chapter_lookup, build_pdf_page_structure_path_lookup, get_pdf_toc,
    )
    from .text_utils import (
        MAX_CHARS_CJK, TARGET_CHARS_CJK, split_long_paragraph,
    )
except ImportError:  # direct `python src/index_from_zotero.py` execution
    from chapter_detect import (
        build_pdf_page_chapter_lookup, build_pdf_page_structure_path_lookup, get_pdf_toc,
    )
    from text_utils import (
        MAX_CHARS_CJK, TARGET_CHARS_CJK, split_long_paragraph,
    )


def find_ndlocr() -> str | None:
    configured = os.environ.get("NDLOCR_BIN", "").strip()
    if configured:
        return configured if Path(configured).expanduser().exists() else None
    return shutil.which("ndlocr-lite")


def _box_size(box: Any) -> tuple[float, float]:
    try:
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        return max(xs) - min(xs), max(ys) - min(ys)
    except (TypeError, ValueError, IndexError):
        return 0.0, 0.0


def _bbox_payload(box: Any) -> dict[str, float] | None:
    """Normalize NDL's polygon bbox without discarding the original line."""
    try:
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
    except (TypeError, ValueError, IndexError):
        return None
    if not xs or not ys:
        return None
    return {"l": min(xs), "t": min(ys), "r": max(xs), "b": max(ys)}


def _union_bbox(rows: list[dict[str, Any]]) -> dict[str, float] | None:
    boxes = [row.get("bbox") for row in rows if row.get("bbox")]
    if not boxes:
        return None
    return {
        "l": min(box["l"] for box in boxes), "t": min(box["t"] for box in boxes),
        "r": max(box["r"] for box in boxes), "b": max(box["b"] for box in boxes),
    }


_PATHOLOGICAL_REPEAT_RE = re.compile(r"(.)\1{19,}")


def _usable_ocr_text(value: Any) -> str:
    """Discard layout artifacts while preserving ordinary textual repetition."""
    text = str(value or "").strip()
    if not text or _PATHOLOGICAL_REPEAT_RE.search(text):
        return ""
    return text


def _trim_adjacent_overlap(previous: str, current: str, *, minimum: int = 12) -> str:
    """Remove OCR region overlap without touching short, possibly intentional repeats."""
    if len(current) >= minimum and current in previous:
        return ""
    limit = min(len(previous), len(current))
    for size in range(limit, minimum - 1, -1):
        if previous.endswith(current[:size]):
            return current[size:].lstrip()
    return current


_HEADING_LABEL_RE = re.compile(r"(?:^|[_\s-])(?:title|heading|section[_\s-]?header)(?:$|[_\s-])|見出し|タイトル", re.I)
_FOOTNOTE_LABEL_RE = re.compile(r"footnote|脚注|註", re.I)
_ENDNOTE_LABEL_RE = re.compile(r"endnote|巻末注|後注", re.I)
_BIBLIOGRAPHY_LABEL_RE = re.compile(r"bibliography|references?|参考文献|引用文献|文献一覧", re.I)
_CAPTION_LABEL_RE = re.compile(r"caption|キャプション", re.I)
_TABLE_LABEL_RE = re.compile(r"table|表組", re.I)
_PAGE_FURNITURE_LABEL_RE = re.compile(r"page[_\s-]?(?:header|footer)|柱|ノンブル", re.I)


def _row_label(row: dict[str, Any]) -> str:
    values = [row.get(key) for key in (
        "block_type", "blockType", "type", "label", "category", "class",
    )]
    return " ".join(str(value or "").strip() for value in values if value).casefold()


def _explicit_block_type(label: str) -> tuple[str, str] | None:
    if _HEADING_LABEL_RE.search(label):
        return "heading", "body"
    if _FOOTNOTE_LABEL_RE.search(label):
        return "footnote", "footnote"
    if _ENDNOTE_LABEL_RE.search(label):
        return "endnote", "endnote"
    if _BIBLIOGRAPHY_LABEL_RE.search(label):
        return "reference", "bibliography"
    if _CAPTION_LABEL_RE.search(label):
        return "caption", "body"
    if _TABLE_LABEL_RE.search(label):
        return "table", "body"
    if _PAGE_FURNITURE_LABEL_RE.search(label):
        return "page_furniture", "other_paratext"
    return None


def _font_size(row: dict[str, Any]) -> float | None:
    for key in ("fontSize", "font_size", "font-size", "size"):
        try:
            value = float(row.get(key))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _ordered_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return de-duplicated OCR lines in NDL's evaluated reading order."""
    candidates: list[dict[str, Any]] = []
    sequence = 0
    for group_index, group in enumerate(payload.get("contents", [])):
        if not isinstance(group, list):
            continue
        for line_index, raw in enumerate(group):
            if not isinstance(raw, dict):
                continue
            text = _usable_ocr_text(raw.get("text"))
            if not text:
                continue
            try:
                order = int(raw.get("id"))
            except (TypeError, ValueError):
                order = 10**9 + sequence
            candidates.append({
                "raw": raw, "text": text, "order": order, "sequence": sequence,
                "group_index": group_index, "line_index": line_index,
                "bbox": _bbox_payload(raw.get("boundingBox")),
            })
            sequence += 1
    candidates.sort(key=lambda row: (row["order"], row["sequence"]))

    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    previous = ""
    for row in candidates:
        normalized = " ".join(row["text"].split())
        if normalized in seen:
            continue
        seen.add(normalized)
        text = _trim_adjacent_overlap(previous, row["text"])
        if not text:
            continue
        row["text"] = text
        output.append(row)
        previous = text
    return output


def blocks_from_ndlocr_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize an NDL page into provenance-preserving semantic blocks.

    NDL's evaluated ``id`` remains the ordering authority.  Heading inference
    is intentionally conservative: a short line must either carry an explicit
    heading label or be visibly larger than the page's ordinary text.
    """
    rows = _ordered_rows(payload)
    if not rows:
        return []
    vertical = sum(
        _box_size(row["raw"].get("boundingBox"))[1]
        > _box_size(row["raw"].get("boundingBox"))[0]
        for row in rows
    ) > len(rows) / 2
    writing_mode = "vertical" if vertical else "horizontal"
    separator = "" if vertical else "\n"
    visual_sizes = [
        min(_box_size(row["raw"].get("boundingBox")))
        for row in rows if min(_box_size(row["raw"].get("boundingBox"))) > 0
    ]
    typical_visual_size = median(visual_sizes) if visual_sizes else 0.0
    font_sizes = [size for row in rows if (size := _font_size(row["raw"])) is not None]
    typical_font_size = median(font_sizes) if font_sizes else 0.0
    page_top = min((row["bbox"]["t"] for row in rows if row["bbox"]), default=0.0)
    page_bottom = max((row["bbox"]["b"] for row in rows if row["bbox"]), default=0.0)
    page_height = max(0.0, page_bottom - page_top)

    for row in rows:
        raw = row["raw"]
        label = _row_label(raw)
        explicit = _explicit_block_type(label)
        compact = re.sub(r"\s+", "", row["text"])
        short = 1 <= len(compact) <= 80
        visual_size = min(_box_size(raw.get("boundingBox")))
        font_size = _font_size(raw)
        larger_visual = bool(
            typical_visual_size and visual_size >= typical_visual_size * 1.35
        )
        larger_font = bool(
            typical_font_size and font_size and font_size >= typical_font_size * 1.25
        )
        near_top = bool(
            row["bbox"] and page_height
            and row["bbox"]["t"] <= page_top + page_height * 0.15
        )
        modestly_larger = bool(
            typical_visual_size and visual_size >= typical_visual_size * 1.15
        )
        inferred_heading = short and (
            larger_visual or larger_font or (near_top and modestly_larger)
        )
        if explicit:
            block_type, zone = explicit
        elif inferred_heading:
            block_type, zone = "heading", "body"
        else:
            block_type, zone = "text", "body"
        row.update({
            "block_type": block_type, "zone": zone, "label": label,
            "writing_mode": writing_mode,
        })

    blocks: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        if not pending:
            return
        first, last = pending[0], pending[-1]
        confidences = []
        provenance = []
        for row in pending:
            try:
                confidences.append(float(row["raw"].get("confidence")))
            except (TypeError, ValueError):
                pass
            provenance.append({
                "reading_order": row["order"],
                "source_group_index": row["group_index"],
                "source_line_index": row["line_index"],
                "bbox": row["bbox"],
                "raw_bounding_box": row["raw"].get("boundingBox"),
                "confidence": row["raw"].get("confidence"),
            })
        blocks.append({
            "text": separator.join(row["text"] for row in pending),
            "block_type": first["block_type"], "zone": first["zone"],
            "writing_mode": first["writing_mode"],
            "reading_order": first["order"], "reading_order_end": last["order"],
            "source_group_index": first["group_index"],
            "bbox": _union_bbox(pending), "provenance": provenance,
            "confidence": (sum(confidences) / len(confidences)) if confidences else 0.0,
            "ndlocr_label": first["label"],
        })
        pending.clear()

    for row in rows:
        # Ordinary lines from one NDL region form one text block.  Semantic
        # rows stay independent even when short.
        can_join = bool(
            pending
            and row["block_type"] == "text"
            and pending[-1]["block_type"] == "text"
            and row["zone"] == pending[-1]["zone"]
            and row["group_index"] == pending[-1]["group_index"]
        )
        if pending and not can_join:
            flush()
        pending.append(row)
        if row["block_type"] != "text":
            flush()
    flush()
    return blocks


def lines_from_ndlocr_payload(payload: dict[str, Any]) -> tuple[list[str], list[float]]:
    """Backward-compatible page text view over provenance-preserving blocks."""
    blocks = blocks_from_ndlocr_payload(payload)
    if not blocks:
        return [], []
    separator = "" if blocks[0]["writing_mode"] == "vertical" else "\n"
    confidences: list[float] = []
    for block in blocks:
        for row in block["provenance"]:
            value = row.get("confidence")
            if value is None:
                continue
            try:
                confidences.append(float(value))
            except (TypeError, ValueError):
                pass
    return [separator.join(block["text"] for block in blocks)], confidences


def _render_pages(pdf_path: Path, image_dir: Path, *, dpi: int) -> tuple[int, dict[int, str]]:
    doc = fitz.open(str(pdf_path))
    labels: dict[int, str] = {}
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    try:
        for index, page in enumerate(doc):
            page_no = index + 1
            try:
                labels[page_no] = str(page.get_label() or "")
            except Exception:
                labels[page_no] = ""
            page.get_pixmap(matrix=matrix, alpha=False).save(
                str(image_dir / f"page-{page_no:05d}.png")
            )
        return doc.page_count, labels
    finally:
        doc.close()


def extract_chunks_from_pdf_with_ndlocr(
    pdf_path: Path,
    attachment_key: str,
    meta_base: Dict[str, Any],
) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], Dict[str, Any]]:
    """OCR a Japanese PDF with NDLOCR-Lite and emit the standard chunk contract."""
    executable = find_ndlocr()
    if not executable:
        raise RuntimeError(
            "NDLOCR-Lite is not installed. Install `ndlocr-lite` or set NDLOCR_BIN."
        )
    dpi = max(72, int(os.environ.get("NDLOCR_DPI", "200")))
    timeout = max(1, int(os.environ.get("NDLOCR_TIMEOUT_SEC", "14400")))
    tmp_root = Path(__file__).resolve().parents[1] / "tmp" / "pdfs"
    tmp_root.mkdir(parents=True, exist_ok=True)
    toc = get_pdf_toc(str(pdf_path))
    toc_lookup = build_pdf_page_chapter_lookup(toc)
    structure_path_lookup = build_pdf_page_structure_path_lookup(toc)
    chunks: List[Tuple[str, str, Dict[str, Any]]] = []
    page_confidences: dict[str, float] = {}
    total_blocks = 0
    heading_blocks = 0
    inferred_path: list[str] = []
    with tempfile.TemporaryDirectory(prefix="ndlocr-", dir=tmp_root) as tmp:
        work = Path(tmp)
        images, output = work / "images", work / "output"
        images.mkdir()
        output.mkdir()
        page_count, page_labels = _render_pages(pdf_path, images, dpi=dpi)
        print(
            f"[INFO] NDLOCR-Lite: {page_count} pages at {dpi} DPI ({pdf_path.name})",
            file=sys.stderr,
        )
        completed = subprocess.run(
            [executable, "--sourcedir", str(images), "--output", str(output), "--device", "cpu"],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        if completed.returncode != 0:
            diagnostic = (completed.stderr or completed.stdout).strip().splitlines()[-5:]
            raise RuntimeError("NDLOCR-Lite failed: " + " | ".join(diagnostic))
        for page_no in range(1, page_count + 1):
            result_path = output / f"page-{page_no:05d}.json"
            if not result_path.exists():
                continue
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            page_blocks = blocks_from_ndlocr_payload(payload)
            confidences = [
                float(line["confidence"])
                for block in page_blocks for line in block["provenance"]
                if isinstance(line.get("confidence"), (int, float))
            ]
            if confidences:
                page_confidences[str(page_no)] = round(sum(confidences) / len(confidences), 4)
            if not page_blocks:
                continue
            chapter_info: dict[str, str] = {}
            if toc_lookup is not None:
                chapter, section = toc_lookup(page_no)
                if chapter:
                    chapter_info["chapter"] = chapter
                if section:
                    chapter_info["section"] = section
            outline_path = structure_path_lookup(page_no) if structure_path_lookup is not None else []
            active_path = list(outline_path or inferred_path)
            page_chunks: List[Tuple[str, str, Dict[str, Any]]] = []
            for block_index, block in enumerate(page_blocks):
                block_text = str(block["text"]).strip()
                if not block_text:
                    continue
                if block["block_type"] == "heading":
                    normalized_heading = " ".join(block_text.split())
                    if not active_path or active_path[-1] != normalized_heading:
                        # Without level evidence NDL headings are siblings, not a
                        # guessed nested hierarchy.  A valid PDF outline remains
                        # the parent path when available.
                        active_path = list(outline_path) + [normalized_heading]
                    heading_blocks += 1
                parts = split_long_paragraph(
                    block_text, max_chars=MAX_CHARS_CJK, target_chars=TARGET_CHARS_CJK,
                ) or [block_text]
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
                        "ocr_confidence": round(float(block["confidence"]), 4),
                        "block_type": block["block_type"], "zone": block["zone"],
                        "reading_order": int(block["reading_order"]),
                        "reading_order_end": int(block["reading_order_end"]),
                        "writing_mode": block["writing_mode"],
                        "ndlocr_label": block["ndlocr_label"],
                        "provenance": json.dumps(
                            block["provenance"], ensure_ascii=False, separators=(",", ":"),
                        ),
                        **chapter_info,
                    })
                    if active_path:
                        metadata["structure_path"] = list(active_path)
                    bbox = block.get("bbox")
                    if bbox:
                        metadata["bbox"] = json.dumps(
                            bbox, ensure_ascii=False, separators=(",", ":"),
                        )
                        for axis in ("l", "t", "r", "b"):
                            metadata[f"bbox_{axis}"] = float(bbox[axis])
                    page_chunks.append((chunk_id, part, metadata))
                    total_blocks += 1
            # Blocks are already source regions.  Do not merge them: doing so
            # would erase exact line provenance, bbox, and semantic block type.
            chunks.extend(page_chunks)
            if not outline_path and active_path:
                inferred_path = list(active_path)
    return chunks, {
        "is_scanned": False, "is_corrupted": False,
        "scanned_pages": [], "corrupted_pages": [],
        "total_pages": page_count, "parser": "ndlocr-lite",
        "ocr_pages": list(range(1, page_count + 1)),
        "page_confidences": page_confidences,
        "blocks": total_blocks, "heading_blocks": heading_blocks,
        "dpi": dpi,
    }


__all__ = [
    "blocks_from_ndlocr_payload", "extract_chunks_from_pdf_with_ndlocr",
    "find_ndlocr", "lines_from_ndlocr_payload",
]
