from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fitz

try:
    from .chapter_detect import build_pdf_page_chapter_lookup, get_pdf_toc
    from .text_utils import (
        HARD_MIN_CHARS, MAX_CHARS_CJK, MIN_CHUNK_CHARS_NO_SPACE,
        TARGET_CHARS_CJK, merge_short_chunk_records, split_long_paragraph,
    )
except ImportError:  # direct `python src/index_from_zotero.py` execution
    from chapter_detect import build_pdf_page_chapter_lookup, get_pdf_toc
    from text_utils import (
        HARD_MIN_CHARS, MAX_CHARS_CJK, MIN_CHUNK_CHARS_NO_SPACE,
        TARGET_CHARS_CJK, merge_short_chunk_records, split_long_paragraph,
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


def lines_from_ndlocr_payload(payload: dict[str, Any]) -> tuple[list[str], list[float]]:
    """Return NDL's evaluated reading order and per-line confidences."""
    rows = [
        row for group in payload.get("contents", []) if isinstance(group, list)
        for row in group if isinstance(row, dict) and _usable_ocr_text(row.get("text"))
    ]
    # IDs are assigned after NDL's PAGE XML reading-order evaluation. Use them
    # instead of naïve y/x sorting, which interleaves Japanese multi-column text.
    rows.sort(key=lambda row: int(row.get("id", 10**9)))
    vertical = sum(_box_size(row.get("boundingBox"))[1] > _box_size(row.get("boundingBox"))[0]
                   for row in rows) > len(rows) / 2
    separator = "" if vertical else "\n"
    texts: list[str] = []
    seen: set[str] = set()
    previous = ""
    for row in rows:
        text = _usable_ocr_text(row.get("text"))
        normalized = " ".join(text.split())
        if normalized in seen:
            continue
        seen.add(normalized)
        text = _trim_adjacent_overlap(previous, text)
        if not text:
            continue
        texts.append(text)
        previous = text
    confidences = []
    for row in rows:
        try:
            confidences.append(float(row.get("confidence")))
        except (TypeError, ValueError):
            pass
    return [separator.join(texts)] if texts else [], confidences


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
    toc_lookup = build_pdf_page_chapter_lookup(get_pdf_toc(str(pdf_path)))
    chunks: List[Tuple[str, str, Dict[str, Any]]] = []
    page_confidences: dict[str, float] = {}
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
            page_texts, confidences = lines_from_ndlocr_payload(payload)
            if confidences:
                page_confidences[str(page_no)] = round(sum(confidences) / len(confidences), 4)
            if not page_texts:
                continue
            page_text = page_texts[0].strip()
            if len(page_text) < HARD_MIN_CHARS:
                continue
            chapter_info: dict[str, str] = {}
            if toc_lookup is not None:
                chapter, section = toc_lookup(page_no)
                if chapter:
                    chapter_info["chapter"] = chapter
                if section:
                    chapter_info["section"] = section
            page_chunks: List[Tuple[str, str, Dict[str, Any]]] = []
            parts = split_long_paragraph(
                page_text, max_chars=MAX_CHARS_CJK, target_chars=TARGET_CHARS_CJK,
            )
            for part_index, part in enumerate(parts):
                if len(part.strip()) < HARD_MIN_CHARS:
                    continue
                chunk_id = f"{attachment_key}:p{page_no}:para0:part{part_index}"
                metadata = dict(meta_base)
                metadata.update({
                    "source_type": "pdf", "locator": f"p{page_no}:para0",
                    "page": page_no, "page_label": page_labels.get(page_no, ""),
                    "pdf_path": str(pdf_path), "path": str(pdf_path),
                    "para_index": 0, "part_index": part_index,
                    "ocr_confidence": page_confidences.get(str(page_no), 0.0),
                    **chapter_info,
                })
                page_chunks.append((chunk_id, part.strip(), metadata))
            chunks.extend(merge_short_chunk_records(
                page_chunks, min_chars=MIN_CHUNK_CHARS_NO_SPACE, max_chars=MAX_CHARS_CJK,
            ))
    return chunks, {
        "is_scanned": False, "is_corrupted": False,
        "scanned_pages": [], "corrupted_pages": [],
        "total_pages": page_count, "parser": "ndlocr-lite",
        "ocr_pages": list(range(1, page_count + 1)),
        "page_confidences": page_confidences,
        "dpi": dpi,
    }
