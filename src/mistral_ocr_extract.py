from __future__ import annotations

import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fitz
import httpx

try:
    from .chapter_detect import (
        build_pdf_page_chapter_lookup, build_pdf_page_structure_path_lookup, get_pdf_toc,
    )
    from .env_utils import load_dotenv_native
    from .text_utils import (
        MAX_CHARS, MAX_CHARS_CJK, TARGET_CHARS, TARGET_CHARS_CJK,
        MIN_CHUNK_CHARS, MIN_CHUNK_CHARS_NO_SPACE, HARD_MIN_CHARS, HARD_MIN_CHARS_CJK,
        is_no_space_language_document, merge_short_chunk_records, split_long_paragraph,
    )
except ImportError:  # direct `python src/index_from_zotero.py` execution
    from chapter_detect import (
        build_pdf_page_chapter_lookup, build_pdf_page_structure_path_lookup, get_pdf_toc,
    )
    from env_utils import load_dotenv_native
    from text_utils import (
        MAX_CHARS, MAX_CHARS_CJK, TARGET_CHARS, TARGET_CHARS_CJK,
        MIN_CHUNK_CHARS, MIN_CHUNK_CHARS_NO_SPACE, HARD_MIN_CHARS, HARD_MIN_CHARS_CJK,
        is_no_space_language_document, merge_short_chunk_records, split_long_paragraph,
    )


DEFAULT_MODEL = "mistral-ocr-latest"
DEFAULT_BASE_URL = "https://api.mistral.ai"

_FOOTNOTE_LABEL_RE = re.compile(r"footnote|脚注|註", re.I)
_ENDNOTE_LABEL_RE = re.compile(r"endnote|巻末注|後注", re.I)
_BIBLIOGRAPHY_LABEL_RE = re.compile(r"bibliography|references?|参考文献|引用文献|文献一覧", re.I)
_TOC_LABEL_RE = re.compile(r"table\s+of\s+contents|contents?|目次|索引|index", re.I)
_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

# Confirmed via a live smoke test (2026-07-21) against pages/blocks that
# actually occurred in the bake-off samples: header, footer, title, text,
# caption, table, image. Treat any other value conservatively as body text
# rather than silently dropping it.
_PAGE_FURNITURE_TYPES = {"header", "footer"}
_SKIP_TYPES = {"image"}  # no text content to chunk


def _blocks_from_markdown(markdown: str) -> List[Dict[str, Any]]:
    """Build coarse blocks when OCR Batch returns Markdown but an empty blocks list."""
    blocks: List[Dict[str, Any]] = []
    paragraph: List[str] = []

    def flush() -> None:
        if paragraph:
            text = "\n".join(paragraph).strip()
            if text:
                block_type = "table" if text.startswith("|") and "|" in text[1:] else "text"
                blocks.append({"type": block_type, "content": text})
            paragraph.clear()

    for line in str(markdown or "").splitlines():
        stripped = line.strip()
        if _MARKDOWN_HEADING_RE.match(stripped):
            flush()
            blocks.append({"type": "title", "content": stripped})
        elif not stripped:
            flush()
        else:
            paragraph.append(line.rstrip())
    flush()
    return blocks


def _zone_from_heading_text(text: str) -> str | None:
    if _BIBLIOGRAPHY_LABEL_RE.search(text):
        return "bibliography"
    if _ENDNOTE_LABEL_RE.search(text):
        return "endnote"
    if _FOOTNOTE_LABEL_RE.search(text):
        return "footnote"
    if _TOC_LABEL_RE.search(text):
        return "toc"
    return None


def _bbox_from_block(block: Dict[str, Any]) -> Dict[str, float] | None:
    try:
        return {
            "l": float(block["top_left_x"]), "t": float(block["top_left_y"]),
            "r": float(block["bottom_right_x"]), "b": float(block["bottom_right_y"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _page_blocks_from_response(page_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize one Mistral OCR page's ``blocks`` into the shared block contract.

    ``blocks`` (confirmed via live smoke test, not just documentation) already
    carries per-region bbox and a type label, unlike the markdown-only
    fallback this module used before verifying the response shape. ``title``
    blocks keep their markdown heading prefix (``#``/``##``/...) in
    ``content``, which is the only place a heading *level* is available.
    """
    raw_blocks = [b for b in (page_payload.get("blocks") or []) if isinstance(b, dict)]
    if not raw_blocks and str(page_payload.get("markdown") or "").strip():
        raw_blocks = _blocks_from_markdown(str(page_payload["markdown"]))
    items: List[Dict[str, Any]] = []
    path_stack: List[Tuple[int, str]] = []  # (heading_level, title)

    for raw in raw_blocks:
        block_type = str(raw.get("type") or "text")
        if block_type in _SKIP_TYPES:
            continue
        text = str(raw.get("content") or "").strip()
        if not text:
            continue
        bbox = _bbox_from_block(raw)

        if block_type == "title":
            heading_match = _MARKDOWN_HEADING_RE.match(text)
            level, title = (len(heading_match.group(1)), heading_match.group(2).strip()) if heading_match else (1, text)
            while path_stack and path_stack[-1][0] >= level:
                path_stack.pop()
            path_stack.append((level, title))
            items.append({
                "block_type": "heading", "zone": "body", "text": title, "bbox": bbox,
                "heading_path": [item[1] for item in path_stack],
            })
            continue

        if block_type in _PAGE_FURNITURE_TYPES:
            items.append({"block_type": "page_furniture", "zone": "other_paratext", "text": text, "bbox": bbox})
            continue

        mapped_type = {"table": "table", "caption": "caption"}.get(block_type, "text")
        items.append({"block_type": mapped_type, "zone": "body", "text": text, "bbox": bbox})

    active_zone = "body"
    for item in items:
        if item["block_type"] == "heading":
            active_zone = _zone_from_heading_text(item["text"]) or "body"
            continue
        if item["zone"] == "body":
            item["zone"] = active_zone
    return items


def _call_mistral_ocr_api(
    pdf_path: Path, *, api_key: str, model: str, base_url: str, timeout: float,
) -> Dict[str, Any]:
    """Call the Mistral OCR endpoint with the PDF inlined as a base64 data URL.

    Request/response shape verified against a live call on 2026-07-21 (not
    just documentation): top-level ``pages`` each carry ``markdown`` and a
    ``blocks`` array with per-region bbox + type (see
    ``_page_blocks_from_response``). Kept isolated from parsing so that
    function stays unit-testable without a key.
    """
    encoded = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "document": {
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{encoded}",
        },
    }
    response = httpx.post(
        f"{base_url.rstrip('/')}/v1/ocr",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload, timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def extract_chunks_from_pdf_with_mistral_ocr(
    pdf_path: Path,
    attachment_key: str,
    meta_base: Dict[str, Any],
) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], Dict[str, Any]]:
    """OCR a PDF with the Mistral OCR cloud API and emit the standard chunk contract.

    This is a cloud engine (no-cloud attachments must never reach it; see
    ``EngineRegistry.select(no_cloud=True)``). Requires ``MISTRAL_OCR_API_KEY``.
    """
    load_dotenv_native(Path(__file__).resolve().parents[1])
    api_key = os.environ.get("MISTRAL_OCR_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("MISTRAL_OCR_API_KEY is not configured; Mistral OCR is a cloud engine and opt-in only.")
    model = os.environ.get("MISTRAL_OCR_MODEL", "").strip() or DEFAULT_MODEL
    base_url = os.environ.get("MISTRAL_OCR_BASE_URL", "").strip() or DEFAULT_BASE_URL
    timeout = max(1.0, float(os.environ.get("MISTRAL_OCR_TIMEOUT_SEC", "120")))

    result = _call_mistral_ocr_api(pdf_path, api_key=api_key, model=model, base_url=base_url, timeout=timeout)
    return extract_chunks_from_mistral_ocr_result(
        pdf_path, attachment_key, meta_base, result, model=model,
    )


#: Meaningful at any length, so they skip the merge/HARD_MIN_CHARS drop below.
_PRESERVE_SHORT_BLOCK_TYPES = {"heading", "page_furniture", "table"}


def extract_chunks_from_mistral_ocr_result(
    pdf_path: Path,
    attachment_key: str,
    meta_base: Dict[str, Any],
    result: Dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    problem_pages: Dict[int | str, List[str]] | None = None,
) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], Dict[str, Any]]:
    """Convert a synchronous or Batch API OCR response to canonical chunks."""
    toc = get_pdf_toc(str(pdf_path))
    toc_lookup = build_pdf_page_chapter_lookup(toc)
    structure_path_lookup = build_pdf_page_structure_path_lookup(toc)

    doc = fitz.open(str(pdf_path))
    try:
        page_labels = {index + 1: str(page.get_label() or "") for index, page in enumerate(doc)}
        page_count = doc.page_count
    finally:
        doc.close()

    pages = result.get("pages") or []

    normalized_problem_pages = {
        int(page): list(reasons)
        for page, reasons in (problem_pages or {}).items()
        if isinstance(reasons, list)
    }
    chunks: List[Tuple[str, str, Dict[str, Any]]] = []
    total_blocks = 0
    heading_blocks = 0
    inferred_path: List[str] = []
    covered_pages: List[int] = []

    for page_payload in pages:
        page_no = int(page_payload.get("index", 0)) + 1
        blocks = _page_blocks_from_response(page_payload)
        if not blocks:
            continue
        covered_pages.append(page_no)

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
        min_chars = MIN_CHUNK_CHARS_NO_SPACE if is_cjk else MIN_CHUNK_CHARS
        # This path never called merge_short_chunk_records at all -- every OCR
        # block became its own chunk regardless of length, unlike the other
        # three extractors. Measured: 34% of mistral_ocr chunks were under 40
        # characters (2026-07-28). page_chunks collects this page's chunks so
        # they can be merged before extending the running `chunks` list.
        page_chunks: List[Tuple[str, str, Dict[str, Any]]] = []

        for block_index, block in enumerate(blocks):
            block_text = block["text"]
            if block["block_type"] == "heading":
                active_path = list(outline_path) + block["heading_path"]
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
                    "block_type": block["block_type"], "zone": block["zone"],
                    "reading_order": block_index,
                    **chapter_info,
                })
                if page_no in normalized_problem_pages:
                    metadata["quality_uncertain"] = True
                    metadata["quality_uncertain_reason"] = ",".join(normalized_problem_pages[page_no])
                if active_path:
                    metadata["structure_path"] = list(active_path)
                if block.get("bbox"):
                    metadata["bbox"] = json.dumps(block["bbox"], ensure_ascii=False, separators=(",", ":"))
                    for axis in ("l", "t", "r", "b"):
                        metadata[f"bbox_{axis}"] = float(block["bbox"][axis])
                page_chunks.append((chunk_id, part, metadata))
                total_blocks += 1
        # Headings and page furniture are meaningful at any length -- a title
        # is not made less of a title by being short -- so they bypass the
        # merge and its trailing HARD_MIN_CHARS drop entirely, the same
        # protection docling_extract.py gives its own short structural labels.
        mergeable = [c for c in page_chunks if c[2].get("block_type") not in _PRESERVE_SHORT_BLOCK_TYPES]
        preserved = [c for c in page_chunks if c[2].get("block_type") in _PRESERVE_SHORT_BLOCK_TYPES]
        merged = merge_short_chunk_records(
            mergeable, min_chars=min_chars, max_chars=max_chars,
            boundary_key=lambda _cid, _text, md: (
                md.get("block_type"), md.get("zone"), tuple(md.get("structure_path") or ()),
            ),
            hard_min_chars=HARD_MIN_CHARS_CJK if is_cjk else HARD_MIN_CHARS,
        )
        chunks.extend(sorted(merged + preserved, key=lambda c: c[2].get("para_index", 0)))
        if not outline_path and active_path:
            inferred_path = list(active_path)

    covered_pages = sorted(set(covered_pages))
    # ocr_pages used to be claimed as every page of the source PDF regardless
    # of what the response actually contained, so a page the API silently
    # skipped (absent from result["pages"], or present but with no blocks)
    # left no trace: quality reported full coverage for a page with zero
    # chunks (2026-07-28). ocr_pages is now the pages that actually produced
    # output; missing_pages is what full coverage would have required.
    missing_pages = sorted(set(range(1, page_count + 1)) - set(covered_pages))
    return chunks, {
        "is_scanned": False, "is_corrupted": False,
        "scanned_pages": [], "corrupted_pages": [],
        "total_pages": page_count, "parser": "mistral_ocr", "model": model,
        "ocr_pages": covered_pages,
        "missing_pages": missing_pages,
        "page_confidences": {},
        "blocks": total_blocks, "heading_blocks": heading_blocks,
        "usage_info": result.get("usage_info") or {},
        "quality_problem_pages": normalized_problem_pages,
    }


def mistral_ocr_available() -> tuple[bool, str]:
    """Whether Mistral OCR can be called at all.

    Only the key is checked. The former ``MISTRAL_OCR_FALLBACK_ENABLE`` switch
    was removed (2026-07-27): every remaining route to this API is already
    guarded by an explicit operator action -- passing a ``target_engine=
    mistral_ocr`` queue to ``--reocr-candidates``, or running
    ``run_mistral_ocr_batch.py --submit`` -- or merely records a candidate for
    a later, separately-invoked send. Automatic ingestion never calls this
    synchronously at all. A second switch in front of an already-explicit
    action bought nothing and produced exactly the confusion it was meant to
    prevent: a configured key with the flag left off, silently doing nothing.
    """
    load_dotenv_native(Path(__file__).resolve().parents[1])
    if not os.environ.get("MISTRAL_OCR_API_KEY", "").strip():
        return False, "MISTRAL_OCR_API_KEY is not configured"
    return True, "ready"


__all__ = [
    "extract_chunks_from_pdf_with_mistral_ocr", "extract_chunks_from_mistral_ocr_result",
    "mistral_ocr_available",
]
