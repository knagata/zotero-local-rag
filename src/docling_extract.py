# src/docling_extract.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict

from text_utils import (
    MIN_CHUNK_CHARS,
    MIN_CHUNK_CHARS_NO_SPACE,
    MAX_CHARS,
    HARD_MIN_CHARS,
    is_no_space_language_document,
    split_long_paragraph,
    merge_short_chunk_records,
)


_DOCLING_CONVERTER = None


def get_docling_converter() -> Any:
    global _DOCLING_CONVERTER
    if _DOCLING_CONVERTER is None:
        from docling.document_converter import DocumentConverter

        _DOCLING_CONVERTER = DocumentConverter()
    return _DOCLING_CONVERTER


def extract_chunks_from_pdf_with_docling(
    pdf_path: Path,
    attachment_key: str,
    meta_base: Dict[str, Any],
) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], Dict[str, Any]]:
    """
    High-fidelity document layout analysis and text/table extraction using IBM Docling.
    This runs vision and layout models to precisely extract multi-column reading order,
    Markdown tables, and outline structures.

    Returns:
        (chunks, quality_info)
    """
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        raise ImportError(
            "\n"
            + "=" * 80
            + "\n"
            "[!] IBM Docling is not installed. To use high-fidelity parsing, please run:\n"
            "    pip install docling\n"
            "Note: Docling is a heavy package containing PyTorch and layout models.\n"
            + "=" * 80
            + "\n"
        )

    # Get page count using PyMuPDF (which is very lightweight) to set expectation
    page_count = 0
    try:
        import fitz

        temp_doc = fitz.open(str(pdf_path))
        page_count = temp_doc.page_count
        temp_doc.close()
    except Exception:
        pass

    if page_count > 0:
        # Estimate: approx 4-8 seconds per page on average CPU, plus some model init overhead.
        est_minutes = max(1, int((page_count * 6) / 60))
        print(
            f"\n[INFO] PDF contains {page_count} pages. Running high-fidelity layout analysis & OCR (Docling).\n"
            f"       This is a heavy process and may take more than {est_minutes} minute(s) depending on your CPU/GPU.\n"
            f"       Please wait...",
            file=sys.stderr,
        )

    # Re-use the single cached Docling converter to save gigabytes of memory and startup overhead
    converter = get_docling_converter()
    result = converter.convert(str(pdf_path))
    doc = result.document

    # Group items by page number
    page_contents = defaultdict(list)
    
    current_chapter = ""
    current_section = ""

    # Iterate through all parsed items in sequential reading order
    for item, level in doc.iterate_items():
        label = getattr(item, "label", None)
        text = getattr(item, "text", "").strip()

        # Capture heading hierarchies
        if label == "heading":
            if level == 0 or level == 1:
                current_chapter = text
                current_section = ""
            else:
                current_section = text

        # Get page number from provenance metadata
        page_no = 1
        prov = getattr(item, "prov", None)
        if prov and len(prov) > 0:
            page_no = prov[0].page_no

        # Convert tables to markdown representation
        item_type_name = type(item).__name__
        if "Table" in item_type_name or label == "table":
            try:
                table_markdown = item.export_to_markdown(doc=doc).strip()
                if table_markdown:
                    text_content = table_markdown
                else:
                    text_content = text
            except Exception:
                text_content = text
        else:
            text_content = text

        if text_content:
            page_contents[page_no].append({
                "text": text_content,
                "chapter": current_chapter,
                "section": current_section,
            })

    chunks: List[Tuple[str, str, Dict[str, Any]]] = []
    total_pages = len(doc.pages) if getattr(doc, "pages", None) else max(page_contents.keys() or [1])

    # Since Docling read it successfully and scanned it with ML, we treat it as 100% healthy
    quality_info = {
        "is_scanned": False,
        "is_corrupted": False,
        "scanned_pages": [],
        "corrupted_pages": [],
        "total_pages": total_pages,
        "parser": "docling",
    }

    # Generate page-by-page chunks preserving locators
    for pi in sorted(page_contents.keys()):
        items_on_page = page_contents[pi]
        if not items_on_page:
            continue

        joined_text = "\n\n".join(item["text"] for item in items_on_page)
        local_min_chunk = MIN_CHUNK_CHARS_NO_SPACE if is_no_space_language_document(joined_text) else MIN_CHUNK_CHARS

        page_chunks: List[Tuple[str, str, Dict[str, Any]]] = []
        for para_index, item_data in enumerate(items_on_page):
            para_text = item_data["text"]
            chapter = item_data["chapter"]
            section = item_data["section"]

            parts = split_long_paragraph(para_text, max_chars=MAX_CHARS)
            for part_index, part in enumerate(parts):
                part = part.strip()
                if len(part) < HARD_MIN_CHARS:
                    continue

                chunk_id = f"{attachment_key}:p{pi}:para{para_index}:part{part_index}"
                md = dict(meta_base)

                chapter_info = {}
                if chapter:
                    chapter_info["chapter"] = chapter
                if section:
                    chapter_info["section"] = section

                md.update({
                    "source_type": "pdf",
                    "locator": f"p{pi}:para{para_index}",
                    "page": int(pi),
                    "page_label": str(pi),
                    "pdf_path": str(pdf_path),
                    "path": str(pdf_path),
                    "para_index": int(para_index),
                    "part_index": int(part_index),
                    **chapter_info,
                })
                page_chunks.append((chunk_id, part, md))

        page_chunks = merge_short_chunk_records(page_chunks, min_chars=local_min_chunk, max_chars=MAX_CHARS)
        chunks.extend(page_chunks)

    return chunks, quality_info
