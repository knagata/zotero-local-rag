# src/docling_extract.py
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict

try:
    from .text_utils import (
        MIN_CHUNK_CHARS, MIN_CHUNK_CHARS_NO_SPACE, MAX_CHARS, TARGET_CHARS,
        MAX_CHARS_CJK, TARGET_CHARS_CJK, HARD_MIN_CHARS,
        is_no_space_language_document, split_long_paragraph, merge_short_chunk_records,
        _cjk_ratio,
    )
except ImportError:  # direct execution with src/ on sys.path
    from text_utils import (
        MIN_CHUNK_CHARS, MIN_CHUNK_CHARS_NO_SPACE, MAX_CHARS, TARGET_CHARS,
        MAX_CHARS_CJK, TARGET_CHARS_CJK, HARD_MIN_CHARS,
        is_no_space_language_document, split_long_paragraph, merge_short_chunk_records,
        _cjk_ratio,
    )


_DOCLING_CONVERTER = None

_HEADING_LABELS = {"title", "section_header"}
_PRESERVE_SHORT_LABELS = {
    "title", "section_header", "table", "caption", "footnote", "reference",
    "formula", "chart", "picture", "document_index",
}
_BIBLIOGRAPHY_HEADING_RE = re.compile(
    r"^(?:references?|bibliography|works\s+cited|参考文献|引用文献|文献一覧)$", re.I,
)
_FOOTNOTE_HEADING_RE = re.compile(r"^(?:footnotes?|脚注|註)$", re.I)
_ENDNOTE_HEADING_RE = re.compile(r"^(?:endnotes?|notes?|巻末注|後注|注)$", re.I)
_INDEX_HEADING_RE = re.compile(r"^(?:index|索引)$", re.I)


def normalize_docling_label(value: Any) -> str:
    """Return a stable lower-case label for DocItemLabel enums and strings."""
    raw = getattr(value, "value", value)
    text = str(raw or "text").strip().lower()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.replace("-", "_").replace(" ", "_")


def _heading_zone(text: str) -> str:
    normalized = " ".join(text.strip().split())
    if _BIBLIOGRAPHY_HEADING_RE.fullmatch(normalized):
        return "bibliography"
    if _FOOTNOTE_HEADING_RE.fullmatch(normalized):
        return "footnote"
    if _ENDNOTE_HEADING_RE.fullmatch(normalized):
        return "endnote"
    if _INDEX_HEADING_RE.fullmatch(normalized):
        return "index"
    return "body"


def zone_for_docling_item(label: str, text: str, section_zone: str = "body") -> str:
    if label == "footnote":
        return "footnote"
    if label == "reference":
        return "bibliography"
    if label == "document_index":
        return "index"
    if label in {"page_header", "page_footer"}:
        return "other_paratext"
    if label in _HEADING_LABELS:
        return _heading_zone(text)
    return section_zone


def _bbox_payload(value: Any) -> Dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        payload = value.model_dump(mode="json")
        return dict(payload) if isinstance(payload, dict) else None
    try:
        return {
            "l": float(value.l), "t": float(value.t),
            "r": float(value.r), "b": float(value.b),
            "coord_origin": str(getattr(getattr(value, "coord_origin", ""), "value", "") or ""),
        }
    except (AttributeError, TypeError, ValueError):
        return None


def docling_provenance(item: Any) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for prov in getattr(item, "prov", None) or []:
        try:
            page_no = int(prov.page_no)
        except (AttributeError, TypeError, ValueError):
            continue
        row: Dict[str, Any] = {"page": page_no}
        bbox = _bbox_payload(getattr(prov, "bbox", None))
        if bbox:
            row["bbox"] = bbox
        charspan = getattr(prov, "charspan", None)
        if isinstance(charspan, (list, tuple)) and len(charspan) == 2:
            row["charspan"] = [int(charspan[0]), int(charspan[1])]
        output.append(row)
    return output


def _heading_depth(item: Any, traversal_level: Any, label: str) -> int:
    if label == "title":
        return 1
    value = getattr(item, "level", None)
    if value is None:
        try:
            value = int(traversal_level) + 1
        except (TypeError, ValueError):
            value = 1
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _item_text(item: Any, label: str, doc: Any) -> str:
    text = str(getattr(item, "text", "") or "").strip()
    if label in {"table", "document_index"} or "table" in type(item).__name__.lower():
        try:
            rendered = str(item.export_to_markdown(doc=doc) or "").strip()
            return rendered or text
        except Exception:
            return text
    return text


def _docling_items(doc: Any) -> Dict[int, List[Dict[str, Any]]]:
    """Normalize a DoclingDocument into source-order, page-addressable blocks."""
    page_contents: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    heading_path: List[str] = []
    section_zone = "body"
    for reading_order, (item, traversal_level) in enumerate(doc.iterate_items()):
        label = normalize_docling_label(getattr(item, "label", None))
        text = _item_text(item, label, doc)
        if not text:
            continue
        if label in _HEADING_LABELS:
            depth = _heading_depth(item, traversal_level, label)
            heading_path = heading_path[: depth - 1]
            heading_path.append(text)
            section_zone = _heading_zone(text)
        zone = zone_for_docling_item(label, text, section_zone)
        provenance = docling_provenance(item)
        page_no = provenance[0]["page"] if provenance else 1
        pages = [row["page"] for row in provenance] or [page_no]
        page_contents[page_no].append({
            "text": text,
            "chapter": heading_path[0] if heading_path else "",
            "section": heading_path[1] if len(heading_path) > 1 else "",
            "structure_path": list(heading_path),
            "block_type": label,
            "zone": zone,
            "reading_order": reading_order,
            "provenance": provenance,
            "page_end": max(pages),
        })
    return page_contents


def _merge_docling_chunks(
    chunks: List[Tuple[str, str, Dict[str, Any]]], *, min_chars: int, max_chars: int,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Merge only ordinary text parts; semantic short blocks stay standalone."""
    output: List[Tuple[str, str, Dict[str, Any]]] = []
    pending: List[Tuple[str, str, Dict[str, Any]]] = []

    def flush() -> None:
        if not pending:
            return
        output.extend(merge_short_chunk_records(
            pending, min_chars=min_chars, max_chars=max_chars,
            boundary_key=lambda _cid, _text, md: (
                md.get("reading_order"), md.get("block_type"), md.get("zone"),
                tuple(md.get("structure_path") or ()),
            ),
        ))
        pending.clear()

    for chunk in chunks:
        if chunk[2].get("block_type") in _PRESERVE_SHORT_LABELS:
            flush()
            output.append(chunk)
        else:
            pending.append(chunk)
    flush()
    return output


# Docling's own per-document timeout (seconds). When exceeded, Docling stops
# requesting further pages but keeps everything already converted, returning
# ConversionStatus.PARTIAL_SUCCESS instead of raising -- unlike an external
# wall-clock kill, this does not discard already-completed work on large
# documents. Kept well under DoclingWorker's external timeout_sec (3600s) so
# this graceful path fires first; the external timeout remains only as a
# backstop for genuine hangs (e.g. a single page stuck inside native OCR code,
# which this in-loop check can't observe).
DOCLING_DOCUMENT_TIMEOUT_SEC = 1700.0


def get_docling_converter() -> Any:
    global _DOCLING_CONVERTER
    if _DOCLING_CONVERTER is None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pipeline_options = PdfPipelineOptions(document_timeout=DOCLING_DOCUMENT_TIMEOUT_SEC)
        _DOCLING_CONVERTER = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
    return _DOCLING_CONVERTER


def extract_chunks_from_pdf_with_docling(
    pdf_path: Path,
    attachment_key: str,
    meta_base: Dict[str, Any],
    converter: Any = None,
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

    # Re-use the single cached Docling converter to save gigabytes of memory and
    # startup overhead. A caller may pass its own -- the Granite runner supplies
    # a VlmPipeline converter so that engine reuses this exact normalisation and
    # chunking rather than growing a parallel implementation of it.
    converter = converter or get_docling_converter()
    result = converter.convert(str(pdf_path))
    doc = result.document

    page_contents = _docling_items(doc)

    chunks: List[Tuple[str, str, Dict[str, Any]]] = []
    total_pages = len(doc.pages) if getattr(doc, "pages", None) else max(page_contents.keys() or [1])

    from docling.datamodel.base_models import ConversionStatus

    is_truncated = result.status == ConversionStatus.PARTIAL_SUCCESS
    if is_truncated:
        print(
            f"[WARN] Docling document_timeout ({DOCLING_DOCUMENT_TIMEOUT_SEC:.0f}s) exceeded for "
            f"{pdf_path.name}; keeping {total_pages}/{page_count or '?'} pages already converted "
            f"instead of discarding them.",
            file=sys.stderr,
        )

    # Since Docling read it successfully and scanned it with ML, we treat it as healthy
    # unless the document_timeout truncated it mid-conversion (partial coverage).
    quality_info = {
        "is_scanned": False,
        "is_corrupted": False,
        "scanned_pages": [],
        "corrupted_pages": [],
        "total_pages": total_pages,
        "processed_pages": total_pages,
        "expected_pages": page_count or total_pages,
        "page_coverage": round(total_pages / max(1, page_count or total_pages), 6),
        "parser": "docling",
        "truncated_by_timeout": is_truncated,
    }

    # Generate page-by-page chunks preserving locators
    for pi in sorted(page_contents.keys()):
        items_on_page = page_contents[pi]
        if not items_on_page:
            continue

        joined_text = "\n\n".join(item["text"] for item in items_on_page)
        is_cjk = is_no_space_language_document(joined_text)
        local_min_chunk = MIN_CHUNK_CHARS_NO_SPACE if is_cjk else MIN_CHUNK_CHARS
        local_max_chars = MAX_CHARS_CJK if is_cjk else MAX_CHARS
        local_target_chars = TARGET_CHARS_CJK if is_cjk else TARGET_CHARS

        page_chunks: List[Tuple[str, str, Dict[str, Any]]] = []
        for para_index, item_data in enumerate(items_on_page):
            para_text = item_data["text"]
            chapter = item_data["chapter"]
            section = item_data["section"]

            parts = split_long_paragraph(para_text, max_chars=local_max_chars, target_chars=local_target_chars)
            for part_index, part in enumerate(parts):
                part = part.strip()
                if len(part) < HARD_MIN_CHARS and item_data["block_type"] not in _PRESERVE_SHORT_LABELS:
                    continue

                chunk_id = f"{attachment_key}:p{pi}:para{para_index}:part{part_index}"
                md = dict(meta_base)

                chapter_info = {}
                if chapter:
                    chapter_info["chapter"] = chapter
                if section:
                    chapter_info["section"] = section
                if item_data.get("structure_path"):
                    chapter_info["structure_path"] = item_data["structure_path"]

                md.update({
                    "source_type": "pdf",
                    "locator": f"p{pi}:para{para_index}",
                    "page": int(pi),
                    "page_label": str(pi),
                    "pdf_path": str(pdf_path),
                    "path": str(pdf_path),
                    "para_index": int(para_index),
                    "part_index": int(part_index),
                    "block_type": item_data["block_type"],
                    "docling_label": item_data["block_type"],
                    "reading_order": int(item_data["reading_order"]),
                    "zone": item_data["zone"],
                    "provenance": json.dumps(
                        item_data["provenance"], ensure_ascii=False, separators=(",", ":"),
                    ),
                    "page_end": int(item_data["page_end"]),
                    **chapter_info,
                })
                if item_data["provenance"] and item_data["provenance"][0].get("bbox"):
                    bbox = item_data["provenance"][0]["bbox"]
                    md["bbox"] = json.dumps(bbox, ensure_ascii=False, separators=(",", ":"))
                    for axis in ("l", "t", "r", "b"):
                        if isinstance(bbox.get(axis), (int, float)):
                            md[f"bbox_{axis}"] = float(bbox[axis])
                page_chunks.append((chunk_id, part, md))

        # Never merge across Docling source blocks. This preserves block type,
        # zone, reading order, and exact provenance even for short captions or notes.
        page_chunks = _merge_docling_chunks(
            page_chunks, min_chars=local_min_chunk, max_chars=local_max_chars,
        )
        chunks.extend(page_chunks)

    return chunks, quality_info


_REFERENCE_HARVEST_ZONES = {"bibliography", "endnote", "footnote"}


def _docling_subset_pages(
    pdf_path: Path, pages: List[int], *, attachment_key: str, meta_base: Dict[str, Any],
) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], Dict[int, int], Dict[str, Any]]:
    """Run Docling on just the given 1-based ``pages`` of a PDF.

    Builds a temporary sub-PDF containing only those pages (in the given
    order), runs Docling on it, and returns the raw chunks alongside a
    sub-page -> original-page map so callers can remap ``page`` metadata back
    onto the source document.  Shared by the reference-section (E2b) and
    scanned-page (E2c) Docling patch paths.
    """
    import os
    import tempfile

    import fitz

    wanted = sorted({int(page) for page in pages if int(page) > 0})
    if not wanted:
        return [], {}, {}
    source = fitz.open(pdf_path)
    subset = fitz.open()
    sub_to_original: Dict[int, int] = {}
    try:
        for sub_index, page in enumerate(wanted, start=1):
            subset.insert_pdf(source, from_page=page - 1, to_page=page - 1)
            sub_to_original[sub_index] = page
        handle = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        handle.close()
        subset.save(handle.name)
    finally:
        subset.close()
        source.close()
    try:
        chunks, quality = extract_chunks_from_pdf_with_docling(
            Path(handle.name), attachment_key, dict(meta_base),
        )
    finally:
        os.unlink(handle.name)
    return chunks, sub_to_original, quality


def extract_reference_sections_with_docling(
    pdf_path: Path, pages: List[int], *, attachment_key: str, meta_base: Dict[str, Any],
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Structure a PDF's reference/note pages with Docling (E2b, note 77).

    Builds a sub-PDF of the given 1-based ``pages``, runs Docling on it, keeps
    only reference/footnote items (each already one entry per chunk), and remaps
    page numbers back to the original document.  Chunk ids are namespaced so they
    never collide with the AI-TOC body chunks they enrich.
    """
    chunks, sub_to_original, _quality = _docling_subset_pages(
        pdf_path, pages, attachment_key=attachment_key, meta_base=meta_base,
    )
    output: List[Tuple[str, str, Dict[str, Any]]] = []
    for chunk_id, text, metadata in chunks:
        if (metadata.get("zone") not in _REFERENCE_HARVEST_ZONES
                and metadata.get("block_type") not in {"reference", "footnote"}):
            continue
        md = dict(metadata)
        md["page"] = sub_to_original.get(int(md.get("page") or 0), int(md.get("page") or 0))
        md["reference_enrichment"] = "docling_page_subset"
        output.append((f"{attachment_key}:docref:{chunk_id}", text, md))
    return output


def _looks_like_scanned_patch_ocr_noise(text: str) -> bool:
    """Quality gate for text recovered by ``_patch_pages_with_docling_ocr``
    (used by both ``patch_scanned_pages_with_docling`` and
    ``patch_corrupted_pages_with_docling``, E2c/E2d).

    Unlike ``text_utils.looks_like_gibberish`` (used elsewhere for normally
    extracted text), this has no length floor: OCR noise recovered from a
    scanned page -- a page PyMuPDF already flagged as scanned/image-only, an
    inherently low-confidence source -- is often short (a mis-OCR'd caption
    fragment), so the usual "don't reject short titles" exemption would let
    garbage straight into the index (E2c, user decision 2026-07-26; found by
    inspecting M5TQ4HLZ's recovered chunks, e.g. "Then10-15CobraHelicopters
    cameandstartedshooting andshooting andshooting."). Also flags
    concatenated-word runs (missing spaces) that pass the character-
    composition check but read as garbled merged words -- CJK text is
    exempt since it has no space-delimited word boundaries by convention.
    Not a complete filter: OCR noise that happens to land on real dictionary
    words with normal spacing (e.g. "blood cloud siagwnus polygons...")
    still slips through; this catches the common concatenated-word and
    non-printable-character failure modes, not all garbling.
    """
    stripped = re.sub(r"\s+", " ", text).strip()
    if not stripped:
        return True
    compact = stripped.replace(" ", "")
    printable_ratio = sum(ch.isprintable() for ch in compact) / max(len(compact), 1)
    letters_ratio = sum(ch.isalpha() for ch in compact) / max(len(compact), 1)
    digits_ratio = sum(ch.isdigit() for ch in compact) / max(len(compact), 1)
    if printable_ratio < 0.90 or (letters_ratio + digits_ratio) < 0.20:
        return True
    if len(stripped) >= 20 and _cjk_ratio(stripped) < 0.3:
        space_count = stripped.count(" ")
        if len(stripped) / max(1, space_count) > 11:
            return True
    return False


def _build_scanned_page_subset(pdf_path: Path, pages: List[int]) -> Tuple[str, Dict[int, int]]:
    """Build a self-contained sub-PDF of ``pages``, rendered from source pixels.

    Some real-world scanned PDFs have an intact, extractable image XObject on
    a page but a broken content stream that makes ``page.get_pixmap()`` (and
    consequently a naive ``insert_pdf`` copy) render blank -- confirmed by
    directly comparing ``doc.extract_image()`` bytes (real photographic
    content) against ``get_pixmap()`` output (all-white) for the same page.
    To make patching robust against this, each wanted page is rendered
    independently: first via direct pixmap render, and if that is blank,
    by re-drawing the page's own extracted embedded image onto a fresh page
    (bypassing whatever in the original content stream breaks rendering).
    """
    import tempfile

    import fitz

    wanted = sorted({int(page) for page in pages if int(page) > 0})
    source = fitz.open(pdf_path)
    subset = fitz.open()
    sub_to_original: Dict[int, int] = {}
    try:
        for sub_index, page_number in enumerate(wanted, start=1):
            page = source[page_number - 1]
            pix = page.get_pixmap(dpi=200)
            non_white = pix.samples.count(255)
            is_blank = (len(pix.samples) - non_white) / max(1, len(pix.samples)) < 0.005
            new_page = subset.new_page(width=page.rect.width, height=page.rect.height)
            if not is_blank:
                new_page.insert_image(new_page.rect, pixmap=pix)
            else:
                # Direct render was blank; fall back to the page's own embedded
                # image bytes, which can be intact even when content-stream
                # composition is broken.
                recovered = False
                for xref, *_rest in page.get_images():
                    try:
                        image = source.extract_image(xref)
                    except Exception:
                        continue
                    if not image.get("image"):
                        continue
                    new_page.insert_image(new_page.rect, stream=image["image"])
                    recovered = True
                    break
                if not recovered:
                    # Nothing recoverable for this page; leave it blank so the
                    # caller's chunk count for it stays zero (fail-closed).
                    pass
            sub_to_original[sub_index] = page_number
        handle = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        handle.close()
        subset.save(handle.name)
    finally:
        subset.close()
        source.close()
    return handle.name, sub_to_original


def _patch_batch_size() -> int:
    """Pages per Docling sub-PDF pass for the E2c/E2d page patches.

    The patches used to build ONE sub-PDF out of every wanted page and hand it
    to Docling whole. That was survivable for the scanned-page counts seen so
    far (<=60), but corrupted-page counts can be far larger while still under
    the is_corrupted document threshold -- AX5CRKJ6 (208 corrupted pages of
    413, ratio 0.504 < 0.6) OOM-killed the whole ingestion run (SIGKILL, no
    traceback, only a leaked-semaphore warning) when its 208-page 200dpi
    sub-PDF hit Docling in one pass (observed 2026-07-26). Bounded batches
    keep peak memory flat and keep each Docling call far away from the
    worker-level external timeout. Default 24 pages; override with
    PDF_PAGE_PATCH_BATCH_PAGES.
    """
    import os

    try:
        value = int(os.environ.get("PDF_PAGE_PATCH_BATCH_PAGES", "24"))
    except ValueError:
        return 24
    return max(1, value)


def _relieve_patch_memory() -> None:
    """Best-effort memory release between patch batches (gc + torch caches)."""
    import gc

    try:
        gc.collect()
    except Exception:
        pass
    try:
        import torch  # type: ignore

        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _patch_pages_with_docling_ocr(
    pdf_path: Path, pages: List[int], *, attachment_key: str, meta_base: Dict[str, Any],
    chunk_namespace: str, patch_marker_key: str, unresolved_block_type: str,
    unresolved_text: Any,
) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], set[int]]:
    """Shared body for ``patch_scanned_pages_with_docling`` and
    ``patch_corrupted_pages_with_docling`` (E2c/E2d, note 77): re-render just
    the wanted pages and OCR them with Docling, independent of whatever made
    PyMuPDF's own extraction unusable on those pages.

    Both callers hit the same underlying failure mode from Docling's point of
    view: the *page image* is fine (or recoverable via ``_build_scanned_page_subset``'s
    embedded-image fallback), and only PyMuPDF's *text extraction* off of it is
    broken -- for scanned pages because there's no text layer at all, for
    corrupted pages because the text layer decodes to garbage (broken font
    ToUnicode/CMap mapping or OCR-derived noise from a prior scan). Re-OCRing
    the rendered page image with Docling bypasses whatever is broken in the
    original content stream / font tables, since Docling reads pixels, not the
    original (possibly-corrupt) text-extraction path.

    Returns ``(chunks, attempted_pages)``.  ``attempted_pages`` is every page
    Docling actually processed, regardless of whether it yielded usable text:
    a page Docling could render but found no (clean) text on is conclusive
    evidence there's nothing to recover there, not an unresolved gap -- forcing
    more OCR (or Mistral) on it risks producing garbled, index-polluting text
    rather than recovering anything real. Callers should treat every attempted
    page as resolved for their respective ratio gate, whether or not it
    produced a chunk. A page that yields no usable text still gets a synthetic
    marker chunk (``unresolved_block_type``, not silently dropped) so it stays
    visible in the document outline/structure browser rather than looking like
    a hole in the extraction. Text Docling *does* recover is screened by
    ``_looks_like_scanned_patch_ocr_noise``: OCR misreads (garbled,
    concatenated-word text) are common enough that "Docling produced a chunk"
    alone isn't sufficient evidence of real content -- a chunk that fails this
    check is treated the same as no text at all (dropped, page becomes the
    unresolved marker instead) rather than indexed as garbage.

    Pages are processed in bounded batches (``_patch_batch_size``, default 24,
    ``PDF_PAGE_PATCH_BATCH_PAGES``): building one 200dpi sub-PDF out of *every*
    wanted page and running Docling on it whole OOM-killed the ingestion run
    on AX5CRKJ6's 208 corrupted pages (2026-07-26). Each batch builds its own
    sub-PDF, runs Docling, collects the results, deletes the sub-PDF, and
    releases memory before the next batch. A batch that raises (Docling crash,
    render failure) doesn't abort the remaining batches: its pages are treated
    as attempted-but-unresolved -- they get the same ``unresolved_block_type``
    marker chunks (visible, not garbage, not silently dropped) and count as
    attempted for the caller's ratio gate, exactly like a page Docling
    processed but found no usable text on.
    """
    import os
    import sys

    wanted = sorted({int(page) for page in pages if int(page) > 0})
    batch_size = _patch_batch_size()
    batches = [wanted[i:i + batch_size] for i in range(0, len(wanted), batch_size)]

    output: List[Tuple[str, str, Dict[str, Any]]] = []
    attempted_pages: set[int] = set()
    pages_with_text: set[int] = set()
    patch_id_occurrences: Dict[str, int] = defaultdict(int)
    for batch_index, batch in enumerate(batches):
        if len(batches) > 1:
            print(
                f"[PROGRESS]   ↳ page patch batch {batch_index + 1}/{len(batches)} "
                f"({len(batch)} page(s), attachment={attachment_key})",
                file=sys.__stderr__,
            )
        try:
            sub_path, sub_to_original = _build_scanned_page_subset(pdf_path, batch)
            try:
                chunks, _quality = extract_chunks_from_pdf_with_docling(
                    Path(sub_path), attachment_key, dict(meta_base),
                )
            finally:
                os.unlink(sub_path)
        except Exception as exc:
            # Fail-open per batch: this batch's pages become
            # attempted-but-unresolved markers below; later batches still run.
            print(
                f"[WARN] page patch batch {batch_index + 1}/{len(batches)} failed "
                f"(pages {batch[0]}-{batch[-1]}, attachment={attachment_key}): {exc}",
                file=sys.__stderr__,
            )
            attempted_pages.update(batch)
            continue
        attempted_pages.update(sub_to_original.values())
        for chunk_id, text, metadata in chunks:
            # Docling can occasionally emit two distinct layout blocks with
            # the same source-relative identifier.  Preserve both blocks: the
            # deterministic occurrence suffix is part of the patch identity,
            # while the batch namespace still distinguishes sub-PDF restarts.
            patch_id = f"{attachment_key}:{chunk_namespace}:b{batch_index}:{chunk_id}"
            occurrence = patch_id_occurrences[patch_id]
            patch_id_occurrences[patch_id] += 1
            if occurrence:
                patch_id = f"{patch_id}:dup{occurrence}"
            if _looks_like_scanned_patch_ocr_noise(text):
                continue
            md = dict(metadata)
            original_page = sub_to_original.get(int(md.get("page") or 0), int(md.get("page") or 0))
            md["page"] = original_page
            md[patch_marker_key] = "docling"
            # Batch index in the id namespace: every batch's sub-PDF restarts
            # page numbering at p1, so raw sub-PDF chunk ids repeat across
            # batches and would collide without it.
            output.append((
                patch_id, text, md,
            ))
            pages_with_text.add(original_page)
        if len(batches) > 1:
            _relieve_patch_memory()

    for page_number in sorted(attempted_pages - pages_with_text):
        md = dict(meta_base)
        md["page"] = page_number
        md[patch_marker_key] = "docling"
        md["block_type"] = unresolved_block_type
        md["zone"] = "body"
        text = (
            unresolved_text(page_number) if callable(unresolved_text)
            else unresolved_text.format(page=page_number)
        )
        output.append((
            f"{attachment_key}:{chunk_namespace}:{unresolved_block_type}:p{page_number}",
            text,
            md,
        ))
    return output, attempted_pages


def patch_scanned_pages_with_docling(
    pdf_path: Path, scanned_pages: List[int], *, attachment_key: str, meta_base: Dict[str, Any],
) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], set[int]]:
    """OCR just a PDF's scanned (image-only) pages with Docling (E2c, note 77).

    For an otherwise clean, embedded-text PDF that has a small number of
    scanned/image-only pages (figure plates, a scanned dedication page, etc.),
    sending the *whole* document to Mistral OCR is wasteful.  This patches only
    the scanned pages via a Docling sub-PDF pass so the caller can splice the
    recovered text into the PyMuPDF-extracted chunk list and clear the
    scanned-page gate that otherwise blocks the PyMuPDF fast-path and AI-TOC
    fast path (``pymupdf_fast_path_passes`` / ``try_ai_toc_fast_path``).  Unlike
    the reference-section patch, this keeps all recovered content (these are
    ordinary body pages, not a specific zone).  Uses ``_build_scanned_page_subset``
    (not the shared ``_docling_subset_pages``) because scanned pages need the
    broken-content-stream recovery fallback described there; reference pages
    (E2b) are ordinary text pages that don't hit this failure mode.

    A page that yields no usable text gets a synthetic ``block_type="figure"``
    marker chunk (user decision 2026-07-25): the page has no text layer at
    all, so "Docling tried and found nothing" reads as a photo/illustration,
    not a gap. See ``_patch_pages_with_docling_ocr`` for the shared mechanics
    (also used by ``patch_corrupted_pages_with_docling``).
    """
    return _patch_pages_with_docling_ocr(
        pdf_path, scanned_pages, attachment_key=attachment_key, meta_base=meta_base,
        chunk_namespace="scanpatch", patch_marker_key="scanned_page_patch",
        unresolved_block_type="figure",
        unresolved_text=lambda page: f"[Figure page {page}]",
    )


def patch_corrupted_pages_with_docling(
    pdf_path: Path, corrupted_pages: List[int], *, attachment_key: str, meta_base: Dict[str, Any],
    chunk_namespace: str = "corruptpatch",
) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], set[int]]:
    """Re-OCR just a PDF's text-corrupted pages with Docling (E2d, note 77).

    ``corrupted_pages`` (``pdf_extract.analyze_text_quality``'s ``is_corrupted``
    per page) means PyMuPDF *did* extract text there, but it's garbled:
    either a font-encoding mismatch (glyph IDs map to the wrong Unicode code
    points -- ``corruption_type="extraction_failure"``) or OCR/linguistic
    nonsense already baked into the PDF's text layer
    (``corruption_type="content_corruption"``). Crucially, in both cases the
    *page renders fine* -- the glyph outlines used for rendering don't depend
    on the broken ToUnicode/CMap table used for text extraction -- so
    re-rendering the page and OCRing the pixels with Docling recovers real
    text instead of the PyMuPDF mojibake (verified on ``AX5CRKJ6`` p.2: garbled
    PyMuPDF text like "先めえつ試尋試..." became coherent Japanese passages
    after this patch, 2026-07-26). Unlike scanned pages, a corrupted page
    already produced (bad) chunks upstream, so the caller must replace those
    chunks with this function's output rather than merely append to them.

    Pages Docling can't recover clean text from get a synthetic
    ``block_type="corrupted_unresolved"`` marker chunk (not silently dropped,
    and not indexed as garbage either) -- unlike a scanned page's
    ``block_type="figure"`` marker, an unresolved corrupted page is not
    necessarily a figure: it's a body-text page that resisted repair, so the
    marker says exactly that instead of implying "this was never text"
    (user decision 2026-07-26). See ``_patch_pages_with_docling_ocr`` for the
    shared mechanics (also used by ``patch_scanned_pages_with_docling``).
    """
    return _patch_pages_with_docling_ocr(
        pdf_path, corrupted_pages, attachment_key=attachment_key, meta_base=meta_base,
        chunk_namespace=chunk_namespace, patch_marker_key="corrupted_page_patch",
        unresolved_block_type="corrupted_unresolved",
        unresolved_text=lambda page: f"[Corrupted page {page} — unresolved]",
    )


__all__ = [
    "docling_provenance", "extract_chunks_from_pdf_with_docling",
    "extract_reference_sections_with_docling", "patch_scanned_pages_with_docling",
    "patch_corrupted_pages_with_docling",
    "get_docling_converter", "normalize_docling_label", "zone_for_docling_item",
]
