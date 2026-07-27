"""Deterministic profiling and page-image derivation for fixed-layout EPUBs."""
from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup  # type: ignore


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _resolve(base: str, href: str) -> str:
    return posixpath.normpath(posixpath.join(posixpath.dirname(base), href.split("#", 1)[0]))


def _package(archive: zipfile.ZipFile) -> tuple[str, ET.Element]:
    container = ET.fromstring(archive.read("META-INF/container.xml"))
    rootfile = next(
        node for node in container.iter() if _local_name(node.tag) == "rootfile"
    )
    opf_path = str(rootfile.attrib["full-path"])
    return opf_path, ET.fromstring(archive.read(opf_path))


def _manifest_and_spine(
    archive: zipfile.ZipFile,
) -> tuple[str, dict[str, dict[str, str]], list[str], str]:
    opf_path, package = _package(archive)
    manifest: dict[str, dict[str, str]] = {}
    spine_ids: list[str] = []
    progression = "default"
    for node in package.iter():
        name = _local_name(node.tag)
        if name == "item" and node.attrib.get("id"):
            manifest[str(node.attrib["id"])] = dict(node.attrib)
        elif name == "spine":
            progression = str(node.attrib.get("page-progression-direction") or "default")
            spine_ids.extend(
                str(child.attrib["idref"])
                for child in node
                if _local_name(child.tag) == "itemref" and child.attrib.get("idref")
            )
    return opf_path, manifest, spine_ids, progression


def _document_image_href(raw: bytes, document_path: str) -> str | None:
    soup = BeautifulSoup(raw, "html.parser")
    candidates: list[str] = []
    for tag in soup.find_all(["img", "image", "object"]):
        href = tag.get("src") or tag.get("href") or tag.get("xlink:href") or tag.get("data")
        if href and not str(href).startswith(("data:", "http:", "https:")):
            candidates.append(_resolve(document_path, str(href)))
    # A fixed-layout wrapper normally has exactly one image.  More than one
    # is deliberately not guessed at here.
    return candidates[0] if len(set(candidates)) == 1 else None


def profile_epub(epub_path: Path) -> dict[str, Any]:
    """Classify an EPUB and resolve page images strictly in OPF spine order."""
    pages: list[dict[str, Any]] = []
    visible_chars = 0
    missing_images: list[int] = []
    duplicate_images: list[int] = []
    with zipfile.ZipFile(epub_path, "r") as archive:
        opf_path, manifest, spine_ids, progression = _manifest_and_spine(archive)
        names = set(archive.namelist())
        seen_images: set[str] = set()
        for spine_index, idref in enumerate(spine_ids):
            item = manifest.get(idref) or {}
            href = str(item.get("href") or "")
            document_path = _resolve(opf_path, href) if href else ""
            raw = archive.read(document_path) if document_path in names else b""
            soup = BeautifulSoup(raw, "html.parser")
            for node in soup(["script", "style", "nav"]):
                node.decompose()
            visible = " ".join(soup.get_text(" ", strip=True).split())
            visible_chars += len(visible)
            image_path = _document_image_href(raw, document_path) if raw else None
            image_exists = bool(image_path and image_path in names)
            if image_path and image_path in seen_images:
                duplicate_images.append(spine_index)
            if image_path:
                seen_images.add(image_path)
            if not image_exists:
                missing_images.append(spine_index)
            pages.append({
                "spine_index": spine_index,
                "idref": idref,
                "document_href": document_path,
                "image_href": image_path,
                "image_exists": image_exists,
                "locator": f"epub:spine{spine_index}",
            })

    count = len(pages)
    image_count = sum(1 for page in pages if page["image_exists"])
    ratio = image_count / count if count else 0.0
    text_per_page = visible_chars / count if count else 0.0
    fixed = count > 0 and ratio >= 0.8 and text_per_page < 200
    return {
        "classification": "fixed_layout_image" if fixed else "html_or_mixed",
        "spine_document_count": count,
        "page_image_count": image_count,
        "page_image_ratio": round(ratio, 6),
        "visible_text_chars": visible_chars,
        "visible_text_chars_per_page": round(text_per_page, 3),
        "page_progression_direction": progression,
        "missing_image_spine_indices": missing_images,
        "duplicate_image_spine_indices": duplicate_images,
        "pages": pages,
    }


def build_fixed_layout_pdf(
    epub_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Build an OCR derivative and return its exact PDF-page/spine mapping.

    Images are embedded at their native pixel dimensions without resampling.
    The function refuses incomplete or duplicate page mappings.
    """
    import fitz  # type: ignore
    from PIL import Image  # type: ignore

    profile = profile_epub(epub_path)
    if profile["classification"] != "fixed_layout_image":
        raise ValueError("EPUB is not classified as fixed_layout_image")
    if profile["missing_image_spine_indices"]:
        raise ValueError("fixed-layout EPUB has missing page images")
    if profile["duplicate_image_spine_indices"]:
        raise ValueError("fixed-layout EPUB reuses a page image")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    mapping: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(epub_path, "r") as archive:
            for pdf_page, page in enumerate(profile["pages"]):
                image_bytes = archive.read(str(page["image_href"]))
                with Image.open(BytesIO(image_bytes)) as image:
                    width, height = image.size
                pdf = document.new_page(width=float(width), height=float(height))
                pdf.insert_image(pdf.rect, stream=image_bytes, keep_proportion=False)
                mapping.append({
                    "pdf_page_index": pdf_page,
                    "spine_index": int(page["spine_index"]),
                    "locator": str(page["locator"]),
                    "image_href": str(page["image_href"]),
                })
        document.set_metadata({})
        document.save(
            str(output_path), garbage=4, deflate=True, clean=True,
            no_new_id=True, preserve_metadata=False,
        )
    finally:
        document.close()
    return {
        "source_path": str(epub_path),
        "source_sha256": hashlib.sha256(epub_path.read_bytes()).hexdigest(),
        "derivative_path": str(output_path),
        "derivative_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "page_progression_direction": profile["page_progression_direction"],
        "pages": mapping,
    }


def save_fixed_layout_derivative(
    epub_path: Path, cache_dir: Path, attachment_key: str,
) -> dict[str, Any]:
    """Create/reuse a source-fingerprinted PDF plus an atomic mapping sidecar."""
    source_sha256 = hashlib.sha256(epub_path.read_bytes()).hexdigest()
    stem = f"{attachment_key}-{source_sha256[:16]}"
    pdf_path = cache_dir / f"{stem}.pdf"
    mapping_path = cache_dir / f"{stem}.json"
    if pdf_path.is_file() and mapping_path.is_file():
        saved = json.loads(mapping_path.read_text(encoding="utf-8"))
        if (
            saved.get("source_sha256") == source_sha256
            and saved.get("derivative_sha256") == hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        ):
            return saved
    result = build_fixed_layout_pdf(epub_path, pdf_path)
    result["mapping_path"] = str(mapping_path)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{mapping_path.name}.", dir=str(mapping_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, mapping_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return result


def remap_ocr_chunks_to_epub(
    chunks: list[tuple[str, str, dict[str, Any]]],
    mapping: dict[str, Any],
    *,
    epub_path: Path,
) -> list[tuple[str, str, dict[str, Any]]]:
    """Restore fixed-layout OCR chunks to EPUB provenance and spine locators."""
    pages = {
        int(row["pdf_page_index"]) + 1: row
        for row in (mapping.get("pages") or []) if isinstance(row, dict)
    }
    remapped: list[tuple[str, str, dict[str, Any]]] = []
    for chunk_id, text, metadata in chunks:
        page = int(metadata.get("page") or 0)
        row = pages.get(page)
        if row is None:
            raise ValueError(f"OCR chunk page {page} has no EPUB spine mapping")
        spine_index = int(row["spine_index"])
        suffix = re.sub(r"^[^:]+:p\d+:", "", str(chunk_id))
        new_id = f"{metadata.get('attachmentKey')}:epub:spine{spine_index}:{suffix}"
        md = dict(metadata)
        old_locator = str(md.get("locator") or "")
        locator_suffix = old_locator.split(":", 1)[1] if ":" in old_locator else old_locator
        md.update({
            "source_type": "epub",
            "path": str(epub_path),
            "pdf_path": str(epub_path),
            "derivative_pdf_path": str(mapping.get("derivative_path") or ""),
            "chapter_index": spine_index,
            "locator": f"epub:spine{spine_index}:{locator_suffix}",
            "image_href": str(row.get("image_href") or ""),
            "epub_source_sha256": str(mapping.get("source_sha256") or ""),
            "epub_derivative_sha256": str(mapping.get("derivative_sha256") or ""),
        })
        remapped.append((new_id, text, md))
    return remapped


# Compatibility for existing staged Mistral Batch results.
remap_mistral_chunks_to_epub = remap_ocr_chunks_to_epub


__all__ = [
    "build_fixed_layout_pdf", "profile_epub", "remap_ocr_chunks_to_epub",
    "remap_mistral_chunks_to_epub",
    "save_fixed_layout_derivative",
]
