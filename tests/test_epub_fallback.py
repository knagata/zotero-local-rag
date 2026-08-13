from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.epub_fallback import (
    build_epub_image_pdf, build_fixed_layout_pdf, profile_epub,
    remap_mistral_chunks_to_epub,
)
from src.html_extract import extract_chunks_from_epub_snapshot, extract_leaf_container_blocks
from src.source_coverage import validate_source_coverage


CONTAINER = """<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
 <rootfiles><rootfile full-path="OPS/package.opf"/></rootfiles>
</container>"""


def _epub(
    path: Path,
    *,
    documents: list[tuple[str, str]],
    images: dict[str, bytes] | None = None,
    progression: str = "default",
) -> Path:
    images = images or {}
    manifest = []
    spine = []
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("META-INF/container.xml", CONTAINER)
        for index, (name, html) in enumerate(documents):
            manifest.append(
                f'<item id="d{index}" href="{name}" media-type="application/xhtml+xml"/>'
            )
            spine.append(f'<itemref idref="d{index}"/>')
            archive.writestr(f"OPS/{name}", html)
        for index, (name, data) in enumerate(images.items()):
            manifest.append(f'<item id="i{index}" href="{name}" media-type="image/png"/>')
            archive.writestr(f"OPS/{name}", data)
        archive.writestr(
            "OPS/package.opf",
            f"""<package xmlns="http://www.idpf.org/2007/opf">
              <manifest>{''.join(manifest)}</manifest>
              <spine page-progression-direction="{progression}">{''.join(spine)}</spine>
            </package>""",
        )
    return path


def _png(width: int = 20, height: int = 30) -> bytes:
    from io import BytesIO
    from PIL import Image

    output = BytesIO()
    Image.new("RGB", (width, height), "white").save(output, format="PNG")
    return output.getvalue()


def test_leaf_div_fallback_owns_only_leaf_text_and_excludes_navigation():
    blocks = extract_leaf_container_blocks(
        """<html><body><h1>Chapter</h1>
        <div class="article"><div>First scholarly paragraph.</div>
        <div><span>Second scholarly paragraph.</span></div></div>
        <div class="navigation">Next page</div></body></html>"""
    )
    assert [row["text"] for row in blocks] == [
        "First scholarly paragraph.", "Second scholarly paragraph.",
    ]
    assert blocks[0]["structure_path"] == ["Chapter"]
    assert blocks[0]["extraction_engine"] == "epub_dom_leaf_fallback"


def test_epub_fragment_toc_paths_survive_full_extraction(tmp_path: Path):
    from ebooklib import epub

    path = tmp_path / "fragment-hierarchy.epub"
    book = epub.EpubBook()
    book.set_identifier("fragment-hierarchy")
    book.set_title("Fragment hierarchy")
    book.set_language("en")
    chapter = epub.EpubHtml(title="Chapter 1", file_name="chapter.xhtml", lang="en")
    prose = "Complete scholarly evidence for this hierarchy. " * 20
    chapter.content = f"""<html><body>
      <h1 id="chapter">Chapter 1</h1><p>{prose}</p>
      <h1 id="section-a">Section A</h1><p>{prose}</p>
      <h1 id="section-b">Section B</h1><p>{prose}</p>
    </body></html>"""
    book.add_item(chapter)
    chapter_link = epub.Link("chapter.xhtml#chapter", "Chapter 1", "chapter-link")
    book.toc = ((
        epub.Section("Part I"),
        ((chapter_link, (
            epub.Link("chapter.xhtml#section-a", "Section A", "section-a-link"),
            epub.Link("chapter.xhtml#section-b", "Section B", "section-b-link"),
        )),),
    ),)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", chapter]
    epub.write_epub(str(path), book)

    chunks, quality = extract_chunks_from_epub_snapshot(path, "ATTACH01", {})
    paths = {tuple(metadata.get("structure_path") or []) for _id, _text, metadata in chunks}
    assert ("Part I", "Chapter 1") in paths
    assert ("Part I", "Chapter 1", "Section A") in paths
    assert ("Part I", "Chapter 1", "Section B") in paths
    assert quality["text_spines"]


def test_epub_carries_toc_parent_across_unlinked_spine_documents(tmp_path: Path):
    from ebooklib import epub

    path = tmp_path / "split-chapter.epub"
    book = epub.EpubBook()
    book.set_identifier("split-chapter")
    book.set_title("Split chapter")
    book.set_language("en")
    prose = "Evidence continuing through one chapter across source files. " * 20
    first = epub.EpubHtml(title="Chapter", file_name="first.xhtml", lang="en")
    first.content = f"<html><body><h1 id='chapter'>Chapter 1</h1><p>{prose}</p></body></html>"
    second = epub.EpubHtml(title="Continuation", file_name="second.xhtml", lang="en")
    second.content = f"<html><body><h2>Section A</h2><p>{prose}</p></body></html>"
    bibliography = epub.EpubHtml(title="Bibliography", file_name="bibliography.xhtml", lang="en")
    bibliography.content = f"<html><body><h1>Bibliography</h1><p>{prose}</p></body></html>"
    book.add_item(first)
    book.add_item(second)
    book.add_item(bibliography)
    book.toc = ((
        epub.Section("Part I"),
        (epub.Link("first.xhtml#chapter", "Chapter 1", "chapter-link"),),
    ),)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", first, second, bibliography]
    epub.write_epub(str(path), book)

    chunks, _quality = extract_chunks_from_epub_snapshot(path, "ATTACH02", {})
    paths = {
        tuple(metadata.get("structure_path") or [])
        for _id, _text, metadata in chunks
    }
    assert ("Part I", "Chapter 1", "Section A") in paths
    assert ("Bibliography",) in paths
    assert not any(path[-1:] == ("Bibliography",) and len(path) > 1 for path in paths)


def test_profile_uses_spine_order_and_preserves_rtl(tmp_path: Path):
    path = _epub(
        tmp_path / "rtl.epub",
        documents=[
            ("z.xhtml", '<html><body><img src="images/z.png"/></body></html>'),
            ("a.xhtml", '<html><body><img src="images/a.png"/></body></html>'),
        ],
        images={"images/z.png": _png(), "images/a.png": _png()},
        progression="rtl",
    )
    profile = profile_epub(path)
    assert profile["classification"] == "fixed_layout_image"
    assert profile["page_progression_direction"] == "rtl"
    assert [page["image_href"] for page in profile["pages"]] == [
        "OPS/images/z.png", "OPS/images/a.png",
    ]


def test_profile_does_not_classify_mixed_text_image_epub(tmp_path: Path):
    article = "Readable article text. " * 50
    path = _epub(
        tmp_path / "mixed.epub",
        documents=[
            ("one.xhtml", f"<html><body><p>{article}</p></body></html>"),
            ("two.xhtml", '<html><body><img src="page.png"/></body></html>'),
        ],
        images={"page.png": _png(1100, 1800)},
    )
    assert profile_epub(path)["classification"] == "html_or_mixed"


def test_epub_records_failed_spine_and_rejects_partial_dom_success(tmp_path: Path):
    first = " ".join([
        "The first complete chapter explains the historical argument in careful detail."
    ] * 8)
    second = " ".join([
        "The second chapter would have supplied the missing comparative evidence."
    ] * 8)
    path = _epub(
        tmp_path / "partially-failing.epub",
        documents=[
            ("one.xhtml", f"<html><body><p>{first}</p></body></html>"),
            ("two.xhtml", f"<html><body><p>{second}</p></body></html>"),
        ],
    )
    from src import html_extract

    actual = html_extract._extract_epub_document_blocks

    def fail_only_second(raw_html: str, **kwargs):
        if "second chapter" in raw_html:
            raise RuntimeError("synthetic chapter parser failure")
        return actual(raw_html, **kwargs)

    with patch.object(html_extract, "_extract_epub_document_blocks", side_effect=fail_only_second):
        chunks, quality = extract_chunks_from_epub_snapshot(path, "ATT", {})

    assert chunks
    assert quality["expected_spines"] == [1, 2]
    assert quality["attempted_spines"] == [1, 2]
    assert quality["text_spines"] == [1]
    assert quality["failed_spines"] == [2]
    assert quality["failure_reason"] == "epub_spine_coverage_incomplete"
    verdict = validate_source_coverage(quality["source_coverage"])
    assert verdict["passed"] is False
    assert "source_unit_failures" in verdict["reasons"]


def test_short_nontext_free_navigation_spine_is_accounted_as_blank(tmp_path: Path):
    article = "Readable source chapter with complete scholarly context. " * 20
    path = _epub(
        tmp_path / "short-nav.epub",
        documents=[
            ("text.xhtml", f"<html><body><p>{article}</p></body></html>"),
            ("nav.xhtml", "<html><body><nav>Contents</nav></body></html>"),
        ],
    )
    from src import html_extract

    actual = html_extract._extract_epub_document_blocks

    def omit_navigation(raw_html: str, **kwargs):
        if "Contents" in raw_html:
            return []
        return actual(raw_html, **kwargs)

    with patch.object(html_extract, "_extract_epub_document_blocks", side_effect=omit_navigation):
        chunks, quality = extract_chunks_from_epub_snapshot(path, "ATT", {})

    assert chunks
    assert quality["blank_spines"] == [2]
    assert quality["failed_spines"] == []
    assert validate_source_coverage(quality["source_coverage"])["passed"] is True


def test_structured_epub_uses_dom_caption_without_ocring_its_image(tmp_path: Path):
    article = " ".join([
        "Readable scholarly chapter text establishes the argument with citations and context."
    ] * 8)
    caption = (
        "Plate seven records the surviving manuscript page and its archival shelfmark."
    )
    path = _epub(
        tmp_path / "mixed-image.epub",
        documents=[
            ("text.xhtml", f"<html><body><p>{article}</p></body></html>"),
            (
                "image.xhtml",
                f'<html><body><p>{caption}</p><img src="page.png"/></body></html>',
            ),
        ],
        images={"page.png": _png(1100, 1800)},
    )
    chunks, quality = extract_chunks_from_epub_snapshot(path, "ATT", {})

    assert chunks
    assert quality["epub_profile"]["image_primary_spine_indices"] == []
    assert any(
        row[2].get("chapter_index") == 1 and caption in row[1]
        for row in chunks
    )
    assert quality["failed_spines"] == []
    assert validate_source_coverage(quality["source_coverage"])["passed"] is True


@pytest.mark.parametrize("dimensions", [(33, 51), (334, 254)])
def test_mixed_epub_logo_or_illustration_does_not_trigger_ocr_or_lose_short_text(
    tmp_path: Path, dimensions: tuple[int, int],
):
    article = "Readable source chapter with complete scholarly context. " * 20
    short_text = "Publisher mark"
    path = _epub(
        tmp_path / f"ordinary-image-{dimensions[0]}.epub",
        documents=[
            ("text.xhtml", f"<html><body><p>{article}</p></body></html>"),
            (
                "mark.xhtml",
                f'<html><body><p>{short_text}</p><img src="mark.png"/></body></html>',
            ),
        ],
        images={"mark.png": _png(*dimensions)},
    )

    chunks, quality = extract_chunks_from_epub_snapshot(path, "ATT", {})

    assert quality["epub_profile"]["classification"] == "html_or_mixed"
    assert quality["epub_profile"]["image_primary_spine_indices"] == []
    assert any(
        row[2].get("chapter_index") == 1 and short_text in row[1]
        for row in chunks
    )
    assert quality["failed_spines"] == []
    assert validate_source_coverage(quality["source_coverage"])["passed"] is True


def test_structured_epub_image_only_cover_is_ignored_without_ocr(tmp_path: Path):
    article = "Readable source chapter with complete scholarly context. " * 20
    path = _epub(
        tmp_path / "unreadable-image.epub",
        documents=[
            ("text.xhtml", f"<html><body><p>{article}</p></body></html>"),
            ("image.xhtml", '<html><body><img src="page.png"/></body></html>'),
        ],
        images={"page.png": _png(1100, 1800)},
    )

    chunks, quality = extract_chunks_from_epub_snapshot(path, "ATT", {})

    assert chunks
    assert quality["epub_profile"]["classification"] == "html_or_mixed"
    assert quality["epub_profile"]["image_primary_spine_indices"] == []
    assert quality["ignored_image_spines"] == [2]
    assert quality["blank_spines"] == [2]
    assert quality["failed_spines"] == []
    assert validate_source_coverage(quality["source_coverage"])["passed"] is True


def test_structured_epub_multiple_image_spine_is_ignored_without_ocr(tmp_path: Path):
    article = "Readable source chapter. " * 20
    path = _epub(
        tmp_path / "multi-image-spine.epub",
        documents=[
            ("text.xhtml", f"<html><body><p>{article}</p></body></html>"),
            (
                "plates.xhtml",
                '<html><body><img src="one.png"/><img src="two.png"/></body></html>',
            ),
        ],
        images={"one.png": _png(), "two.png": _png()},
    )

    chunks, quality = extract_chunks_from_epub_snapshot(path, "ATT", {})

    assert chunks
    assert quality["ignored_image_spines"] == [2]
    assert quality["blank_spines"] == [2]
    assert quality["failed_spines"] == []
    assert validate_source_coverage(quality["source_coverage"])["passed"] is True


def test_fixed_layout_epub_never_returns_wrapper_dom_as_canonical_chunks(tmp_path: Path):
    incidental_wrapper_text = "Long image caption metadata. " * 9
    path = _epub(
        tmp_path / "fixed-wrapper-text.epub",
        documents=[
            (
                "one.xhtml",
                f'<html><body><p>{incidental_wrapper_text}</p>'
                '<img src="one.png"/></body></html>',
            ),
            ("two.xhtml", '<html><body><p>Page two</p><img src="two.png"/></body></html>'),
        ],
        images={"one.png": _png(), "two.png": _png()},
    )
    chunks, quality = extract_chunks_from_epub_snapshot(path, "ATT", {})

    assert chunks == []
    assert quality["epub_profile"]["classification"] == "fixed_layout_image"
    assert quality["epub_profile"]["image_primary_spine_indices"] == [1]
    assert quality["failure_reason"] == "fixed_layout_epub_requires_ocr"
    assert quality["failed_spines"] == [1, 2]
    assert validate_source_coverage(quality["source_coverage"])["passed"] is False


@pytest.mark.parametrize("mode", ["missing", "duplicate"])
def test_derivative_refuses_incomplete_or_duplicate_pages(tmp_path: Path, mode: str):
    second = "missing.png" if mode == "missing" else "page.png"
    images = {"page.png": _png()}
    path = _epub(
        tmp_path / f"{mode}.epub",
        documents=[
            ("one.xhtml", '<html><body><img src="page.png"/></body></html>'),
            ("two.xhtml", f'<html><body><img src="{second}"/></body></html>'),
        ],
        images=images,
    )
    with pytest.raises(ValueError):
        build_fixed_layout_pdf(path, tmp_path / f"{mode}.pdf")


def test_derivative_mapping_is_exact_and_repeatable(tmp_path: Path):
    path = _epub(
        tmp_path / "pages.epub",
        documents=[
            ("one.xhtml", '<html><body><img src="one.png"/></body></html>'),
            ("two.xhtml", '<html><body><img src="two.png"/></body></html>'),
        ],
        images={"one.png": _png(20, 30), "two.png": _png(40, 50)},
    )
    first = build_fixed_layout_pdf(path, tmp_path / "first.pdf")
    second = build_fixed_layout_pdf(path, tmp_path / "second.pdf")
    assert first["derivative_sha256"] == second["derivative_sha256"]
    assert [row["locator"] for row in first["pages"]] == [
        "epub:spine0", "epub:spine1",
    ]


def test_mixed_derivative_contains_only_requested_image_spines(tmp_path: Path):
    article = "Readable source chapter. " * 20
    path = _epub(
        tmp_path / "mixed-subset.epub",
        documents=[
            ("text.xhtml", f"<html><body><p>{article}</p></body></html>"),
            ("image.xhtml", '<html><body><img src="one.png"/></body></html>'),
            ("image2.xhtml", '<html><body><img src="two.png"/></body></html>'),
        ],
        images={"one.png": _png(20, 30), "two.png": _png(40, 50)},
    )
    mapping = build_epub_image_pdf(
        path, tmp_path / "subset.pdf", spine_indices=[2, 1],
    )

    assert mapping["requested_spine_indices"] == [1, 2]
    assert [
        (row["pdf_page_index"], row["spine_index"])
        for row in mapping["pages"]
    ] == [(0, 1), (1, 2)]


def test_mixed_derivative_refuses_image_reused_outside_selection(tmp_path: Path):
    path = _epub(
        tmp_path / "ambiguous-subset.epub",
        documents=[
            ("one.xhtml", '<html><body><img src="page.png"/></body></html>'),
            ("two.xhtml", '<html><body><img src="page.png"/></body></html>'),
        ],
        images={"page.png": _png()},
    )

    with pytest.raises(ValueError, match="reuses a page image"):
        build_epub_image_pdf(
            path, tmp_path / "ambiguous.pdf", spine_indices=[0],
        )


def test_mistral_chunks_are_remapped_to_epub_spine(tmp_path: Path):
    mapping = {
        "source_sha256": "source", "derivative_sha256": "derived",
        "derivative_path": "/cache/book.pdf",
        "pages": [{
            "pdf_page_index": 0, "spine_index": 7,
            "locator": "epub:spine7", "image_href": "OPS/page7.jpg",
        }],
    }
    chunks = [(
        "ATT:p1:para2:part0", "Body text",
        {"attachmentKey": "ATT", "page": 1, "locator": "p1:para2", "source_type": "pdf"},
    )]
    remapped = remap_mistral_chunks_to_epub(
        chunks, mapping, epub_path=tmp_path / "book.epub",
    )
    chunk_id, _, metadata = remapped[0]
    assert chunk_id == "ATT:epub:spine7:para2:part0"
    assert metadata["locator"] == "epub:spine7:para2"
    assert metadata["source_type"] == "epub"
    assert metadata["image_href"] == "OPS/page7.jpg"


def test_remap_refuses_duplicate_mapping_rows(tmp_path: Path):
    mapping = {
        "pages": [
            {"pdf_page_index": 0, "spine_index": 1},
            {"pdf_page_index": 1, "spine_index": 1},
        ],
    }
    with pytest.raises(ValueError, match="repeats"):
        remap_mistral_chunks_to_epub(
            [], mapping, epub_path=tmp_path / "book.epub",
        )
