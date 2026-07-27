from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from src.epub_fallback import (
    build_fixed_layout_pdf, profile_epub, remap_mistral_chunks_to_epub,
)
from src.html_extract import extract_leaf_container_blocks


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
        images={"page.png": _png()},
    )
    assert profile_epub(path)["classification"] == "html_or_mixed"


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
