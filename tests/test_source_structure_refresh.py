from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.source_structure_refresh import refresh_source_structure_metadata


def _old_rows():
    return [{
        "id": "ATTACH01:epub:block9:part0",
        "text": "unchanged indexed text",
        "metadata": {
            "attachmentKey": "ATTACH01", "source_type": "epub",
            "locator": "epub:spine2:block7", "chapter": "Old flat section",
        },
    }]


def test_epub_refresh_maps_by_stable_spine_block_without_replacing_text():
    fresh = [(
        "different-new-chunk-id", "different freshly chunked text",
        {"locator": "epub:spine2:block6", "locator_end": "epub:spine2:block8",
         "structure_path": ["Chapter", "Section"],
         "structure_roles": ["chapter", "section"], "chapter": "Section"},
    )]
    with patch("src.source_structure_refresh._epub_source_path", return_value=Path("book.epub")), \
            patch("src.source_structure_refresh.extract_chunks_from_epub_snapshot",
                  return_value=(fresh, {})):
        rows, reports = refresh_source_structure_metadata(_old_rows())

    assert rows[0]["id"] == _old_rows()[0]["id"]
    assert rows[0]["text"] == "unchanged indexed text"
    assert rows[0]["metadata"]["structure_path"] == ["Chapter", "Section"]
    assert rows[0]["metadata"]["structure_roles"] == ["chapter", "section"]
    assert reports[0]["metadata_changed"] == 1


def test_epub_refresh_rejects_unmapped_existing_chunk():
    fresh = [(
        "new", "text", {"locator": "epub:spine3:block1", "structure_path": ["Other"]},
    )]
    with patch("src.source_structure_refresh._epub_source_path", return_value=Path("book.epub")), \
            patch("src.source_structure_refresh.extract_chunks_from_epub_snapshot",
                  return_value=(fresh, {})):
        with pytest.raises(RuntimeError, match="no fresh EPUB structure match"):
            refresh_source_structure_metadata(_old_rows())


def test_epub_refresh_rejects_ambiguous_structure_for_one_locator():
    fresh = [
        ("new:1", "a", {"locator": "epub:spine2:block7", "structure_path": ["One"]}),
        ("new:2", "b", {"locator": "epub:spine2:block7", "structure_path": ["Two"]}),
    ]
    with patch("src.source_structure_refresh._epub_source_path", return_value=Path("book.epub")), \
            patch("src.source_structure_refresh.extract_chunks_from_epub_snapshot",
                  return_value=(fresh, {})):
        with pytest.raises(RuntimeError, match="ambiguous fresh EPUB structure"):
            refresh_source_structure_metadata(_old_rows())


def test_epub_refresh_uses_start_heading_when_old_chunk_crosses_new_boundary():
    rows = _old_rows()
    rows[0]["metadata"]["locator_end"] = "epub:spine2:block8"
    fresh = [
        ("new:1", "chapter", {"locator": "epub:spine2:block7",
                              "structure_path": ["Chapter"]}),
        ("new:2", "section", {"locator": "epub:spine2:block8",
                              "structure_path": ["Chapter", "Section"]}),
    ]
    with patch("src.source_structure_refresh._epub_source_path", return_value=Path("book.epub")), \
            patch("src.source_structure_refresh.extract_chunks_from_epub_snapshot",
                  return_value=(fresh, {})):
        refreshed, _reports = refresh_source_structure_metadata(rows)
    assert refreshed[0]["metadata"]["structure_path"] == ["Chapter"]


def test_pdf_refresh_reuses_existing_chunks_and_resolves_same_page_headings():
    rows = [
        {"id": f"PDFKEY01:p1:{index}", "text": text,
         "metadata": {"attachmentKey": "PDFKEY01", "source_type": "pdf", "page": 1,
                      "reading_order": index, "chapter": "Old flat heading"}}
        for index, text in enumerate(("Paper", "Introduction", "body argument"))
    ]
    with patch("src.source_structure_refresh._pdf_source_path", return_value=Path("paper.pdf")), \
            patch("src.source_structure_refresh.get_pdf_toc", return_value=[
                (1, "Paper", 1), (2, "Introduction", 1),
            ]):
        refreshed, reports = refresh_source_structure_metadata(rows)

    assert [row["text"] for row in refreshed] == [row["text"] for row in rows]
    assert refreshed[0]["metadata"]["structure_path"] == ["Paper"]
    assert refreshed[1]["metadata"]["structure_path"] == ["Paper", "Introduction"]
    assert refreshed[2]["metadata"]["structure_roles"] == ["chapter", "section"]
    assert reports[0]["outline_entries"] == 2


def test_pdf_refresh_clears_old_outline_metadata_when_toc_disappears():
    rows = [{
        "id": "PDFKEY01:p1:0", "text": "body",
        "metadata": {
            "attachmentKey": "PDFKEY01", "source_type": "pdf", "page": 1,
            "structure_path": ["Old chapter"], "structure_roles": ["chapter"],
            "chapter": "Old chapter", "section": "Old section", "zone": "bibliography",
        },
    }]
    with patch("src.source_structure_refresh._pdf_source_path", return_value=Path("paper.pdf")), \
            patch("src.source_structure_refresh.get_pdf_toc", return_value=[]):
        refreshed, reports = refresh_source_structure_metadata(rows)

    for key in ("structure_path", "structure_roles", "chapter", "section", "zone"):
        assert key not in refreshed[0]["metadata"]
    assert reports[0]["metadata_changed"] == 1


@pytest.mark.parametrize("intrinsic_zone", ["corrupted", "footnote"])
def test_pdf_refresh_preserves_intrinsic_zone_without_toc(intrinsic_zone):
    rows = [{
        "id": "PDFKEY01:p1:0", "text": "body",
        "metadata": {
            "attachmentKey": "PDFKEY01", "source_type": "pdf", "page": 1,
            "structure_path": ["Old chapter"], "zone": intrinsic_zone,
        },
    }]
    with patch("src.source_structure_refresh._pdf_source_path", return_value=Path("paper.pdf")), \
            patch("src.source_structure_refresh.get_pdf_toc", return_value=[]):
        refreshed, _reports = refresh_source_structure_metadata(rows)
    assert refreshed[0]["metadata"]["zone"] == intrinsic_zone


def test_pdf_refresh_clears_stale_zone_when_refreshed_path_is_empty():
    rows = [{
        "id": "PDFKEY01:p1:0", "text": "front matter before heading",
        "metadata": {
            "attachmentKey": "PDFKEY01", "source_type": "pdf", "page": 1,
            "structure_path": ["Bibliography"], "chapter": "Bibliography",
            "zone": "bibliography",
        },
    }]
    with patch("src.source_structure_refresh._pdf_source_path", return_value=Path("paper.pdf")), \
            patch("src.source_structure_refresh.get_pdf_toc", return_value=[
                (1, "Introduction", 2),
            ]), patch(
                "src.source_structure_refresh._resolve_record_structure_paths",
                return_value=[[]],
            ), patch(
                "src.source_structure_refresh.build_pdf_page_structure_path_lookup",
                return_value=lambda _page: [],
            ):
        refreshed, _reports = refresh_source_structure_metadata(rows)
    assert "zone" not in refreshed[0]["metadata"]
