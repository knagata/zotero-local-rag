from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from index_from_zotero import (  # noqa: E402
    _epub_ocr_quality_from_mapping,
)
from epub_fallback import add_fixed_layout_terminal_markers  # noqa: E402
from source_coverage import validate_source_coverage  # noqa: E402


def _mapping() -> dict:
    return {
        "pages": [
            {"pdf_page_index": 0, "spine_index": 4},
            {"pdf_page_index": 1, "spine_index": 8},
        ],
    }


def test_remapped_ocr_quality_uses_opf_spine_units_and_rejects_missing_output():
    chunks = [(
        "ATT:epub:spine4:para0", "Recovered first page text",
        {"chapter_index": 4},
    )]
    quality = _epub_ocr_quality_from_mapping(
        {"ocr_pages": [1], "missing_pages": [2], "total_pages": 2},
        chunks, _mapping(),
    )

    assert quality["expected_spines"] == [5, 9]
    assert quality["attempted_spines"] == [5]
    assert quality["text_spines"] == [5]
    assert quality["failed_spines"] == [9]
    assert validate_source_coverage(quality["source_coverage"])["passed"] is False


def test_remapped_ocr_quality_accepts_complete_spine_coverage():
    chunks = [
        ("ATT:epub:spine4:para0", "Recovered first page text", {"chapter_index": 4}),
        ("ATT:epub:spine8:para0", "Recovered second page text", {"chapter_index": 8}),
    ]
    quality = _epub_ocr_quality_from_mapping(
        {"ocr_pages": [1, 2], "total_pages": 2}, chunks, _mapping(),
    )

    assert validate_source_coverage(quality["source_coverage"])["passed"] is True


def test_terminal_marker_accounts_only_for_returned_empty_fixed_layout_page():
    mapping = {
        "pages": [
            {"pdf_page_index": 0, "spine_index": 0},
            {"pdf_page_index": 1, "spine_index": 1},
            {"pdf_page_index": 2, "spine_index": 2},
        ],
    }
    chunks = [("ATT:p1", "Recovered first page", {"page": 1})]
    updated, quality = add_fixed_layout_terminal_markers(
        chunks, {"ocr_pages": [1], "missing_pages": [2, 3]}, mapping,
        returned_pdf_pages={1, 2},
        attachment_key="ATT",
        metadata={"attachmentKey": "ATT"},
    )

    assert [row[2]["page"] for row in updated] == [1, 2]
    assert updated[1][2]["block_type"] == "figure"
    assert updated[1][2]["quality_uncertain_reason"] == "empty_ocr_output"
    assert quality["nontext_marker_pages"] == [2]
    assert quality["missing_pages"] == [3]
