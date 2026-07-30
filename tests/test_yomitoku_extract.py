from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from src.yomitoku_extract import extract_chunks_from_pdf_with_yomitoku


def _payload(paragraphs):
    result = Mock()
    result.model_dump.return_value = {"words": [], "paragraphs": paragraphs, "tables": []}
    return result, None, None


def _paragraph(order, text):
    return {
        "order": order, "contents": text, "role": "", "box": [0, 0, 10, 10],
        "direction": "horizontal",
    }


def test_ocr_pages_only_lists_pages_that_actually_produced_chunks():
    # 2026-07-30 regression: ocr_pages used to be claimed as every page of
    # the source PDF regardless of whether that page produced any text --
    # the same bug already found and fixed in mistral_ocr_extract.py and
    # ndlocr_extract.py. Page 2 here yields no paragraphs at all.
    fake_doc = MagicMock()
    fake_doc.page_count = 2
    fake_page = MagicMock()
    fake_page.get_label.return_value = ""
    fake_doc.__iter__.return_value = iter([fake_page, fake_page])

    analyzer_calls = {"n": 0}

    def fake_analyzer(_image):
        analyzer_calls["n"] += 1
        if analyzer_calls["n"] == 1:
            return _payload([_paragraph(0, "Real recovered text on page one.")])
        return _payload([])  # page two: nothing usable

    with (
        patch("src.yomitoku_extract.fitz.open", return_value=fake_doc),
        patch("src.yomitoku_extract._build_analyzer", return_value=fake_analyzer),
        patch("yomitoku.data.functions.load_pdf", return_value=[object(), object()]),
    ):
        chunks, quality = extract_chunks_from_pdf_with_yomitoku(
            Path("missing-fixture.pdf"), "ATT", {"itemKey": "ITEM"},
        )

    assert quality["ocr_pages"] == [1]
    assert quality["missing_pages"] == [2]
    assert {md["page"] for _cid, _text, md in chunks} == {1}
