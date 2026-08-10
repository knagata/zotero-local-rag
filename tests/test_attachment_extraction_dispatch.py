"""Deterministic contracts for ordinary attachment extractor dispatch."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import index_from_zotero as module  # noqa: E402


def _dispatch(source_type: str, extract_pdf: Mock | None = None):
    pdf = extract_pdf or Mock(
        return_value=module.PdfExtraction(
            [("pdf:1", "pdf text", {"source_type": "pdf"})],
            {"parser": "pymupdf"},
        )
    )
    result = module._extract_attachment_by_source_type(
        source_type=source_type,
        file_path=Path(f"synthetic.{source_type}"),
        attachment_key="ATT",
        metadata={"itemKey": "ITEM"},
        extract_pdf=pdf,
    )
    return result, pdf


def test_html_dispatch_calls_only_the_html_extractor():
    expected = (
        [("html:1", "html text", {"source_type": "html"})],
        {"parser": "html"},
    )
    with (
        patch.object(module, "extract_chunks_from_html_snapshot", return_value=expected) as html,
        patch.object(module, "extract_chunks_from_epub_snapshot") as epub,
    ):
        result, pdf = _dispatch("html")

    assert result.chunks == expected[0]
    assert result.quality == expected[1]
    html.assert_called_once_with(
        Path("synthetic.html"), "ATT", {"itemKey": "ITEM"},
    )
    epub.assert_not_called()
    pdf.assert_not_called()


def test_epub_dispatch_calls_only_the_epub_extractor():
    expected = (
        [("epub:1", "epub text", {"source_type": "epub"})],
        {"parser": "epub"},
    )
    with (
        patch.object(module, "extract_chunks_from_html_snapshot") as html,
        patch.object(module, "extract_chunks_from_epub_snapshot", return_value=expected) as epub,
    ):
        result, pdf = _dispatch("epub")

    assert result.chunks == expected[0]
    assert result.quality == expected[1]
    epub.assert_called_once_with(
        Path("synthetic.epub"), "ATT", {"itemKey": "ITEM"},
    )
    html.assert_not_called()
    pdf.assert_not_called()


def test_pdf_dispatch_preserves_the_orchestrator_result_and_deferred_state():
    deferred = module.PdfExtraction([], {}, deferred=True)
    pdf = Mock(return_value=deferred)
    with (
        patch.object(module, "extract_chunks_from_html_snapshot") as html,
        patch.object(module, "extract_chunks_from_epub_snapshot") as epub,
    ):
        result, called_pdf = _dispatch("pdf", pdf)

    assert result is deferred
    called_pdf.assert_called_once_with()
    html.assert_not_called()
    epub.assert_not_called()
