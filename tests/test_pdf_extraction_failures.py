from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src import pdf_extract
from src.extraction_engine import (
    pymupdf_fast_path_passes,
    pymupdf_fast_path_rejection_reason,
)


class _Rect:
    x0 = 0.0
    y0 = 0.0
    x1 = 600.0
    y1 = 800.0
    width = 600.0


class _Page:
    rect = _Rect()

    def __init__(self, *, image: bool = False, text_failure: bool = False):
        self._image = image
        self._text_failure = text_failure

    def get_label(self):
        return ""

    def get_images(self):
        return [(1,)] if self._image else []

    def get_text(self, kind, sort=False):
        if self._text_failure:
            raise RuntimeError("simulated text extraction failure")
        text = "Readable first-page body text. " * 20
        if kind == "blocks":
            return [(50.0, 50.0, 550.0, 700.0, text, 0, 0)]
        return text


class _Document:
    page_count = 2

    def __init__(self, failure: str):
        self._failure = failure
        # This makes stale-page use observable: page 2 must not inherit this
        # image flag when its own loading or text extraction fails.
        self._first_page = _Page(image=True)
        self._second_page = _Page(text_failure=failure == "text")

    def load_page(self, index):
        if index == 0:
            return self._first_page
        if self._failure == "load":
            raise RuntimeError("simulated page loading failure")
        return self._second_page

    def close(self):
        pass


class PdfExtractionFailureTests(unittest.TestCase):
    def test_blank_page_requires_near_white_rendered_pixels(self):
        class Rendered:
            def __init__(self, samples):
                self.samples = samples

        white = Mock()
        white.get_pixmap.return_value = Rendered(bytes([255, 252, 250]))
        vector_content = Mock()
        vector_content.get_pixmap.return_value = Rendered(bytes([255, 249, 255]))
        broken = Mock()
        broken.get_pixmap.side_effect = RuntimeError("render failed")

        self.assertTrue(pdf_extract._rendered_page_is_visually_blank(white))
        self.assertFalse(pdf_extract._rendered_page_is_visually_blank(vector_content))
        self.assertFalse(pdf_extract._rendered_page_is_visually_blank(broken))

    def test_page_load_and_text_failures_are_not_classified_as_blank_or_scanned(self):
        for failure in ("load", "text"):
            with self.subTest(failure=failure), \
                 patch.object(pdf_extract, "get_pdf_toc", return_value=[]), \
                 patch.object(pdf_extract, "classify_pdf_source", return_value=None), \
                 patch.object(pdf_extract.fitz, "open", return_value=_Document(failure)), \
                 patch.object(pdf_extract, "PDF_OCR_FALLBACK", False):
                _chunks, quality = pdf_extract.extract_chunks_from_pdf(
                    Path("synthetic.pdf"), "ATT", {"title": "Synthetic"},
                )

            self.assertEqual(quality["extraction_failure_pages"], [2])
            self.assertEqual(quality["extraction_failure_ratio"], 0.5)
            self.assertNotIn(2, quality["scanned_pages"])
            self.assertNotIn(2, quality["empty_pages"])
            self.assertFalse(pymupdf_fast_path_passes(quality))
            self.assertEqual(
                pymupdf_fast_path_rejection_reason(quality),
                "extraction_failure_ratio_nonzero",
            )


if __name__ == "__main__":
    unittest.main()
