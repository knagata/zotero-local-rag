from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from src import pdf_extract


class HasOutlineQualityInfoTests(unittest.TestCase):
    """Feeds the EngineRegistry.select() fast-path gate
    (src/extraction_engine.pymupdf_fast_path_passes), which requires an
    embedded outline before treating PyMuPDF's own result as canonical."""

    def _extract(self, *, with_toc: bool) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.pdf"
            document = fitz.open()
            page = document.new_page()
            page.insert_textbox(
                fitz.Rect(50, 50, 500, 400),
                "Body text long enough to survive minimum-chunk-size filtering in the extractor.",
                fontsize=11,
            )
            if with_toc:
                document.set_toc([[1, "Chapter 1", 1]])
            document.save(path)
            document.close()
            with patch.object(pdf_extract, "PDF_OCR_FALLBACK", False):
                _chunks, quality = pdf_extract.extract_chunks_from_pdf(path, "ATT", {"title": "Test"})
        return quality

    def test_has_outline_true_when_pdf_carries_a_toc(self):
        self.assertTrue(self._extract(with_toc=True)["has_outline"])

    def test_has_outline_false_when_pdf_has_no_toc(self):
        self.assertFalse(self._extract(with_toc=False)["has_outline"])


if __name__ == "__main__":
    unittest.main()
