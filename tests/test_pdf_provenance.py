"""Tests for PDF source classification and deterministic text-defect detection.

Thresholds asserted here come from the measured library distribution recorded
in dev-notes/current/79_embedding_gates.md, so the cases are pinned to the
values that actually separated real documents rather than to invented ones.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pdf_provenance import (  # noqa: E402
    BORN_DIGITAL, SCANNED_NO_TEXT, SCANNED_OCR_LAYER,
    classify_pdf_source, detect_text_defects,
)


def _make_pdf(path: Path, pages: list[dict]) -> None:
    """Build a PDF where each page is {"text": str} and/or {"image": True}."""
    doc = fitz.open()
    for spec in pages:
        page = doc.new_page(width=400, height=600)
        if spec.get("image"):
            # A single image covering the whole page, i.e. a scanned page.
            pix = fitz.Pixmap(fitz.csGRAY, fitz.IRect(0, 0, 80, 120), False)
            pix.clear_with(200)
            page.insert_image(page.rect, pixmap=pix)
        if spec.get("text"):
            page.insert_textbox(
                fitz.Rect(10, 10, 390, 590), spec["text"], fontsize=7,
            )
    doc.save(str(path))
    doc.close()


BODY = ("The quick brown fox jumps over the lazy dog and then continues along "
        "the riverbank until the evening light fades entirely away. " * 4)


class ClassifyPdfSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="pdf-provenance-")
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_text_only_document_is_born_digital(self):
        path = self.tmp / "digital.pdf"
        _make_pdf(path, [{"text": BODY} for _ in range(20)])
        result = classify_pdf_source(path)
        self.assertEqual(result.kind, BORN_DIGITAL)
        self.assertFalse(result.text_layer_is_derived)

    def test_image_only_document_is_scan_without_text(self):
        path = self.tmp / "scan.pdf"
        _make_pdf(path, [{"image": True} for _ in range(20)])
        result = classify_pdf_source(path)
        self.assertEqual(result.kind, SCANNED_NO_TEXT)
        self.assertTrue(result.text_layer_is_derived)

    def test_image_pages_carrying_text_are_a_scan_with_an_ocr_layer(self):
        # Every page is an image *and* carries text: the signature of a scan
        # that has been through OCR.  The library's own measurement found 67
        # such documents, none of which the old is_scanned flag caught.
        path = self.tmp / "ocr.pdf"
        _make_pdf(path, [{"image": True, "text": BODY} for _ in range(20)])
        result = classify_pdf_source(path)
        self.assertEqual(result.kind, SCANNED_OCR_LAYER)
        self.assertTrue(result.text_layer_is_derived)

    def test_plate_heavy_book_with_typeset_front_matter_is_born_digital(self):
        # An art book: front matter is real typeset text, the plates are
        # full-page images.  Judged on the image ratio alone this looks like a
        # scan; the pure-text front matter is what distinguishes it.
        path = self.tmp / "art.pdf"
        _make_pdf(
            path,
            [{"text": BODY} for _ in range(6)] + [{"image": True} for _ in range(40)],
        )
        result = classify_pdf_source(path)
        self.assertEqual(result.kind, BORN_DIGITAL)
        self.assertGreater(result.pure_text_page_ratio, 0.15)

    def test_front_loaded_sampling_reaches_front_matter_of_a_long_book(self):
        # The front matter is 6 pages out of 300.  An evenly spread sample
        # would hit it at most once and misfile the book as a scan.
        path = self.tmp / "long_art.pdf"
        _make_pdf(
            path,
            [{"text": BODY} for _ in range(6)] + [{"image": True} for _ in range(294)],
        )
        result = classify_pdf_source(path)
        self.assertEqual(result.kind, BORN_DIGITAL)


class DetectTextDefectsTests(unittest.TestCase):
    def _latin_filler(self, count: int) -> str:
        return ("alpha beta gamma delta epsilon zeta eta theta iota kappa " * (count // 10 + 1))

    def test_letter_spacing_is_detected_above_threshold(self):
        spaced = "p r o b l e m s   m e a n i n g   a c t i o n " * 200
        result = detect_text_defects(spaced + self._latin_filler(2000))
        self.assertIn("letter_spacing", result["text_defects"])
        self.assertGreater(result["letter_spacing_ratio"], 0.05)

    def test_clean_latin_text_reports_no_defect(self):
        result = detect_text_defects(self._latin_filler(3000))
        self.assertEqual(result["text_defects"], [])
        self.assertTrue(result["text_defect_scope_applicable"])

    def test_standalone_a_and_i_do_not_count_as_letter_spacing(self):
        text = ("a cat and i saw a dog and i left a note " * 500)
        result = detect_text_defects(text)
        self.assertEqual(result["text_defects"], [])

    def test_dropped_ligature_words_are_detected(self):
        text = (self._latin_filler(3000)
                + " sofware afer ofen lef thef shif craf " * 40)
        result = detect_text_defects(text)
        self.assertIn("dropped_ligature", result["text_defects"])
        self.assertGreater(result["dropped_ligature_count"], 100)

    def test_preserved_ligature_codepoints_are_not_reported(self):
        # U+FB01 and friends are already decomposed by clean_extracted_text's
        # NFKC pass, so they must not be flagged here as needing re-extraction.
        result = detect_text_defects(self._latin_filler(3000) + " ﬁrst ﬂow ")
        self.assertEqual(result["text_defects"], [])

    def test_cjk_document_is_out_of_scope_for_letter_spacing(self):
        # Sparse Latin tokens in a Japanese document are mostly initials, so
        # the ratio is meaningless there: two real documents measured 38.8%
        # and 84.5% off ~200 tokens.  The guard must exclude them.
        text = "庭園と住居のありやうと見せかた見えかた" * 400 + " a b c d e f g h " * 20
        result = detect_text_defects(text)
        self.assertFalse(result["text_defect_scope_applicable"])
        self.assertEqual(result["text_defects"], [])

    def test_short_latin_text_is_out_of_scope(self):
        result = detect_text_defects("p r o b l e m s")
        self.assertFalse(result["text_defect_scope_applicable"])
        self.assertEqual(result["text_defects"], [])


if __name__ == "__main__":
    unittest.main()
