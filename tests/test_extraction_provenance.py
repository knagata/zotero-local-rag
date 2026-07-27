from __future__ import annotations

import json
import unittest

from src.extraction_engine import resolve_extraction_engine, summarize_extraction_quality


class ResolveExtractionEngineTests(unittest.TestCase):
    def test_existing_engine_is_preserved(self):
        # EPUB/HTML DOM paths (and adapter engines) already stamp their own
        # extraction_engine; that must win over anything derived from quality_info.
        self.assertEqual(
            resolve_extraction_engine({"parser": "docling"}, "epub_dom"), "epub_dom",
        )
        self.assertEqual(
            resolve_extraction_engine({"parser": "docling"}, "html_dom"), "html_dom",
        )

    def test_ndlocr_lite_parser_is_normalized(self):
        self.assertEqual(
            resolve_extraction_engine({"parser": "ndlocr-lite"}, None), "ndlocr_lite",
        )

    def test_fallback_engine_is_preferred_over_parser(self):
        quality_info = {"parser": "pymupdf-attempt", "fallback_engine": "mistral_ocr"}
        self.assertEqual(resolve_extraction_engine(quality_info, None), "mistral_ocr")

    def test_plain_pymupdf_defaults_when_nothing_present(self):
        # Plain PyMuPDF quality_info has no "parser" key at all.
        self.assertEqual(resolve_extraction_engine({}, None), "pymupdf")
        self.assertEqual(resolve_extraction_engine({"is_scanned": False}, None), "pymupdf")

    def test_existing_falsy_string_is_not_treated_as_set(self):
        self.assertEqual(resolve_extraction_engine({"parser": "docling"}, ""), "docling")

    def test_docling_and_yomitoku_parsers_pass_through_unchanged(self):
        self.assertEqual(resolve_extraction_engine({"parser": "docling"}, None), "docling")
        self.assertEqual(resolve_extraction_engine({"parser": "yomitoku"}, None), "yomitoku")


class SummarizeExtractionQualityTests(unittest.TestCase):
    def test_only_present_allow_listed_keys_are_included(self):
        quality_info = {
            "parser": "docling",
            "is_scanned": True,
            "some_unrelated_internal_field": "should not leak",
            "structure_v3": {"status": "ok"},
        }
        summary = json.loads(summarize_extraction_quality(quality_info))
        self.assertEqual(summary, {"parser": "docling", "is_scanned": True})
        self.assertNotIn("some_unrelated_internal_field", summary)
        self.assertNotIn("structure_v3", summary)

    def test_result_is_valid_json_string_of_scalars(self):
        quality_info = {
            "parser": "ndlocr-lite", "total_pages": 12, "scanned_ratio": 0.5,
            "has_outline": False, "dpi": 300,
        }
        raw = summarize_extraction_quality(quality_info)
        self.assertIsInstance(raw, str)
        summary = json.loads(raw)
        for value in summary.values():
            self.assertIsInstance(value, (str, int, float, bool, type(None)))

    def test_empty_quality_info_yields_empty_json_object(self):
        self.assertEqual(summarize_extraction_quality({}), "{}")

    def test_uses_compact_separators_and_preserves_unicode(self):
        quality_info = {"parser": "docling", "model": "モデル名"}
        raw = summarize_extraction_quality(quality_info)
        self.assertNotIn(", ", raw)
        self.assertNotIn(": ", raw)
        self.assertIn("モデル名", raw)


if __name__ == "__main__":
    unittest.main()
