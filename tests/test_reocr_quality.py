from __future__ import annotations

import unittest

from src.reocr_quality import (
    candidate_assessment, evaluate_adoption_gate, same_engine_version,
    summary_report_flags_text_disorder, text_metrics,
)


class ReocrQualityTests(unittest.TestCase):
    def test_scanned_source_provenance_survives_clean_local_ocr_metrics(self):
        assessment = candidate_assessment(
            quality={"total_pages": 2, "scanned_ratio": 0, "source_class": "scanned_no_text"},
            chunks=[
                {"text": "Readable recovered English body text. " * 20, "metadata": {"page": page}}
                for page in (1, 2)
            ],
            structure_status="exact",
            summary_reports=[],
            current_engine="docling",
            current_version="1",
            target_engine="mistral_ocr",
            target_version="2",
        )
        self.assertTrue(assessment["candidate"])
        self.assertIn("scan_source_without_native_text", assessment["reasons"])

    def test_gate_passes_exact_initial_thresholds(self):
        result = evaluate_adoption_gate(
            {"characters": 100, "gibberish_rate": 0.25},
            {"characters": 80, "gibberish_rate": 0.25, "repeat_artifacts": 0, "page_coverage": 1},
        )
        self.assertTrue(result["passed"])

    def test_gate_reports_every_deterministic_failure(self):
        result = evaluate_adoption_gate(
            {"characters": 100, "gibberish_rate": 0},
            {"characters": 151, "gibberish_rate": 0.5, "repeat_artifacts": 1, "page_coverage": .5},
        )
        self.assertFalse(result["passed"])
        self.assertEqual(
            {failure["code"] for failure in result["failures"]},
            {"page_coverage", "repeat_artifacts", "gibberish_worsened", "character_ratio"},
        )

    def test_text_metrics_detects_repeat_and_coverage(self):
        metrics = text_metrics([
            {"text": "x" * 20, "metadata": {"page": 1}},
            {"text": "normal readable text " * 5, "metadata": {"page": 2}},
        ], total_pages=2)
        self.assertEqual(metrics["repeat_artifacts"], 1)
        self.assertEqual(metrics["page_coverage"], 1)

    def test_candidate_does_not_use_unrelated_summary_complaint(self):
        assessment = candidate_assessment(
            quality={"total_pages": 1},
            chunks=[{"text": "readable English text " * 10, "metadata": {"page": 1, "lang": "en"}}],
            structure_status="exact",
            summary_reports=[{"reason": "wrong_number", "details": "The summary says 10 but source says 12."}],
            current_engine="pymupdf", current_version="1",
            target_engine="docling", target_version="2",
        )
        self.assertFalse(assessment["candidate"])

    def test_summary_report_requires_explicit_text_disorder(self):
        self.assertTrue(summary_report_flags_text_disorder({"details": "OCR reading order is broken"}))
        self.assertFalse(summary_report_flags_text_disorder({"details": "The interpretation is misleading"}))

    def test_same_engine_requires_known_matching_version(self):
        self.assertTrue(same_engine_version("Docling", "1", "docling", "1"))
        self.assertFalse(same_engine_version("docling", None, "docling", "1"))


if __name__ == "__main__":
    unittest.main()
