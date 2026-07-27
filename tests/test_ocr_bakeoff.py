from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_ocr_bakeoff import _evaluation_source, main
from src.ocr_bakeoff import (
    SCORE_WEIGHTS, markdown_report, score_result, validate_manifest, validate_tree,
)


class OcrBakeoffScoringTests(unittest.TestCase):
    def setUp(self):
        self.truth = {
            "sample_id": "fixture",
            "regions": [
                {"id": "h", "text_anchor": "Chapter One", "order": 0,
                 "heading_path": ["Chapter One"], "zone": "body",
                 "kind": "heading", "locator_required": True, "bbox_required": True},
                {"id": "t", "text_anchor": "Revenue 2025", "order": 1,
                 "heading_path": ["Chapter One", "Results"], "zone": "body",
                 "kind": "table", "locator_required": True, "bbox_required": True,
                 "expected_text": "Revenue 2025"},
                {"id": "b", "text_anchor": "Smith 2020", "order": 2,
                 "zone": "bibliography", "kind": "bibliography",
                 "locator_required": True, "bbox_required": False},
            ],
        }
        self.result = {
            "blocks": [
                {"id": "1", "ordinal": 0, "text": "Chapter One", "metadata": {
                    "structure_path": ["Chapter One"], "zone": "body", "block_type": "heading",
                    "locator": "p1:b1", "bbox": [0, 0, 10, 10]}},
                {"id": "2", "ordinal": 1, "text": "Revenue 2025", "metadata": {
                    "structure_path": ["Chapter One", "Results"], "zone": "body", "block_type": "table",
                    "locator": "p2:b1", "bbox": [0, 0, 10, 10]}},
                {"id": "3", "ordinal": 2, "text": "Smith 2020", "metadata": {
                    "structure_path": ["References"], "zone": "bibliography", "block_type": "bibliography",
                    "locator": "p3:b1"}},
            ]
        }

    def test_perfect_structure_scores_one(self):
        scored = score_result(self.truth, self.result)
        self.assertEqual(scored["total_score"], 1.0)
        self.assertEqual(scored["matched_regions"], 3)
        self.assertAlmostEqual(sum(SCORE_WEIGHTS.values()), 1.0)

    def test_short_anchor_matches_inside_a_much_longer_paragraph_with_markdown_emphasis(self):
        # Regression: found 2026-07-21 testing Mistral OCR's no_outline sample.
        # A block wraps part of the anchor in markdown emphasis (`*...*`),
        # which breaks exact substring containment, and the block is a long
        # multi-sentence paragraph rather than a single-sentence block. The
        # naive full-string SequenceMatcher.ratio() fallback used to score
        # this near 0.09 (diluted by the paragraph's unrelated length) well
        # below the 0.6 threshold, silently dropping a correct match.
        truth = {
            "sample_id": "fixture",
            "regions": [{
                "id": "p", "text_anchor": "The Politics of Operations examines how particular operations",
                "order": 0, "zone": "body", "kind": "text",
                "locator_required": True, "bbox_required": False,
            }],
        }
        long_paragraph = (
            "*The Politics of Operations* examines how particular operations of capital "
            "'hit the ground' not simply to furnish an analysis of their local or wider "
            "effects but also to supply an analytical prism through which to investigate "
            "how their meshing, and conflicting, with other operations of capital remake "
            "the world. We imagine this as a means of excavating contemporary capitalism."
        )
        result = {"blocks": [{
            "id": "1", "ordinal": 0, "text": long_paragraph,
            "metadata": {"zone": "body", "block_type": "text", "locator": "p1:b1"},
        }]}
        scored = score_result(truth, result)
        self.assertEqual(scored["matched_regions"], 1)

    def test_wrong_order_and_zone_are_penalized_deterministically(self):
        damaged = json.loads(json.dumps(self.result))
        damaged["blocks"][1], damaged["blocks"][2] = damaged["blocks"][2], damaged["blocks"][1]
        damaged["blocks"][1]["ordinal"] = 1
        damaged["blocks"][2]["ordinal"] = 2
        damaged["blocks"][1]["metadata"]["zone"] = "body"
        first = score_result(self.truth, damaged)
        second = score_result(self.truth, damaged)
        self.assertEqual(first, second)
        self.assertLess(first["metrics"]["reading_order"], 1.0)
        self.assertLess(first["metrics"]["zone_classification"], 1.0)

    def test_tree_validation_finds_duplicates_non_monotonic_and_missing_locator(self):
        blocks = [
            {"id": "same", "ordinal": 1, "text": "a", "metadata": {"locator": "p1"}},
            {"id": "same", "ordinal": 1, "text": "b", "metadata": {}},
        ]
        errors = validate_tree(blocks)
        self.assertTrue(any("duplicate" in error for error in errors))
        self.assertTrue(any("strictly increasing" in error for error in errors))
        self.assertTrue(any("missing locator" in error for error in errors))

    def test_markdown_report_is_stable_and_contains_score(self):
        text = markdown_report({"runs": [{
            "sample_id": "fixture", "engine": "fake", "status": "completed",
            "score": {"total_score": 0.875},
        }]})
        self.assertIn("| fixture | fake | 0.875 | completed |", text)
        self.assertIn("`heading_hierarchy`: 22%", text)

    def test_manifest_rejects_committed_pdf_paths_and_duplicate_ids(self):
        sample = {
            "id": "one", "category": "embedded_text", "path_env": "OCR_BAKEOFF_ONE_PDF",
            "ground_truth": "annotations/one.json", "path": "/private/book.pdf",
        }
        with self.assertRaisesRegex(ValueError, "embeds a PDF path"):
            validate_manifest({"version": "ocr-bakeoff-v3.1", "samples": [sample]})
        sample.pop("path")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_manifest({"version": "ocr-bakeoff-v3.1", "samples": [sample, sample]})


class OcrBakeoffCliTests(unittest.TestCase):
    def test_evaluation_source_slices_declared_pages_without_changing_original(self):
        import fitz

        with tempfile.TemporaryDirectory() as tmp_value:
            root = Path(tmp_value)
            source = root / "source.pdf"
            document = fitz.open()
            for index in range(3):
                page = document.new_page()
                page.insert_text((72, 72), f"page {index + 1}")
            document.save(source)
            document.close()
            subset = _evaluation_source(source, {"id": "one", "pages": [2]}, root / "out")
            original_doc = fitz.open(source)
            subset_doc = fitz.open(subset)
            try:
                self.assertEqual(original_doc.page_count, 3)
                self.assertEqual(subset_doc.page_count, 1)
                self.assertIn("page 2", subset_doc[0].get_text())
            finally:
                original_doc.close()
                subset_doc.close()

    def test_dry_run_writes_reports_without_parsing_pdf(self):
        with tempfile.TemporaryDirectory() as tmp_value:
            tmp = Path(tmp_value)
            pdf = tmp / "sample.pdf"
            pdf.write_bytes(b"not parsed in dry run")
            truth = tmp / "truth.json"
            truth.write_text(json.dumps({
                "sample_id": "one", "regions": [{
                    "id": "r1", "text_anchor": "test", "order": 0,
                }],
            }), encoding="utf-8")
            manifest = tmp / "manifest.json"
            manifest.write_text(json.dumps({
                "version": "ocr-bakeoff-v3.1",
                "samples": [{
                    "id": "one", "category": "embedded_text", "path_env": "OCR_BAKEOFF_TEST_PDF",
                    "ground_truth": "annotations/one.json",
                }],
            }), encoding="utf-8")
            annotation_dir = tmp / "annotations"
            annotation_dir.mkdir()
            truth.rename(annotation_dir / "one.json")
            output = tmp / "output"
            with patch.dict(os.environ, {"OCR_BAKEOFF_TEST_PDF": str(pdf)}):
                rc = main([
                    "--manifest", str(manifest), "--engine", "pymupdf", "--sample", "one",
                    "--dry-run", "--output", str(output),
                ])
            self.assertEqual(rc, 0)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["runs"][0]["status"], "ready")
            self.assertTrue((output / "report.md").is_file())


if __name__ == "__main__":
    unittest.main()
