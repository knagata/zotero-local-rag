from __future__ import annotations

import argparse
import unittest

from scripts.list_reocr_candidates import build_candidates, render_markdown
from scripts.run_reocr_queue import compare_result, validate_adoption_args


def _no_status(_item_key: str):
    return []


class ReocrCandidateTests(unittest.TestCase):
    def test_uses_v3_quality_structure_and_summary_report_signals(self):
        manifest = {"files": {"ATT": {"quality": {
            "scanned_ratio": 0.5, "total_pages": 2, "parser": "pymupdf",
            "parser_version": "1.24",
        }}}}
        chunks = [{
            "text": "短い", "metadata": {
                "attachmentKey": "ATT", "lang": "ja", "page": 1,
                "extraction_engine": "pymupdf", "extraction_version": "1.24",
            },
        }]
        rows = build_candidates(
            manifest, {"items": [{"sections": [{"cases": ["ignored"]}]}]},
            item_keys=["ITEM"], chunk_loader=lambda _: chunks,
            structure_loader=lambda _: {"status": "flat_fallback"},
            node_loader=lambda _: [], status_loader=_no_status,
            summary_reports=[{
                "item_key": "ITEM", "reason": "other",
                "details": "OCR文字化けにより原文乱れが見える",
            }],
            target_engine="docling", target_version="v3-adapter-1",
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["language"], "ja")
        self.assertEqual(rows[0]["current_engine"], "pymupdf")
        self.assertIn("scanned_pages", rows[0]["reasons"])
        self.assertIn("low_characters_per_page", rows[0]["reasons"])
        self.assertIn("flat_fallback", rows[0]["reasons"])
        self.assertIn("summary_quality_text_disorder", rows[0]["reasons"])
        self.assertNotIn("grounding", render_markdown(rows).casefold())

    def test_excludes_same_engine_and_version(self):
        manifest = {"files": {"ATT": {"quality": {"scanned_ratio": 1, "total_pages": 1}}}}
        chunks = [{
            "text": "ordinary text " * 20,
            "metadata": {
                "attachmentKey": "ATT", "lang": "en", "page": 1,
                "extraction_engine": "docling", "extraction_version": "v3-adapter-1",
            },
        }]
        rows = build_candidates(
            manifest, item_keys=["ITEM"], chunk_loader=lambda _: chunks,
            structure_loader=lambda _: {"status": "exact"}, node_loader=lambda _: [],
            status_loader=_no_status, target_engine="docling", target_version="v3-adapter-1",
        )
        self.assertEqual(rows, [])

    def test_engine_version_change_restores_candidate(self):
        manifest = {"files": {"ATT": {"quality": {"scanned_ratio": 1, "total_pages": 1}}}}
        chunks = [{
            "text": "ordinary text " * 20,
            "metadata": {
                "attachmentKey": "ATT", "lang": "en", "page": 1,
                "extraction_engine": "docling", "extraction_version": "old",
            },
        }]
        rows = build_candidates(
            manifest, item_keys=["ITEM"], chunk_loader=lambda _: chunks,
            structure_loader=lambda _: {"status": "exact"}, node_loader=lambda _: [],
            status_loader=_no_status, target_engine="docling", target_version="new",
        )
        self.assertEqual(len(rows), 1)


class ReocrQueueTests(unittest.TestCase):
    def test_compare_result_applies_gate_without_writes(self):
        old = [{
            "text": "ordinary readable source text " * 4, "metadata": {
                "attachmentKey": "ATT", "lang": "en", "page": 1,
            },
        }]
        prepared = {
            "engine": "docling", "version": "new", "quality": {"total_pages": 1},
            "structure_status": "recovered", "heading_count": 2,
            "blocks": [{
                "text": "ordinary readable output text " * 4,
                "metadata": {"lang": "en", "page": 1},
            }],
        }
        result = compare_result(
            {"item_key": "ITEM", "attachment_key": "ATT", "structure_status": "flat_fallback"},
            prepared, chunk_loader=lambda _: old,
        )
        self.assertTrue(result["quality_gate"]["passed"])
        self.assertTrue(result["structure"]["flat_fallback_resolved"])

    def test_force_adopt_requires_explicit_item(self):
        parser = argparse.ArgumentParser()
        args = argparse.Namespace(force_adopt=True, adopt=False, item=None, limit=1, results="x")
        with self.assertRaises(SystemExit):
            validate_adoption_args(args, parser)


if __name__ == "__main__":
    unittest.main()
