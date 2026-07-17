from __future__ import annotations

import unittest

from scripts.list_reocr_candidates import build_candidates, render_markdown


class ReocrCandidateTests(unittest.TestCase):
    def test_combines_language_quality_and_grounding_signals(self):
        manifest = {"files": {"ATT": {"quality": {
            "is_scanned": False, "is_corrupted": True, "parser": "pymupdf",
        }}}}
        report = {"items": [{
            "item_key": "ITEM", "sections": [{
                "verification": {
                    "total_generated": 4, "total_discarded": 3, "suspicious_section": True,
                    "cases": {"reasons": {"evidence_not_in_chunk": 1}},
                },
                "cases": [{"evidence_quote": "前半 後半"}],
            }],
        }]}
        chunks = [
            {"text": "前半", "metadata": {"attachmentKey": "ATT", "lang": "ja"}},
            {"text": "後半", "metadata": {"attachmentKey": "ATT", "lang": "ja"}},
        ]
        rows = build_candidates(manifest, report, chunk_loader=lambda _: chunks)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["lang"], "ja")
        self.assertEqual(rows[0]["parser"], "pymupdf")
        self.assertEqual(rows[0]["discard_rate"], 0.75)
        self.assertEqual(rows[0]["evidence_not_in_chunk"], 1)
        self.assertEqual(rows[0]["cross_chunk_evidence"], 1)
        self.assertEqual(rows[0]["recommendation"], "benchmark_ja_ocr")
        self.assertIn("ITEM", render_markdown(rows))

    def test_omits_items_without_reocr_signals(self):
        manifest = {"files": {"ATT": {"quality": {}}}}
        report = {"items": [{"item_key": "ITEM", "sections": []}]}
        chunks = [{
            "text": "ordinary text", "metadata": {"attachmentKey": "ATT", "lang": "en"},
        }]
        self.assertEqual(build_candidates(manifest, report, chunk_loader=lambda _: chunks), [])


if __name__ == "__main__":
    unittest.main()
