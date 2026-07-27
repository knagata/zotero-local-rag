from __future__ import annotations

import unittest

from src.v3_migration import is_ocr_derived, reuse_ocr_chunks_for_v3


class V3MigrationTests(unittest.TestCase):
    def test_ocr_detection_is_conservative(self):
        self.assertTrue(is_ocr_derived({"parser": "ndlocr-lite"}))
        self.assertTrue(is_ocr_derived({"parser": "pymupdf", "scanned_ratio": 0.2}))
        self.assertFalse(is_ocr_derived({"parser": "pymupdf", "scanned_ratio": 0}))

    def test_reuse_preserves_boundaries_and_records_provenance(self):
        chunks = [
            {"id": "old1", "text": "a" * 300,
             "metadata": {"attachmentKey": "ATT", "structure_path": ["One"], "zone": "body"}},
            {"id": "old2", "text": "b" * 300,
             "metadata": {"attachmentKey": "ATT", "structure_path": ["Two"], "zone": "body"}},
        ]
        result, quality = reuse_ocr_chunks_for_v3(
            chunks, "ATT", {"itemKey": "ITEM", "attachmentKey": "ATT"},
            original_quality={"parser": "ndlocr-lite", "parser_version": "1"},
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][2]["structure_path"], ["One"])
        self.assertEqual(result[0][2]["original_extraction_engine"], "ndlocr-lite")
        self.assertTrue(quality["ocr_text_reused"])


if __name__ == "__main__":
    unittest.main()
