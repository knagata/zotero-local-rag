from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import compare_summary_models


class SummaryComparisonTests(unittest.TestCase):
    def test_zero_generated_evidence_is_accepted(self):
        result = {
            "status": "generated", "llm_summary": "資料の内容を要約した。",
            "verification": {"total_generated": 0, "discard_rate": 0.0},
        }
        self.assertEqual(compare_summary_models._classify(result), "accepted")

    def test_checkpoint_reuses_matching_input_and_invalidates_changed_input(self):
        section = {
            "section_id": "c0", "chapter": "Chapter",
            "chunks": [{"id": "a", "text": "original text", "metadata": {}}],
        }
        generated = {
            "section_id": "c0", "chapter": "Chapter", "status": "generated",
            "model": "deepseek:deepseek-v4-pro", "extractive_summary": "original text",
            "llm_summary": "要約", "cases": [], "chapter_authors": [],
            "first_publication_note": None,
            "verification": {"total_generated": 0, "discard_rate": 0.0},
            "classification": "accepted",
        }
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            compare_summary_models.build_summaries, "_excluded_from_llm", return_value=(False, None)
        ), patch.object(
            compare_summary_models, "get_item_chunks", return_value=section["chunks"]
        ), patch.object(
            compare_summary_models.build_summaries, "split_sections", return_value=[section]
        ), patch.object(
            compare_summary_models, "_compare_section", return_value=generated
        ) as compare:
            checkpoint_dir = Path(tmp)
            first = compare_summary_models.compare_item(
                "ITEM", max_sections=1, checkpoint_dir=checkpoint_dir,
                llm_spec="deepseek:deepseek-v4-pro",
            )
            second = compare_summary_models.compare_item(
                "ITEM", max_sections=1, checkpoint_dir=checkpoint_dir,
                llm_spec="deepseek:deepseek-v4-pro",
            )
            self.assertEqual(compare.call_count, 1)
            self.assertEqual(first, second)

            section["chunks"][0]["text"] = "changed text"
            compare_summary_models.compare_item(
                "ITEM", max_sections=1, checkpoint_dir=checkpoint_dir,
                llm_spec="deepseek:deepseek-v4-pro",
            )
            self.assertEqual(compare.call_count, 2)


if __name__ == "__main__":
    unittest.main()
