from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts import compare_summary_models


class SummaryComparisonTests(unittest.TestCase):
    def test_parallel_comparison_preserves_section_order(self):
        sections = [
            {"section_id": "s0", "chapter": "A", "chunks": []},
            {"section_id": "s1", "chapter": "B", "chunks": []},
        ]

        def compare(section):
            return {"section_id": section["section_id"], "status": "generated"}

        with patch.object(
            compare_summary_models.build_summaries, "_excluded_from_llm",
            return_value=(False, None),
        ), patch.object(
            compare_summary_models, "get_item_chunks", return_value=[]
        ), patch.object(
            compare_summary_models.build_summaries, "split_sections", return_value=sections
        ), patch.object(compare_summary_models, "_compare_section", side_effect=compare):
            result = compare_summary_models.compare_item(
                "ITEM", max_sections=2, workers=2,
            )

        self.assertEqual(
            [section["section_id"] for section in result["sections"]], ["s0", "s1"]
        )


if __name__ == "__main__":
    unittest.main()
