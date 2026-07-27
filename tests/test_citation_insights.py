from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import citation_insights, db_relations


class CitationInsightsTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "relations.db"
        self.db_patch = patch.object(db_relations, "DB_PATH", str(self.db_path))
        self.db_patch.start()
        db_relations._db_initialized = False
        db_relations.save_item_summary("ITEM", "item summary", "deepseek:flash")
        db_relations.save_section_summary(
            "ITEM", "w0", "first section summary", chapter="Chapter 1",
            model="deepseek:flash", chunk_count=2,
        )
        db_relations.save_section_summary(
            "ITEM", "w10", "later section summary", chapter="Chapter 10",
            model="deepseek:flash", chunk_count=1,
        )

    def tearDown(self):
        self.db_patch.stop()
        db_relations._db_initialized = False
        self.tempdir.cleanup()

    def test_overview_counts_and_generation_states(self):
        result = citation_insights.get_item_insights("ITEM")
        self.assertEqual(result["summary"]["kind"], "llm")
        self.assertEqual(result["sections"], {"status": "available", "count": 2})

    def test_sections_are_naturally_sorted_filtered_and_paged(self):
        first = citation_insights.list_sections("ITEM", limit=1)
        self.assertEqual(first["items"][0]["section_id"], "w0")
        self.assertEqual(first["next_cursor"], "1")
        second = citation_insights.list_sections("ITEM", cursor="1", limit=1)
        self.assertEqual(second["items"][0]["section_id"], "w10")
        filtered = citation_insights.list_sections("ITEM", query="later")
        self.assertEqual([row["section_id"] for row in filtered["items"]], ["w10"])

    def test_processed_empty_is_distinct_from_not_processed(self):
        db_relations.mark_insight_generation_status("EMPTY", "sections", 0)
        self.assertEqual(
            citation_insights.get_item_insights("EMPTY")["sections"]["status"],
            "processed_empty",
        )

    def test_section_source_uses_summary_grouping(self):
        chunks = [
            {"id": "chunk-1", "text": "one", "metadata": {"chapter": "Chapter 1", "page": 3}},
            {"id": "chunk-2", "text": "two", "metadata": {"chapter": "Chapter 1", "page": 4}},
        ]
        with patch.object(citation_insights, "get_item_chunks", return_value=chunks):
            source = citation_insights.get_section_source("ITEM", "c0")
        self.assertEqual([row["chunk_id"] for row in source["chunks"]], ["chunk-1", "chunk-2"])
        self.assertEqual(source["chunks"][0]["page"], 3)


if __name__ == "__main__":
    unittest.main()
