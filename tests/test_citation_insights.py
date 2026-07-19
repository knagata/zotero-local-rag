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
        db_relations.replace_item_case_annotations("ITEM", [{
            "section_id": "w0",
            "cases": [
                {
                    "description": "Confirmed coastal practice",
                    "region": "Okinawa", "quality_status": "confirmed", "confidence": 0.95,
                    "evidence": [{"field_name": "description", "chunk_id": "chunk-1",
                                  "evidence_quote": "coastal practice"}],
                },
                {
                    "description": "Partial farming practice",
                    "group": "Farmers", "quality_status": "partial", "confidence": 0.8,
                    "evidence": [{"field_name": "description", "chunk_id": "chunk-2",
                                  "evidence_quote": "farming practice"}],
                },
                {
                    "description": "Candidate ritual",
                    "quality_status": "candidate", "confidence": 0.6,
                    "evidence": [{"field_name": "description", "chunk_id": "chunk-3",
                                  "evidence_quote": "candidate ritual"}],
                },
            ],
        }], model="deepseek:test")

    def tearDown(self):
        self.db_patch.stop()
        db_relations._db_initialized = False
        self.tempdir.cleanup()

    def test_overview_counts_and_generation_states(self):
        result = citation_insights.get_item_insights("ITEM")
        self.assertEqual(result["summary"]["kind"], "llm")
        self.assertEqual(result["sections"], {"status": "available", "count": 2})
        self.assertEqual(
            result["cases"]["counts"], {"confirmed": 1, "partial": 1, "candidate": 1},
        )
        self.assertEqual(result["cases"]["status"], "available")

    def test_sections_are_naturally_sorted_filtered_and_paged(self):
        first = citation_insights.list_sections("ITEM", limit=1)
        self.assertEqual(first["items"][0]["section_id"], "w0")
        self.assertEqual(first["next_cursor"], "1")
        second = citation_insights.list_sections("ITEM", cursor="1", limit=1)
        self.assertEqual(second["items"][0]["section_id"], "w10")
        filtered = citation_insights.list_sections("ITEM", query="later")
        self.assertEqual([row["section_id"] for row in filtered["items"]], ["w10"])

    def test_cases_filter_evidence_and_current_report_status(self):
        partial = citation_insights.list_cases("ITEM", statuses=["partial"])
        self.assertEqual(partial["total"], 1)
        case_id = partial["items"][0]["case_id"]
        evidence = citation_insights.get_case_evidence(case_id)
        self.assertEqual(evidence["evidence"][0]["chunk_id"], "chunk-2")
        db_relations.submit_case_quality_report(
            case_id=case_id, reason="unsupported_field",
            details="The group field is not grounded by this evidence.",
            evidence_chunk_ids=["chunk-2"], reporter="citation-graph",
        )
        reported = citation_insights.list_cases("ITEM", statuses=["partial"])
        self.assertEqual(reported["items"][0]["report_status"], "pending")

    def test_disabled_current_case_is_excluded(self):
        row = citation_insights.list_cases("ITEM", statuses=["candidate"])["items"][0]
        report = db_relations.submit_case_quality_report(
            case_id=row["case_id"], reason="not_a_case",
            details="This is an abstract claim and not an empirical case.",
        )
        db_relations.resolve_case_quality_report(report["report_id"], "disable")
        self.assertEqual(citation_insights.list_cases("ITEM", statuses=["candidate"])["total"], 0)
        self.assertEqual(citation_insights.get_item_insights("ITEM")["cases"]["counts"]["candidate"], 0)
        with self.assertRaises(KeyError):
            citation_insights.get_case_evidence(row["case_id"])

    def test_processed_empty_is_distinct_from_not_processed(self):
        db_relations.replace_item_case_annotations("EMPTY", [], model="deepseek:test")
        self.assertEqual(
            citation_insights.get_item_insights("EMPTY")["cases"]["status"],
            "processed_empty",
        )
        self.assertEqual(
            citation_insights.get_item_insights("UNKNOWN")["cases"]["status"],
            "not_processed",
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
