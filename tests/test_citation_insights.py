from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from citation_graph import insights as citation_insights
from src import db_relations
from tests.v3_summary_fixtures import seed_v3_summaries


class CitationInsightsTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "relations.db"
        self.db_patch = patch.object(db_relations, "DB_PATH", str(self.db_path))
        self.db_patch.start()
        db_relations._db_initialized = False
        fixture = seed_v3_summaries(
            db_relations,
            sections=[
                ("w0", "Chapter 1", "first section summary", ["chunk-1", "chunk-2"]),
                ("w10", "Chapter 10", "later section summary", ["chunk-10"]),
            ],
        )
        self.section_ids = fixture["section_ids"]

    def tearDown(self):
        self.db_patch.stop()
        db_relations._db_initialized = False
        self.tempdir.cleanup()

    def test_overview_counts_and_generation_states(self):
        result = citation_insights.get_item_insights("ITEM")
        self.assertEqual(result["summary"]["kind"], "llm")
        self.assertEqual(result["sections"], {"status": "available", "count": 2})

    def test_summary_trust_exposes_sentence_verification_and_labels_legacy_rows(self):
        legacy = citation_insights.get_item_insights("ITEM")["summary"]
        self.assertEqual(legacy["trust"]["level"], "legacy")
        db_relations.save_document_node_summary(
            legacy["node_id"], "ITEM", legacy["summary"], summary_kind="llm",
            model=legacy["model"], prompt_version=legacy["prompt_version"],
            source_fingerprint=legacy["source_fingerprint"],
            source_chunk_count=legacy["source_chunk_count"], source_chars=legacy["source_chars"],
            quality_status="candidate", input_scope={"verification": {
                "accepted": False, "generated_sentences": 3,
                "kept_sentences": 1, "discarded_sentences": 2,
                "discard_rate": 0.6667,
            }},
        )
        trust = citation_insights.get_item_insights("ITEM")["summary"]["trust"]
        self.assertEqual(trust["level"], "limited")
        self.assertEqual((trust["kept_sentences"], trust["generated_sentences"]), (1, 3))

    def test_summary_trust_distinguishes_verified_unavailable_and_absent(self):
        self.assertIsNone(citation_insights._summary_trust(None))
        self.assertEqual(
            citation_insights._summary_trust({"summary_kind": "extractive"})["level"],
            "unavailable",
        )
        trust = citation_insights._summary_trust({
            "summary_kind": "llm", "input_scope": {"verification": {
                "accepted": True, "generated_sentences": 2, "kept_sentences": 2,
            }},
        })
        self.assertEqual((trust["level"], trust["label"]), ("verified", "根拠確認済み"))

    def test_summary_trust_does_not_overstate_candidate_with_accepted_final_reduction(self):
        trust = citation_insights._summary_trust({
            "summary_kind": "llm", "quality_status": "candidate",
            "input_scope": {"verification": {
                "accepted": True, "generated_sentences": 3, "kept_sentences": 3,
            }},
        })

        self.assertEqual((trust["level"], trust["label"]), ("limited", "限定的"))

    def test_sections_are_naturally_sorted_filtered_and_paged(self):
        first = citation_insights.list_sections("ITEM", limit=1)
        self.assertEqual(first["items"][0]["section_id"], self.section_ids["w0"])
        self.assertEqual(first["next_cursor"], "1")
        second = citation_insights.list_sections("ITEM", cursor="1", limit=1)
        self.assertEqual(second["items"][0]["section_id"], self.section_ids["w10"])
        filtered = citation_insights.list_sections("ITEM", query="later")
        self.assertEqual(
            [row["section_id"] for row in filtered["items"]],
            [self.section_ids["w10"]],
        )

    def test_sections_treat_malformed_legacy_input_scope_as_unverified(self):
        conn = db_relations.get_db_connection()
        try:
            conn.execute(
                "UPDATE document_node_summaries SET input_scope_json='{' WHERE node_id=?",
                (self.section_ids["w0"],),
            )
            conn.commit()
        finally:
            conn.close()
        row = citation_insights.list_sections("ITEM")["items"][0]
        self.assertEqual(row["trust"]["level"], "legacy")

    def test_processed_empty_is_distinct_from_not_processed(self):
        db_relations.mark_artifact_status("EMPTY", "summary", "empty")
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
            source = citation_insights.get_section_source(
                "ITEM", self.section_ids["w0"],
            )
        self.assertEqual([row["chunk_id"] for row in source["chunks"]], ["chunk-1", "chunk-2"])
        self.assertEqual(source["chunks"][0]["page"], 3)

    def test_outline_loads_all_summary_parts_in_one_query(self):
        section_id = self.section_ids["w0"]
        db_relations.replace_document_node_summary_parts(
            section_id,
            [{
                "child_node_ids": ["child-a"],
                "summary": "intermediate reduction",
                "model": "deepseek:test",
            }],
            prompt_version="test",
            source_fingerprint="fixture",
        )
        with patch.object(
            db_relations,
            "get_document_node_summary_parts_for_nodes",
            wraps=db_relations.get_document_node_summary_parts_for_nodes,
        ) as batch, patch.object(
            db_relations,
            "get_document_node_summary_parts",
            side_effect=AssertionError("per-node summary-part query"),
        ):
            outline = citation_insights.get_document_outline("ITEM")
        batch.assert_called_once()
        row = next(node for node in outline["nodes"] if node["node_id"] == section_id)
        self.assertEqual(row["summary_parts"][0]["child_node_ids"], ["child-a"])

    def test_outline_hides_internal_roots_and_untitled_segments(self):
        outline = citation_insights.get_document_outline("ITEM")
        self.assertTrue(outline["nodes"])
        self.assertFalse(any(
            node["node_type"] in {"item_root", "semantic_segment"}
            for node in outline["nodes"]
        ))


if __name__ == "__main__":
    unittest.main()
