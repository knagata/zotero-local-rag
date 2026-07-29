from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import db_relations  # noqa: E402
import rag_mcp_server  # noqa: E402
from tests.v3_summary_fixtures import seed_v3_summaries  # noqa: E402


class McpRelationReportingTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "relations.db"
        self.db_patch = patch.object(db_relations, "DB_PATH", str(self.db_path))
        self.db_patch.start()
        db_relations._db_initialized = False
        db_relations.insert_reference(
            cited_paper_id="S2-WRONG", cited_title="Suspicious work", cited_year=2002,
            context_snippet=None, citing_item_key="ITEM", citing_chunk_id=None,
            similarity_distance=None, source="s2", s2_status="no_context",
        )
        fixture = seed_v3_summaries(
            db_relations, root_summary="A current summary.",
            sections=[(
                "w0", "Chapter", "A current section summary.", ["chunk-1"],
            )],
        )
        self.section_id = fixture["section_ids"]["w0"]

    def tearDown(self):
        self.db_patch.stop()
        db_relations._db_initialized = False
        self.tempdir.cleanup()

    def test_claude_can_report_but_cannot_decide(self):
        result = rag_mcp_server.report_citation_relation(
            "references:ITEM:S2-WRONG",
            "not_in_source",
            "The title is absent from the source bibliography.",
        )
        self.assertEqual(result["status"], "reported")
        self.assertEqual(result["report"]["status"], "pending")
        listed = rag_mcp_server.list_citation_relation_reports("pending")
        self.assertEqual(listed["report_count"], 1)
        self.assertIn("human", listed["review_note"].lower())
        # No MCP tool exposes review_relation_report; the relation remains visible.
        self.assertEqual(len(db_relations.get_reference_relations_for_item("ITEM")), 1)

    def test_report_requires_stable_key_and_concrete_details(self):
        malformed = rag_mcp_server.report_citation_relation(
            "ITEM:S2-WRONG", "other", "This is sufficiently detailed.",
        )
        self.assertEqual(malformed["status"], "error")
        vague = rag_mcp_server.report_citation_relation(
            "references:ITEM:S2-WRONG", "other", "odd",
        )
        self.assertEqual(vague["status"], "error")

    def test_claude_can_report_summary_problem_for_automated_triage(self):
        result = rag_mcp_server.report_summary_quality(
            "ITEM", "unsupported_claim",
            "The source chunk directly contradicts the summary claim.",
            section_id=self.section_id, evidence_chunk_ids=["chunk-1"],
        )
        self.assertEqual(result["status"], "reported")
        self.assertEqual(result["report"]["status"], "pending")
        listed = rag_mcp_server.list_summary_quality_reports("pending")
        self.assertEqual(listed["report_count"], 1)
        self.assertEqual(listed["reports"][0]["section_id"], self.section_id)

if __name__ == "__main__":
    unittest.main()
