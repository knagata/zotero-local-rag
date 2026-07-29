from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from citation_graph import server as show_citation_graph
from src import db_relations


def _json(response):
    return json.loads(response.body.decode("utf-8"))


class CitationInsightsApiTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "relations.db"
        self.db_patch = patch.object(db_relations, "DB_PATH", str(self.db_path))
        self.db_patch.start()
        db_relations._db_initialized = False
        db_relations.save_item_summary("ITEM", "item summary", "deepseek:flash")
        db_relations.save_section_summary(
            "ITEM", "w0", "section summary", chapter="Chapter", model="deepseek:flash",
        )

    def tearDown(self):
        self.db_patch.stop()
        db_relations._db_initialized = False
        self.tempdir.cleanup()

    def test_overview_and_sections_routes(self):
        overview = _json(show_citation_graph._route_node_insights("ITEM"))
        self.assertEqual(overview["sections"]["count"], 1)
        sections = _json(show_citation_graph._route_node_sections("ITEM", "", "", 50))
        self.assertEqual(sections["items"][0]["section_id"], "w0")

    def test_abstract_route_uses_the_packaged_database_module(self):
        response = _json(show_citation_graph._route_node_abstract("ITEM"))
        self.assertEqual(response["summary"]["summary"], "item summary")

    def test_summary_generation_route_reaches_packaged_llm_module(self):
        request = show_citation_graph._SummaryRequest(
            item_key="MISSING", force=True,
        )
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
            response = show_citation_graph._route_generate_summary(request)
        self.assertEqual(response.status_code, 503)
        self.assertIn("DEEPSEEK_API_KEY", _json(response)["error"])

    def test_processing_status_distinguishes_degraded_and_blocked_work(self):
        db_relations.mark_artifact_status(
            "ITEM", "structure", "degraded", reason_code="flat_fallback",
            fallback_kind="contiguous_semantic_segments",
        )
        db_relations.mark_artifact_status(
            "ITEM", "summary", "blocked", reason_code="no_cloud",
        )
        response = _json(show_citation_graph._route_node_processing_status("ITEM"))
        self.assertEqual(response["overall"], "needs_attention")
        states = {row["artifact_type"]: row for row in response["artifacts"]}
        self.assertEqual(states["structure"]["fallback_kind"], "contiguous_semantic_segments")
        self.assertEqual(states["summary"]["reason_code"], "no_cloud")

    def test_outline_route_is_empty_until_structure_exists(self):
        outline = _json(show_citation_graph._route_node_outline("ITEM"))
        self.assertIsNone(outline["structure"])
        self.assertEqual(outline["nodes"], [])

    def test_quality_report_route_queues_without_hiding(self):
        response = show_citation_graph._route_quality_report(
            show_citation_graph._QualityReportRequest(
                target_type="item_summary", item_key="ITEM", reason="unsupported_claim",
                details="The saved summary claim is not supported by the source.",
                evidence_chunk_ids=["chunk-1"],
            )
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(db_relations.get_summary_quality_reports("pending")[0]["reporter"], "citation-graph")

    def test_generated_html_contains_accessible_lazy_insights_ui(self):
        html = show_citation_graph._build_sigma_html(
            n_items=1, n_nodes=1, n_edges=0, n_citer=0, n_ref=0,
            palette=show_citation_graph._PALETTE,
            css_root=show_citation_graph._CSS_ROOT,
            js_theme=show_citation_graph._JS_THEME,
        )
        self.assertIn('role="tablist" aria-label="資料の詳細"', html)
        self.assertIn("/api/node/sections?key=", html)
        self.assertNotIn('data-insight-tab="cases"', html)
        self.assertIn('data-insight-tab="processing"', html)
        self.assertIn('aria-modal="true"', html)
        self.assertIn("var MIN_W = 280, MAX_W = 700", html)
        self.assertIn('.replace(/"/g, \'&quot;\')', html)
        self.assertIn(".replace(/'/g, '&#39;')", html)
        self.assertIn("encodeURIComponent(doi)", html)


if __name__ == "__main__":
    unittest.main()
