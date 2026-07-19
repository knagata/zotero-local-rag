from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import show_citation_graph
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
        db_relations.replace_item_case_annotations("ITEM", [{
            "section_id": "w0", "cases": [{
                "description": "Grounded observation", "quality_status": "partial",
                "evidence": [{"field_name": "description", "chunk_id": "chunk-1",
                              "evidence_quote": "Grounded observation"}],
            }],
        }], model="deepseek:test")

    def tearDown(self):
        self.db_patch.stop()
        db_relations._db_initialized = False
        self.tempdir.cleanup()

    def test_overview_sections_cases_and_evidence_routes(self):
        overview = _json(show_citation_graph._route_node_insights("ITEM"))
        self.assertEqual(overview["sections"]["count"], 1)
        sections = _json(show_citation_graph._route_node_sections("ITEM", "", "", 50))
        self.assertEqual(sections["items"][0]["section_id"], "w0")
        cases = _json(show_citation_graph._route_node_cases(
            "ITEM", "partial", "", "", "", 20,
        ))
        case_id = cases["items"][0]["case_id"]
        evidence = _json(show_citation_graph._route_case_evidence(case_id))
        self.assertEqual(evidence["evidence"][0]["chunk_id"], "chunk-1")

    def test_invalid_status_is_a_client_error(self):
        response = show_citation_graph._route_node_cases(
            "ITEM", "unknown", "", "", "", 20,
        )
        self.assertEqual(response.status_code, 400)

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

    def test_quality_report_route_queues_without_hiding(self):
        case_id = db_relations.get_case_annotations("ITEM")[0]["case_id"]
        response = show_citation_graph._route_quality_report(
            show_citation_graph._QualityReportRequest(
                target_type="case", case_id=case_id, reason="unsupported_field",
                details="The saved field is not supported by this source quote.",
                evidence_chunk_ids=["chunk-1"],
            )
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(db_relations.get_case_quality_reports("pending")[0]["reporter"], "citation-graph")
        self.assertEqual(len(db_relations.get_case_annotations("ITEM")), 1)

    def test_generated_html_contains_accessible_lazy_insights_ui(self):
        html = show_citation_graph._build_sigma_html(
            n_items=1, n_nodes=1, n_edges=0, n_citer=0, n_ref=0,
            palette=show_citation_graph._PALETTE,
            css_root=show_citation_graph._CSS_ROOT,
            js_theme=show_citation_graph._JS_THEME,
        )
        self.assertIn('role="tablist" aria-label="資料の詳細"', html)
        self.assertIn("/api/node/sections?key=", html)
        self.assertIn("/api/case/evidence?case_id=", html)
        self.assertIn('data-insight-tab="processing"', html)
        self.assertIn('aria-modal="true"', html)
        self.assertIn("var MIN_W = 280, MAX_W = 700", html)


if __name__ == "__main__":
    unittest.main()
