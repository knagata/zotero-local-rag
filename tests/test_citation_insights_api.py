from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from citation_graph import server as show_citation_graph
from src import db_relations
from tests.v3_summary_fixtures import seed_v3_summaries


def _json(response):
    return json.loads(response.body.decode("utf-8"))


def _browser_document() -> str:
    """The HTML shell plus the CSS/JS the browser actually loads with it.

    The UI's markup and behaviour used to live entirely inside
    _build_sigma_html's string literals, so asserting against that function
    (or its output) covered the whole interface. The CSS and JS now live in
    citation_graph/static/, so a check that only looks at the shell would
    silently stop covering the thing it was written to guard (2026-08-04).
    """
    html = show_citation_graph._build_sigma_html(
        n_items=1, n_nodes=1, n_edges=0, n_citer=0, n_ref=0,
        palette=show_citation_graph._PALETTE,
        css_root=show_citation_graph._CSS_ROOT,
        js_theme=show_citation_graph._JS_THEME,
    )
    static_dir = show_citation_graph._STATIC_DIR
    return "\n".join([
        html,
        (static_dir / "app.css").read_text(encoding="utf-8"),
        (static_dir / "app.js").read_text(encoding="utf-8"),
    ])


class CitationInsightsApiTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "relations.db"
        self.db_patch = patch.object(db_relations, "DB_PATH", str(self.db_path))
        self.db_patch.start()
        db_relations._db_initialized = False
        fixture = seed_v3_summaries(db_relations)
        self.section_id = fixture["section_ids"]["w0"]

    def tearDown(self):
        self.db_patch.stop()
        db_relations._db_initialized = False
        self.tempdir.cleanup()

    def test_overview_and_sections_routes(self):
        overview = _json(show_citation_graph._route_node_insights("ITEM"))
        self.assertEqual(overview["sections"]["count"], 1)
        sections = _json(show_citation_graph._route_node_sections("ITEM", "", "", 50))
        self.assertEqual(sections["items"][0]["section_id"], self.section_id)

    def test_abstract_route_uses_the_packaged_database_module(self):
        response = _json(show_citation_graph._route_node_abstract("ITEM"))
        self.assertEqual(response["summary"]["summary"], "item summary")

    def test_browser_summary_writes_are_not_exposed(self):
        routes = {
            (method, route.path)
            for route in show_citation_graph.app.routes
            for method in getattr(route, "methods", set())
        }
        self.assertNotIn(("POST", "/api/node/summary"), routes)
        self.assertNotIn(("PUT", "/api/node/summary"), routes)

    def test_browser_summary_ui_is_read_only(self):
        document = _browser_document()
        self.assertNotIn("AI要約を生成", document)
        self.assertNotIn("sum-edit-btn", document)
        self.assertNotIn("fetch('/api/node/summary'", document)
        self.assertIn("Maintenance Widgetで要約更新", document)

    @patch(
        "src.crossref_client.fetch_crossref_by_doi",
        return_value={"abstract": "Crossref abstract"},
    )
    def test_external_abstract_doi_route_uses_packaged_crossref(self, fetch):
        response = show_citation_graph._route_external_abstract(
            doi="10.1038/nphys1170",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(_json(response)["abstract"], "Crossref abstract")
        fetch.assert_called_once_with("10.1038/nphys1170")

    @patch(
        "src.citation_mapper.s2_request",
        return_value={"abstract": "S2 abstract", "tldr": {"text": "S2 TLDR"}},
    )
    def test_external_abstract_paper_route_uses_packaged_s2(self, request):
        response = show_citation_graph._route_external_abstract(
            paper_id="CorpusId:1",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(_json(response)["abstract"], "S2 abstract")
        self.assertIn("CorpusId%3A1", request.call_args.args[0])

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
        outline = _json(show_citation_graph._route_node_outline("MISSING"))
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
        html = _browser_document()
        self.assertIn('role="tablist" aria-label="資料の詳細"', html)
        self.assertIn("/api/node/sections?key=", html)
        self.assertNotIn('data-insight-tab="cases"', html)
        self.assertIn('data-insight-tab="processing"', html)
        self.assertIn('aria-modal="true"', html)
        self.assertIn("var MIN_W = 280, MAX_W = 700", html)
        self.assertIn('.replace(/"/g, \'&quot;\')', html)
        self.assertIn(".replace(/'/g, '&#39;')", html)
        self.assertIn("encodeURIComponent(doi)", html)
        self.assertIn("label.textContent = c.label", html)
        self.assertIn("keywords.textContent = kw", html)
        self.assertNotIn("'<span class=\"cl-label\">' + c.label", html)


if __name__ == "__main__":
    unittest.main()
