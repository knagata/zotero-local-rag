"""A Zotero-owned work must be one node, even when S2 gives it no DOI.

Observed 2026-08-06 with Bratton "The Stack": S2's record carries only
MAG/CorpusId in externalIds -- no DOI, no ISBN -- so the DOI/ISBN merge could
never collapse the external paper node into the owned item's node, and the
work rendered twice. About 13% of external papers on both the citation and
the reference side have no DOI, and books are concentrated there.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import citation_graph.server as server


PID = "38c017d80bbad14d75d65033afd75f8770bce8e3"


def _build(items, citers=(), refs=()):
    """Run build_graph_data with layout work stubbed and its cache in a temp dir."""
    with tempfile.TemporaryDirectory() as tmp:
        # build_graph_data derives its layout cache from Path(__file__); point
        # that at a scratch tree so the real data/layout_cache.json is untouched.
        stand_in = Path(tmp) / "citation_graph" / "server.py"
        (Path(tmp) / "data").mkdir(parents=True)
        stand_in.parent.mkdir(parents=True)
        with mock.patch.object(server, "get_item_vectors", return_value={}), \
             mock.patch.object(server, "compute_layout", return_value={}), \
             mock.patch.object(server, "Path", lambda _path: stand_in):
            return server.build_graph_data(list(items), list(citers), list(refs))


class OwnedWorkMergeTests(unittest.TestCase):
    def _item(self, key="VBRN4WYR", **over):
        row = {
            "item_key": key, "citer_count": 0, "context_count": 0,
            "s2_status": "mapped", "s2_paper_id": PID, "s2_year": 2016,
            "s2_citation_count": 636, "doi": "", "isbn": "",
        }
        row.update(over)
        return row

    def _ref(self, citing_item_key="JQYWTQP8", **over):
        row = {
            "citing_item_key": citing_item_key, "cited_paper_id": PID,
            "cited_title": "The Stack: On Software and Sovereignty",
            "cited_year": 2016, "cited_citation_count": 635,
            "cited_doi": None, "cited_authors": "B. Bratton", "context_count": 1,
        }
        row.update(over)
        return row

    def test_doiless_owned_work_is_not_duplicated_as_an_external_node(self):
        graph = _build([self._item(), self._item("JQYWTQP8", s2_paper_id="OTHER")],
                       refs=[self._ref()])
        ids = [n["id"] for n in graph["nodes"]]

        self.assertIn("item:VBRN4WYR", ids)
        self.assertNotIn(f"ref:{PID}", ids)
        self.assertNotIn(f"paper:{PID}", ids)

    def test_the_reference_edge_points_at_the_owned_item_node(self):
        graph = _build([self._item(), self._item("JQYWTQP8", s2_paper_id="OTHER")],
                       refs=[self._ref()])
        targets = [e["target"] for e in graph["edges"]]

        self.assertEqual(targets, ["item:VBRN4WYR"])

    def test_incoming_citation_from_another_owned_work_merges_too(self):
        citer = {
            "cited_item_key": "JQYWTQP8", "citing_paper_id": PID,
            "citing_title": "The Stack: On Software and Sovereignty",
            "citing_year": 2016, "citing_citation_count": 636,
            "citing_doi": None, "citing_authors": "B. Bratton", "context_count": 2,
        }
        graph = _build([self._item(), self._item("JQYWTQP8", s2_paper_id="OTHER")],
                       citers=[citer])
        ids = [n["id"] for n in graph["nodes"]]

        self.assertNotIn(f"paper:{PID}", ids)
        self.assertEqual([e["source"] for e in graph["edges"]], ["item:VBRN4WYR"])

    def test_unowned_paper_still_renders_as_an_external_node(self):
        graph = _build([self._item("JQYWTQP8", s2_paper_id="OTHER")],
                       refs=[self._ref()])
        ids = [n["id"] for n in graph["nodes"]]

        self.assertIn(f"ref:{PID}", ids)
        self.assertNotIn("item:VBRN4WYR", ids)

    def test_item_without_an_s2_paper_id_does_not_swallow_external_nodes(self):
        # An empty s2_paper_id must never become a merge key: every DOI-less
        # external paper would collapse into whichever item happened to lack one.
        graph = _build([self._item("JQYWTQP8", s2_paper_id=None),
                        self._item("AAAAAAAA", s2_paper_id="")],
                       refs=[self._ref()])
        ids = [n["id"] for n in graph["nodes"]]

        self.assertIn(f"ref:{PID}", ids)

    def test_absorbed_paper_is_recorded_on_the_owning_node(self):
        # s2_paper_id is an unverified stored value; when it is wrong the merge
        # swallows another work silently, and the duplicate node that used to
        # reveal the problem is gone. The absorbed ids must stay inspectable.
        graph = _build([self._item(), self._item("JQYWTQP8", s2_paper_id="OTHER")],
                       refs=[self._ref()])
        node = next(n for n in graph["nodes"] if n["id"] == "item:VBRN4WYR")

        self.assertEqual(node["absorbedPapers"], [
            {"paperId": PID, "title": "The Stack: On Software and Sovereignty"},
        ])

    def test_node_that_absorbed_nothing_carries_no_marker(self):
        graph = _build([self._item("JQYWTQP8", s2_paper_id="OTHER")], refs=[self._ref()])
        node = next(n for n in graph["nodes"] if n["id"] == "item:JQYWTQP8")

        self.assertNotIn("absorbedPapers", node)

    def test_doi_based_merge_is_not_reported_as_a_paper_id_merge(self):
        graph = _build(
            [self._item("AAAAAAAA", s2_paper_id=None, doi="10.1/x"),
             self._item("JQYWTQP8", s2_paper_id="OTHER")],
            refs=[self._ref(cited_paper_id="SOMEPID", cited_doi="10.1/X")],
        )
        node = next(n for n in graph["nodes"] if n["id"] == "item:AAAAAAAA")

        self.assertNotIn("absorbedPapers", node)

    def test_owned_item_keeps_the_zotero_colour_whatever_s2_reported(self):
        # The legend calls this colour "Zotero アイテム" and has no entry for
        # grey, so coluring an owned work grey put it in no category at all.
        # Whether S2 could identify the work is S2's coverage, not ownership.
        for status in ("mapped", "s2_done", "not_found", "pending", "error", None):
            with self.subTest(status=status):
                graph = _build([self._item(s2_status=status)])
                node = next(n for n in graph["nodes"] if n["id"] == "item:VBRN4WYR")
                self.assertEqual(node["color"], server._PALETTE["nodeZotero"])

    def test_s2_outcome_is_still_visible_in_the_tooltip(self):
        graph = _build([self._item(s2_status="not_found")])
        node = next(n for n in graph["nodes"] if n["id"] == "item:VBRN4WYR")

        self.assertIn("not_found", node["tooltip"])

    def test_no_self_loop_when_an_owned_work_cites_itself(self):
        graph = _build([self._item()], refs=[self._ref(citing_item_key="VBRN4WYR")])

        self.assertEqual(graph["edges"], [])


if __name__ == "__main__":
    unittest.main()
