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

    def test_paper_id_claimed_by_two_items_merges_into_neither(self):
        # An original and its translation can resolve to the same S2 paper.
        # Letting the last one win routes every edge to it and silently strips
        # them from the other, so the ambiguous id is left unmerged instead and
        # the external node stays visible.
        graph = _build([self._item("AAAAAAAA"), self._item("BBBBBBBB")], refs=[self._ref()])
        ids = [n["id"] for n in graph["nodes"]]

        self.assertIn(f"ref:{PID}", ids)
        for node in graph["nodes"]:
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

    def test_external_node_shape_and_edge_are_symmetric_by_direction(self):
        """Citer and reference paths share the same external-node contract."""
        for direction in ("citer", "reference"):
            with self.subTest(direction=direction):
                item = self._item("VBRN4WYR", s2_paper_id="OWNED")
                if direction == "citer":
                    row = {
                        "cited_item_key": "VBRN4WYR", "citing_paper_id": "EXTERNAL",
                        "citing_title": "An external work", "citing_year": 2020,
                        "citing_citation_count": 12, "citing_doi": "10.1/ext",
                        "citing_authors": "A. Author", "context_count": 2,
                    }
                    graph = _build([item], citers=[row])
                    node_id = "paper:EXTERNAL"
                    group = "external"
                    expected_edge = (node_id, "item:VBRN4WYR")
                else:
                    row = self._ref(
                        citing_item_key="VBRN4WYR", cited_paper_id="EXTERNAL",
                        cited_title="An external work", cited_year=2020,
                        cited_citation_count=12, cited_doi="10.1/ext",
                        cited_authors="A. Author", context_count=2,
                    )
                    graph = _build([item], refs=[row])
                    node_id = "ref:EXTERNAL"
                    group = "reference"
                    expected_edge = ("item:VBRN4WYR", node_id)

                node = next(n for n in graph["nodes"] if n["id"] == node_id)
                self.assertEqual(node["group"], group)
                self.assertEqual(
                    set(node),
                    {"id", "label", "size", "color", "tooltip", "group", "cc",
                     "fullTitle", "year", "doi", "isbn", "authors", "x", "y"},
                )
                self.assertEqual(
                    (graph["edges"][0]["source"], graph["edges"][0]["target"]),
                    expected_edge,
                )
                self.assertEqual(graph["edges"][0]["externalPaperId"], "EXTERNAL")

    def test_identifier_override_is_applied_to_both_external_directions(self):
        override = {
            "doi": "10.9/corrected", "isbn": "", "title": "Corrected title",
            "year": "2001", "authors": "C. Corrector", "citations": "17",
        }
        for direction in ("citer", "reference"):
            with self.subTest(direction=direction):
                item = self._item("VBRN4WYR", s2_paper_id="OWNED")
                key = "paper:EXTERNAL" if direction == "citer" else "ref:EXTERNAL"
                if direction == "citer":
                    row = {
                        "cited_item_key": "VBRN4WYR", "citing_paper_id": "EXTERNAL",
                        "citing_title": "Original title", "citing_year": 2020,
                        "citing_citation_count": 12, "citing_doi": "10.1/ext",
                        "citing_authors": "A. Author", "context_count": 1,
                    }
                    kwargs = {"citers": [row]}
                else:
                    row = self._ref(
                        citing_item_key="VBRN4WYR", cited_paper_id="EXTERNAL",
                        cited_title="Original title", cited_year=2020,
                        cited_citation_count=12, cited_doi="10.1/ext",
                        cited_authors="A. Author", context_count=1,
                    )
                    kwargs = {"refs": [row]}
                with mock.patch.object(
                    server, "_load_identifier_overrides", return_value={key: override}
                ):
                    graph = _build([item], **kwargs)
                node = next(n for n in graph["nodes"] if n["id"] in {"paper:EXTERNAL", "ref:EXTERNAL"})
                self.assertEqual(node["fullTitle"], "Corrected title")
                self.assertEqual(node["year"], "2001")
                self.assertEqual(node["authors"], "C. Corrector")
                self.assertEqual(node["doi"], "10.9/corrected")
                self.assertEqual(node["cc"], 17)

    def test_self_loop_and_owned_identifier_merge_are_symmetric_by_direction(self):
        for direction in ("citer", "reference"):
            with self.subTest(direction=direction):
                item = self._item("VBRN4WYR", s2_paper_id="OWNED", doi="10.1/owned")
                if direction == "citer":
                    row = {
                        "cited_item_key": "VBRN4WYR", "citing_paper_id": "EXTERNAL",
                        "citing_title": "Same work", "citing_year": 2020,
                        "citing_citation_count": 12, "citing_doi": "10.1/OWNED",
                        "citing_authors": "A. Author", "context_count": 1,
                    }
                    graph = _build([item], citers=[row])
                else:
                    row = self._ref(
                        citing_item_key="VBRN4WYR", cited_paper_id="EXTERNAL",
                        cited_title="Same work", cited_year=2020,
                        cited_citation_count=12, cited_doi="10.1/OWNED",
                        cited_authors="A. Author", context_count=1,
                    )
                    graph = _build([item], refs=[row])
                self.assertEqual([n["id"] for n in graph["nodes"]], ["item:VBRN4WYR"])
                self.assertEqual(graph["edges"], [])

    def test_duplicate_external_identifiers_are_deduplicated_in_both_directions(self):
        for direction in ("citer", "reference"):
            with self.subTest(direction=direction):
                item = self._item("VBRN4WYR", s2_paper_id="OWNED")
                if direction == "citer":
                    rows = [
                        {
                            "cited_item_key": "VBRN4WYR", "citing_paper_id": "FIRST",
                            "citing_title": "First title", "citing_year": 2020,
                            "citing_citation_count": 12, "citing_doi": "10.1/shared",
                            "citing_authors": "A. Author", "context_count": 1,
                        },
                        {
                            "cited_item_key": "VBRN4WYR", "citing_paper_id": "SECOND",
                            "citing_title": "Second title", "citing_year": 2021,
                            "citing_citation_count": 8, "citing_doi": "10.1/SHARED",
                            "citing_authors": "B. Author", "context_count": 1,
                        },
                    ]
                    graph = _build([item], citers=rows)
                    external_ids = {"paper:FIRST", "paper:SECOND"}
                    expected_external_id = "paper:FIRST"
                else:
                    rows = [
                        self._ref(
                            citing_item_key="VBRN4WYR", cited_paper_id="FIRST",
                            cited_title="First title", cited_year=2020,
                            cited_citation_count=12, cited_doi="10.1/shared",
                        ),
                        self._ref(
                            citing_item_key="VBRN4WYR", cited_paper_id="SECOND",
                            cited_title="Second title", cited_year=2021,
                            cited_citation_count=8, cited_doi="10.1/SHARED",
                        ),
                    ]
                    graph = _build([item], refs=rows)
                    external_ids = {"ref:FIRST", "ref:SECOND"}
                    expected_external_id = "ref:FIRST"
                external_nodes = [n["id"] for n in graph["nodes"] if n["id"] in external_ids]
                self.assertEqual(external_nodes, [expected_external_id])
                self.assertEqual(len(graph["edges"]), 2)


if __name__ == "__main__":
    unittest.main()
