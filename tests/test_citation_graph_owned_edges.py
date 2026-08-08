"""Citations between two owned works must survive every render-side reduction.

S2's /citations endpoint has no sort parameter -- `sort=citationCount:desc` and
an invented parameter both return byte-identical results -- so a fetch-side cap
keeps an arbitrary prefix. Reduction therefore happens at render time, ordered
by citation count. That ranking is exactly what pushes an owned work off the
end: the Japanese and book-length scholarship this library is made of is
lightly cited elsewhere, while being the relation the user can least get
anywhere else.
"""
from __future__ import annotations

import sqlite3
import unittest

import citation_graph.server as server


def _db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE item_citation_status (item_key TEXT, s2_paper_id TEXT);
        CREATE TABLE relation_reports (
            direction TEXT, item_key TEXT, external_paper_id TEXT, status TEXT);
        CREATE TABLE global_citations (
            cited_item_key TEXT, citing_paper_id TEXT, citing_title TEXT,
            citing_year INTEGER, citing_citation_count INTEGER,
            citing_doi TEXT, citing_authors TEXT);
        CREATE TABLE global_references (
            citing_item_key TEXT, cited_paper_id TEXT, cited_title TEXT,
            cited_year INTEGER, cited_citation_count INTEGER,
            cited_doi TEXT, cited_authors TEXT);
    """)
    # OWNED is a second Zotero item; it is barely cited elsewhere.
    conn.execute("INSERT INTO item_citation_status VALUES ('OTHERITEM', 'OWNED')")
    conn.execute("INSERT INTO global_citations VALUES ('ITEM','OWNED','Owned work',2010,1,NULL,NULL)")
    for i in range(30):
        conn.execute(
            "INSERT INTO global_citations VALUES ('ITEM',?,?,2020,?,NULL,NULL)",
            (f"EXT{i}", f"External {i}", 500 + i),
        )
    conn.execute("INSERT INTO global_references VALUES ('ITEM','OWNED','Owned work',2010,1,NULL,NULL)")
    for i in range(30):
        conn.execute(
            "INSERT INTO global_references VALUES ('ITEM',?,?,2020,?,NULL,NULL)",
            (f"REXT{i}", f"External ref {i}", 500 + i),
        )
    conn.commit()
    return conn

class OwnedEdgesSurviveReductionTests(unittest.TestCase):
    def setUp(self):
        self.conn = _db()

    def tearDown(self):
        self.conn.close()

    def test_owned_citer_survives_a_cap_that_drops_everything_else(self):
        rows = server.get_citers(self.conn, ["ITEM"], per_item=1)
        owned = [r for r in rows if r["citing_paper_id"] == "OWNED"]
        self.assertEqual(len(owned), 1)
        self.assertEqual(len(rows), 2)  # the owned one plus one capped external

    def test_owned_citer_survives_a_citation_count_floor(self):
        # cc=1 is far below the floor; without the exemption it disappears.
        rows = server.get_citers(self.conn, ["ITEM"], per_item=9999, min_cc=100)
        self.assertIn("OWNED", {r["citing_paper_id"] for r in rows})

    def test_owned_reference_survives_both_reductions(self):
        rows = server.get_refs(self.conn, ["ITEM"], per_item=1, min_cc=100)
        self.assertIn("OWNED", {r["cited_paper_id"] for r in rows})

    def test_unowned_papers_are_still_reduced(self):
        rows = server.get_citers(self.conn, ["ITEM"], per_item=3)
        self.assertEqual(sum(1 for r in rows if not r["is_owned"]), 3)

    def test_the_cap_keeps_the_most_cited_of_the_unowned(self):
        rows = server.get_citers(self.conn, ["ITEM"], per_item=2)
        kept = [r["citing_citation_count"] for r in rows if not r["is_owned"]]
        self.assertEqual(kept, [529, 528])

    def test_a_disabled_relation_is_still_excluded_even_when_owned(self):
        # A reviewed-and-rejected edge stays rejected; ownership is not a bypass.
        self.conn.execute(
            "INSERT INTO relation_reports VALUES ('citations','ITEM','OWNED','disabled')")
        rows = server.get_citers(self.conn, ["ITEM"], per_item=9999)
        self.assertNotIn("OWNED", {r["citing_paper_id"] for r in rows})


if __name__ == "__main__":
    unittest.main()


class RenderCapDefaultsTests(unittest.TestCase):
    """No single owned item should dominate the overview.

    Uncapped, one item carried 2,644 edges against a median of 26 -- a hundred
    times the typical node. Owned items are all equally the user's own library;
    how heavily the outside world cites one of them should not decide how big
    its node is.
    """

    def test_per_item_caps_are_bounded_by_default(self):
        import pathlib
        import subprocess
        import sys as _sys

        root = pathlib.Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [_sys.executable, "citation_graph/server.py", "--help"],
            capture_output=True, text=True, cwd=str(root),
        )
        self.assertEqual(result.returncode, 0, result.stderr[:400])
        for flag in ("--citers", "--refs"):
            with self.subTest(flag=flag):
                line = next(
                    ln for ln in result.stdout.splitlines() if flag in ln
                    or (ln.strip().startswith("1アイテム") and flag in result.stdout)
                )
                self.assertIn("default: 100", result.stdout)
                self.assertTrue(line)
