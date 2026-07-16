from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import db_relations
from src import recommendations


class RecommendationAggregationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tempdir.name) / "relations.db")
        self.db_patch = patch.object(db_relations, "DB_PATH", self.db_path)
        self.db_patch.start()
        db_relations._db_initialized = False
        connection = db_relations.get_db_connection()
        connection.executemany(
            """INSERT INTO item_citation_status
               (item_key, s2_status, s2_paper_id, doi) VALUES (?, 'mapped', ?, ?)""",
            [("OWN1", "owned-s2", "10.1/owned"), ("OWN2", None, None)],
        )
        references = [
            ("candidate-s2", "Candidate", "A. Author", "10.1/candidate", "OWN1", 20),
            ("candidate-s2", "Candidate", "A. Author", "10.1/candidate", "OWN2", 20),
            ("owned-s2", "Already Owned", "B. Author", "10.1/owned", "OWN1", 30),
            ("owned-s2", "Already Owned", "B. Author", "10.1/owned", "OWN2", 30),
            ("single-s2", "Single Source", "C. Author", None, "OWN1", 5),
        ]
        connection.executemany(
            """INSERT INTO global_references
               (cited_paper_id, cited_title, cited_authors, cited_doi,
                citing_item_key, cited_citation_count, context_snippet)
               VALUES (?, ?, ?, ?, ?, ?, random())""",
            references,
        )
        connection.executemany(
            """INSERT INTO global_citations
               (citing_paper_id, citing_title, citing_authors, citing_doi,
                cited_item_key, citing_citation_count, context_snippet)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                ("neighbor-s2", "Neighbor Study", "D. Author", "10.1/neighbor", "OWN1", 8, "a"),
                ("neighbor-s2", "Neighbor Study", "D. Author", "10.1/neighbor", "OWN2", 8, "b"),
            ],
        )
        connection.commit()
        connection.close()

    def tearDown(self):
        self.db_patch.stop()
        db_relations._db_initialized = False
        self.tempdir.cleanup()

    def test_aggregate_excludes_owned_and_applies_minimum(self):
        results = db_relations.aggregate_unowned_works(min_citing_items=2)
        self.assertEqual([row["title"] for row in results], ["Candidate"])
        self.assertEqual(results[0]["adjacent_item_count"], 2)
        self.assertEqual(set(results[0]["adjacent_item_keys"]), {"OWN1", "OWN2"})

    def test_scope_changes_distinct_item_threshold(self):
        results = db_relations.aggregate_unowned_works(
            ["OWN1"], min_citing_items=1
        )
        self.assertEqual({row["title"] for row in results}, {"Candidate", "Single Source"})

    def test_normalized_title_alone_cannot_exclude_unowned_work(self):
        results = db_relations.aggregate_unowned_works(
            min_citing_items=2,
            normalized_owned_titles={db_relations.normalize_work_title("Candidate!")},
        )
        self.assertEqual([row["title"] for row in results], ["Candidate"])

    def test_citations_direction_finds_shared_external_citer(self):
        results = db_relations.aggregate_unowned_works(
            direction="citations", min_citing_items=2
        )
        self.assertEqual([row["title"] for row in results], ["Neighbor Study"])
        self.assertEqual(results[0]["adjacent_item_count"], 2)

    def test_coupling_and_cocitation_pairs(self):
        coupling = db_relations.get_coupling_pairs("OWN1")
        cocitation = db_relations.get_cocitation_pairs("OWN1")
        self.assertEqual(coupling[0]["item_key"], "OWN2")
        self.assertGreaterEqual(coupling[0]["shared_reference_count"], 2)
        self.assertEqual(cocitation, [{"item_key": "OWN2", "shared_citer_count": 1}])

    def test_case_overlap_pairs(self):
        connection = db_relations.get_db_connection()
        connection.executemany('''
            INSERT INTO case_annotations
                (item_key, description, region, practices, phenomena)
            VALUES (?, ?, ?, ?, ?)
        ''', [
            ("OWN1", "a", "Melanesia", "kula; gift", "reciprocity"),
            ("OWN2", "b", "Melanesia", "kula", "prestige"),
            ("OWN3", "c", "Europe", "market", "price"),
        ])
        connection.commit()
        connection.close()
        rows = db_relations.get_case_overlap_pairs("OWN1")
        self.assertEqual(rows[0]["item_key"], "OWN2")
        self.assertEqual(set(rows[0]["shared_case_terms"]), {"melanesia", "kula"})


class RelatedItemsTests(unittest.TestCase):
    def test_hybrid_rrf_combines_three_methods_and_evidence(self):
        with patch.object(
            recommendations, "get_network_item_keys", return_value=["A", "B", "C"]
        ), patch.object(
            recommendations,
            "get_coupling_pairs",
            return_value=[{"item_key": "B", "shared_reference_count": 3}],
        ), patch.object(
            recommendations,
            "get_cocitation_pairs",
            return_value=[{"item_key": "C", "shared_citer_count": 2}],
        ), patch.object(
            recommendations,
            "get_item_vectors",
            return_value={"A": [1.0, 0.0], "B": [0.8, 0.6], "C": [0.0, 1.0]},
        ), patch.object(
            recommendations,
            "get_item_meta",
            return_value={"B": {"title": "Book B"}, "C": {"title": "Book C"}},
        ):
            rows = recommendations.related_items("A", method="hybrid", k=2)

        self.assertEqual(rows[0]["item_key"], "B")
        self.assertEqual(rows[0]["title"], "Book B")
        self.assertEqual(
            {entry["method"] for entry in rows[0]["evidence"]},
            {"coupling", "semantic"},
        )

    def test_invalid_method_is_rejected(self):
        with self.assertRaises(ValueError):
            recommendations.related_items("A", method="unknown")

    def test_case_overlap_evidence(self):
        with patch.object(
            recommendations, "get_network_item_keys", return_value=["A", "B"]
        ), patch.object(
            recommendations, "get_case_overlap_pairs", return_value=[{
                "item_key": "B", "shared_case_terms": ["kula"], "case_overlap_score": 0.5,
            }]
        ), patch.object(
            recommendations, "get_item_meta", return_value={"B": {"title": "Book B"}}
        ):
            rows = recommendations.related_items("A", method="case_overlap", k=1)
        self.assertEqual(rows[0]["evidence"][0]["method"], "case_overlap")


if __name__ == "__main__":
    unittest.main()
