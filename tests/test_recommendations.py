from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import db_relations


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

    def test_normalized_title_can_exclude_owned_work(self):
        results = db_relations.aggregate_unowned_works(
            min_citing_items=2,
            normalized_owned_titles={db_relations.normalize_work_title("Candidate!")},
        )
        self.assertEqual(results, [])

    def test_citations_direction_finds_shared_external_citer(self):
        results = db_relations.aggregate_unowned_works(
            direction="citations", min_citing_items=2
        )
        self.assertEqual([row["title"] for row in results], ["Neighbor Study"])
        self.assertEqual(results[0]["adjacent_item_count"], 2)


if __name__ == "__main__":
    unittest.main()
