"""Regression tests for NULL-safe relation identity and S2 pagination status."""
from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import citation_mapper  # noqa: E402
import db_relations  # noqa: E402


class RelationIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tempdir.name) / "relations.db")
        self.patch = patch.object(db_relations, "DB_PATH", self.db_path)
        self.patch.start()
        db_relations._db_initialized = False

    def tearDown(self):
        self.patch.stop()
        db_relations._db_initialized = False
        self.tempdir.cleanup()

    def test_insert_helpers_are_idempotent_when_identity_fields_are_null(self):
        db_relations.insert_citation(
            "CITER", "first", 2024, None, "ITEM", None, None, None,
        )
        db_relations.insert_citation(
            "CITER", "updated", 2025, None, "ITEM", "chunk", 0.1, "3",
        )
        db_relations.insert_reference(
            None, "same work", 2025, None, "ITEM", None, None,
            raw_reference_text=None,
        )
        db_relations.insert_reference(
            None, "same work", 2025, None, "ITEM", "chunk", 0.1,
            raw_reference_text=None,
        )

        connection = db_relations.get_db_connection()
        try:
            citation = connection.execute("SELECT * FROM global_citations").fetchone()
            reference = connection.execute("SELECT * FROM global_references").fetchone()
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM global_citations").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM global_references").fetchone()[0], 1)
            self.assertEqual(citation["citing_title"], "updated")
            self.assertEqual(citation["cited_chunk_id"], "chunk")
            self.assertEqual(reference["cited_title"], "same work")
            self.assertEqual(reference["citing_chunk_id"], "chunk")
        finally:
            connection.close()

    def test_legacy_null_duplicates_are_compacted_before_unique_indexes(self):
        connection = sqlite3.connect(self.db_path)
        connection.executescript("""
            CREATE TABLE global_citations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                citing_paper_id TEXT, citing_title TEXT, citing_year INTEGER,
                context_snippet TEXT, cited_item_key TEXT, cited_chunk_id TEXT,
                similarity_distance REAL, page_hint TEXT, created_at TIMESTAMP,
                UNIQUE(citing_paper_id, cited_item_key, context_snippet)
            );
            CREATE TABLE global_references (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cited_paper_id TEXT, cited_title TEXT, cited_year INTEGER,
                context_snippet TEXT, citing_item_key TEXT, citing_chunk_id TEXT,
                similarity_distance REAL, page_hint TEXT, source TEXT DEFAULT 's2',
                raw_reference_text TEXT, created_at TIMESTAMP,
                UNIQUE(cited_paper_id, citing_item_key, context_snippet, raw_reference_text)
            );
            INSERT INTO global_citations
                (citing_paper_id, cited_item_key, context_snippet) VALUES
                ('CITER', 'ITEM', NULL), ('CITER', 'ITEM', NULL);
            INSERT INTO global_references
                (cited_paper_id, cited_title, cited_year, citing_item_key,
                 context_snippet, raw_reference_text) VALUES
                (NULL, 'Same work', 2020, 'ITEM', NULL, NULL),
                (NULL, 'Same work', 2020, 'ITEM', NULL, NULL);
        """)
        db_relations._init_db(connection)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM global_citations").fetchone()[0], 1)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM global_references").fetchone()[0], 1)
        with self.assertRaises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO global_citations (citing_paper_id, cited_item_key, context_snippet) "
                "VALUES ('CITER', 'ITEM', NULL)"
            )
        connection.close()

    def test_distinct_unidentified_references_are_not_collapsed(self):
        db_relations.insert_reference(
            None, "First work", 2001, None, "ITEM", None, None,
        )
        db_relations.insert_reference(
            None, "Second work", 2002, None, "ITEM", None, None,
        )
        connection = db_relations.get_db_connection()
        try:
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM global_references"
            ).fetchone()[0], 2)
        finally:
            connection.close()

    def test_clean_rebuild_reset_removes_chunk_relations_and_derived_state(self):
        connection = db_relations.get_db_connection()
        connection.execute(
            "INSERT INTO global_citations "
            "(citing_paper_id, cited_item_key, context_snippet) VALUES ('C', 'ITEM', 'ctx')"
        )
        connection.execute(
            "INSERT INTO item_summaries (item_key, summary) VALUES ('ITEM', 'old')"
        )
        connection.execute(
            "INSERT INTO artifact_processing_status "
            "(item_key, attachment_key, artifact_type, status) "
            "VALUES ('ITEM', 'ATT', 'extraction', 'success')"
        )
        connection.commit()
        connection.close()

        counts = db_relations.reset_ingestion_derived_state()

        connection = db_relations.get_db_connection()
        try:
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM global_citations"
            ).fetchone()[0], 0)
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM item_summaries"
            ).fetchone()[0], 0)
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM artifact_processing_status"
            ).fetchone()[0], 0)
            self.assertEqual(counts["item_summaries"], 1)
        finally:
            connection.close()


class CitationPaginationTests(unittest.TestCase):
    def setUp(self):
        # The debug logs used to be module constants pointing into the real
        # data directory, so every test here had to patch them aside. They now
        # resolve beside whichever relations database the run is using, which
        # under pytest is a scratch file, so there is nothing left to patch.
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

    @staticmethod
    def _citation_page(*, next_offset=None):
        result = {"data": [{
            "citingPaper": {"paperId": "CITER", "title": "Citing", "authors": []},
            "contexts": ["citing context"],
        }]}
        if next_offset is not None:
            result["next"] = next_offset
        return result

    @staticmethod
    def _reference_page(*, next_offset=None, contexts=None):
        result = {"data": [{
            "citedPaper": {"paperId": "REF", "title": "Reference", "authors": []},
            "contexts": ["reference context"] if contexts is None else contexts,
        }]}
        if next_offset is not None:
            result["next"] = next_offset
        return result

    def _map_with(self, responses, *, max_citations=50):
        statuses = []
        with patch.object(citation_mapper, "find_s2_paper_id", return_value={"paperId": "TARGET"}), \
             patch.object(citation_mapper, "s2_request", side_effect=responses), \
             patch.object(citation_mapper, "search_chunks", return_value=[]), \
             patch.object(citation_mapper, "insert_citation") as save_citation, \
             patch.object(db_relations, "insert_reference") as save_reference, \
             patch.object(citation_mapper, "update_item_citation_status", side_effect=lambda *args, **kwargs: statuses.append(args)):
            result = citation_mapper.map_item_global_citations(
                "ITEM", title="Target", max_citations=max_citations,
            )
        return result, statuses, save_citation, save_reference

    def test_later_citation_page_failure_is_retryable_not_done(self):
        result, statuses, saved, _references = self._map_with([
            self._citation_page(next_offset=1000),
            None,
            {"data": []},
        ])

        self.assertEqual(result["status"], "error")
        self.assertTrue(result["retryable"])
        self.assertEqual(result["incomplete_parts"], ["citations"])
        self.assertEqual(statuses[-1], ("ITEM", "error"))
        self.assertNotIn(("ITEM", "s2_done"), statuses)
        saved.assert_called_once()

    def test_later_reference_page_failure_is_retryable_not_done(self):
        result, statuses, saved, _references = self._map_with([
            self._citation_page(),
            {"data": [{
                "citedPaper": {"paperId": "REF", "title": "Reference", "authors": []},
                "contexts": [],
            }], "next": 1000},
            None,
        ])

        self.assertEqual(result["status"], "error")
        self.assertTrue(result["retryable"])
        self.assertEqual(result["incomplete_parts"], ["references"])
        self.assertEqual(statuses[-1], ("ITEM", "error"))
        self.assertNotIn(("ITEM", "s2_done"), statuses)
        saved.assert_called_once()

    def test_empty_citations_still_fetches_outgoing_references(self):
        result, statuses, saved, references = self._map_with([
            {"data": []},
            {"data": [{
                "citedPaper": {"paperId": "REF", "title": "Reference", "authors": []},
                "contexts": [],
            }]},
        ])

        self.assertEqual(result["status"], "success")
        saved.assert_not_called()
        references.assert_called_once()
        self.assertEqual(statuses[-1], ("ITEM", "s2_done"))

    def test_empty_contexts_are_saved_symmetrically(self):
        citation = self._citation_page()
        citation["data"][0]["contexts"] = []
        result, statuses, saved, references = self._map_with([
            citation,
            self._reference_page(contexts=[]),
        ])
        saved.assert_called_once()
        references.assert_called_once()
        self.assertEqual(saved.call_args.kwargs["chunk_status"], "no_context")
        self.assertEqual(references.call_args.kwargs["s2_status"], "no_context")
        self.assertEqual(saved.call_args.kwargs["context_snippet"], "")
        self.assertEqual(references.call_args.kwargs["context_snippet"], "")
        self.assertEqual(result["status"], "success")
        self.assertEqual(statuses[-1], ("ITEM", "s2_done"))

    def test_multiple_pages_and_page_hints_are_symmetric(self):
        citation_first = self._citation_page(next_offset=10)
        citation_first["data"][0]["contexts"] = ["p. 12 citation"]
        citation_second = self._citation_page()
        citation_second["data"][0]["citingPaper"]["paperId"] = "CITER2"
        citation_second["data"][0]["contexts"] = ["page 13 citation"]
        reference_first = self._reference_page(next_offset=20, contexts=["pp. 22 reference"])
        reference_second = self._reference_page(contexts=["note: 23 reference"])
        result, _statuses, saved, references = self._map_with([
            citation_first, citation_second, reference_first, reference_second,
        ])

        self.assertEqual(result["status"], "success")
        self.assertEqual(saved.call_count, 2)
        self.assertEqual(references.call_count, 2)
        self.assertEqual(
            [call.kwargs["page_hint"] for call in saved.call_args_list], ["12", "13"],
        )
        self.assertEqual(
            [call.kwargs["page_hint"] for call in references.call_args_list], ["22", "23"],
        )

    def test_reference_pagination_limit_is_not_marked_done(self):
        result, statuses, _saved, references = self._map_with([
            {"data": []},
            self._reference_page(next_offset=1),
        ], max_citations=1)

        self.assertEqual(result["status"], "error")
        self.assertFalse(result["retryable"])
        self.assertEqual(result["incomplete_parts"], ["references"])
        self.assertEqual(statuses[-1], ("ITEM", "limited"))
        self.assertNotIn(("ITEM", "s2_done"), statuses)
        references.assert_called_once()

    def test_pagination_limit_is_not_marked_done(self):
        result, statuses, saved, _references = self._map_with([
            self._citation_page(next_offset=1),
            {"data": []},
        ], max_citations=1)

        self.assertEqual(result["status"], "error")
        self.assertFalse(result["retryable"])
        self.assertEqual(result["incomplete_parts"], ["citations"])
        self.assertEqual(statuses[-1], ("ITEM", "limited"))
        self.assertNotIn(("ITEM", "s2_done"), statuses)
        saved.assert_called_once()


if __name__ == "__main__":
    unittest.main()
