from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

from src import db_relations, reference_agent


class _FakeLLM:
    provider = "fake"
    model = "test"

    def generate_json(self, prompt, *, schema, timeout):
        return {"references": [
            {"raw": "Mauss (1925). Essai sur le don.", "authors": ["Mauss"], "title": "Essai sur le don"},
            {"raw": "Invented (2000). Missing.", "authors": ["Invented"], "title": "Missing"},
        ]}


class _FakeStructureLLM:
    provider = "deepseek"
    model = "test-structure"

    def __init__(self, decisions):
        self.decisions = decisions

    def generate_json(self, prompt, *, schema, timeout):
        return {"items": self.decisions}


class _FakeCompoundLLM(_FakeStructureLLM):
    model = "test-compound"


class ReferenceAgentTests(unittest.TestCase):
    def test_reference_schema_is_strict_for_codex(self):
        schema = reference_agent.REFERENCE_SCHEMA
        self.assertFalse(schema["additionalProperties"])
        item = schema["properties"]["references"]["items"]
        self.assertFalse(item["additionalProperties"])
        self.assertEqual(set(item["required"]), set(item["properties"]))

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(
            db_relations, "DB_PATH", str(Path(self.tempdir.name) / "relations.db")
        )
        self.db_patch.start()
        db_relations._db_initialized = False

    def tearDown(self):
        self.db_patch.stop()
        db_relations._db_initialized = False
        self.tempdir.cleanup()

    def test_detects_heading_and_following_chunks(self):
        chunks = [
            {"id": "1", "text": "body"},
            {"id": "2", "text": "References"},
            {"id": "3", "text": "Mauss (1925). Essai sur le don."},
        ]
        self.assertEqual(
            [row["id"] for row in reference_agent.detect_reference_sections(chunks)],
            ["2", "3"],
        )

    def test_llm_output_must_exist_in_source(self):
        text = "References\nMauss (1925). Essai sur le don."
        with patch.object(reference_agent, "get_llm", return_value=_FakeLLM()):
            rows, model = reference_agent.extract_references(text)
        self.assertEqual(model, "fake:test")
        self.assertEqual([row["title"] for row in rows], ["Essai sur le don"])

    def test_strict_llm_does_not_hide_provider_failure(self):
        failing = _FakeLLM()
        failing.generate_json = lambda *args, **kwargs: (_ for _ in ()).throw(
            reference_agent.LLMError("failed")
        )
        with patch.object(reference_agent, "get_llm", return_value=failing):
            with self.assertRaises(reference_agent.LLMError):
                reference_agent.extract_references(
                    "References\nAuthor (2020). Title.", fallback_heuristic=False,
                )

    def test_identifier_resolution_is_cached(self):
        reference = {
            "raw": "Example", "authors": ["Author"], "title": "Work",
            "year": 2020, "doi": "10.1234/example", "isbn": None,
        }
        first = reference_agent.resolve_reference(reference)
        second = reference_agent.resolve_reference(reference)
        self.assertEqual(first, second)
        self.assertEqual(first["confidence"], 1.0)

    def test_external_outage_does_not_cache_unresolved_result(self):
        reference = {"raw": "Example", "authors": ["Author"], "title": "Work", "year": 2020}
        failure = RuntimeError("offline")
        with patch.object(reference_agent, "search_cinii", side_effect=failure), patch.object(
            reference_agent, "search_ndl", side_effect=failure
        ), patch.object(
            reference_agent, "search_crossref", side_effect=failure
        ), patch.object(reference_agent, "save_resolver_cache") as save:
            result = reference_agent.resolve_reference(reference)
        self.assertEqual(result["source"], "unresolved")
        save.assert_not_called()

    def test_partial_external_outage_does_not_cache_unresolved_result(self):
        reference = {"raw": "Example", "authors": ["Author"], "title": "Work", "year": 2020}
        with patch.object(
            reference_agent, "search_crossref", side_effect=RuntimeError("offline")
        ), patch.object(reference_agent, "search_cinii", return_value=[]), patch.object(
            reference_agent, "search_ndl", return_value=[]
        ), patch.object(reference_agent, "save_resolver_cache") as save:
            result = reference_agent.resolve_reference(reference)
        self.assertEqual(result["source"], "unresolved")
        save.assert_not_called()

    def test_candidate_score_penalizes_conflicting_authors(self):
        reference = {"title": "The Gift", "authors": ["Marcel Mauss"], "year": 1925}
        matching = {"title": "The Gift", "authors": "Marcel Mauss", "year": 1925}
        conflicting = {"title": "The Gift", "authors": "Lewis Hyde", "year": 1925}
        self.assertGreater(
            reference_agent._candidate_score(reference, matching),
            reference_agent._candidate_score(reference, conflicting),
        )

    def test_identifies_unique_external_record_without_literal_identifier(self):
        reference = {
            "raw": "Alice Smith. A Distinctive Article. Example Journal 12 (2020): 1-9.",
            "title": "A Distinctive Article", "authors": ["Alice Smith"], "year": 2020,
        }
        candidate = {
            "title": "A Distinctive Article", "authors": "Alice Smith", "year": 2020,
            "doi": "10.1234/example", "source": "crossref",
        }
        with patch.object(reference_agent, "search_crossref", return_value=[candidate]), patch.object(
            reference_agent, "search_cinii", return_value=[]
        ), patch.object(reference_agent, "search_ndl", return_value=[]):
            result = reference_agent.identify_reference_metadata(reference)
        self.assertEqual(result["status"], "matched")
        self.assertTrue(result["evidence"]["title_supported"])
        self.assertTrue(result["evidence"]["author_supported"])

    def test_resolved_work_uses_trusted_external_bibliographic_metadata(self):
        reference = {
            "raw": "A. Smith. A Distinctive Article. Example Journal (2020).",
            "title": "A Distinctive Article", "authors": ["A. Smith"], "year": 2020,
        }
        candidate = {
            "title": "A Distinctive Article", "authors": "Alice Smith", "year": 2020,
            "container": "Example Journal", "type": "journal-article",
            "doi": "10.1234/external-metadata", "source": "crossref",
        }
        with patch.object(reference_agent, "search_crossref", return_value=[candidate]), patch.object(
            reference_agent, "search_cinii", return_value=[]
        ), patch.object(reference_agent, "search_ndl", return_value=[]):
            result = reference_agent.resolve_reference(reference)
        connection = db_relations.get_db_connection()
        try:
            work = connection.execute(
                "SELECT authors, container, work_type FROM works WHERE work_id = ?",
                (result["work_id"],),
            ).fetchone()
        finally:
            connection.close()
        self.assertEqual(work["authors"], "Alice Smith")
        self.assertEqual(work["container"], "Example Journal")
        self.assertEqual(work["work_type"], "journal-article")

    def test_competing_external_records_remain_ambiguous(self):
        reference = {
            "raw": "Alice Smith. A Distinctive Article. Example Journal 12 (2020): 1-9.",
            "title": "A Distinctive Article", "authors": ["Alice Smith"], "year": 2020,
        }
        candidates = [
            {"title": "A Distinctive Article", "authors": "Alice Smith", "year": 2020,
             "doi": f"10.1234/{suffix}", "source": "crossref"}
            for suffix in ("one", "two")
        ]
        with patch.object(reference_agent, "search_crossref", return_value=candidates), patch.object(
            reference_agent, "search_cinii", return_value=[]
        ), patch.object(reference_agent, "search_ndl", return_value=[]):
            result = reference_agent.identify_reference_metadata(reference)
        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(result["evidence"]["margin"], 0.0)

    def test_primary_title_match_accepts_subtitle_difference(self):
        reference = {
            "raw": "M. Eriksson (2019). Spotify Teardown: Inside the Black Box of Streaming Music.",
            "title": "Spotify Teardown: Inside the Black Box of Streaming Music",
            "authors": ["M. Eriksson"], "year": 2019, "type": "book",
        }
        candidate = {
            "title": "Spotify Teardown", "authors": "Maria Eriksson, Rasmus Fleischer",
            "year": 2019, "doi": "10.7551/mitpress/example", "source": "crossref",
            "type": "book",
        }
        evidence = reference_agent.assess_metadata_candidate(reference, candidate)
        self.assertTrue(evidence["accepted"])
        self.assertEqual(evidence["title_match_mode"], "primary")
        self.assertTrue(evidence["strong_author_supported"])

    def test_simple_primary_title_is_allowed_with_strong_author_and_year(self):
        reference = {
            "raw": "Alice B. Smith (2020). Introduction: A Longer Subtitle.",
            "title": "Introduction: A Longer Subtitle", "authors": ["Alice B. Smith"],
            "year": 2020,
        }
        candidate = {
            "title": "Introduction", "authors": "Alice Smith", "year": 2020,
            "doi": "10.1234/introduction", "source": "crossref",
        }
        self.assertTrue(reference_agent.assess_metadata_candidate(reference, candidate)["accepted"])

    def test_primary_title_match_rejects_different_author_initial(self):
        reference = {
            "raw": "Alice Smith (2020). Introduction: A Longer Subtitle.",
            "title": "Introduction: A Longer Subtitle", "authors": ["Alice Smith"],
            "year": 2020,
        }
        candidate = {
            "title": "Introduction", "authors": "Bob Smith", "year": 2020,
            "doi": "10.1234/wrong", "source": "crossref",
        }
        self.assertFalse(reference_agent.assess_metadata_candidate(reference, candidate)["accepted"])

    def test_primary_title_match_rejects_different_full_name_with_same_initial(self):
        reference = {
            "raw": "Alice Smith (2020). Introduction: A Longer Subtitle.",
            "title": "Introduction: A Longer Subtitle", "authors": ["Alice Smith"],
            "year": 2020,
        }
        candidate = {
            "title": "Introduction", "authors": "Andrew Smith", "year": 2020,
            "doi": "10.1234/wrong-given-name", "source": "crossref",
        }
        self.assertFalse(reference_agent.assess_metadata_candidate(reference, candidate)["accepted"])

    def test_primary_title_match_rejects_bibliographic_type_conflict(self):
        reference = {
            "raw": "Alice Smith (2020). Introduction: A Longer Subtitle. Example Press.",
            "title": "Introduction: A Longer Subtitle", "authors": ["Alice Smith"],
            "year": 2020, "type": "book",
        }
        candidate = {
            "title": "Introduction", "authors": "Alice Smith", "year": 2020,
            "doi": "10.1234/wrong-type", "source": "crossref", "type": "journal-article",
        }
        self.assertFalse(reference_agent.assess_metadata_candidate(reference, candidate)["accepted"])

    def test_restructures_legacy_epub_candidates_without_graph_writes(self):
        candidates = [
            {
                "raw": "Alice Smith (2020). First Work. Example Journal.",
                "title": None, "source_kind": "epub-unverified",
            },
            {
                "raw": "Smith, First Work, 42.", "title": None,
                "source_kind": "epub-unverified",
            },
            {
                "raw": "This is an ordinary explanatory paragraph rather than a citation.",
                "title": None, "source_kind": "epub-unverified",
            },
        ]
        db_relations.stage_reference_candidates("ITEM", "epub-unverified", candidates)
        rows = db_relations.get_reference_review_candidates("pending")
        for row in rows:
            db_relations.set_reference_review_status(
                row["review_id"], "rejected", "unresolved: insufficient stable identifier",
            )
        decisions = [
            {
                "review_id": rows[0]["review_id"], "classification": "full_reference",
                "authors": ["Alice Smith"], "title": "First Work", "year": 2020,
                "container": "Example Journal", "publisher": None,
                "doi": None, "isbn": None, "type": "journal-article",
            },
            {
                "review_id": rows[1]["review_id"], "classification": "short_citation",
                "authors": [], "title": None, "year": None, "container": None,
                "publisher": None, "doi": None, "isbn": None, "type": None,
            },
            {
                "review_id": rows[2]["review_id"], "classification": "commentary_or_body",
                "authors": [], "title": None, "year": None, "container": None,
                "publisher": None, "doi": None, "isbn": None, "type": None,
            },
        ]
        with patch.object(
            reference_agent, "get_llm", return_value=_FakeStructureLLM(decisions)
        ):
            report = reference_agent.restructure_unparsed_epub_references(limit=10)
        self.assertEqual(report["valid"], 3)
        self.assertEqual(report["full_reference"], 1)
        applied = reference_agent.apply_reference_structure_report(
            report["results"], model=report["model"],
        )
        self.assertEqual(applied["applied"], 3)
        updated = {
            row["structure_classification"]: row
            for row in db_relations.get_reference_review_candidates("rejected")
        }
        self.assertEqual(updated["full_reference"]["title"], "First Work")
        self.assertEqual(updated["full_reference"]["authors"], ["Alice Smith"])
        self.assertIsNone(updated["short_citation"]["title"])
        connection = db_relations.get_db_connection()
        try:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM work_edges").fetchone()[0], 0)
        finally:
            connection.close()

    def test_structure_report_revalidates_all_rows_before_applying(self):
        db_relations.stage_reference_candidates("ITEM", "epub-unverified", [
            {
                "raw": "Alice Smith (2020). First Work.", "title": None,
                "source_kind": "epub-unverified",
            },
            {
                "raw": "Bob Jones (2021). Second Work.", "title": None,
                "source_kind": "epub-unverified",
            },
        ])
        rows = db_relations.get_reference_review_candidates("pending")
        for row in rows:
            db_relations.set_reference_review_status(
                row["review_id"], "rejected", "unresolved: insufficient stable identifier",
            )
        results = [
            {
                "review_id": rows[0]["review_id"], "classification": "short_citation",
                "authors": [], "title": None, "year": None, "container": None,
                "publisher": None, "doi": None, "isbn": None, "type": None, "valid": True,
            },
            {
                "review_id": rows[1]["review_id"], "classification": "full_reference",
                "authors": ["Bob Jones"], "title": "Invented Work", "year": 2021,
                "container": None, "publisher": None, "doi": None, "isbn": None,
                "type": None, "valid": True,
            },
        ]
        with self.assertRaisesRegex(ValueError, "title is not present"):
            reference_agent.apply_reference_structure_report(results, model="deepseek:test")
        unchanged = db_relations.get_reference_review_candidates("rejected")
        self.assertTrue(all(row["structure_classification"] is None for row in unchanged))

    def test_splits_only_verbatim_complete_compound_references(self):
        raw = "1 Alice Smith (2020). First Work. 2 Bob Jones (2021). Second Work."
        db_relations.stage_reference_candidates("ITEM", "epub-unverified", [{
            "raw": raw, "title": None, "source_kind": "epub-unverified",
        }])
        parent = db_relations.get_reference_review_candidates("pending")[0]
        db_relations.set_reference_review_status(
            parent["review_id"], "rejected", "compound reference: requires splitting",
        )
        decision = {
            "review_id": parent["review_id"],
            "classification": "multiple_full_references",
            "references": [
                {
                    "raw": "1 Alice Smith (2020). First Work.",
                    "authors": ["Alice Smith"], "title": "First Work", "year": 2020,
                    "container": None, "publisher": None, "doi": None, "isbn": None,
                    "type": "book",
                },
                {
                    "raw": "2 Bob Jones (2021). Second Work.",
                    "authors": ["Bob Jones"], "title": "Second Work", "year": 2021,
                    "container": None, "publisher": None, "doi": None, "isbn": None,
                    "type": "book",
                },
            ],
        }
        with patch.object(
            reference_agent, "get_llm", return_value=_FakeCompoundLLM([decision])
        ):
            report = reference_agent.split_compound_reference_candidates(limit=10)
        self.assertEqual(report["valid"], 1)
        self.assertEqual(report["multiple_full_references"], 1)
        applied = reference_agent.apply_compound_reference_report(
            report["results"], model=report["model"],
        )
        self.assertEqual(applied["parents"], 1)
        self.assertEqual(applied["children_staged"], 2)
        rejected = db_relations.get_reference_review_candidates("rejected")
        updated_parent = next(row for row in rejected if row["review_id"] == parent["review_id"])
        children = [row for row in rejected if row.get("parent_review_id") == parent["review_id"]]
        self.assertEqual(updated_parent["structure_classification"], "compound_parent")
        self.assertEqual({row["title"] for row in children}, {"First Work", "Second Work"})
        self.assertTrue(all(row["source_kind"] == "epub-compound-child" for row in children))
        connection = db_relations.get_db_connection()
        try:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM work_edges").fetchone()[0], 0)
        finally:
            connection.close()

    def test_compound_commentary_is_classified_without_staging_children(self):
        db_relations.stage_reference_candidates("ITEM", "epub-unverified", [{
            "raw": "The author discusses Smith (2020) and Jones (2021).",
            "title": None, "source_kind": "epub-unverified",
        }])
        parent = db_relations.get_reference_review_candidates("pending")[0]
        db_relations.set_reference_review_status(
            parent["review_id"], "rejected", "compound reference: requires splitting",
        )
        decision = {
            "review_id": parent["review_id"], "classification": "commentary_or_body",
            "references": [],
        }
        with patch.object(
            reference_agent, "get_llm", return_value=_FakeCompoundLLM([decision])
        ):
            report = reference_agent.split_compound_reference_candidates(limit=10)
        applied = reference_agent.apply_compound_reference_report(
            report["results"], model=report["model"],
        )
        self.assertEqual(applied["safe_splits"], 0)
        self.assertEqual(len(db_relations.get_reference_review_candidates("rejected")), 1)

    def test_compound_split_rejects_nonverbatim_child(self):
        row = {
            "review_id": 7,
            "raw_reference": "Alice Smith (2020). First Work. Bob Jones (2021). Second Work.",
        }
        decision = {
            "review_id": 7, "classification": "multiple_full_references",
            "references": [
                {"raw": "Alice Smith (2020). Rewritten Work.", "authors": ["Alice Smith"],
                 "title": "Rewritten Work", "year": 2020, "doi": None, "isbn": None},
                {"raw": "Bob Jones (2021). Second Work.", "authors": ["Bob Jones"],
                 "title": "Second Work", "year": 2021, "doi": None, "isbn": None},
            ],
        }
        with self.assertRaisesRegex(ValueError, "sequential verbatim"):
            reference_agent._validate_compound_split(row, decision)

    def test_compound_split_batch_is_atomic(self):
        for item, raw in (
            ("A", "Alice Smith (2020). First Work. Bob Jones (2021). Second Work."),
            ("B", "Carol White (2022). Third Work. Dan Black (2023). Fourth Work."),
        ):
            db_relations.stage_reference_candidates(item, "epub-unverified", [{
                "raw": raw, "title": None, "source_kind": "epub-unverified",
            }])
        parents = db_relations.get_reference_review_candidates("pending")
        for parent in parents:
            db_relations.set_reference_review_status(
                parent["review_id"], "rejected", "compound reference: requires splitting",
            )
        valid = {
            "review_id": parents[0]["review_id"], "classification": "multiple_full_references",
            "references": [
                {"raw": "Alice Smith (2020). First Work.", "authors": ["Alice Smith"],
                 "title": "First Work", "year": 2020, "doi": None, "isbn": None},
                {"raw": "Bob Jones (2021). Second Work.", "authors": ["Bob Jones"],
                 "title": "Second Work", "year": 2021, "doi": None, "isbn": None},
            ],
        }
        invalid = {
            "review_id": parents[1]["review_id"], "classification": "multiple_full_references",
            "references": [
                {"raw": "Invented", "authors": ["Carol White"], "title": "Third Work",
                 "year": 2022, "doi": None, "isbn": None},
                {"raw": "Dan Black (2023). Fourth Work.", "authors": ["Dan Black"],
                 "title": "Fourth Work", "year": 2023, "doi": None, "isbn": None},
            ],
        }
        with self.assertRaisesRegex(ValueError, "child raw"):
            db_relations.apply_compound_reference_splits(
                [valid, invalid], model="deepseek:test",
            )
        rejected = db_relations.get_reference_review_candidates("rejected")
        self.assertEqual(len(rejected), 2)
        self.assertTrue(all(row["structure_classification"] is None for row in rejected))
