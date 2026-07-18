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

    def test_reference_exclusion_fails_closed_when_tags_cannot_be_checked(self):
        with patch.dict(os.environ, {"EXTRACT_EXCLUDE_TAGS": "private"}, clear=False), patch.object(
            reference_agent.httpx, "get", side_effect=RuntimeError("offline")
        ):
            excluded, reason = reference_agent._item_excluded("ITEM")
        self.assertTrue(excluded)
        self.assertIn("could not verify", reason)

    def test_reference_exclusion_requires_an_explicit_cloud_policy(self):
        with patch.dict(os.environ, {}, clear=True):
            excluded, reason = reference_agent._item_excluded("ITEM")
        self.assertTrue(excluded)
        self.assertIn("not configured", reason)

    def test_reference_exclusion_allows_explicit_cloud_opt_in(self):
        with patch.dict(os.environ, {"EXTRACT_ALLOW_CLOUD_ALL": "1"}, clear=True):
            self.assertEqual(reference_agent._item_excluded("ITEM"), (False, None))


if __name__ == "__main__":
    unittest.main()
