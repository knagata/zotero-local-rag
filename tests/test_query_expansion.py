from __future__ import annotations

import json
import sqlite3
import unittest
from unittest.mock import patch

from src.llm_client import ProviderUnavailable
from src.db_relations import _init_db
from src import query_expansion
from src.query_expansion import expand_queries


class FakeLLM:
    provider = "fake"
    model = "test"

    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = 0

    def generate_json(self, prompt, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return self.payload


class QueryExpansionTests(unittest.TestCase):
    def setUp(self):
        query_expansion._negative_until.clear()

    def test_database_migration_creates_cache_table(self):
        connection = sqlite3.connect(":memory:")
        _init_db(connection)
        row = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='query_expansion_cache'"
        ).fetchone()
        connection.close()
        self.assertEqual(row[0], "query_expansion_cache")

    def test_bilingual_default_query_skips_llm(self):
        with patch("src.query_expansion.get_llm") as get_llm:
            result = expand_queries(["贈与 互酬性", "gift reciprocity"])
        self.assertEqual(result, ["贈与 互酬性", "gift reciprocity"])
        get_llm.assert_not_called()

    def test_case_mode_combines_all_expansion_levels(self):
        payload = {
            "queries": ["gift exchange"],
            "hyde": ["首長が豚を各世帯へ分配した。"],
            "broader": ["互酬性"],
            "narrower": ["儀礼的再分配"],
        }
        fake = FakeLLM(payload)
        with patch("src.query_expansion.get_llm", return_value=fake), patch(
            "src.query_expansion.get_query_expansion", return_value=None
        ), patch("src.query_expansion.save_query_expansion") as save:
            result = expand_queries(["威信財の再分配"], mode="case")
        self.assertEqual(result[0], "威信財の再分配")
        self.assertIn("首長が豚を各世帯へ分配した。", result)
        self.assertEqual(fake.calls, 1)
        save.assert_called_once()

    def test_cache_hit_avoids_generation(self):
        cached = json.dumps({
            "queries": ["gift"], "hyde": [], "broader": [], "narrower": []
        })
        fake = FakeLLM()
        with patch("src.query_expansion.get_llm", return_value=fake), patch(
            "src.query_expansion.get_query_expansion", return_value=cached
        ):
            result = expand_queries(["贈与"])
        self.assertEqual(result, ["贈与", "gift"])
        self.assertEqual(fake.calls, 0)

    def test_provider_failure_returns_original(self):
        fake = FakeLLM(error=ProviderUnavailable("missing key"))
        with patch("src.query_expansion.get_llm", return_value=fake), patch(
            "src.query_expansion.get_query_expansion", return_value=None
        ):
            self.assertEqual(expand_queries(["贈与"]), ["贈与"])

    def test_provider_failure_is_temporarily_suppressed(self):
        fake = FakeLLM(error=ProviderUnavailable("missing key"))
        with patch("src.query_expansion.get_llm", return_value=fake), patch(
            "src.query_expansion.get_query_expansion", return_value=None
        ), patch("src.query_expansion.time.monotonic", return_value=100.0):
            self.assertEqual(expand_queries(["贈与"]), ["贈与"])
            self.assertEqual(expand_queries(["交換"]), ["交換"])
        self.assertEqual(fake.calls, 1)

    def test_cache_failure_returns_original(self):
        fake = FakeLLM()
        with patch("src.query_expansion.get_llm", return_value=fake), patch(
            "src.query_expansion.get_query_expansion", side_effect=sqlite3.OperationalError("locked")
        ):
            self.assertEqual(expand_queries(["贈与"]), ["贈与"])


if __name__ == "__main__":
    unittest.main()
