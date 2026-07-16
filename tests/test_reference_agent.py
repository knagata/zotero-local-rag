from __future__ import annotations

import tempfile
import unittest
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

    def test_identifier_resolution_is_cached(self):
        reference = {
            "raw": "Example", "authors": ["Author"], "title": "Work",
            "year": 2020, "doi": "10.1234/example", "isbn": None,
        }
        first = reference_agent.resolve_reference(reference)
        second = reference_agent.resolve_reference(reference)
        self.assertEqual(first, second)
        self.assertEqual(first["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
