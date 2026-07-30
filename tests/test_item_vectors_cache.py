from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.item_vectors import get_item_vectors


def _embeddings(count: int, value: float) -> list[list[float]]:
    return [[value, value] for _ in range(count)]


class FakeCollection:
    """Minimal Chroma collection stand-in: two chunks for ITEM at first,
    reduced to one chunk after a simulated re-ingestion."""

    def __init__(self, chunk_count: int, value: float):
        self.name = "zotero_paragraphs_v3"
        self.chunk_count = chunk_count
        self.value = value

    def peek(self, _n):
        return {"embeddings": [[self.value, self.value]]}

    def get(self, *, where=None, include=None):
        ids = [f"chunk-{i}" for i in range(self.chunk_count)]
        result = {"ids": ids}
        if include and "embeddings" in include:
            result["embeddings"] = _embeddings(self.chunk_count, self.value)
        return result


class ItemVectorsCacheTests(unittest.TestCase):
    def test_first_call_computes_and_persists_chunk_count(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.json"
            collection = FakeCollection(chunk_count=2, value=1.0)
            with patch(
                "src.item_vectors._open_collection",
                return_value=(Mock(), collection),
            ):
                result = get_item_vectors(["ITEM"], cache_path=cache_path)
            self.assertIn("ITEM", result)
            saved = json.loads(cache_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["ITEM"]["n"], 2)

    def test_stale_vector_is_recomputed_after_reingestion_changes_chunk_count(self):
        # 2026-07-30 regression: a per-item vector used to be cached forever
        # once computed (only a collection/dimension change invalidated the
        # whole cache), so re-OCRing an item with a materially different
        # embedding kept serving the stale pre-reingestion vector to
        # related_items() indefinitely, with no error surfaced.
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.json"
            first_collection = FakeCollection(chunk_count=2, value=1.0)
            with patch(
                "src.item_vectors._open_collection",
                return_value=(Mock(), first_collection),
            ):
                first_result = get_item_vectors(["ITEM"], cache_path=cache_path)

            # Simulate a re-ingestion that changes the item's chunk count and
            # embedding content (e.g. via --force-reparse).
            second_collection = FakeCollection(chunk_count=1, value=-1.0)
            with patch(
                "src.item_vectors._open_collection",
                return_value=(Mock(), second_collection),
            ):
                second_result = get_item_vectors(["ITEM"], cache_path=cache_path)

            self.assertNotEqual(first_result["ITEM"], second_result["ITEM"])
            saved = json.loads(cache_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["ITEM"]["n"], 1)

    def test_legacy_plain_list_cache_entry_is_migrated_on_next_request(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.json"
            cache_path.write_text(json.dumps({
                "__meta__": {"embedding_dim": 2, "collection": "zotero_paragraphs_v3"},
                "ITEM": [0.5, 0.5],  # old format: no "n" chunk count
            }), encoding="utf-8")
            collection = FakeCollection(chunk_count=2, value=1.0)
            with patch(
                "src.item_vectors._open_collection",
                return_value=(Mock(), collection),
            ):
                result = get_item_vectors(["ITEM"], cache_path=cache_path)
            self.assertIn("ITEM", result)
            saved = json.loads(cache_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["ITEM"]["n"], 2)


if __name__ == "__main__":
    unittest.main()
