"""search_chunks holds one item's vectors and scores them with a single matmul.

It used to cache every item it had ever seen, keyed by item_key and never
evicted, and to re-normalise every chunk vector on every query. Over a run of
the library that reached a 26 GB footprint on 513,683 chunks and drove the
machine into swap; the per-query renormalisation allocated one throwaway array
per chunk per citation context and dominated the runtime.
"""
from __future__ import annotations

import sys
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import citation_mapper  # noqa: E402


class UnitRowsTests(unittest.TestCase):
    def test_rows_become_unit_length(self):
        matrix = citation_mapper._unit_rows([[3.0, 4.0], [1.0, 0.0]])
        self.assertTrue(np.allclose(np.linalg.norm(matrix, axis=1), 1.0))

    def test_a_zero_vector_stays_zero_rather_than_becoming_nan(self):
        # Dividing by a zero norm makes every component inf/NaN, which then
        # propagates through the dot product and corrupts the ranking of every
        # other chunk, not just this one.
        matrix = citation_mapper._unit_rows([[0.0, 0.0], [3.0, 4.0]])
        self.assertTrue(np.isfinite(matrix).all())
        self.assertTrue(np.array_equal(matrix[0], np.zeros(2, dtype=np.float32)))

    def test_the_callers_array_is_not_rewritten_in_place(self):
        # np.asarray hands back the caller's own array when it is already
        # float32, and normalising into it would corrupt the source vectors.
        source = np.array([[3.0, 4.0]], dtype=np.float32)
        citation_mapper._unit_rows(source)
        self.assertTrue(np.array_equal(source, np.array([[3.0, 4.0]], dtype=np.float32)))

    def test_a_single_vector_is_accepted_as_one_row(self):
        self.assertEqual(citation_mapper._unit_rows([3.0, 4.0]).shape, (1, 2))


class SearchChunksTests(unittest.TestCase):
    def setUp(self):
        self._saved = (
            citation_mapper._ITEM_CHUNKS_CACHE_KEY,
            citation_mapper._ITEM_CHUNK_IDS,
            citation_mapper._ITEM_CHUNK_MATRIX,
        )
        rng = np.random.default_rng(0)
        self.vectors = rng.normal(size=(40, 16)).astype(np.float32)
        self.ids = [f"c{i}" for i in range(40)]
        citation_mapper._ITEM_CHUNKS_CACHE_KEY = "ITEM"
        citation_mapper._ITEM_CHUNK_IDS = self.ids
        citation_mapper._ITEM_CHUNK_MATRIX = citation_mapper._unit_rows(self.vectors)

    def tearDown(self):
        (citation_mapper._ITEM_CHUNKS_CACHE_KEY,
         citation_mapper._ITEM_CHUNK_IDS,
         citation_mapper._ITEM_CHUNK_MATRIX) = self._saved

    def _search(self, query, n_results):
        with unittest.mock.patch.object(
            citation_mapper, "_get_emb_fn", lambda: (lambda texts: [query])
        ), unittest.mock.patch.object(
            citation_mapper, "_get_segment_meta", lambda: {"metadata_segment_id": "s"}
        ):
            return citation_mapper.search_chunks("q", "ITEM", n_results=n_results)

    def _reference(self, query, n_results):
        """The pre-matmul implementation, kept here as the oracle."""
        unit = query / np.linalg.norm(query)
        scored = []
        for cid, vec in zip(self.ids, self.vectors):
            norm = np.linalg.norm(vec)
            vec = vec / norm if norm > 0 else vec
            scored.append({"id": cid, "distance": 1.0 - float(np.dot(unit, vec))})
        scored.sort(key=lambda row: row["distance"])
        return scored[:n_results]

    def test_matches_the_previous_implementation(self):
        query = np.random.default_rng(7).normal(size=16).astype(np.float32)
        for n in (1, 3, 40, 100):
            with self.subTest(n_results=n):
                got, want = self._search(query, n), self._reference(query, n)
                self.assertEqual([r["id"] for r in got], [r["id"] for r in want])
                for a, b in zip(got, want):
                    self.assertAlmostEqual(a["distance"], b["distance"], places=4)

    def test_an_item_with_no_chunks_returns_nothing(self):
        citation_mapper._ITEM_CHUNK_IDS, citation_mapper._ITEM_CHUNK_MATRIX = [], None
        self.assertEqual(self._search(np.ones(16, dtype=np.float32), 1), [])

    def test_scoring_emits_no_warnings(self):
        # matmul on macOS/Accelerate raises spurious FP warnings; unsuppressed
        # that is three lines per citation context.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._search(np.random.default_rng(1).normal(size=16).astype(np.float32), 1)


class OnlyOneItemIsHeldTests(unittest.TestCase):
    """The cache used to be a dict keyed by item_key that was never evicted."""

    def setUp(self):
        self._saved = (
            citation_mapper._ITEM_CHUNKS_CACHE_KEY,
            citation_mapper._ITEM_CHUNK_IDS,
            citation_mapper._ITEM_CHUNK_MATRIX,
        )

    def tearDown(self):
        (citation_mapper._ITEM_CHUNKS_CACHE_KEY,
         citation_mapper._ITEM_CHUNK_IDS,
         citation_mapper._ITEM_CHUNK_MATRIX) = self._saved

    def _load(self, item_key, rows, embed=None):
        class _Cursor:
            def fetchall(self_inner):
                return rows

        class _Conn:
            def execute(self_inner, *_a):
                return _Cursor()

            def close(self_inner):
                pass

        with unittest.mock.patch.object(
            citation_mapper, "_get_segment_meta", lambda: {"metadata_segment_id": "s"}
        ), unittest.mock.patch.object(
            citation_mapper, "_get_emb_fn",
            lambda: (embed or (lambda texts: [[1.0, 0.0] for _ in texts])),
        ), unittest.mock.patch.object(
            citation_mapper.sqlite3, "connect", lambda *a, **k: _Conn()
        ):
            return citation_mapper._load_chunks_for_item(item_key)

    def test_moving_to_the_next_item_releases_the_previous_one(self):
        self._load("ITEM_A", [("a1", "text")])
        self._load("ITEM_B", [("b1", "text"), ("b2", "text")])

        self.assertEqual(citation_mapper._ITEM_CHUNKS_CACHE_KEY, "ITEM_B")
        self.assertEqual(citation_mapper._ITEM_CHUNK_IDS, ["b1", "b2"])
        self.assertEqual(len(citation_mapper._ITEM_CHUNK_MATRIX), 2)

    def test_the_same_item_is_not_embedded_twice(self):
        calls = []

        def _embed(texts):
            calls.append(len(texts))
            return [[1.0, 0.0] for _ in texts]

        self._load("ITEM_A", [("a1", "text")], embed=_embed)
        self._load("ITEM_A", [("a1", "text")], embed=_embed)   # cached
        self.assertEqual(len(calls), 1)

    def test_an_item_with_no_rows_is_remembered_as_empty(self):
        ids, matrix = self._load("EMPTY", [])
        self.assertEqual(ids, [])
        self.assertIsNone(matrix)
        self.assertEqual(citation_mapper._ITEM_CHUNKS_CACHE_KEY, "EMPTY")


class StoredVectorsArePreferredTests(unittest.TestCase):
    """Re-embedding is a fallback, not the normal path.

    _load_chunks_for_item used to embed every chunk's text from scratch, on the
    stated belief that ChromaDB 1.5+ no longer persists the id->label mapping.
    It does: index_metadata.pickle holds all 513,683 entries. Measured on one
    2,861-chunk item, re-embedding took 131s against 1.9s to read the stored
    vectors, and produced the same ids and the same nearest neighbour 30 times
    out of 30.
    """

    def setUp(self):
        self._saved = (
            citation_mapper._ITEM_CHUNKS_CACHE_KEY,
            citation_mapper._ITEM_CHUNK_IDS,
            citation_mapper._ITEM_CHUNK_MATRIX,
        )
        citation_mapper._ITEM_CHUNKS_CACHE_KEY = None

    def tearDown(self):
        (citation_mapper._ITEM_CHUNKS_CACHE_KEY,
         citation_mapper._ITEM_CHUNK_IDS,
         citation_mapper._ITEM_CHUNK_MATRIX) = self._saved

    def test_stored_vectors_are_used_without_embedding(self):
        embedded = []
        with unittest.mock.patch.object(
            citation_mapper, "_stored_item_vectors",
            lambda key: (["a", "b"], citation_mapper._unit_rows([[1.0, 0.0], [0.0, 1.0]])),
        ), unittest.mock.patch.object(
            citation_mapper, "_get_emb_fn", lambda: embedded.append(1),
        ), unittest.mock.patch.object(
            citation_mapper, "_get_segment_meta", lambda: {"metadata_segment_id": "s"},
        ):
            ids, matrix = citation_mapper._load_chunks_for_item("ITEM")

        self.assertEqual(ids, ["a", "b"])
        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(embedded, [])

    def test_embedding_still_runs_when_no_vectors_are_stored(self):
        # A stale or missing index must still yield an answer rather than
        # silently returning nothing.
        rows = [("c1", "text one"), ("c2", "text two")]

        class _Conn:
            def execute(self_inner, *_a):
                class _C:
                    def fetchall(self_c):
                        return rows
                return _C()

            def close(self_inner):
                pass

        with unittest.mock.patch.object(
            citation_mapper, "_stored_item_vectors", lambda key: None,
        ), unittest.mock.patch.object(
            citation_mapper, "_get_segment_meta", lambda: {"metadata_segment_id": "s"},
        ), unittest.mock.patch.object(
            citation_mapper.sqlite3, "connect", lambda *a, **k: _Conn(),
        ), unittest.mock.patch.object(
            citation_mapper, "_get_emb_fn",
            lambda: (lambda texts: [[1.0, 0.0] for _ in texts]),
        ):
            ids, matrix = citation_mapper._load_chunks_for_item("ITEM")

        self.assertEqual(ids, ["c1", "c2"])
        self.assertEqual(matrix.shape, (2, 2))

    def test_an_unreadable_index_returns_none_instead_of_raising(self):
        # A missing or unreadable Chroma directory must degrade to the embedding
        # fallback, not abort the citation run.
        with unittest.mock.patch.object(
            citation_mapper, "CHROMA_DIR", Path("/nonexistent/chroma"),
        ):
            self.assertIsNone(citation_mapper._stored_item_vectors("NOPE"))


if __name__ == "__main__":
    import unittest.mock  # noqa: F401
    unittest.main()
