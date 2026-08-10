"""What the read path does with a bad request, a broken store, and a big k.

Nothing here can corrupt the library -- these are queries. What they can do is
answer wrongly and look right, which for a retrieval tool is the same damage
one step later: a search that quietly returns nothing is indistinguishable from
a library that contains nothing, and the user's next move is to conclude the
material is not there.

Three shapes, each of which was present:

* an unbounded internal fetch (``k_internal = max(k * 10, 100)``), so a large
  ``k`` asked Chroma for an arbitrarily large result set in one call while the
  sibling ``rag_search`` had been given a cap for exactly that;
* a request error reported as a store error -- an empty string in the query
  list reached the embedding function, raised, and came back as the HNSW index
  message, blaming the index for what the request said;
* a store error reported as an absence -- a Chroma failure in ``_chunk_by_id``
  became "chunk_id not found in the active collection", told to the reader who
  had just found damaged text and wanted to report it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import rag_mcp_server as server  # noqa: E402


class _Collection:
    """Enough of a Chroma collection to answer one query."""

    def __init__(self, count=5000):
        self._count = count
        self.n_results_seen: list[int] = []

    def count(self):
        return self._count

    def _embedding_function(self, queries):
        return [[0.0, 1.0] for _ in queries]

    def query(self, *, query_embeddings, n_results, where, include):
        self.n_results_seen.append(n_results)
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}


def _search(query="a query", k=10, collection=None):
    collection = collection or _Collection()
    with (
        patch.object(server, "_col", return_value=collection),
        patch.object(server, "_check_indexing_lock", return_value=(False, None)),
    ):
        return server.search_items(query=query, k=k), collection


def test_a_huge_k_does_not_ask_the_store_for_everything():
    _result, collection = _search(k=100_000)
    assert collection.n_results_seen, "the query never ran"
    assert max(collection.n_results_seen) <= 10_000, (
        f"asked Chroma for {max(collection.n_results_seen)} chunks in one call; "
        "the cap that rag_search has is what stops this"
    )


def test_the_internal_fetch_never_exceeds_the_collection():
    _result, collection = _search(k=500, collection=_Collection(count=37))
    assert max(collection.n_results_seen) <= 37, (
        "asked for more chunks than the collection holds"
    )


def test_an_empty_query_is_a_request_error_not_an_index_error():
    result, collection = _search(query=["", "   "])
    assert not collection.n_results_seen, "an empty query reached the store"
    assert "query" in (result.get("warning") or "").lower(), (
        f"the caller is told {result.get('warning')!r}, which does not say the "
        "request was the problem"
    )
    assert "error" not in result, "a request error is reported as a store error"


def test_a_store_failure_is_not_reported_as_a_missing_chunk():
    class Broken:
        def get(self, **_kwargs):
            raise RuntimeError("chroma is down")

    with patch.object(server, "_col", return_value=Broken()):
        result = server.report_chunk_quality(
            item_key="ITEM", chunk_id="CHUNK:1", reason="ocr_garbled",
            details="the text is unreadable",
        )
    assert result["status"] == "error"
    assert "not found" not in result["message"], (
        "a broken store is reported to the reader as a wrong chunk id, so the "
        f"damage report is dropped and blamed on them: {result['message']!r}"
    )
    assert "chroma is down" in result["message"]


def test_an_unreadable_lock_is_not_read_as_no_lock(tmp_path):
    # Serving during a write is the thing the lock exists to prevent, and a
    # lock this process cannot parse is not evidence that nobody is writing.
    lock = tmp_path / "indexing.lock"
    lock.write_text("{ this is not json", encoding="utf-8")
    with patch.object(server, "INDEXING_LOCK_PATH", str(lock)):
        blocked, message = server._check_indexing_lock()
    assert blocked, "an unreadable lock was treated as an absent one"
    assert str(lock) in (message or "")


def test_an_old_unreadable_lock_is_still_treated_as_stale(tmp_path):
    # The other half: a corrupt file must not take search down forever, so the
    # evidence that remains -- the file's age -- decides.
    import os
    import time

    lock = tmp_path / "indexing.lock"
    lock.write_text("{ this is not json", encoding="utf-8")
    old = time.time() - (server._LOCK_STALE_HOURS + 1) * 3600
    os.utime(lock, (old, old))
    with patch.object(server, "INDEXING_LOCK_PATH", str(lock)):
        blocked, _message = server._check_indexing_lock()
    assert not blocked


def test_a_lock_that_vanishes_mid_check_does_not_block_the_answer(tmp_path):
    # The narrow race the age fallback has to get right: the indexer releases
    # the lock between the read that fails and the stat that would date it.
    # Treating that as "recently written" told the caller to hand-delete a file
    # that no longer exists.
    lock = tmp_path / "indexing.lock"
    lock.write_text("{ this is not json", encoding="utf-8")
    with (
        patch.object(server, "INDEXING_LOCK_PATH", str(lock)),
        patch("os.path.getmtime", side_effect=OSError("gone")),
    ):
        blocked, message = server._check_indexing_lock()
    assert not blocked
    assert message is None


def test_the_server_asks_the_same_place_where_the_lock_is():
    import indexing_lock

    assert Path(server.INDEXING_LOCK_PATH) == indexing_lock.default_path(Path(server.ROOT))


def test_an_explicitly_unvalidated_generation_cannot_open_for_queries(tmp_path):
    manifest = tmp_path / "manifest_v3.json"
    manifest.write_text('{"hnsw_validated": false}', encoding="utf-8")
    absent_lock = tmp_path / "indexing.lock"

    # A cached collection must not bypass the durable generation state. This
    # is the post-crash shape: the process lock is gone, but the clean rebuild
    # never reached its completeness and HNSW publication boundary.
    with patch.object(server, "MANIFEST_PATH", str(manifest)), \
            patch.object(server, "INDEXING_LOCK_PATH", str(absent_lock)), \
            patch.object(server, "_COL", object()):
        blocked, message = server._check_indexing_lock()
        assert blocked
        assert "incomplete" in (message or "")
        with pytest.raises(RuntimeError, match="incomplete.*HNSW"):
            server._col()


def test_a_validated_generation_remains_queryable(tmp_path):
    manifest = tmp_path / "manifest_v3.json"
    manifest.write_text('{"hnsw_validated": true}', encoding="utf-8")
    collection = object()

    with patch.object(server, "MANIFEST_PATH", str(manifest)), \
            patch.object(server, "_COL", collection), \
            patch.object(server, "_db_mtimes", return_value=(0.0, 0.0)), \
            patch.object(server, "_COL_INIT_DB_MTIME", 0.0), \
            patch.object(server, "_COL_INIT_MANIFEST_MTIME", 0.0):
        assert server._col() is collection


def test_generation_guard_distinguishes_absent_and_corrupt_manifests(tmp_path):
    manifest = tmp_path / "manifest_v3.json"
    with patch.object(server, "MANIFEST_PATH", str(manifest)):
        assert server._index_generation_problem() is None
        manifest.write_text("{not json", encoding="utf-8")
        assert "unreadable" in (server._index_generation_problem() or "")
        manifest.write_text("[]", encoding="utf-8")
        assert "root" in (server._index_generation_problem() or "")


def test_server_status_surfaces_an_unvalidated_generation(tmp_path):
    manifest = tmp_path / "manifest_v3.json"
    manifest.write_text('{"hnsw_validated": false}', encoding="utf-8")
    absent_chroma = tmp_path / "missing-chroma"

    with patch.object(server, "MANIFEST_PATH", str(manifest)), \
            patch.object(server, "CHROMA_DIR", str(absent_chroma)), \
            patch.object(server, "_check_writer_lock", return_value=(False, None)):
        report = server.server_status()

    assert report["status"] == "error"
    assert report["index_generation_ready"] is False
    assert any("incomplete" in error for error in report["errors"])
