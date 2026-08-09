"""Break a whole indexing run partway through and check what survives.

The re-OCR adoption was the sharpest case of "five stores, no transaction, a
hand-written except block". The ingest loop is the largest one: it writes the
same stores, over many attachments, in flushed batches, and the only end-to-end
check it had ran the real library through a child process and could compare
nothing but printed output.

With the run callable in-process against a synthetic library, a failure can be
injected into a specific phase and the plane read directly afterwards. The
question each case asks is the one a repair script exists to answer: **do the
stores still agree, and does the manifest claim only what is actually there.**

The invariant is not "nothing was written". Ingest commits per flushed batch on
purpose, so a run that dies after the first batch is meant to keep it -- that is
what makes a re-run cheap. What must never happen is a manifest that records an
attachment whose chunks are absent, or chunks that no manifest entry admits to,
because every consistency repair in this project starts from one of those.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tests.data_plane_fixture import temporary_data_plane  # noqa: E402
from tests.synthetic_library import (  # noqa: E402
    SyntheticZoteroSource, build_library, deterministic_embedding_function,
)


class BreakableCollection:
    """Wraps the real Chroma collection and fails a chosen call."""

    def __init__(self, inner, fail_on: tuple[str, int] | None):
        self._inner = inner
        self._fail_call, self._fail_at = fail_on or ("", 0)
        self._counts: dict[str, int] = {}

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _count(self, name: str) -> None:
        self._counts[name] = self._counts.get(name, 0) + 1
        if name == self._fail_call and self._counts[name] == self._fail_at:
            raise RuntimeError(f"injected failure in collection.{name}")

    def upsert(self, *args, **kwargs):
        self._count("upsert")
        return self._inner.upsert(*args, **kwargs)

    def delete(self, *args, **kwargs):
        self._count("delete")
        return self._inner.delete(*args, **kwargs)


def _open_collection(fail_on=None):
    def factory(**kwargs):
        from embedder import open_chroma_collection

        collection = open_chroma_collection(
            kwargs["chroma_dir"],
            kwargs.get("chroma_collection_env") or kwargs["chroma_collection_default"],
            deterministic_embedding_function(),
        )
        collection._zotero_embedder_config = {
            "provider": "synthetic", "model_name": "synthetic-hash-32",
            "dimension": 32, "collection": collection.name,
        }
        return BreakableCollection(collection, fail_on)

    return factory


def _run(source, *argv: str, fail_on=None, flush_size=1):
    import index_from_zotero

    with patch.object(sys, "argv", ["index_from_zotero.py", *argv]):
        args = index_from_zotero.parse_args()
    # One attachment per flush, so a failure lands between commits rather than
    # in the single end-of-run batch three documents would otherwise produce.
    with patch.object(index_from_zotero, "FLUSH_SIZE", flush_size):
        return asyncio.run(index_from_zotero.main_async(
            args, source=source, open_collection=_open_collection(fail_on),
        ))


def _chunk_ids(plane) -> set[str]:
    import chromadb

    client = chromadb.PersistentClient(path=str(plane.chroma_dir))
    try:
        collection = client.get_collection(plane.collection_name)
    except Exception:
        return set()
    return set(collection.get(include=[])["ids"])


def _lexical_ids(plane) -> set[str]:
    path = plane.project_root / "lexical_v3.sqlite3"
    if not path.exists():
        return set()
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        try:
            rows = connection.execute("SELECT chunk_id FROM chunks_fts").fetchall()
        except sqlite3.OperationalError:
            return set()
    return {row[0] for row in rows}


def _manifest(plane) -> dict:
    if not plane.manifest_path.exists():
        return {"files": {}}
    return json.loads(plane.manifest_path.read_text(encoding="utf-8"))


def _attachments_with_chunks(plane) -> set[str]:
    import chromadb

    client = chromadb.PersistentClient(path=str(plane.chroma_dir))
    try:
        collection = client.get_collection(plane.collection_name)
    except Exception:
        return set()
    rows = collection.get(include=["metadatas"])
    return {
        str(metadata.get("attachmentKey"))
        for metadata in rows["metadatas"] if metadata.get("attachmentKey")
    }


def _assert_stores_agree(plane, where: str) -> None:
    chroma, lexical = _chunk_ids(plane), _lexical_ids(plane)
    assert chroma == lexical, (
        f"{where}: Chroma and the lexical index disagree -- "
        f"{len(chroma - lexical)} only in Chroma, {len(lexical - chroma)} only in the index"
    )
    recorded = set(_manifest(plane).get("files") or {})
    present = _attachments_with_chunks(plane)
    assert recorded <= present, (
        f"{where}: the manifest records attachments with no chunks: "
        f"{sorted(recorded - present)}. A re-run trusts the manifest and will "
        "skip them, so the gap is permanent."
    )
    assert present <= recorded, (
        f"{where}: chunks exist for attachments the manifest does not record: "
        f"{sorted(present - recorded)}. Nothing will ever delete or refresh them."
    )


PHASES = [
    ("chroma upsert, first flush", ("upsert", 1)),
    ("chroma upsert, second flush", ("upsert", 2)),
    ("chroma delete, second flush", ("delete", 2)),
]


@pytest.mark.parametrize("name,fail_on", PHASES, ids=[row[0] for row in PHASES])
def test_a_run_that_dies_partway_leaves_the_stores_agreeing(tmp_path, name, fail_on):
    with temporary_data_plane(tmp_path) as plane:
        source = SyntheticZoteroSource(build_library(tmp_path))
        with pytest.raises(BaseException):
            _run(source, fail_on=fail_on)
        _assert_stores_agree(plane, name)


def test_a_re_run_after_a_failure_finishes_the_job(tmp_path):
    # The point of committing per batch: the second run does the rest rather
    # than starting over, and ends with everything indexed exactly once.
    with temporary_data_plane(tmp_path) as plane:
        source = SyntheticZoteroSource(build_library(tmp_path))
        with pytest.raises(BaseException):
            _run(source, fail_on=("upsert", 2))
        _assert_stores_agree(plane, "after the failed run")

        # No lock cleanup between the two runs: main_async releases in a
        # finally now, so a failed run leaves the lock free for the next one.
        # This line is the check -- without the release, the second run exits
        # with "another indexer is running" naming this very process.
        _run(source)
        _assert_stores_agree(plane, "after the re-run")
        assert set(_manifest(plane)["files"]) == {"SYNPDF001", "SYNHTML01", "SYNEPUB01"}, (
            "the re-run did not pick up the attachments the failed run missed"
        )


def test_a_failed_run_does_not_leave_attachments_marked_in_flight(tmp_path):
    # inflight_attachments is the recovery marker: an entry that outlives the
    # run it describes tells the next one to redo work that is already done, or
    # worse, is read as evidence of a partial write that is not there.
    with temporary_data_plane(tmp_path) as plane:
        source = SyntheticZoteroSource(build_library(tmp_path))
        with pytest.raises(BaseException):
            _run(source, fail_on=("upsert", 2))
        inflight = _manifest(plane).get("inflight_attachments") or []
        present = _attachments_with_chunks(plane)
        assert not (set(inflight) & present), (
            f"attachments {sorted(set(inflight) & present)} are both written and "
            "still marked in flight"
        )


def test_a_failed_run_does_not_keep_the_indexing_lock(tmp_path):
    # The lock is what stops the MCP server serving mid-write, so a run that
    # dies holding it takes search down until someone deletes a file by hand.
    import index_from_zotero

    with temporary_data_plane(tmp_path) as plane:
        source = SyntheticZoteroSource(build_library(tmp_path))
        with pytest.raises(BaseException):
            _run(source, fail_on=("upsert", 2))
        assert not plane.indexing_lock_path.exists(), (
            "the lock outlived the run that took it"
        )
        assert index_from_zotero._HELD_LOCK is None
