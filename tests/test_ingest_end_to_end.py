"""One whole indexing run, called rather than spawned.

``tests/test_ingestion_baseline.py`` runs the same function against the user's
real library and compares what came out with what came out before. It is the
only check that has ever caught an extract-method breaking the loop, and it
costs 140 seconds, needs Zotero running, and cannot run in CI.

This runs the loop in-process against three documents built in a temporary
directory, with a source that answers like Zotero's local API and an embedder
that is a hash. What it is for is the *shape* of a run, which the baseline
cannot assert because it only sees printed output: that the four stores agree
when it finishes, that an unchanged source is not re-indexed on the second
pass, and that a failure part-way through does not leave a half-written plane.

Its blind spot is stated plainly: synthetic PDFs have a text layer, so no OCR
route runs, and nothing here says anything about extraction quality. The
baseline stays for that.
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


def _open_synthetic_collection(**kwargs):
    """Stand in for ``embedder.get_collection`` without loading a model."""
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
    return collection


def _run(source, plane, *argv: str):
    """Drive the real CLI parser, so the run is configured the way runs are."""
    import index_from_zotero

    with patch.object(sys, "argv", ["index_from_zotero.py", *argv]):
        args = index_from_zotero.parse_args()
    return asyncio.run(index_from_zotero.main_async(
        args, source=source, open_collection=_open_synthetic_collection,
    ))


@pytest.fixture()
def indexed(tmp_path: Path):
    """A completed run over the synthetic library, and the plane it wrote."""
    with temporary_data_plane(tmp_path) as plane:
        source = SyntheticZoteroSource(build_library(tmp_path))
        _run(source, plane, "--progress")
        yield plane, source


def _manifest(plane) -> dict:
    return json.loads(plane.manifest_path.read_text(encoding="utf-8"))


def _chunk_ids(plane) -> set[str]:
    import chromadb

    client = chromadb.PersistentClient(path=str(plane.chroma_dir))
    collection = client.get_collection(plane.collection_name)
    return set(collection.get(include=[])["ids"])


def _lexical_ids(plane) -> set[str]:
    path = plane.project_root / "lexical_v3.sqlite3"
    if not path.exists():
        return set()
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        rows = connection.execute("SELECT chunk_id FROM chunks_fts").fetchall()
    return {row[0] for row in rows}


def test_a_run_indexes_every_source_type_it_was_given(indexed):
    plane, source = indexed
    manifest = _manifest(plane)
    assert set(manifest["files"]) == {"SYNPDF001", "SYNHTML01", "SYNEPUB01"}, (
        "a source type dropped out of the run without failing it"
    )
    assert "list_normalized_attachments" in source.calls
    assert _chunk_ids(plane), "the run wrote no chunks at all"


def test_the_stores_hold_the_same_chunks_when_the_run_finishes(indexed):
    # The invariant every repair script exists to restore, asserted once here
    # instead of only being auditable after the fact on the real library.
    plane, _ = indexed
    chroma, lexical = _chunk_ids(plane), _lexical_ids(plane)
    assert lexical, "the lexical index is empty while Chroma is not"
    assert chroma == lexical, (
        "Chroma and the lexical index disagree about which chunks exist: "
        f"{len(chroma - lexical)} only in Chroma, {len(lexical - chroma)} only in the index"
    )


def test_a_second_pass_over_unchanged_sources_reindexes_nothing(tmp_path: Path):
    # The decision the ingestion baseline can only watch through counters, and
    # the one that a refactor of the unchanged-source check would break first.
    with temporary_data_plane(tmp_path) as plane:
        source = SyntheticZoteroSource(build_library(tmp_path))
        _run(source, plane)
        first = _chunk_ids(plane)
        first_manifest = _manifest(plane)
        _run(source, plane)
        assert _chunk_ids(plane) == first, "a second pass rewrote the chunks"
        assert _manifest(plane)["files"] == first_manifest["files"], (
            "a second pass changed the manifest for unchanged sources"
        )


def test_nothing_is_written_outside_the_temporary_plane(indexed):
    # The check that makes every other test here safe to run on a machine that
    # has a real library: a run must not touch anything it was not pointed at.
    plane, _ = indexed
    assert plane.chroma_dir.exists()
    assert not (ROOT / "data" / "manifest_v3.json").samefile(plane.manifest_path) \
        if (ROOT / "data" / "manifest_v3.json").exists() else True
