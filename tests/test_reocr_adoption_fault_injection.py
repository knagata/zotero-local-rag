"""Break the re-OCR adoption at every phase and check what it leaves behind.

Adoption writes five stores by hand: Chroma, the lexical index, the manifest,
the document tree, and the artifact status rows. There is no transaction across
them, so what stands in for one is an ``except`` block that undoes the writes
that already happened. 83% of the ``except`` handlers in this project have never
been executed by a test, and the defects this project has actually shipped live
in exactly these blocks -- the rollback that ran in the wrong order and dropped
an attachment's chunks for good (F1, 2026-07-30), the read failure that passed
as success (F4), the delete failure that was swallowed (F5).

So rather than pick the failure that seems interesting, this fails *every* phase
in turn and asserts the same thing each time: an adoption either happened or did
not, and the stores agree about which. The table is generated from the phases,
so a new write in ``adopt_prepared_reocr`` that is not compensated shows up here
without anyone deciding to test it.

Two facts this found, both of which the code did not survive when it was
written:

* a failure while restoring Chroma skipped the lexical and manifest restores
  *and* the ``failed`` status write, because they were sequential statements in
  one ``except`` block;
* a failure after the document tree was replaced rolled the chunks back but left
  the tree, which is the mixed state the rollback exists to prevent.
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from src.manifest import load_manifest, save_manifest  # noqa: E402
from src.reocr_adoption import adopt_prepared_reocr  # noqa: E402


class RecordingCollection:
    """A Chroma stand-in that can be told to fail its Nth call."""

    def __init__(self):
        self.rows: dict[str, tuple[str, dict]] = {}
        self.fail_on: dict[str, int] = {}
        self.calls: dict[str, int] = {"delete": 0, "upsert": 0}

    def _maybe_fail(self, name: str) -> None:
        self.calls[name] += 1
        if self.fail_on.get(name) == self.calls[name]:
            raise RuntimeError(f"collection.{name} failed on call {self.calls[name]}")

    def delete(self, *, where):
        self._maybe_fail("delete")
        attachment = where["attachmentKey"]
        self.rows = {
            key: row for key, row in self.rows.items()
            if row[1].get("attachmentKey") != attachment
        }

    def upsert(self, *, ids, documents, metadatas):
        self._maybe_fail("upsert")
        for chunk_id, document, metadata in zip(ids, documents, metadatas, strict=False):
            self.rows[chunk_id] = (document, dict(metadata))


def _old_chunks() -> list[dict[str, Any]]:
    return [{
        "id": "ATT:p1:old", "text": "Old readable source text. " * 8,
        "metadata": {
            "itemKey": "ITEM", "attachmentKey": "ATT", "source_type": "pdf",
            "page": 1, "locator": "p1", "lang": "en", "title": "Test",
        },
    }]


def _prepared() -> dict[str, Any]:
    return {
        "engine": "docling", "version": "2",
        "quality": {"total_pages": 1, "page_coverage": 1},
        "blocks": [{
            "text": "New readable OCR source text. " * 8,
            "metadata": {"page": 1, "locator": "p1", "structure_path": ["Chapter 1"]},
        }],
    }


class Plane:
    """The five stores, as a test can see them."""

    def __init__(self, root: Path):
        self.root = root
        self.collection = RecordingCollection()
        rows = _old_chunks()
        self.collection.upsert(
            ids=[rows[0]["id"]], documents=[rows[0]["text"]],
            metadatas=[rows[0]["metadata"]],
        )
        self.old_rows = rows
        # Seeding the old rows is not part of the run under test; counting it
        # would aim every injection one call too early.
        self.collection.calls = {"delete": 0, "upsert": 0}
        self.manifest_path = root / "manifest_v3.json"
        self.lexical_path = root / "lexical.sqlite3"
        # The lexical index holds the old rows before an adoption starts, so a
        # rollback restores them rather than inventing them. Seeding an empty
        # index instead made a correct restore look like a change.
        from lexical_index import upsert_chunks

        upsert_chunks(
            [rows[0]["id"]], [rows[0]["text"]], [rows[0]["metadata"]],
            path=self.lexical_path,
        )
        save_manifest(self.manifest_path, {
            "version": 1, "files": {"ATT": {"title": "Before"}}, "notes": {},
        })
        self.structure_replaced = 0
        self.summaries_deleted = 0
        self.statuses: list[tuple] = []

    def lexical_ids(self) -> set[str]:
        if not self.lexical_path.exists():
            return set()
        with sqlite3.connect(f"file:{self.lexical_path}?mode=ro", uri=True) as connection:
            try:
                rows = connection.execute("SELECT chunk_id FROM chunks_fts").fetchall()
            except sqlite3.OperationalError:
                return set()
        return {row[0] for row in rows}

    def snapshot(self) -> dict[str, Any]:
        return {
            "chroma": set(self.collection.rows),
            "lexical": self.lexical_ids(),
            "manifest": load_manifest(self.manifest_path),
            "structure_replaced": self.structure_replaced,
        }

    def adopt(self):
        def replace_structure(*args, **kwargs):
            self.structure_replaced += 1
            if self.fail_structure:
                raise RuntimeError("structure write failed")

        def delete_summaries(*args, **kwargs):
            self.summaries_deleted += 1
            if self.fail_summaries:
                raise RuntimeError("summary cascade failed")

        def status_writer(*args, **kwargs):
            self.statuses.append((args, kwargs))
            if self.fail_status_on and len(self.statuses) == self.fail_status_on:
                raise RuntimeError("status write failed")

        with (
            patch("src.reocr_adoption.replace_document_structure", replace_structure),
            patch("src.reocr_adoption.delete_document_node_summaries", delete_summaries),
        ):
            return adopt_prepared_reocr(
                item_key="ITEM", attachment_key="ATT", prepared=_prepared(),
                collection=self.collection, old_item_chunks=self.old_rows,
                manifest_path=self.manifest_path, lexical_path=self.lexical_path,
                status_writer=status_writer,
            )

    fail_structure = False
    fail_summaries = False
    fail_status_on = 0


#: Every phase that can fail, how to make it fail, and what must still hold.
#:
#: ``after_canonical`` marks the phases that happen once Chroma, the lexical
#: index, the manifest and the tree are all consistently new -- past that point
#: the adoption has succeeded and bookkeeping must not undo it.
#:
#: ``must_restore`` is which stores the rollback is still answerable for. When
#: the injected failure *is* Chroma refusing a call, Chroma cannot be restored
#: by calling Chroma; what is not excused is abandoning the other stores and
#: the record that the run failed, which is what a sequential except block did.
PHASES = [
    ("chroma delete", {"collection_fail": ("delete", 1)}, False, {"chroma", "lexical", "manifest"}),
    ("chroma upsert", {"collection_fail": ("upsert", 1)}, False, {"chroma", "lexical", "manifest"}),
    ("structure write", {"fail_structure": True}, False, {"chroma", "lexical", "manifest"}),
    ("rollback's chroma delete",
     {"collection_fail": ("delete", 2), "fail_structure": True}, False, {"lexical", "manifest"}),
    ("rollback's chroma upsert",
     {"collection_fail": ("upsert", 2), "fail_structure": True}, False, {"lexical", "manifest"}),
    # The cascade is part of writing the tree, not bookkeeping about it: stale
    # summaries of a replaced structure are wrong answers, not a missing record.
    ("summary cascade", {"fail_summaries": True}, False, {"chroma", "lexical", "manifest"}),
    ("first status write", {"fail_status_on": 2}, True, set()),
    ("last status write", {"fail_status_on": 6}, True, set()),
]


@pytest.mark.parametrize(
    "name,injection,after_canonical,must_restore", PHASES, ids=[row[0] for row in PHASES],
)
def test_a_failure_leaves_the_stores_agreeing(name, injection, after_canonical, must_restore):
    with tempfile.TemporaryDirectory() as directory:
        plane = Plane(Path(directory))
        before = plane.snapshot()
        if "collection_fail" in injection:
            call, ordinal = injection["collection_fail"]
            plane.collection.fail_on[call] = ordinal
        plane.fail_structure = injection.get("fail_structure", False)
        plane.fail_summaries = injection.get("fail_summaries", False)
        plane.fail_status_on = injection.get("fail_status_on", 0)

        result = None
        try:
            result = plane.adopt()
            failed = False
        except Exception:
            failed = True

        after = plane.snapshot()
        if after_canonical:
            # The canonical stores were all written before this phase, so the
            # adoption stands. Bookkeeping that fails afterwards must be
            # reported, not compensated by undoing correct data.
            assert after["chroma"] != before["chroma"], (
                f"{name}: the adoption was undone even though every canonical "
                "store had already been written consistently"
            )
            # And it has to be reported as the success it is. Raising here
            # tells the caller the adoption failed, and the caller's answer to
            # a failed adoption is to run the whole expensive thing again.
            assert not failed, (
                f"{name}: bookkeeping failed after the data was written and the "
                "call raised, so a completed adoption looks like a failed one"
            )
            assert result["status_write_errors"], (
                f"{name}: the failed status write was neither raised nor reported"
            )
            return

        assert failed, f"{name}: the failure was swallowed and the run reported success"
        for store in must_restore:
            assert after[store] == before[store], f"{name}: {store} was left changed"
        assert ("extraction", "failed") in [args[1:3] for args, _ in plane.statuses], (
            f"{name}: nothing recorded that the adoption failed, so the next run "
            "has no reason to retry it"
        )


def test_a_failed_adoption_always_records_that_it_failed(monkeypatch):
    # The status row is how the next run knows to retry. It was written as the
    # last statement of a sequential except block, so any earlier compensation
    # failing meant no record at all -- the run vanished.
    with tempfile.TemporaryDirectory() as directory:
        plane = Plane(Path(directory))
        plane.fail_structure = True
        plane.collection.fail_on["delete"] = 2  # the rollback's own delete
        with pytest.raises(Exception):
            plane.adopt()
        kinds = [args[1:3] for args, _ in plane.statuses]
        assert ("extraction", "failed") in kinds, (
            "a rollback that itself failed left no record that the adoption failed"
        )


def test_a_rollback_failure_names_both_what_failed_and_what_is_inconsistent():
    # One error carrying the original failure and every compensation that could
    # not be completed: without it the operator sees the rollback's error and
    # never learns which write started it, or which store is now wrong.
    with tempfile.TemporaryDirectory() as directory:
        plane = Plane(Path(directory))
        plane.fail_structure = True
        plane.collection.fail_on["upsert"] = 2
        with pytest.raises(Exception) as caught:
            plane.adopt()
        message = str(caught.value)
        assert "structure write failed" in message, (
            "the original failure is not in the error the caller sees"
        )
        assert "collection.upsert" in message, (
            "the compensation that failed is not named"
        )
