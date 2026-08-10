from __future__ import annotations

import unittest
from unittest.mock import patch

from src.note_extract import index_notes


class FakeCollection:
    def __init__(self, delete_error=None):
        self.deleted = []
        self.delete_error = delete_error

    def delete(self, **kwargs):
        if self.delete_error:
            raise self.delete_error
        self.deleted.append(kwargs)


def _identity_dedupe(ids, docs, metas):
    return ids, docs, metas


def _note(note_key="N1", version=1, html=None):
    return {
        "noteKey": note_key,
        "version": version,
        "parentItemKey": "ITEM1",
        "title": "A note",
        "year": 2024,
        "creators": ["Ada Author", "Grace Writer"],
        "language": "en",
        "note_html": html if html is not None else (
            "<html><body><p>"
            + "This is a meaningful note sentence for retrieval. " * 12
            + "</p></body></html>"
        ),
    }


class NoteExtractContractTests(unittest.TestCase):
    def _run(self, notes, manifest=None, *, collection=None, lexical_delete=None,
             delete_stale=True, strict_lexical=False, upsert=None):
        collection = collection or FakeCollection()
        upsert = upsert or (lambda *args, **kwargs: None)
        return index_notes(
            notes,
            col=collection,
            notes_manifest={} if manifest is None else manifest,
            batch_size=10,
            show_progress=False,
            dedupe_fn=_identity_dedupe,
            upsert_fn=upsert,
            lexical_delete_fn=lexical_delete,
            delete_stale=delete_stale,
            strict_lexical=strict_lexical,
        )

    def test_valid_note_writes_stable_id_and_metadata(self):
        writes = []

        def capture(*args, **kwargs):
            writes.append((args, kwargs))

        manifest, stats = self._run([_note()], upsert=capture)

        self.assertEqual(manifest["N1"], {"version": 1})
        self.assertEqual(stats["updated_notes"], 1)
        self.assertEqual(len(writes), 1)
        args, kwargs = writes[0]
        self.assertEqual(args[1], ["N1:note:para0:part0"])
        self.assertEqual(args[3][0]["noteKey"], "N1")
        self.assertEqual(args[3][0]["source_type"], "note")
        self.assertEqual(args[3][0]["itemKey"], "ITEM1")
        self.assertEqual(args[3][0]["locator"], "note:para0")
        self.assertEqual(args[3][0]["para_index"], 0)
        self.assertEqual(args[3][0]["part_index"], 0)

    def test_same_version_skips_without_deleting_or_upserting(self):
        collection = FakeCollection()
        writes = []
        manifest, stats = self._run(
            [_note(version=7)], {"N1": {"version": 7}},
            collection=collection, upsert=lambda *args, **kwargs: writes.append(kwargs),
        )

        self.assertEqual(manifest, {"N1": {"version": 7}})
        self.assertEqual(stats["skipped_notes"], 1)
        self.assertEqual(collection.deleted, [])
        self.assertEqual(writes, [])

    def test_empty_and_gibberish_notes_update_manifest_without_chunks(self):
        notes = [_note("EMPTY", 2, ""), _note("GIB", 3, "<p>unusable</p>")]
        writes = []
        with patch("src.note_extract.looks_like_gibberish", return_value=True):
            manifest, stats = self._run(
                notes, upsert=lambda *args, **kwargs: writes.append(kwargs),
            )

        self.assertEqual(manifest["EMPTY"], {"version": 2})
        self.assertEqual(manifest["GIB"], {"version": 3})
        self.assertEqual(stats["updated_notes"], 2)
        self.assertEqual(writes, [])

    def test_stale_note_is_removed_from_chroma_lexical_and_manifest(self):
        collection = FakeCollection()
        lexical_deleted = []
        manifest, stats = self._run(
            [], {"STALE": {"version": 1}}, collection=collection,
            lexical_delete=lexical_deleted.append,
        )

        self.assertEqual(collection.deleted, [{"where": {"noteKey": "STALE"}}])
        self.assertEqual(lexical_deleted, ["STALE"])
        self.assertEqual(manifest, {})
        self.assertEqual(stats["deleted_stale_notes"], 1)

    def test_non_strict_delete_errors_are_best_effort(self):
        lexical_error = RuntimeError("lexical unavailable")
        collection = FakeCollection(delete_error=RuntimeError("chroma unavailable"))
        manifest, stats = self._run(
            [_note("EMPTY", 2, "")], collection=collection,
            lexical_delete=lambda _key: (_ for _ in ()).throw(lexical_error),
        )

        self.assertEqual(manifest["EMPTY"], {"version": 2})
        self.assertEqual(stats["updated_notes"], 1)

    def test_strict_chroma_delete_error_is_propagated_unchanged(self):
        error = ValueError("chroma unavailable")
        with self.assertRaises(ValueError) as caught:
            self._run(
                [_note("EMPTY", 2, "")], collection=FakeCollection(delete_error=error),
                strict_lexical=True,
            )
        self.assertIs(caught.exception, error)

    def test_strict_lexical_delete_error_is_propagated_unchanged(self):
        error = KeyError("lexical unavailable")
        with self.assertRaises(KeyError) as caught:
            self._run(
                [_note("EMPTY", 2, "")],
                lexical_delete=lambda _key: (_ for _ in ()).throw(error),
                strict_lexical=True,
            )
        self.assertIs(caught.exception, error)

    def test_strict_mode_is_forwarded_to_upsert(self):
        calls = []
        self._run(
            [_note()], strict_lexical=True,
            upsert=lambda *args, **kwargs: calls.append(kwargs),
        )
        self.assertEqual(calls[0]["strict_lexical"], True)


if __name__ == "__main__":
    unittest.main()
