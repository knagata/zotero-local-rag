from __future__ import annotations

import unittest

from src.note_extract import index_notes


class FakeCollection:
    def __init__(self):
        self.deleted = []

    def delete(self, **kwargs):
        self.deleted.append(kwargs)


class NoteScopeTests(unittest.TestCase):
    def test_partial_scope_does_not_delete_unseen_manifest_notes(self):
        collection = FakeCollection()
        manifest, stats = index_notes(
            [], col=collection, notes_manifest={"OUTSIDE": {"version": 1}},
            batch_size=10, show_progress=False,
            dedupe_fn=lambda ids, docs, metas: (ids, docs, metas),
            upsert_fn=lambda *args, **kwargs: None,
            delete_stale=False,
        )
        self.assertEqual(manifest, {"OUTSIDE": {"version": 1}})
        self.assertEqual(stats["deleted_stale_notes"], 0)
        self.assertEqual(collection.deleted, [])


if __name__ == "__main__":
    unittest.main()
