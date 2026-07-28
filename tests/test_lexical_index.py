from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.lexical_index import (
    delete_by_attachment_keys,
    delete_by_note_key,
    search_chunks,
    upsert_chunks,
)


class LexicalIndexTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "lexical.sqlite3"
        upsert_chunks(
            ["a", "b", "n"],
            ["贈与と互酬性の民族誌", "gift exchange and reciprocity", "贈与に関するメモ"],
            [
                {"itemKey": "I1", "lang": "ja", "source_type": "pdf", "attachmentKey": "A1"},
                {"itemKey": "I2", "lang": "en", "source_type": "pdf", "attachmentKey": "A2"},
                {"itemKey": "I1", "lang": "ja", "source_type": "note", "noteKey": "N1"},
            ],
            path=self.path,
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_trigram_search_supports_japanese(self):
        rows = search_chunks("贈与", path=self.path)
        self.assertEqual([row["chunk_id"] for row in rows], ["a"])

    def test_multiword_query_matches_non_adjacent_terms(self):
        rows = search_chunks("gift reciprocity", path=self.path)
        self.assertEqual([row["chunk_id"] for row in rows], ["b"])

    def test_notes_and_item_filters(self):
        self.assertEqual(search_chunks("メモ", path=self.path), [])
        rows = search_chunks("メモ", include_notes=True, item_keys=["I1"], path=self.path)
        self.assertEqual(rows[0]["chunk_id"], "n")

    def test_upsert_replaces_and_delete_removes(self):
        upsert_chunks(
            ["a"], ["交換ではなく市場"],
            [{"itemKey": "I1", "source_type": "pdf", "attachmentKey": "A1"}],
            path=self.path,
        )
        self.assertEqual(search_chunks("贈与", path=self.path), [])
        delete_by_attachment_keys(["A2"], path=self.path)
        self.assertEqual(search_chunks("gift", path=self.path), [])
        delete_by_note_key("N1", path=self.path)
        self.assertEqual(search_chunks("メモ", include_notes=True, path=self.path), [])


if __name__ == "__main__":
    unittest.main()


class ShortTokenSearchTests(unittest.TestCase):
    """A short word in a query must not silence the whole search.

    FTS5's trigram tokenizer indexes three-character sequences, so a shorter
    token has no trigram and cannot be matched through MATCH at all. The length
    test used to be applied to the entire query string, so a single-term search
    took a LIKE path and worked while a multi-term one went to MATCH and ANDed
    in an unmatchable token: "移民" found rows and "移民 国家" found none, as did
    "landscape of power" and "a landscape". Multi-word search was broken in both
    languages -- two-character words carry meaning in Japanese, and English
    articles and prepositions are unavoidable (2026-07-28).
    """

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "lexical.sqlite3"
        upsert_chunks(
            ["a", "b"],
            [
                "移民と国家をめぐる都市の空間編成について",
                "the landscape of power in colonial cities",
            ],
            [
                {"itemKey": "I1", "lang": "ja", "source_type": "pdf", "attachmentKey": "A1"},
                {"itemKey": "I2", "lang": "en", "source_type": "pdf", "attachmentKey": "A2"},
            ],
            path=self.path,
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def _ids(self, query):
        return [row["chunk_id"] for row in search_chunks(query, k=10, path=self.path)]

    def test_two_japanese_terms_together_still_match(self):
        self.assertEqual(self._ids("移民 国家"), ["a"])

    def test_an_english_preposition_does_not_silence_the_query(self):
        self.assertEqual(self._ids("landscape of power"), ["b"])

    def test_an_article_alone_beside_a_word_still_matches(self):
        self.assertEqual(self._ids("a landscape"), ["b"])

    def test_a_single_short_term_still_matches(self):
        self.assertEqual(self._ids("移民"), ["a"])

    def test_short_terms_still_narrow_rather_than_widen(self):
        # The short token is matched with LIKE, not dropped: silently ignoring
        # it would return documents the user did not ask for.
        self.assertEqual(self._ids("landscape of colonial"), ["b"])
        self.assertEqual(self._ids("landscape of zebra"), [])

    def test_a_long_term_that_matches_nothing_still_excludes(self):
        self.assertEqual(self._ids("移民 資本主義"), [])
