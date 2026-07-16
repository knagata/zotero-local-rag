from __future__ import annotations

import unittest

from src.backfill_languages import annotate_batch


class LanguageBackfillTests(unittest.TestCase):
    def test_preserves_metadata_and_adds_language(self):
        metadatas, counts = annotate_batch(
            ["贈与と交換について論じる。", "Gift exchange and reciprocity."],
            [{"itemKey": "JA"}, {"itemKey": "EN", "year": 2020}],
        )
        self.assertEqual(metadatas[0], {"itemKey": "JA", "lang": "ja"})
        self.assertEqual(metadatas[1]["year"], 2020)
        self.assertEqual(counts["ja"], 1)
        self.assertEqual(counts["en"], 1)


if __name__ == "__main__":
    unittest.main()
