from __future__ import annotations

import unittest

from src.search_fusion import language_balanced_order


def _hit(chunk_id: str, lang: str):
    return chunk_id, {"metadata": {"lang": lang}}


class LanguageBalanceTests(unittest.TestCase):
    def test_reserves_minority_language_hits(self):
        hits = [_hit(f"en-{i}", "en") for i in range(6)] + [
            _hit("ja-1", "ja"),
            _hit("ja-2", "ja"),
        ]
        top = language_balanced_order(hits, 5)[:5]
        self.assertEqual(sum(hit[1]["metadata"]["lang"] == "ja" for hit in top), 2)
        self.assertEqual([hit[0] for hit in top], ["en-0", "en-1", "en-2", "ja-1", "ja-2"])

    def test_no_change_when_only_one_language_exists(self):
        hits = [_hit("en-1", "en"), _hit("en-2", "en")]
        self.assertEqual(language_balanced_order(hits, 2), hits)

    def test_small_k_keeps_both_languages(self):
        hits = [_hit("en-1", "en"), _hit("en-2", "en"), _hit("ja-1", "ja")]
        top = language_balanced_order(hits, 2)[:2]
        self.assertEqual({hit[1]["metadata"]["lang"] for hit in top}, {"ja", "en"})


if __name__ == "__main__":
    unittest.main()
