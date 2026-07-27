from __future__ import annotations

import unittest

from scripts.suggest_ocr_bakeoff_samples import score_profile


class SuggestOcrSamplesTests(unittest.TestCase):
    def test_category_scores_prefer_matching_profiles(self):
        scanned_ja = {"pages": 200, "scanned": True, "japanese_ratio": .8, "chars_per_sampled_page": 20}
        embedded_en = {"pages": 20, "scanned": False, "japanese_ratio": 0, "chars_per_sampled_page": 2000}
        self.assertGreater(score_profile(scanned_ja, "ja_vertical"), score_profile(embedded_en, "ja_vertical"))
        self.assertGreater(score_profile(embedded_en, "embedded_text_pair"), score_profile(scanned_ja, "embedded_text_pair"))


if __name__ == "__main__":
    unittest.main()
