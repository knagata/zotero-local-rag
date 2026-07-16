from __future__ import annotations

import unittest

from src.text_utils import detect_lang


class DetectLanguageTests(unittest.TestCase):
    def test_japanese_kana(self):
        self.assertEqual(detect_lang("贈与と互酬性について考える。"), "ja")

    def test_chinese_han_without_kana(self):
        self.assertEqual(detect_lang("这是关于社会关系和礼物交换的研究。"), "zh")

    def test_english(self):
        self.assertEqual(detect_lang("Gift exchange creates durable social relations."), "en")

    def test_metadata_hint_wins(self):
        self.assertEqual(detect_lang("Anthropology", "ja-JP"), "ja")
        self.assertEqual(detect_lang("本文", "English"), "en")

    def test_empty_is_other(self):
        self.assertEqual(detect_lang(""), "other")


if __name__ == "__main__":
    unittest.main()
