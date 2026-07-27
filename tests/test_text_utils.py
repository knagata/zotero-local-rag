from __future__ import annotations

import unittest

from src.text_utils import analyze_text_quality, detect_lang, merge_short_chunk_records


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

    def test_short_chunks_do_not_merge_across_structure_boundary(self):
        chunks = [
            ("a", "a" * 400, {"node_id": "one", "zone": "body", "locator": "1"}),
            ("b", "b" * 400, {"node_id": "two", "zone": "body", "locator": "2"}),
        ]
        result = merge_short_chunk_records(
            chunks, min_chars=1_000, max_chars=2_000,
            boundary_key=lambda _cid, _text, md: (md["node_id"], md["zone"]),
        )
        self.assertEqual([row[0] for row in result], ["a", "b"])

    def test_short_adjacent_records_merge_before_hard_min_filter(self):
        # A chronology often has one short event per block.  Dropping it before
        # the merger silently deletes source content; two adjacent rows should
        # instead become one searchable record.
        chunks = [
            ("a", "a" * 20, {"node_id": "one", "zone": "body", "locator": "1"}),
            ("b", "b" * 25, {"node_id": "one", "zone": "body", "locator": "2"}),
        ]
        result = merge_short_chunk_records(
            chunks, min_chars=100, max_chars=2_000,
            boundary_key=lambda _cid, _text, md: (md["node_id"], md["zone"]),
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "a")
        self.assertEqual(result[0][1], "a" * 20 + "\n\n" + "b" * 25)

    def test_contents_and_index_are_not_classified_as_corruption(self):
        contents = (
            "目次\n第1章 序論 ................................ 1\n"
            "第2章 方法 ................................ 20\n"
            "第3章 結論 ................................ 50\n"
        )
        index = (
            "Index\nAdorno, Theodor 1, 4, 9\nBenjamin, Walter 20, 31\n"
            "Culture 2, 8, 15\nMedia 7, 18, 25\n"
        )
        for text in (contents, index):
            quality = analyze_text_quality(text)
            self.assertTrue(quality["structured_listing"])
            self.assertEqual(quality["extraction_failure_score"], 0.0)
            self.assertEqual(quality["content_corruption_score"], 0.0)

    def test_numeric_table_is_not_font_encoding_failure(self):
        table = (
            "表1 台湾糖業の成長 1901/02-1918/19年期\n"
            "工場数 1 2 3 5 8 10 17 71 118 191\n"
            "生産量 300 350 390 1326 1556 2560 9140\n"
        )
        quality = analyze_text_quality(table)
        self.assertTrue(quality["structured_listing"])
        self.assertEqual(quality["extraction_failure_score"], 0.0)

    def test_figure_credit_is_not_linguistic_corruption(self):
        caption = (
            "Richard Misrach, Abandoned Trailerhome, Mississippi River, 1998. "
            "Courtesy of Fraenkel Gallery, San Francisco; Pace Gallery, New York."
        )
        quality = analyze_text_quality(caption)
        self.assertTrue(quality["structured_listing"])
        self.assertEqual(quality["content_corruption_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
