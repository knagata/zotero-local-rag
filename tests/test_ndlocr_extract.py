from __future__ import annotations

import unittest

from src.ndlocr_extract import lines_from_ndlocr_payload


class NdlocrExtractTests(unittest.TestCase):
    def test_vertical_lines_follow_ndl_evaluated_id_order_without_spaces(self):
        payload = {"contents": [[
            {"id": 1, "text": "左の列。", "confidence": 0.8,
             "boundingBox": [[10, 0], [10, 100], [20, 0], [20, 100]]},
            {"id": 0, "text": "右の列から", "confidence": 0.9,
             "boundingBox": [[30, 0], [30, 100], [40, 0], [40, 100]]},
        ]]}
        texts, confidences = lines_from_ndlocr_payload(payload)
        self.assertEqual(texts, ["右の列から左の列。"])
        self.assertEqual(confidences, [0.9, 0.8])

    def test_horizontal_lines_keep_line_boundaries(self):
        payload = {"contents": [[
            {"id": 0, "text": "First line", "boundingBox": [[0, 0], [0, 10], [100, 0], [100, 10]]},
            {"id": 1, "text": "Second line", "boundingBox": [[0, 20], [0, 30], [100, 20], [100, 30]]},
        ]]}
        texts, _ = lines_from_ndlocr_payload(payload)
        self.assertEqual(texts, ["First line\nSecond line"])

    def test_discards_layout_repeat_artifacts_and_exact_duplicate_lines(self):
        payload = {"contents": [[
            {"id": 0, "text": "本文です。", "boundingBox": [[0, 0], [0, 10], [100, 0], [100, 10]]},
            {"id": 1, "text": "子" * 40, "boundingBox": [[0, 20], [0, 30], [100, 20], [100, 30]]},
            {"id": 2, "text": "本文です。", "boundingBox": [[0, 40], [0, 50], [100, 40], [100, 50]]},
            {"id": 3, "text": "次の行です。", "boundingBox": [[0, 60], [0, 70], [100, 60], [100, 70]]},
        ]]}
        texts, _ = lines_from_ndlocr_payload(payload)
        self.assertEqual(texts, ["本文です。\n次の行です。"])

    def test_trims_long_adjacent_region_overlap(self):
        payload = {"contents": [[
            {"id": 0, "text": "前半から重複する十分に長い文章です。", "boundingBox": [[0, 0], [0, 10], [100, 0], [100, 10]]},
            {"id": 1, "text": "重複する十分に長い文章です。後半へ続きます。", "boundingBox": [[0, 20], [0, 30], [100, 20], [100, 30]]},
        ]]}
        texts, _ = lines_from_ndlocr_payload(payload)
        self.assertEqual(texts, ["前半から重複する十分に長い文章です。\n後半へ続きます。"])


if __name__ == "__main__":
    unittest.main()
