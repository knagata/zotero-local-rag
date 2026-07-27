from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.ndlocr_extract import (
    blocks_from_ndlocr_payload, extract_chunks_from_pdf_with_ndlocr,
    lines_from_ndlocr_payload,
)


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

    def test_preserves_line_provenance_bbox_and_reading_order_in_text_block(self):
        payload = {"contents": [[
            {"id": 4, "text": "本文一行目", "confidence": 0.9,
             "boundingBox": [[10, 20], [110, 20], [110, 30], [10, 30]]},
            {"id": 5, "text": "本文二行目", "confidence": 0.7,
             "boundingBox": [[10, 35], [100, 35], [100, 45], [10, 45]]},
        ]]}
        blocks = blocks_from_ndlocr_payload(payload)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["text"], "本文一行目\n本文二行目")
        self.assertEqual(blocks[0]["reading_order"], 4)
        self.assertEqual(blocks[0]["reading_order_end"], 5)
        self.assertEqual(blocks[0]["bbox"], {"l": 10.0, "t": 20.0, "r": 110.0, "b": 45.0})
        self.assertEqual(len(blocks[0]["provenance"]), 2)
        self.assertEqual(blocks[0]["provenance"][1]["source_line_index"], 1)
        self.assertAlmostEqual(blocks[0]["confidence"], 0.8)

    def test_separates_explicit_heading_from_body_and_preserves_vertical_order(self):
        payload = {"contents": [[
            {"id": 0, "text": "第一章", "label": "section_header", "confidence": 0.9,
             "boundingBox": [[90, 0], [90, 60], [105, 0], [105, 60]]},
            {"id": 1, "text": "縦書き本文の一行目", "confidence": 0.8,
             "boundingBox": [[70, 0], [70, 100], [80, 0], [80, 100]]},
            {"id": 2, "text": "縦書き本文の二行目", "confidence": 0.8,
             "boundingBox": [[50, 0], [50, 100], [60, 0], [60, 100]]},
        ]]}
        blocks = blocks_from_ndlocr_payload(payload)
        self.assertEqual([block["block_type"] for block in blocks], ["heading", "text"])
        self.assertEqual(blocks[0]["text"], "第一章")
        self.assertEqual(blocks[1]["text"], "縦書き本文の一行目縦書き本文の二行目")
        self.assertTrue(all(block["writing_mode"] == "vertical" for block in blocks))

    def test_infers_heading_only_with_short_text_and_size_evidence(self):
        payload = {"contents": [
            [{"id": 0, "text": "序論", "fontSize": 20,
              "boundingBox": [[0, 0], [100, 0], [100, 20], [0, 20]]}],
            [{"id": 1, "text": "本文です。", "fontSize": 10,
              "boundingBox": [[0, 40], [100, 40], [100, 50], [0, 50]]}],
            [{"id": 2, "text": "短い本文", "fontSize": 10,
              "boundingBox": [[0, 60], [100, 60], [100, 70], [0, 70]]}],
        ]}
        blocks = blocks_from_ndlocr_payload(payload)
        self.assertEqual(blocks[0]["block_type"], "heading")
        self.assertEqual(blocks[1]["block_type"], "text")
        self.assertEqual(blocks[2]["block_type"], "text")

    def test_uses_only_explicit_labels_for_non_body_zones(self):
        payload = {"contents": [
            [{"id": 0, "text": "注の本文", "type": "footnote",
              "boundingBox": [[0, 0], [100, 0], [100, 10], [0, 10]]}],
            [{"id": 1, "text": "参考文献項目", "category": "reference",
              "boundingBox": [[0, 20], [100, 20], [100, 30], [0, 30]]}],
            [{"id": 2, "text": "通常本文", "boundingBox": [[0, 40], [100, 40], [100, 50], [0, 50]]}],
        ]}
        blocks = blocks_from_ndlocr_payload(payload)
        self.assertEqual(
            [(block["block_type"], block["zone"]) for block in blocks],
            [("footnote", "footnote"), ("reference", "bibliography"), ("text", "body")],
        )

    def test_extractor_emits_v3_block_metadata_without_flattening(self):
        payload = {"contents": [[
            {"id": 0, "text": "節見出し", "label": "section_header", "confidence": 0.95,
             "boundingBox": [[0, 0], [100, 0], [100, 20], [0, 20]]},
            {"id": 1, "text": "本文の一行目です。", "confidence": 0.8,
             "boundingBox": [[0, 30], [100, 30], [100, 40], [0, 40]]},
        ]]}

        def fake_run(command, **_kwargs):
            output = Path(command[command.index("--output") + 1])
            (output / "page-00001.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8",
            )
            return SimpleNamespace(returncode=0, stderr="", stdout="")

        with (
            patch("src.ndlocr_extract.find_ndlocr", return_value="ndlocr-lite"),
            patch("src.ndlocr_extract._render_pages", return_value=(1, {1: "1"})),
            patch("src.ndlocr_extract.get_pdf_toc", return_value=[(1, "第一部", 1)]),
            patch("src.ndlocr_extract.subprocess.run", side_effect=fake_run),
        ):
            chunks, quality = extract_chunks_from_pdf_with_ndlocr(
                Path("dummy.pdf"), "ATT", {"itemKey": "ITEM", "lang": "ja"},
            )

        self.assertEqual(len(chunks), 2)
        heading_md, body_md = chunks[0][2], chunks[1][2]
        self.assertEqual(heading_md["block_type"], "heading")
        self.assertEqual(body_md["block_type"], "text")
        self.assertEqual(body_md["structure_path"], ["第一部", "節見出し"])
        self.assertEqual([heading_md["reading_order"], body_md["reading_order"]], [0, 1])
        self.assertEqual(json.loads(body_md["bbox"]), {"l": 0.0, "t": 30.0, "r": 100.0, "b": 40.0})
        self.assertEqual(json.loads(body_md["provenance"])[0]["source_line_index"], 1)
        self.assertEqual(body_md["zone"], "body")
        self.assertEqual(quality["blocks"], 2)
        self.assertEqual(quality["heading_blocks"], 1)


if __name__ == "__main__":
    unittest.main()
