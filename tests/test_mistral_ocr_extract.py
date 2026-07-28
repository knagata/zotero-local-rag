from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from src import mistral_ocr_extract
from src.mistral_ocr_extract import _page_blocks_from_response, mistral_ocr_available


def _block(block_type: str, content: str, *, y: int = 0) -> dict:
    return {
        "type": block_type, "content": content,
        "top_left_x": 0, "top_left_y": y, "bottom_right_x": 100, "bottom_right_y": y + 10,
    }


class PageBlocksFromResponseTests(unittest.TestCase):
    """Fixtures mirror the verified live response shape (2026-07-21 smoke
    test): blocks carry bbox + type in {header, footer, title, text,
    caption, table, image}; title content keeps its markdown ``#`` prefix."""

    def test_heading_stack_builds_nested_structure_path(self):
        page = {"blocks": [
            _block("title", "# Part One", y=0),
            _block("title", "## Chapter 1", y=10),
            _block("text", "Body text here.", y=20),
        ]}
        blocks = _page_blocks_from_response(page)
        types = [b["block_type"] for b in blocks]
        self.assertEqual(types, ["heading", "heading", "text"])
        self.assertEqual(blocks[1]["heading_path"], ["Part One", "Chapter 1"])
        self.assertEqual(blocks[2]["zone"], "body")

    def test_heading_at_same_or_shallower_level_pops_stack(self):
        page = {"blocks": [
            _block("title", "# Chapter 1", y=0),
            _block("text", "first", y=10),
            _block("title", "# Chapter 2", y=20),
            _block("text", "second", y=30),
        ]}
        blocks = _page_blocks_from_response(page)
        headings = [b for b in blocks if b["block_type"] == "heading"]
        self.assertEqual(headings[0]["heading_path"], ["Chapter 1"])
        self.assertEqual(headings[1]["heading_path"], ["Chapter 2"])

    def test_bibliography_heading_propagates_zone_to_following_blocks(self):
        page = {"blocks": [
            _block("title", "# Bibliography", y=0),
            _block("text", "Smith, J. (2020). Some Book.", y=10),
            _block("table", "| a | b |\n|---|---|\n| 1 | 2 |", y=20),
        ]}
        blocks = _page_blocks_from_response(page)
        by_type = {b["block_type"]: b for b in blocks}
        self.assertEqual(by_type["text"]["zone"], "bibliography")
        self.assertEqual(by_type["table"]["zone"], "bibliography")

    def test_zone_resets_after_a_body_heading(self):
        page = {"blocks": [
            _block("title", "# References", y=0),
            _block("text", "Smith 2020.", y=10),
            _block("title", "# Discussion", y=20),
            _block("text", "Ordinary body text.", y=30),
        ]}
        blocks = _page_blocks_from_response(page)
        texts = [b for b in blocks if b["block_type"] == "text"]
        self.assertEqual(texts[0]["zone"], "bibliography")
        self.assertEqual(texts[1]["zone"], "body")

    def test_header_and_footer_become_page_furniture(self):
        page = {"blocks": [
            _block("header", "Running Title", y=0),
            _block("text", "Body.", y=10),
            _block("footer", "12", y=20),
        ]}
        blocks = _page_blocks_from_response(page)
        self.assertEqual(blocks[0]["block_type"], "page_furniture")
        self.assertEqual(blocks[0]["zone"], "other_paratext")
        self.assertEqual(blocks[-1]["block_type"], "page_furniture")

    def test_image_blocks_are_skipped(self):
        page = {"blocks": [_block("image", ""), _block("text", "Caption-less figure page.")]}
        blocks = _page_blocks_from_response(page)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["block_type"], "text")

    def test_bbox_is_carried_through(self):
        page = {"blocks": [_block("text", "Body.", y=5)]}
        blocks = _page_blocks_from_response(page)
        self.assertEqual(blocks[0]["bbox"], {"l": 0.0, "t": 5.0, "r": 100.0, "b": 15.0})

    def test_empty_blocks_yield_no_output(self):
        self.assertEqual(_page_blocks_from_response({"blocks": []}), [])
        self.assertEqual(_page_blocks_from_response({}), [])

    def test_empty_batch_blocks_fall_back_to_markdown(self):
        page = {
            "blocks": [],
            "markdown": "# Chapter One\n\nFirst paragraph.\n\n## Notes\n\nSecond paragraph.",
        }
        blocks = _page_blocks_from_response(page)
        self.assertEqual(
            [block["block_type"] for block in blocks],
            ["heading", "text", "heading", "text"],
        )
        self.assertEqual(blocks[2]["heading_path"], ["Chapter One", "Notes"])


class MistralOcrAvailabilityTests(unittest.TestCase):
    """Only the key gates Mistral OCR now (2026-07-27).

    Both former gates were removed. Per-item Zotero-tag exclusion protected
    nothing it was still indexing, since anything indexed reaches the assistant
    through search. ``MISTRAL_OCR_FALLBACK_ENABLE`` sat in front of routes that
    are already explicit operator actions -- passing a queue to
    ``--reocr-candidates``, or running ``--submit`` -- while automatic
    ingestion never calls this synchronously at all, so it only ever produced
    the failure it was meant to prevent: a configured key silently doing
    nothing.
    """

    def test_removed_gates_are_gone(self):
        self.assertFalse(hasattr(mistral_ocr_extract, "_item_excluded_from_cloud_fallback"))
        self.assertFalse(hasattr(mistral_ocr_extract, "mistral_ocr_fallback_available"))

    def test_missing_api_key_blocks(self):
        with mock.patch.dict(os.environ, {"MISTRAL_OCR_API_KEY": ""}, clear=False):
            ready, reason = mistral_ocr_available()
        self.assertFalse(ready)
        self.assertIn("MISTRAL_OCR_API_KEY", reason)

    def test_ready_with_a_key(self):
        with mock.patch.dict(os.environ, {"MISTRAL_OCR_API_KEY": "k"}, clear=False):
            ready, reason = mistral_ocr_available()
        self.assertTrue(ready)
        self.assertEqual(reason, "ready")


if __name__ == "__main__":
    unittest.main()


class ExtractChunksMergeTests(unittest.TestCase):
    """This path never called merge_short_chunk_records at all.

    Every OCR block became its own chunk regardless of length, unlike the
    other three extractors. Measured: 34% of mistral_ocr chunks were under 40
    characters (2026-07-28).
    """

    def _pdf(self, directory, page_count=1):
        import fitz
        path = str(Path(directory) / "test.pdf")
        doc = fitz.open()
        for _ in range(page_count):
            doc.new_page(width=600, height=800)
        doc.save(path)
        doc.close()
        return path

    def test_consecutive_short_blocks_on_a_page_are_merged(self):
        import tempfile
        from src.mistral_ocr_extract import extract_chunks_from_mistral_ocr_result

        with tempfile.TemporaryDirectory() as directory:
            pdf_path = self._pdf(directory)
            result = {"pages": [{
                "index": 0,
                "blocks": [
                    {"type": "text", "content": "one short line of text"},
                    {"type": "text", "content": "another short line of text"},
                    {"type": "text", "content": "a third short line of text"},
                ],
            }]}
            chunks, _quality = extract_chunks_from_mistral_ocr_result(
                Path(pdf_path), "ATT", {}, result,
            )
        self.assertLess(len(chunks), 3)
        self.assertGreaterEqual(sum(len(text) for _cid, text, _md in chunks), 40)

    def test_a_block_type_change_still_forms_a_new_group(self):
        import tempfile
        from src.mistral_ocr_extract import extract_chunks_from_mistral_ocr_result

        with tempfile.TemporaryDirectory() as directory:
            pdf_path = self._pdf(directory)
            result = {"pages": [{
                "index": 0,
                "blocks": [
                    {"type": "text", "content": "short body text here today"},
                    {"type": "title", "content": "# A Heading"},
                    {"type": "text", "content": "short body text again today"},
                ],
            }]}
            chunks, _quality = extract_chunks_from_mistral_ocr_result(
                Path(pdf_path), "ATT", {}, result,
            )
        block_types = [md.get("block_type") for _cid, _text, md in chunks]
        self.assertIn("heading", block_types)


class OcrPageCoverageTests(unittest.TestCase):
    """P2-04 (2026-07-28): ocr_pages claimed every page of the source PDF
    regardless of what the response actually contained, so a page the API
    silently skipped left no trace -- quality reported full coverage for a
    page with zero chunks."""

    def _pdf(self, directory, page_count=1):
        import fitz
        path = str(Path(directory) / "test.pdf")
        doc = fitz.open()
        for _ in range(page_count):
            doc.new_page(width=600, height=800)
        doc.save(path)
        doc.close()
        return path

    def test_a_page_missing_from_the_response_is_not_claimed_as_covered(self):
        import tempfile
        from src.mistral_ocr_extract import extract_chunks_from_mistral_ocr_result

        with tempfile.TemporaryDirectory() as directory:
            pdf_path = self._pdf(directory, page_count=3)
            # Page index 1 (page 2) is entirely absent from the response.
            result = {"pages": [
                {"index": 0, "blocks": [{"type": "text", "content": "first page body text here"}]},
                {"index": 2, "blocks": [{"type": "text", "content": "third page body text here"}]},
            ]}
            _chunks, quality = extract_chunks_from_mistral_ocr_result(
                Path(pdf_path), "ATT", {}, result,
            )
        self.assertEqual(quality["ocr_pages"], [1, 3])
        self.assertEqual(quality["missing_pages"], [2])

    def test_a_page_present_but_with_no_blocks_is_also_reported_missing(self):
        import tempfile
        from src.mistral_ocr_extract import extract_chunks_from_mistral_ocr_result

        with tempfile.TemporaryDirectory() as directory:
            pdf_path = self._pdf(directory, page_count=2)
            result = {"pages": [
                {"index": 0, "blocks": [{"type": "text", "content": "only page with content"}]},
                {"index": 1, "blocks": []},
            ]}
            _chunks, quality = extract_chunks_from_mistral_ocr_result(
                Path(pdf_path), "ATT", {}, result,
            )
        self.assertEqual(quality["ocr_pages"], [1])
        self.assertEqual(quality["missing_pages"], [2])

    def test_full_coverage_reports_no_missing_pages(self):
        import tempfile
        from src.mistral_ocr_extract import extract_chunks_from_mistral_ocr_result

        with tempfile.TemporaryDirectory() as directory:
            pdf_path = self._pdf(directory, page_count=2)
            result = {"pages": [
                {"index": 0, "blocks": [{"type": "text", "content": "first page body text here"}]},
                {"index": 1, "blocks": [{"type": "text", "content": "second page body text here"}]},
            ]}
            _chunks, quality = extract_chunks_from_mistral_ocr_result(
                Path(pdf_path), "ATT", {}, result,
            )
        self.assertEqual(quality["ocr_pages"], [1, 2])
        self.assertEqual(quality["missing_pages"], [])
