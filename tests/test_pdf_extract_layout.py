from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from src import pdf_extract


class _Rect:
    width = 600.0


class _Page:
    rect = _Rect()

    def __init__(self, blocks):
        self.blocks = blocks

    def get_text(self, kind, sort=False):
        if kind == "blocks":
            return self.blocks
        return ""


def _block(x0, y0, x1, y1, text, number):
    return (x0, y0, x1, y1, text, number, 0)


class PdfLayoutTests(unittest.TestCase):
    def test_two_columns_are_read_left_then_right_between_full_width_blocks(self):
        page = _Page([
            _block(40, 30, 560, 65, "Full width title", 0),
            # Deliberately interleaved in raw PDF order.
            _block(330, 100, 555, 130, "right one", 1),
            _block(45, 100, 270, 130, "left one", 2),
            _block(330, 190, 555, 220, "right two", 3),
            _block(45, 190, 270, 220, "left two", 4),
            _block(40, 700, 560, 725, "Full width footer", 5),
        ])
        rows = pdf_extract.extract_layout_blocks_from_pdf_page(page)
        self.assertEqual([row["text"] for row in rows], [
            "Full width title", "left one", "left two",
            "right one", "right two", "Full width footer",
        ])
        self.assertEqual([row["reading_order"] for row in rows], list(range(6)))
        self.assertEqual([row["column"] for row in rows], [
            "full", "left", "left", "right", "right", "full",
        ])
        self.assertEqual(rows[1]["bbox"], [45.0, 100.0, 270.0, 130.0])
        self.assertEqual(rows[1]["block_type"], "text")
        self.assertEqual(rows[1]["source_block_indices"], [2])

    def test_single_column_uses_geometry_not_raw_block_order(self):
        page = _Page([
            _block(50, 200, 550, 230, "second paragraph", 0),
            _block(50, 50, 550, 80, "first paragraph", 1),
        ])
        rows = pdf_extract.extract_layout_blocks_from_pdf_page(page)
        self.assertEqual([row["text"] for row in rows], ["first paragraph", "second paragraph"])
        self.assertEqual([row["bbox"][1] for row in rows], [50.0, 200.0])

    def test_adjacent_blocks_merge_with_union_bbox_and_source_indices(self):
        page = _Page([
            _block(50, 50, 550, 80, "first line of one paragraph", 0),
            _block(50, 86, 550, 116, "second line of one paragraph", 1),
        ])
        rows = pdf_extract.extract_layout_blocks_from_pdf_page(page)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["bbox"], [50.0, 50.0, 550.0, 116.0])
        self.assertEqual(rows[0]["source_block_indices"], [0, 1])
        self.assertEqual(rows[0]["reading_order"], 0)

    def test_public_extractor_preserves_layout_metadata_on_real_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layout.pdf"
            document = fitz.open()
            page = document.new_page(width=600, height=800)
            page.insert_textbox(
                fitz.Rect(40, 40, 560, 80),
                "A deterministic full width title for layout testing.", fontsize=11,
            )
            page.insert_textbox(
                fitz.Rect(45, 120, 270, 165),
                "Left column contains enough substantive text for one extracted paragraph.", fontsize=10,
            )
            page.insert_textbox(
                fitz.Rect(330, 120, 555, 165),
                "Right column contains enough substantive text for another paragraph.", fontsize=10,
            )
            page.insert_textbox(
                fitz.Rect(45, 220, 270, 265),
                "The second left block makes the column structure geometrically unambiguous.", fontsize=10,
            )
            page.insert_textbox(
                fitz.Rect(330, 220, 555, 265),
                "The second right block provides the matching column population.", fontsize=10,
            )
            document.save(path)
            document.close()

            with patch.object(pdf_extract, "PDF_OCR_FALLBACK", False):
                chunks, quality = pdf_extract.extract_chunks_from_pdf(path, "ATT", {"title": "Test"})

        self.assertEqual(quality["total_pages"], 1)
        self.assertGreaterEqual(len(chunks), 5)
        metadata = [row[2] for row in chunks]
        self.assertTrue(all(row["block_type"] == "text" for row in metadata))
        self.assertTrue(all(len(row["bbox"]) == 4 for row in metadata))
        self.assertEqual([row["reading_order"] for row in metadata], sorted(
            row["reading_order"] for row in metadata
        ))
        self.assertIn(metadata[0]["column"], {"full", "left"})
        self.assertEqual({row["column"] for row in metadata[1:]}, {"left", "right"})

    def test_a_heading_from_the_pdf_outline_stamps_a_zone(self):
        """This was the one PDF path that assigned no zone at all beyond corrupted.

        Paratext classification existed only for AI-TOC and Docling documents,
        which most PDFs never route through -- an estimated 10.6 million
        characters across 217 items sat under headings like "Bibliography"
        while indexed as zone=body (2026-07-28).
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outline.pdf"
            document = fitz.open()
            intro = document.new_page(width=600, height=800)
            intro.insert_textbox(
                fitz.Rect(50, 50, 550, 750), "This is the introduction body text. " * 30, fontsize=11,
            )
            refs = document.new_page(width=600, height=800)
            refs.insert_textbox(
                fitz.Rect(50, 50, 550, 750),
                "Smith, J. 2020. A book. Jones, K. 2019. Another book. " * 30, fontsize=11,
            )
            document.set_toc([[1, "Introduction", 1], [1, "Bibliography", 2]])
            document.save(str(path))
            document.close()

            with patch.object(pdf_extract, "PDF_OCR_FALLBACK", False):
                chunks, _quality = pdf_extract.extract_chunks_from_pdf(path, "ATT", {"title": "Test"})

        by_page = {}
        for _cid, _text, metadata in chunks:
            by_page.setdefault(metadata.get("page"), []).append(metadata.get("zone"))

        self.assertTrue(all(zone in (None, "body") for zone in by_page[1]))
        self.assertTrue(chunks)
        self.assertTrue(all(zone == "bibliography" for zone in by_page[2]))

    def test_hard_min_chars_is_cjk_aware(self):
        """Every other length constant here has a CJK counterpart already.

        The final drop-or-keep decision inside merge_short_chunk_records
        stayed language-blind: a genuinely short but meaningful CJK fragment
        was measured by the same floor as English (2026-07-28).
        """
        from src.text_utils import HARD_MIN_CHARS, HARD_MIN_CHARS_CJK

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outline.pdf"
            document = fitz.open()
            page = document.new_page(width=600, height=800)
            page.insert_textbox(
                fitz.Rect(50, 50, 550, 750), "some ordinary English body text. " * 30, fontsize=11,
            )
            document.save(str(path))
            document.close()

            captured = {}
            real_merge = pdf_extract.merge_short_chunk_records

            def spy(*args, **kwargs):
                captured["hard_min_chars"] = kwargs.get("hard_min_chars")
                return real_merge(*args, **kwargs)

            with patch.object(pdf_extract, "PDF_OCR_FALLBACK", False), \
                 patch.object(pdf_extract, "merge_short_chunk_records", side_effect=spy), \
                 patch.object(pdf_extract, "is_no_space_language_document", return_value=True):
                pdf_extract.extract_chunks_from_pdf(path, "ATT", {"title": "Test"})

        self.assertEqual(captured.get("hard_min_chars"), HARD_MIN_CHARS_CJK)
        self.assertNotEqual(HARD_MIN_CHARS_CJK, HARD_MIN_CHARS)


def test_a_page_whose_lines_are_all_too_short_is_rescued_not_erased():
    """A filter that removes a page's entire content has misfired.

    Merge boundaries are keyed per record, so nothing on a page combines. When
    _merge_layout_blocks falls back to one line per block (~34 characters),
    every line is under HARD_MIN_CHARS and the merge drops the lot -- 13 pages
    of one document went missing (4CY8EIIB, 2026-07-28). Ordinary pages keep
    their existing granularity; only the all-or-nothing case is retried.
    """
    from src.text_utils import merge_short_chunk_records

    rows = [
        (f"A:p1:para{i}:part0", "the line of body text here again"[:34],
         {"page": 1, "reading_order": i, "chapter": "Chapter One"})
        for i in range(12)
    ]
    per_record = merge_short_chunk_records(
        rows, min_chars=200, max_chars=1200,
        boundary_key=lambda _cid, _text, md: (md.get("page"), md.get("reading_order")),
    )
    assert per_record == [], "precondition: the per-record key erases the page"

    rescued = merge_short_chunk_records(
        rows, min_chars=200, max_chars=1200,
        boundary_key=lambda _cid, _text, md: (md.get("page"), md.get("chapter")),
    )
    assert rescued
    assert sum(len(row[1]) for row in rescued) >= sum(len(row[1]) for row in rows)


if __name__ == "__main__":
    unittest.main()
