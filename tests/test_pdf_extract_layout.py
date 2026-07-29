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

    def test_narrow_journal_gutter_is_still_read_column_by_column(self):
        blocks = [
            _block(88, 42, 542, 57, "running header", 0),
            _block(68, 75, 299, 143, "left continuation", 1),
            _block(313, 75, 544, 101, "right continuation", 2),
            _block(313, 114, 424, 127, "right heading", 3),
            _block(313, 140, 544, 445, "right body", 4),
            _block(68, 156, 182, 168, "left heading", 5),
            _block(68, 182, 299, 487, "left body", 6),
            _block(68, 500, 299, 721, "left body continued", 7),
        ]

        records = [
            {
                "text": block[4],
                "bbox": [float(value) for value in block[:4]],
                "block_type": "text",
                "source_block_index": index,
                "source_block_indices": [index],
            }
            for index, block in enumerate(blocks)
        ]
        rows = pdf_extract._order_layout_blocks(records, 612.0)

        self.assertEqual(
            [row["text"] for row in rows],
            [
                "running header",
                "left continuation",
                "left heading",
                "left body",
                "left body continued",
                "right continuation",
                "right heading",
                "right body",
            ],
        )
        self.assertEqual(
            [row["column"] for row in rows],
            ["full", "left", "left", "left", "left", "right", "right", "right"],
        )

    def test_one_long_left_block_still_establishes_two_columns(self):
        page = _Page([
            _block(38, 59, 289, 736, "left conclusion continuation", 0),
            _block(307, 59, 387, 67, "Acknowledgements", 1),
            _block(307, 80, 558, 151, "acknowledgement body", 2),
            _block(307, 164, 462, 172, "Supplementary material", 3),
            _block(307, 216, 352, 224, "References", 4),
            _block(307, 238, 558, 739, "reference entries", 5),
        ])

        rows = pdf_extract.extract_layout_blocks_from_pdf_page(page)

        self.assertEqual(rows[0]["text"], "left conclusion continuation")
        self.assertEqual([row["column"] for row in rows], [
            "left", "right", "right", "right", "right", "right",
        ])

    def test_same_page_outline_events_start_at_heading_records(self):
        records = [
            {"text": "left conclusion continuation"},
            {"text": "Acknowledgements"},
            {"text": "support statement"},
            {"text": "Appendix A. Supplementary material"},
            {"text": "supplement link"},
            {"text": "References"},
            {"text": "AIF, 2014"},
        ]
        events = pdf_extract._outline_events_by_page([
            (1, "Paper", 1),
            (2, "Conclusion", 13),
            (2, "Acknowledgements", 14),
            (2, "Supplementary material", 14),
            (2, "References", 14),
        ])[14]

        paths = pdf_extract._resolve_record_structure_paths(
            records,
            previous_path=["Paper", "Conclusion"],
            page_events=events,
        )

        self.assertEqual(paths[0], ["Paper", "Conclusion"])
        self.assertEqual(paths[2], ["Paper", "Acknowledgements"])
        self.assertEqual(paths[4], ["Paper", "Supplementary material"])
        self.assertEqual(paths[6], ["Paper", "References"])

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

    def test_a_page_emptied_by_repeated_header_removal_is_restored(self):
        """Header filtering must not turn an extracted page into a silent gap."""
        running_header = "Running Header Example Text"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "repeated_header.pdf"
            document = fitz.open()
            first = document.new_page(width=600, height=800)
            first.insert_textbox(fitz.Rect(40, 20, 560, 40), running_header, fontsize=10)
            first.insert_textbox(
                fitz.Rect(40, 60, 560, 750),
                "Ordinary body text that makes the first page substantive. " * 10,
                fontsize=11,
            )
            for _ in range(4):
                page = document.new_page(width=600, height=800)
                page.insert_textbox(fitz.Rect(40, 20, 560, 40), running_header, fontsize=10)
            document.save(str(path))
            document.close()

            with patch.object(pdf_extract, "PDF_OCR_FALLBACK", False), \
                 patch.object(pdf_extract, "PDF_DROP_REPEATED_LINES", True):
                chunks, quality = pdf_extract.extract_chunks_from_pdf(path, "ATT", {"title": "Test"})

        self.assertEqual(quality["repeated_header_dropped_pages"], [])
        self.assertEqual(quality["repeated_margin_lines_removed_pages"], [1])
        self.assertEqual(sorted(quality["repeated_header_restored_pages"]), [2, 3, 4, 5])
        self.assertIn(running_header, "\n".join(text for _chunk_id, text, _metadata in chunks))

    def test_repeated_body_refrain_in_page_middle_is_never_removed(self):
        """Frequency alone is not evidence that a paragraph is a header."""
        refrain = "This repeated refrain is intentional body prose."
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "refrain.pdf"
            document = fitz.open()
            for page_number in range(5):
                page = document.new_page(width=600, height=800)
                page.insert_textbox(
                    fitz.Rect(50, 330, 550, 370), refrain, fontsize=11,
                )
                page.insert_textbox(
                    fitz.Rect(50, 410, 550, 700),
                    f"Page {page_number + 1} has enough distinct body text to remain substantive. " * 8,
                    fontsize=11,
                )
            document.save(path)
            document.close()

            with patch.object(pdf_extract, "PDF_OCR_FALLBACK", False), \
                 patch.object(pdf_extract, "PDF_DROP_REPEATED_LINES", True):
                chunks, _quality = pdf_extract.extract_chunks_from_pdf(path, "ATT", {"title": "Test"})

        extracted = "\n".join(text for _chunk_id, text, _metadata in chunks)
        self.assertIn(refrain, extracted)

    def test_repeated_top_header_is_removed_without_removing_body(self):
        running_header = "Journal of Safe Extraction"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "running_header.pdf"
            document = fitz.open()
            for page_number in range(5):
                page = document.new_page(width=600, height=800)
                page.insert_textbox(fitz.Rect(50, 20, 550, 40), running_header, fontsize=10)
                page.insert_textbox(
                    fitz.Rect(50, 170, 550, 700),
                    f"Body text unique to page {page_number + 1} and long enough for indexing. " * 12,
                    fontsize=11,
                )
            document.save(path)
            document.close()

            with patch.object(pdf_extract, "PDF_OCR_FALLBACK", False), \
                 patch.object(pdf_extract, "PDF_DROP_REPEATED_LINES", True):
                chunks, _quality = pdf_extract.extract_chunks_from_pdf(path, "ATT", {"title": "Test"})

        extracted = "\n".join(text for _chunk_id, text, _metadata in chunks)
        self.assertNotIn(running_header, extracted)
        self.assertIn("Body text unique to page 3", extracted)

    def test_text_only_repeat_helpers_fail_closed_without_position_evidence(self):
        repeated = "A repeated body paragraph must remain."
        pages = [[repeated] for _ in range(5)]
        self.assertEqual(pdf_extract.detect_repeated_lines(pages), set())
        self.assertEqual(pdf_extract.detect_repeated_prefixes(pages), set())
        self.assertEqual(
            pdf_extract.drop_repeated_lines_from_paras([repeated], {repeated}), [repeated],
        )
        self.assertEqual(
            pdf_extract.strip_repeated_prefix_from_first_para([repeated], {"A repeated"}), [repeated],
        )

    def test_page_label_joined_to_repeated_footer_is_canonicalized(self):
        footer = (
            "This report is available at no cost from the National Renewable "
            "Energy Laboratory at www.nrel.gov/publications."
        )
        pages = [
            [{
                "text": f"{label} {footer}",
                "bbox": [83, 733, 531, 771],
                "page_y0": 0,
                "page_y1": 792,
            }]
            for label in ("iii", "4", "5", "38")
        ]

        self.assertEqual(pdf_extract.detect_repeated_lines(pages), {footer})

    def test_prefix_stripping_requires_a_first_block_at_page_top(self):
        prefix = "Journal Header Metadata Shared Prefix 2026 "
        detected_prefix = prefix[:40]
        top_pages = [[{
            "text": f"{prefix}{page_number}: actual opening paragraph",
            "bbox": [50, 20, 550, 40], "page_y0": 0, "page_y1": 800,
        }] for page_number in range(5)]
        prefixes = pdf_extract.detect_repeated_prefixes(top_pages)
        self.assertEqual(prefixes, {detected_prefix})
        stripped = pdf_extract._strip_repeated_prefix_from_first_record(top_pages[0], prefixes)
        self.assertEqual(stripped[0]["text"], f"{prefix[40:]}0: actual opening paragraph")

        middle_pages = [[{
            "text": f"{prefix}{page_number}: intentional body refrain",
            "bbox": [50, 330, 550, 350], "page_y0": 0, "page_y1": 800,
        }] for page_number in range(5)]
        self.assertEqual(pdf_extract.detect_repeated_prefixes(middle_pages), set())

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
