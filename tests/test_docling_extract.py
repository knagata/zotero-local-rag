from __future__ import annotations

from enum import Enum
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import importlib.util

from src import docling_extract


class _Label(Enum):
    TITLE = "title"
    SECTION_HEADER = "section_header"
    TEXT = "text"
    TABLE = "table"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    REFERENCE = "reference"


class _BBox:
    l, t, r, b = 1.0, 2.0, 30.0, 40.0
    coord_origin = SimpleNamespace(value="TOPLEFT")


class _Item:
    def __init__(self, label, text, *, level=None, page=1, markdown=None):
        self.label = label
        self.text = text
        self.prov = [SimpleNamespace(page_no=page, bbox=_BBox(), charspan=(2, 12))]
        self._markdown = markdown
        if level is not None:
            self.level = level

    def export_to_markdown(self, *, doc):
        return self._markdown or self.text


class _FakeDoclingWorker:
    """Stands in for the run's DoclingWorker subprocess in patch-path tests."""

    def __init__(self, result=None, *, side_effect=None):
        self._result = result
        self._side_effect = side_effect

    def extract(self, pdf_path, attachment_key, meta_base):
        if self._side_effect is not None:
            return self._side_effect(pdf_path, attachment_key, meta_base)
        return self._result


class _Doc:
    pages = {1: object(), 2: object()}

    def __init__(self, items):
        self.items = items

    def iterate_items(self):
        return iter(self.items)


class DoclingExtractTests(unittest.TestCase):
    def test_label_enum_and_qualified_string_are_normalized(self):
        self.assertEqual(docling_extract.normalize_docling_label(_Label.TABLE), "table")
        self.assertEqual(
            docling_extract.normalize_docling_label("DocItemLabel.SECTION_HEADER"),
            "section_header",
        )

    def test_provenance_serializes_page_bbox_and_charspan(self):
        rows = docling_extract.docling_provenance(_Item(_Label.TEXT, "text", page=2))
        self.assertEqual(rows[0]["page"], 2)
        self.assertEqual(rows[0]["bbox"]["coord_origin"], "TOPLEFT")
        self.assertEqual(rows[0]["charspan"], [2, 12])

    @unittest.skipUnless(importlib.util.find_spec("docling"), "optional docling dependency not installed")
    def test_extract_preserves_structure_zone_block_type_and_reading_order(self):
        items = [
            (_Item(_Label.TITLE, "A Study"), 0),
            (_Item(_Label.SECTION_HEADER, "Introduction", level=1), 1),
            (_Item(_Label.TEXT, "This is an ordinary body paragraph long enough for chunk retention."), 2),
            (_Item(_Label.TABLE, "fallback", markdown="| A | B |\n|---|---|\n| 1 | 2 |"), 2),
            (_Item(_Label.CAPTION, "Table 1"), 2),
            (_Item(_Label.FOOTNOTE, "Short note"), 2),
            (_Item(_Label.SECTION_HEADER, "References", level=1, page=2), 1),
            (_Item(_Label.REFERENCE, "Smith 2020" , page=2), 2),
        ]
        from docling.datamodel.base_models import ConversionStatus

        converter = SimpleNamespace(
            convert=lambda _path: SimpleNamespace(document=_Doc(items), status=ConversionStatus.SUCCESS)
        )
        with patch.object(docling_extract, "get_docling_converter", return_value=converter):
            chunks, quality = docling_extract.extract_chunks_from_pdf_with_docling(
                Path("missing-fixture.pdf"), "ATT", {"itemKey": "ITEM"},
            )

        metadata = [md for _chunk_id, _text, md in chunks]
        texts = [text for _chunk_id, text, _md in chunks]
        by_type = {md["block_type"]: md for md in metadata}
        self.assertIn("A Study", texts)  # short title is not discarded
        self.assertIn("Table 1", texts)  # short caption is not merged into table/body
        self.assertIn("Short note", texts)
        self.assertEqual(by_type["text"]["structure_path"], ["Introduction"])
        self.assertEqual(by_type["table"]["zone"], "body")
        self.assertEqual(by_type["caption"]["block_type"], "caption")
        self.assertEqual(by_type["footnote"]["zone"], "footnote")
        self.assertEqual(by_type["reference"]["zone"], "bibliography")
        self.assertEqual(by_type["reference"]["page"], 2)
        self.assertEqual(json.loads(by_type["reference"]["bbox"])["l"], 1.0)
        self.assertEqual(by_type["reference"]["bbox_l"], 1.0)
        self.assertEqual(json.loads(by_type["reference"]["provenance"])[0]["page"], 2)
        self.assertEqual(
            [md["reading_order"] for md in metadata],
            sorted(md["reading_order"] for md in metadata),
        )
        from chromadb.api.types import validate_metadata
        for md in metadata:
            validate_metadata(md)
        self.assertEqual(quality["parser"], "docling")

    @unittest.skipUnless(importlib.util.find_spec("docling"), "optional docling dependency not installed")
    def test_consecutive_references_stay_one_chunk_per_entry(self):
        # Part D (dev-notes/current/77): reference/footnote items are in
        # _PRESERVE_SHORT_LABELS, so consecutive short references must NOT be
        # merged into a single chunk -- each entry keeps its own boundary.
        items = [
            (_Item(_Label.SECTION_HEADER, "References", level=1), 1),
            (_Item(_Label.REFERENCE, "Barrat J (2013) Our Final Invention. Press."), 2),
            (_Item(_Label.REFERENCE, "Bostrom N (2014) Superintelligence. Oxford."), 2),
            (_Item(_Label.REFERENCE, "Dennett D (2017) From Bacteria to Bach. Norton."), 2),
        ]
        from docling.datamodel.base_models import ConversionStatus

        converter = SimpleNamespace(
            convert=lambda _path: SimpleNamespace(document=_Doc(items), status=ConversionStatus.SUCCESS)
        )
        with patch.object(docling_extract, "get_docling_converter", return_value=converter):
            chunks, _quality = docling_extract.extract_chunks_from_pdf_with_docling(
                Path("missing-fixture.pdf"), "ATT", {"itemKey": "ITEM"},
            )
        reference_texts = [text for _cid, text, md in chunks if md.get("block_type") == "reference"]
        self.assertEqual(len(reference_texts), 3)
        self.assertTrue(reference_texts[0].startswith("Barrat J (2013)"))
        self.assertTrue(reference_texts[1].startswith("Bostrom N (2014)"))
        self.assertTrue(reference_texts[2].startswith("Dennett D (2017)"))

    @unittest.skipUnless(importlib.util.find_spec("docling"), "optional docling dependency not installed")
    def test_a_page_of_all_short_text_fragments_is_not_lost_entirely(self):
        # P2-01 (2026-07-29): each fragment below HARD_MIN_CHARS used to be
        # dropped before _merge_docling_chunks ever ran, so a page whose text
        # items all happened to be short one-liners (e.g. one line per block)
        # produced zero chunks -- the whole page vanished from the index. The
        # fix lets short fragments reach the merge pass, where consecutive
        # same-block_type/zone/heading items can combine into a real chunk.
        items = [
            (_Item(_Label.TEXT, "one short line"), 1),
            (_Item(_Label.TEXT, "another short line"), 1),
            (_Item(_Label.TEXT, "a third short line here"), 1),
            (_Item(_Label.TEXT, "and a fourth short line"), 1),
        ]
        from docling.datamodel.base_models import ConversionStatus

        converter = SimpleNamespace(
            convert=lambda _path: SimpleNamespace(document=_Doc(items), status=ConversionStatus.SUCCESS)
        )
        with patch.object(docling_extract, "get_docling_converter", return_value=converter):
            chunks, _quality = docling_extract.extract_chunks_from_pdf_with_docling(
                Path("missing-fixture.pdf"), "ATT", {"itemKey": "ITEM"},
            )
        self.assertGreater(len(chunks), 0)
        combined = " ".join(text for _cid, text, _md in chunks)
        for fragment in ("one short line", "another short line", "a third short line here", "and a fourth short line"):
            self.assertIn(fragment, combined)

    def test_patch_scanned_pages_remaps_pages_and_namespaces_ids(self):
        # E2c (dev-notes/current/77, user decision 2026-07-25): patching just the
        # scanned pages must (a) remap sub-PDF page numbers back to the original
        # document's page numbers, (b) namespace chunk ids so they never collide
        # with the PyMuPDF body chunks they patch alongside, and (c) report every
        # attempted page (not just ones that yielded text) so the caller can
        # treat pages Docling tried but found no text on (figures/photos) as
        # resolved rather than still-scanned.
        fake_chunks = [
            ("ATT:p1:para0:part0", "OCR text from first scanned page",
             {"page": 1, "block_type": "text"}),
        ]
        sub_to_original = ("missing-fixture-sub.pdf", {1: 5, 2: 12})
        with patch.object(
            docling_extract, "_build_scanned_page_subset", return_value=sub_to_original,
        ) as mocked_subset, patch("os.unlink"):
            output, attempted = docling_extract.patch_scanned_pages_with_docling(
                Path("missing-fixture.pdf"), [5, 12], attachment_key="ATT", meta_base={"itemKey": "ITEM"},
                worker=_FakeDoclingWorker((fake_chunks, {"parser": "docling"})),
            )
        mocked_subset.assert_called_once()
        # Page 5 yielded a real text chunk; page 12 (a figure Docling found no
        # text on) still gets a visible marker chunk, not a silent drop
        # (user decision 2026-07-26).
        self.assertEqual(len(output), 2)
        by_page = {md["page"]: (text, md) for _cid, text, md in output}
        self.assertEqual(by_page[5][0], "OCR text from first scanned page")
        self.assertEqual(by_page[5][1]["scanned_page_patch"], "docling")
        self.assertIn(":scanpatch:", output[0][0])
        self.assertEqual(by_page[12][1]["block_type"], "figure")
        # Unlike the corrupted-page marker, a figure marker stays zone="body"
        # (user decision 2026-07-25): these are ordinary body pages, not a
        # specific zone -- see docstring.
        self.assertEqual(by_page[12][1]["zone"], "body")
        self.assertEqual(by_page[12][1]["scanned_page_patch"], "docling")
        self.assertIn(":scanpatch:figure:", [cid for cid, _t, md in output if md["page"] == 12][0])
        # Both pages were attempted, so both count as resolved for the gate,
        # even though page 12 produced no real-text chunk.
        self.assertEqual(attempted, {5, 12})

    def test_scanned_patch_ocr_noise_filter_catches_concatenated_words(self):
        # E2c (user decision 2026-07-26): real garbled OCR text pulled from
        # M5TQ4HLZ's recovered chunks -- missing spaces merge multiple words
        # into unbroken runs, which the character-composition check alone
        # (looks_like_gibberish) wouldn't flag since letters_ratio stays high.
        noise = (
            "Then10-15CobraHelicopterscameandstartedshooting "
            "andshooting andshooting."
        )
        self.assertTrue(docling_extract._looks_like_scanned_patch_ocr_noise(noise))
        # Legitimate short captions (names, titles) must not be rejected --
        # this is exactly the case looks_like_gibberish's 50-char floor was
        # designed to protect, and this stricter check must still spare it.
        self.assertFalse(docling_extract._looks_like_scanned_patch_ocr_noise("Sven Lutticken"))
        self.assertFalse(
            docling_extract._looks_like_scanned_patch_ocr_noise(
                "Theory and the Young Girl This is a shot."
            )
        )

    def test_patch_scanned_pages_drops_ocr_noise_chunks_as_figure_pages(self):
        # A chunk that fails the noise filter must not reach the index as
        # real text -- the page it came from should fall back to a figure
        # marker instead, same as a page Docling found literally nothing on.
        fake_chunks = [
            ("ATT:p1:para0:part0", "Sven Lutticken", {"page": 1, "block_type": "text"}),
            (
                "ATT:p2:para0:part0",
                "Then10-15CobraHelicopterscameandstartedshooting andshooting andshooting.",
                {"page": 2, "block_type": "text"},
            ),
        ]
        sub_to_original = ("missing-fixture-sub.pdf", {1: 24, 2: 92})
        with patch.object(
            docling_extract, "_build_scanned_page_subset", return_value=sub_to_original,
        ), patch("os.unlink"):
            output, attempted = docling_extract.patch_scanned_pages_with_docling(
                Path("missing-fixture.pdf"), [24, 92], attachment_key="ATT", meta_base={"itemKey": "ITEM"},
                worker=_FakeDoclingWorker((fake_chunks, {"parser": "docling"})),
            )
        by_page = {md["page"]: (text, md) for _cid, text, md in output}
        self.assertEqual(by_page[24][0], "Sven Lutticken")
        self.assertEqual(by_page[92][1]["block_type"], "figure")
        self.assertNotIn("CobraHelicopters", by_page[92][0])
        self.assertEqual(attempted, {24, 92})

    def test_patch_corrupted_pages_remaps_pages_and_namespaces_ids(self):
        # E2d (dev-notes/current/77, user decision 2026-07-26): patching just
        # the corrupted (garbled-text) pages must (a) remap sub-PDF page
        # numbers back to the original document, (b) namespace chunk ids
        # under corruptpatch so they never collide with the PyMuPDF body
        # chunks they replace, and (c) report every attempted page so the
        # caller can drop the pre-patch garbled chunks for those pages.
        fake_chunks = [
            ("ATT:p1:para0:part0", "Recovered coherent text from a corrupted page",
             {"page": 1, "block_type": "text"}),
        ]
        sub_to_original = ("missing-fixture-sub.pdf", {1: 2, 2: 9})
        with patch.object(
            docling_extract, "_build_scanned_page_subset", return_value=sub_to_original,
        ) as mocked_subset, patch("os.unlink"):
            output, attempted = docling_extract.patch_corrupted_pages_with_docling(
                Path("missing-fixture.pdf"), [2, 9], attachment_key="ATT", meta_base={"itemKey": "ITEM"},
                worker=_FakeDoclingWorker((fake_chunks, {"parser": "docling"})),
            )
        mocked_subset.assert_called_once()
        self.assertEqual(len(output), 2)
        by_page = {md["page"]: (text, md) for _cid, text, md in output}
        self.assertEqual(by_page[2][0], "Recovered coherent text from a corrupted page")
        self.assertEqual(by_page[2][1]["corrupted_page_patch"], "docling")
        self.assertIn(":corruptpatch:", output[0][0])
        # Page 9 got no clean text back from Docling -> unresolved marker,
        # distinct from the scanned-page patch's "figure" marker since a
        # corrupted page is body text that resisted repair, not a photo.
        self.assertEqual(by_page[9][1]["block_type"], "corrupted_unresolved")
        # 2026-07-30 regression: the unresolved marker must be excluded from
        # ordinary retrieval/summaries like any other known-garbage text, not
        # indexed as zone="body" alongside genuinely recovered content.
        self.assertEqual(by_page[9][1]["zone"], "corrupted")
        self.assertEqual(by_page[9][1]["corrupted_page_patch"], "docling")
        self.assertIn(
            ":corruptpatch:corrupted_unresolved:",
            [cid for cid, _t, md in output if md["page"] == 9][0],
        )
        self.assertIn("Corrupted page 9", by_page[9][0])
        self.assertEqual(attempted, {2, 9})

    def test_patch_corrupted_pages_drops_ocr_noise_chunks_as_unresolved(self):
        # Same noise filter as the scanned-page patch: garbled OCR output
        # from Docling must not replace one kind of garbage with another.
        fake_chunks = [
            ("ATT:p1:para0:part0", "A properly recovered sentence with real words.",
             {"page": 1, "block_type": "text"}),
            (
                "ATT:p2:para0:part0",
                "Then10-15CobraHelicopterscameandstartedshooting andshooting andshooting.",
                {"page": 2, "block_type": "text"},
            ),
        ]
        sub_to_original = ("missing-fixture-sub.pdf", {1: 30, 2: 31})
        with patch.object(
            docling_extract, "_build_scanned_page_subset", return_value=sub_to_original,
        ), patch("os.unlink"):
            output, attempted = docling_extract.patch_corrupted_pages_with_docling(
                Path("missing-fixture.pdf"), [30, 31], attachment_key="ATT", meta_base={"itemKey": "ITEM"},
                worker=_FakeDoclingWorker((fake_chunks, {"parser": "docling"})),
            )
        by_page = {md["page"]: (text, md) for _cid, text, md in output}
        self.assertEqual(by_page[30][0], "A properly recovered sentence with real words.")
        self.assertEqual(by_page[31][1]["block_type"], "corrupted_unresolved")
        self.assertEqual(by_page[31][1]["zone"], "corrupted")
        self.assertNotIn("CobraHelicopters", by_page[31][0])
        self.assertEqual(attempted, {30, 31})

    def test_patch_corrupted_pages_distinguishes_duplicate_docling_source_ids(self):
        # A real XGEQTPS3 patch returned two different layout blocks with the
        # same Docling-relative id.  Both are content, not retries, so the
        # patch namespace must keep both deterministically rather than letting
        # the later structure/write boundary see conflicting duplicate IDs.
        fake_chunks = [
            ("XGEQTPS3:p1:para0:part0", "First recovered block with enough text.",
             {"page": 1, "block_type": "text"}),
            ("XGEQTPS3:p1:para0:part0", "Second distinct recovered block with enough text.",
             {"page": 1, "block_type": "text"}),
        ]
        with patch.object(
            docling_extract, "_build_scanned_page_subset",
            return_value=("sub.pdf", {1: 7}),
        ), patch("os.unlink"):
            output, attempted = docling_extract.patch_corrupted_pages_with_docling(
                Path("missing.pdf"), [7], attachment_key="XGEQTPS3", meta_base={"itemKey": "TA4V4875"},
                worker=_FakeDoclingWorker((fake_chunks, {"parser": "docling"})),
            )

        self.assertEqual(attempted, {7})
        self.assertEqual([text for _cid, text, _md in output], [row[1] for row in fake_chunks])
        self.assertEqual(
            [cid for cid, _text, _md in output],
            [
                "XGEQTPS3:corruptpatch:b0:XGEQTPS3:p1:para0:part0",
                "XGEQTPS3:corruptpatch:b0:XGEQTPS3:p1:para0:part0:dup1",
            ],
        )

    def test_scan_repair_and_corruption_repair_have_disjoint_provenance_ids(self):
        # The scan-derived repair path deliberately uses the same Docling
        # helper as the later text-corruption repair.  Both can inspect p1 in
        # separate invocations, so a per-call duplicate counter alone cannot
        # protect the composed extraction list.
        fake_chunks = [
            ("XGEQTPS3:p1:para0:part0", "Recovered text from the shared page.",
             {"page": 1, "block_type": "text"}),
        ]
        with patch.object(
            docling_extract, "_build_scanned_page_subset",
            return_value=("sub.pdf", {1: 1}),
        ), patch("os.unlink"):
            worker = _FakeDoclingWorker((fake_chunks, {"parser": "docling"}))
            scan_repair, _ = docling_extract.patch_corrupted_pages_with_docling(
                Path("missing.pdf"), [1], attachment_key="XGEQTPS3",
                meta_base={"itemKey": "TA4V4875"}, worker=worker,
                chunk_namespace="scanrepair",
            )
            corruption_repair, _ = docling_extract.patch_corrupted_pages_with_docling(
                Path("missing.pdf"), [1], attachment_key="XGEQTPS3",
                meta_base={"itemKey": "TA4V4875"}, worker=worker,
            )

        merged_ids = [cid for cid, _text, _md in scan_repair + corruption_repair]
        self.assertEqual(len(merged_ids), len(set(merged_ids)))
        self.assertEqual(
            merged_ids,
            [
                "XGEQTPS3:scanrepair:b0:XGEQTPS3:p1:para0:part0",
                "XGEQTPS3:corruptpatch:b0:XGEQTPS3:p1:para0:part0",
            ],
        )

    def test_patch_batches_cover_all_pages_with_unique_chunk_ids(self):
        # OOM fix (2026-07-26, AX5CRKJ6: a single 208-page 200dpi sub-PDF
        # killed the run): pages are processed in bounded batches
        # (PDF_PAGE_PATCH_BATCH_PAGES). Every wanted page must still be
        # attempted exactly once across batches, and chunk ids must stay
        # unique even though each batch's sub-PDF restarts at page 1.
        import os as _os

        built_batches: list[list[int]] = []

        def fake_subset(_pdf, batch):
            built_batches.append(list(batch))
            return "sub.pdf", {i + 1: page for i, page in enumerate(batch)}

        def fake_docling(_path, _att, _meta):
            # Same sub-PDF-relative chunk id in every batch on purpose.
            return (
                [("ATT:p1:para0:part0", "Recovered text with plenty of words here.",
                  {"page": 1, "block_type": "text"})],
                {"parser": "docling"},
            )

        pages = list(range(1, 8))  # 7 pages, batch size 3 -> 3 batches
        with patch.dict(_os.environ, {"PDF_PAGE_PATCH_BATCH_PAGES": "3"}), \
             patch.object(docling_extract, "_build_scanned_page_subset", side_effect=fake_subset), \
             patch.object(docling_extract, "_relieve_patch_memory"), \
             patch("os.unlink"):
            output, attempted = docling_extract.patch_corrupted_pages_with_docling(
                Path("missing.pdf"), pages, attachment_key="ATT", meta_base={"itemKey": "I"},
                worker=_FakeDoclingWorker(side_effect=fake_docling),
            )
        self.assertEqual(built_batches, [[1, 2, 3], [4, 5, 6], [7]])
        self.assertEqual(attempted, set(pages))
        chunk_ids = [cid for cid, _t, _m in output]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))  # no collisions
        # First page of each batch got real text; the rest are markers.
        text_pages = {md["page"] for _c, _t, md in output if md["block_type"] == "text"}
        marker_pages = {md["page"] for _c, _t, md in output
                        if md["block_type"] == "corrupted_unresolved"}
        self.assertEqual(text_pages, {1, 4, 7})
        self.assertEqual(marker_pages, {2, 3, 5, 6})

    def test_patch_failed_batch_continues_and_marks_pages_unresolved(self):
        # A batch that raises (Docling crash / render failure) must not abort
        # the remaining batches: its pages become attempted-but-unresolved
        # markers, later batches still recover text.
        import os as _os

        call_count = {"n": 0}

        def fake_subset(_pdf, batch):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("docling exploded on this batch")
            return "sub.pdf", {i + 1: page for i, page in enumerate(batch)}

        def fake_docling(_path, _att, _meta):
            return (
                [("ATT:p1:para0:part0", "Recovered text with plenty of words here.",
                  {"page": 1, "block_type": "text"})],
                {"parser": "docling"},
            )

        with patch.dict(_os.environ, {"PDF_PAGE_PATCH_BATCH_PAGES": "2"}), \
             patch.object(docling_extract, "_build_scanned_page_subset", side_effect=fake_subset), \
             patch.object(docling_extract, "_relieve_patch_memory"), \
             patch("os.unlink"):
            output, attempted = docling_extract.patch_scanned_pages_with_docling(
                Path("missing.pdf"), [10, 11, 20, 21], attachment_key="ATT",
                meta_base={"itemKey": "I"},
                worker=_FakeDoclingWorker(side_effect=fake_docling),
            )
        # All four pages attempted despite batch 1 failing.
        self.assertEqual(attempted, {10, 11, 20, 21})
        by_page = {md["page"]: md for _c, _t, md in output}
        # Failed batch's pages -> figure markers (scanned patch), not dropped.
        self.assertEqual(by_page[10]["block_type"], "figure")
        self.assertEqual(by_page[11]["block_type"], "figure")
        # Second batch succeeded: page 20 has text, page 21 is a marker.
        self.assertEqual(by_page[20]["block_type"], "text")
        self.assertEqual(by_page[21]["block_type"], "figure")

    def test_patch_worker_crash_does_not_abort_remaining_batches(self):
        # LX6XGB67 (2026-07-31): Docling segfaulted inside OpenCV's kleidicv
        # NEON resize during a page patch. Running it in-process killed the
        # whole indexing run; through DoclingWorker the same crash arrives as
        # a RuntimeError, so only that batch's pages go unresolved.
        import os as _os

        calls = {"n": 0}

        def crashing_worker(_path, _att, _meta):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("Docling worker crashed while processing: sub.pdf")
            return (
                [("ATT:p1:para0:part0", "Recovered text with plenty of words here.",
                  {"page": 1, "block_type": "text"})],
                {"parser": "docling"},
            )

        with patch.dict(_os.environ, {"PDF_PAGE_PATCH_BATCH_PAGES": "2"}), \
             patch.object(
                 docling_extract, "_build_scanned_page_subset",
                 side_effect=lambda _pdf, batch: (
                     "sub.pdf", {i + 1: page for i, page in enumerate(batch)}
                 ),
             ), \
             patch.object(docling_extract, "_relieve_patch_memory"), \
             patch("os.unlink"):
            output, attempted = docling_extract.patch_corrupted_pages_with_docling(
                Path("missing.pdf"), [3, 4, 5, 6], attachment_key="ATT",
                meta_base={"itemKey": "I"},
                worker=_FakeDoclingWorker(side_effect=crashing_worker),
            )

        self.assertEqual(attempted, {3, 4, 5, 6})
        by_page = {md["page"]: md for _c, _t, md in output}
        self.assertEqual(by_page[3]["block_type"], "corrupted_unresolved")
        self.assertEqual(by_page[4]["block_type"], "corrupted_unresolved")
        self.assertEqual(by_page[5]["block_type"], "text")

    def test_patch_batch_size_env_parsing(self):
        import os as _os

        with patch.dict(_os.environ, {"PDF_PAGE_PATCH_BATCH_PAGES": "10"}):
            self.assertEqual(docling_extract._patch_batch_size(), 10)
        with patch.dict(_os.environ, {"PDF_PAGE_PATCH_BATCH_PAGES": "0"}):
            self.assertEqual(docling_extract._patch_batch_size(), 1)  # floor
        with patch.dict(_os.environ, {"PDF_PAGE_PATCH_BATCH_PAGES": "junk"}):
            self.assertEqual(docling_extract._patch_batch_size(), 24)  # default
        _os.environ.pop("PDF_PAGE_PATCH_BATCH_PAGES", None)
        self.assertEqual(docling_extract._patch_batch_size(), 24)

    def test_reference_sections_still_filters_to_reference_zones_after_refactor(self):
        # Regression: extract_reference_sections_with_docling must keep its
        # reference/footnote-only filtering after sharing _docling_subset_pages.
        fake_chunks = [
            ("ATT:p1:para0:part0", "Smith 2020 reference entry",
             {"page": 1, "zone": "bibliography", "block_type": "reference"}),
            ("ATT:p1:para1:part0", "Ordinary body paragraph, not a reference",
             {"page": 1, "zone": "body", "block_type": "text"}),
        ]
        with patch.object(
            docling_extract, "_docling_subset_pages",
            return_value=(fake_chunks, {1: 20}, {"parser": "docling"}),
        ):
            output = docling_extract.extract_reference_sections_with_docling(
                Path("missing-fixture.pdf"), [20], attachment_key="ATT", meta_base={"itemKey": "ITEM"},
                worker=_FakeDoclingWorker(([], {})),
            )
        self.assertEqual(len(output), 1)
        chunk_id, text, md = output[0]
        self.assertIn(":docref:", chunk_id)
        self.assertEqual(md["page"], 20)
        self.assertIn("Smith 2020", text)

    def test_zone_heading_context_resets_at_next_body_section(self):
        self.assertEqual(
            docling_extract.zone_for_docling_item("section_header", "参考文献"),
            "bibliography",
        )
        self.assertEqual(
            docling_extract.zone_for_docling_item("text", "entry", "bibliography"),
            "bibliography",
        )
        self.assertEqual(
            docling_extract.zone_for_docling_item("section_header", "Conclusion", "bibliography"),
            "body",
        )


if __name__ == "__main__":
    unittest.main()


class MergeDoclingChunksTests(unittest.TestCase):
    """The merge boundary must not key on a per-chunk-unique value.

    reading_order used to be part of the boundary key, and it is unique per
    chunk -- so every chunk formed its own group and no two could ever merge.
    The rescue this call exists to perform (join short fragments into a real
    chunk) was a no-op by construction for every Docling document
    (2026-07-28).
    """

    def _chunk(self, cid, text, *, reading_order, block_type="paragraph",
               zone="body", structure_path=("Chapter One",)):
        return (cid, text, {
            "reading_order": reading_order, "block_type": block_type,
            "zone": zone, "structure_path": list(structure_path),
        })

    def test_consecutive_short_paragraphs_under_the_same_heading_merge(self):
        chunks = [
            self._chunk(f"c{i}", "short line of text here", reading_order=i)
            for i in range(6)
        ]
        merged = docling_extract._merge_docling_chunks(chunks, min_chars=100, max_chars=1000)
        self.assertLess(len(merged), len(chunks))
        self.assertGreaterEqual(sum(len(text) for _cid, text, _md in merged), 100)

    def test_a_heading_change_still_forms_a_new_group(self):
        # Long enough to survive HARD_MIN_CHARS on its own, short of min_chars
        # so it would merge with an adjacent chunk under the same boundary.
        text = "word " * 20
        chunks = [
            self._chunk("a", text, reading_order=0, structure_path=("Chapter One",)),
            self._chunk("b", text, reading_order=1, structure_path=("Chapter Two",)),
        ]
        merged = docling_extract._merge_docling_chunks(chunks, min_chars=1000, max_chars=2000)
        # Both stay separate (below min_chars, above HARD_MIN_CHARS) but must
        # not merge across headings, even though they are adjacent in
        # reading order.
        self.assertEqual(len(merged), 2)

    def test_a_zone_change_still_forms_a_new_group(self):
        text = "word " * 20
        chunks = [
            self._chunk("a", text, reading_order=0, zone="body"),
            self._chunk("b", text, reading_order=1, zone="endnote"),
        ]
        merged = docling_extract._merge_docling_chunks(chunks, min_chars=1000, max_chars=2000)
        self.assertEqual(len(merged), 2)

    def test_preserve_short_labels_are_never_merged_into(self):
        chunks = [
            self._chunk("a", "short text", reading_order=0),
            ("t", "A Table Caption", {
                "reading_order": 1, "block_type": "caption", "zone": "body",
                "structure_path": ["Chapter One"],
            }),
            self._chunk("b", "more short text", reading_order=2),
        ]
        merged = docling_extract._merge_docling_chunks(chunks, min_chars=100, max_chars=1000)
        ids = [cid for cid, _text, _md in merged]
        self.assertIn("t", ids)
