"""Tests for scan-derived page repair selection (note 79, U2).

The distinction under test: in a scan-derived document a page without usable
text is an OCR failure and must be re-attempted, whereas in a born-digital
document the same page is a figure. Pages already carrying a marker from a
previous repair must not be re-attempted, or every run would loop on them.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import index_from_zotero as izo  # noqa: E402


def _chunk(page: int, text: str, block_type: str | None = None):
    metadata = {"page": page}
    if block_type:
        metadata["block_type"] = block_type
    return (f"AK:p{page}", text, metadata)


BODY = "x" * 400


class ScanPagesNeedingRepairTests(unittest.TestCase):
    def test_pages_with_no_text_are_selected(self):
        chunks = [_chunk(1, BODY), _chunk(3, BODY)]
        self.assertEqual(izo._scan_pages_needing_repair(chunks, 4), [2, 4])

    def test_pages_with_almost_no_text_are_selected(self):
        # Below PDF_SCAN_PAGE_REPAIR_MIN_CHARS: OCR produced a stray fragment
        # rather than the page's actual content.
        chunks = [_chunk(1, BODY), _chunk(2, "12")]
        self.assertEqual(izo._scan_pages_needing_repair(chunks, 2), [2])

    def test_healthy_document_selects_nothing(self):
        chunks = [_chunk(page, BODY) for page in range(1, 6)]
        self.assertEqual(izo._scan_pages_needing_repair(chunks, 5), [])

    def test_text_split_across_chunks_on_one_page_counts_together(self):
        chunks = [_chunk(1, "y" * 50), _chunk(1, "z" * 50)]
        self.assertEqual(izo._scan_pages_needing_repair(chunks, 1), [])

    def test_already_marked_pages_are_not_reattempted(self):
        # A previous run left markers here. Re-selecting them would re-run
        # Docling on the same pages on every subsequent ingest.
        chunks = [
            _chunk(1, BODY),
            _chunk(2, "[Corrupted page 2 - unresolved]", "corrupted_unresolved"),
            _chunk(3, "[Figure page 3]", "figure"),
        ]
        self.assertEqual(izo._scan_pages_needing_repair(chunks, 3), [])

    def test_no_pages_selected_when_page_count_unknown(self):
        self.assertEqual(izo._scan_pages_needing_repair([_chunk(1, BODY)], 0), [])

    def test_chunks_without_a_page_are_ignored(self):
        chunks = [("AK:x", BODY, {}), _chunk(1, BODY)]
        self.assertEqual(izo._scan_pages_needing_repair(chunks, 1), [])


class AuditReusedOcrChunksTests(unittest.TestCase):
    """Reused legacy OCR text must not skip the quality checks (note 79).

    It is OCR output by definition, so it is the most likely input to be
    degraded; accepting it unmeasured would carry old damage into V3.
    """

    def setUp(self) -> None:
        self._classify = izo.classify_pdf_source
        self._detect = izo.detect_text_defects
        self._audit = izo.audit_ocr_text_layer
        izo.classify_pdf_source = lambda path: type(
            "SC", (), {"as_metadata": lambda self: {"source_class": "scanned_ocr_layer"}},
        )()
        izo.detect_text_defects = lambda text: {"text_defects": []}

    def tearDown(self) -> None:
        izo.classify_pdf_source = self._classify
        izo.detect_text_defects = self._detect
        izo.audit_ocr_text_layer = self._audit

    def _run(self):
        return izo._audit_reused_ocr_chunks(
            [_chunk(1, BODY)], {"parser": "legacy-ocr-reuse"}, Path("x.pdf"),
            item_key="ITEM", prev=None, mtime=1.0, size=2, show_progress=False,
        )

    def test_acceptable_text_is_still_reused_and_gains_provenance(self):
        izo.audit_ocr_text_layer = lambda path, key: {"ocr_layer_quality": "acceptable"}
        chunks, quality, reuse_ok = self._run()
        self.assertTrue(reuse_ok)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(quality["source_class"], "scanned_ocr_layer")
        self.assertEqual(quality["ocr_layer_quality"], "acceptable")

    def test_degraded_text_abandons_reuse_so_normal_extraction_runs(self):
        izo.audit_ocr_text_layer = lambda path, key: {
            "ocr_layer_quality": "degraded", "ocr_layer_error_rate": 0.031,
        }
        chunks, quality, reuse_ok = self._run()
        self.assertFalse(reuse_ok)
        self.assertEqual(chunks, [])
        self.assertEqual(quality["ocr_layer_quality"], "degraded")

    def test_deterministic_text_defect_also_abandons_reuse(self):
        izo.detect_text_defects = lambda text: {"text_defects": ["letter_spacing"]}
        izo.audit_ocr_text_layer = lambda path, key: {"ocr_layer_quality": "acceptable"}
        chunks, _quality, reuse_ok = self._run()
        self.assertFalse(reuse_ok)
        self.assertEqual(chunks, [])

    def test_unverified_audit_does_not_block_reuse(self):
        # Failing to measure is not evidence of a problem.
        izo.audit_ocr_text_layer = lambda path, key: {
            "ocr_layer_quality": "unverified", "ocr_layer_audit_reason": "cloud_not_allowed:no-cloud",
        }
        chunks, _quality, reuse_ok = self._run()
        self.assertTrue(reuse_ok)
        self.assertEqual(len(chunks), 1)


class SourceProvenanceCarryTests(unittest.TestCase):
    def test_local_ocr_replacement_keeps_scan_provenance_for_later_gates(self):
        carried = izo._attach_pdf_source_provenance(
            {"parser": "rapidocr", "total_pages": 2},
            {"source_class": "scanned_no_text", "pdf_producer": "ScanSnap"},
        )
        self.assertEqual(carried["parser"], "rapidocr")
        self.assertEqual(carried["source_class"], "scanned_no_text")
        self.assertEqual(carried["pdf_producer"], "ScanSnap")

    def test_other_ocr_routes_share_the_same_provenance_contract(self):
        for parser in ("docling", "mistral_ocr", "legacy-ocr-reuse"):
            carried = izo._attach_pdf_source_provenance(
                {"parser": parser}, {"source_class": "scanned_ocr_layer"},
            )
            self.assertEqual(carried["parser"], parser)
            self.assertEqual(carried["source_class"], "scanned_ocr_layer")


if __name__ == "__main__":
    unittest.main()
