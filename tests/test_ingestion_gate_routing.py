"""Tests for the note-78 ingestion-gate audit fixes (P1/P2/P5) in
index_from_zotero's routing helpers (user approval 2026-07-26)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import index_from_zotero as module  # noqa: E402


class PriorNoStructureAiTocStatusTests(unittest.TestCase):
    # P1: a prior "the LLM said there are no headings" verdict for the exact
    # same source file is final -- reingest skips the AI-TOC fast path.
    def _prev(self, status: str, *, mtime: float = 100.0, size: int = 2048) -> dict:
        return {
            "mtime": mtime, "size": size,
            "quality": {"ai_toc_recovery_status": status},
        }

    def test_no_structure_verdict_for_same_file_is_returned(self):
        prev = self._prev("insufficient_inferred_headings")
        self.assertEqual(
            module._prior_no_structure_ai_toc_status(prev, mtime=100.0, size=2048),
            "insufficient_inferred_headings",
        )
        prev = self._prev("insufficient_body_headings")
        self.assertEqual(
            module._prior_no_structure_ai_toc_status(prev, mtime=100.0, size=2048),
            "insufficient_body_headings",
        )

    def test_changed_source_file_invalidates_the_cached_verdict(self):
        prev = self._prev("insufficient_inferred_headings")
        self.assertIsNone(module._prior_no_structure_ai_toc_status(prev, mtime=200.0, size=2048))
        self.assertIsNone(module._prior_no_structure_ai_toc_status(prev, mtime=100.0, size=4096))

    def test_alignment_failures_and_errors_are_not_cached(self):
        # Alignment logic improves across code versions and errors are
        # retryable -- only the LLM's own "no headings" verdict is final.
        for status in (
            "body_coverage_below_threshold", "structured_chunk_ratio_below_threshold",
            "recovery_error:boom", "accepted", "",
        ):
            prev = self._prev(status)
            self.assertIsNone(
                module._prior_no_structure_ai_toc_status(prev, mtime=100.0, size=2048),
                status,
            )

    def test_missing_or_malformed_manifest_entry_returns_none(self):
        self.assertIsNone(module._prior_no_structure_ai_toc_status(None, mtime=1.0, size=1))
        self.assertIsNone(module._prior_no_structure_ai_toc_status({}, mtime=1.0, size=1))
        self.assertIsNone(module._prior_no_structure_ai_toc_status(
            {"mtime": "not-a-number", "size": 1, "quality": {}}, mtime=1.0, size=1,
        ))


class MistralCandidateResumeTests(unittest.TestCase):
    def test_normal_resumption_skips_but_explicit_repair_routes_again(self):
        ordinary = dict(
            stype="pdf", reocr_route=None, force_reparse=False,
            reparse_corrupted=False, use_docling=False,
        )
        self.assertTrue(module._skip_current_mistral_toc_candidate(**ordinary))
        for override in ("force_reparse", "reparse_corrupted", "use_docling"):
            case = dict(ordinary)
            case[override] = True
            self.assertFalse(module._skip_current_mistral_toc_candidate(**case))
        self.assertFalse(module._skip_current_mistral_toc_candidate(
            **{**ordinary, "reocr_route": {"target_engine": "mistral_ocr"}},
        ))


class ScannedPdfOcrRoutingTests(unittest.TestCase):
    """Regression fixtures for the 2026-07-27 scan OCR routing contract."""

    def _route(self, quality: dict, pages: int) -> tuple[str | None, str]:
        return module._scanned_pdf_ocr_route(
            quality, total_pages=pages, item_key="ITEM",
        )

    def test_each_size_bucket_uses_its_configured_engine(self):
        # Choice (C) (note 80): the operator picks per bucket rather than the
        # router inferring an engine from length.
        env = {
            "PDF_STRUCTURE_ENGINE_SHORT": "docling",
            "PDF_STRUCTURE_ENGINE_LONG": "granite",
            "PDF_STRUCTURE_ENGINE_PAGE_BOUNDARY": "30",
        }
        with patch.dict("os.environ", env):
            self.assertEqual(
                self._route({"source_class": "scanned_no_text"}, 29),
                ("docling", "structure_engine_docling"),
            )
            self.assertEqual(
                self._route({"source_class": "scanned_no_text"}, 30),
                ("granite", "structure_engine_granite"),
            )

    def test_mistral_is_reached_through_the_batch_queue(self):
        env = {"PDF_STRUCTURE_ENGINE_LONG": "mistral",
               "PDF_MISTRAL_TOC_QUEUE_ENABLE": "1"}
        with patch.dict("os.environ", env):
            self.assertEqual(
                self._route({"source_class": "scanned_no_text"}, 300),
                ("mistral_batch", "structure_engine_mistral"),
            )

    def test_choosing_mistral_with_the_queue_off_falls_back_to_docling(self):
        # The document still gets structured rather than failing outright.
        env = {"PDF_STRUCTURE_ENGINE_LONG": "mistral",
               "PDF_MISTRAL_TOC_QUEUE_ENABLE": "0"}
        with patch.dict("os.environ", env):
            self.assertEqual(
                self._route({"source_class": "scanned_no_text"}, 300),
                ("docling", "mistral_batch_queue_disabled"),
            )

    def test_the_boundary_is_configurable(self):
        env = {"PDF_STRUCTURE_ENGINE_SHORT": "docling",
               "PDF_STRUCTURE_ENGINE_LONG": "granite",
               "PDF_STRUCTURE_ENGINE_PAGE_BOUNDARY": "100"}
        with patch.dict("os.environ", env):
            self.assertEqual(
                self._route({"source_class": "scanned_no_text"}, 99)[0], "docling",
            )
            self.assertEqual(
                self._route({"source_class": "scanned_no_text"}, 100)[0], "granite",
            )

    def test_accepted_ocr_layer_is_adopted_without_replacement(self):
            self.assertEqual(
                self._route({
                    "source_class": "scanned_ocr_layer",
                    "ocr_layer_quality": "acceptable",
                }, 300),
                (None, "not_scan_ocr_replacement"),
            )

    def test_ocr_layer_waits_for_audit_then_accepted_result_is_adopted(self):
        # This models the two calls made by main_async: the initial branch must
        # not replace a present OCR layer, and the post-audit branch must adopt
        # an acceptable verdict without a cloud/local replacement.
        extracted = {"source_class": "scanned_ocr_layer", "total_pages": 80}
        self.assertEqual(
            module._initial_scanned_pdf_ocr_route(
                extracted, total_pages=80, item_key="ITEM",
            ),
            (None, "awaiting_ocr_layer_audit"),
        )
        audited = {**extracted, "ocr_layer_quality": "acceptable"}
        self.assertEqual(self._route(audited, 80), (None, "not_scan_ocr_replacement"))

    def test_ocr_layer_waits_for_audit_then_degraded_result_uses_threshold(self):
        extracted = {"source_class": "scanned_ocr_layer", "total_pages": 30}
        with patch.dict("os.environ", {
            "PDF_MISTRAL_TOC_QUEUE_ENABLE": "1",
            "PDF_STRUCTURE_ENGINE_LONG": "mistral",
        }):
            self.assertEqual(
                module._initial_scanned_pdf_ocr_route(
                    extracted, total_pages=30, item_key="ITEM",
                ),
                (None, "awaiting_ocr_layer_audit"),
            )
            self.assertEqual(self._route({
                **extracted, "ocr_layer_quality": "degraded",
            }, 30), ("mistral_batch", "structure_engine_mistral"))

    def test_rejected_or_unverified_ocr_layer_uses_same_threshold(self):
        # An unexplained "unverified" (no reason recorded) is still treated
        # conservatively; a *reasoned* one is not -- see AuditOutcomeTests.
        for verdict in ("degraded", "unverified", ""):
            with patch.dict("os.environ", {
                "PDF_MISTRAL_TOC_QUEUE_ENABLE": "1",
                "PDF_STRUCTURE_ENGINE_LONG": "mistral",
            }):
                self.assertEqual(
                    self._route({
                        "source_class": "scanned_ocr_layer",
                        "ocr_layer_quality": verdict,
                    }, 30),
                    ("mistral_batch", "structure_engine_mistral"),
                )

    def test_queue_disabled_keeps_long_scan_local(self):
        with patch.dict("os.environ", {
            "PDF_MISTRAL_TOC_QUEUE_ENABLE": "0",
            "PDF_STRUCTURE_ENGINE_LONG": "mistral",
        }):
            self.assertEqual(
                self._route({"source_class": "scanned_no_text"}, 30),
                ("docling", "mistral_batch_queue_disabled"),
            )


class DeferredOcrAuditCacheTests(unittest.TestCase):
    def test_deferral_retains_only_audit_and_source_provenance(self):
        previous = {
            "quality": {"parser": "mistral_ocr", "blocks": 99, "old": "kept"},
        }
        fresh = {
            "parser": "pymupdf", "blocks": 3,
            "source_class": "scanned_ocr_layer", "pdf_producer": "Paper Capture",
            "ocr_layer_quality": "degraded", "ocr_layer_error_rate": 0.02,
            "ocr_layer_audit_reason": "measured", "ocr_layer_sampled_pages": [50, 60, 70],
        }
        merged = module._merge_deferred_ocr_audit_quality(previous, fresh)
        self.assertEqual(merged["parser"], "mistral_ocr")
        self.assertEqual(merged["blocks"], 99)
        self.assertEqual(merged["old"], "kept")
        self.assertEqual(merged["source_class"], "scanned_ocr_layer")
        self.assertEqual(merged["ocr_layer_quality"], "degraded")
        self.assertNotEqual(merged["parser"], fresh["parser"])

    def test_docling_replacement_keeps_stage2_audit_for_manifest_reuse(self):
        # A short, degraded scan takes the direct Docling branch rather than
        # the generic escalation branch. Its replacement quality must retain
        # the measured fields so the next run sees a persisted audit instead
        # of reporting audit_not_persisted and requeueing the attachment.
        replacement = {"parser": "docling", "total_pages": 12}
        prior = {
            "parser": "pymupdf", "ocr_layer_quality": "degraded",
            "ocr_layer_error_rate": 0.031, "ocr_layer_audit_reason": "measured",
            "ocr_layer_verified_count": 7, "ocr_layer_rejected_count": 2,
        }
        carried = module._carry_ocr_layer_audit(replacement, prior)
        self.assertEqual(carried["parser"], "docling")
        self.assertEqual(carried["ocr_layer_quality"], "degraded")
        self.assertEqual(carried["ocr_layer_error_rate"], 0.031)
        self.assertEqual(carried["ocr_layer_audit_reason"], "measured")
        self.assertEqual(carried["ocr_layer_verified_count"], 7)


class DoclingEscalationGateTests(unittest.TestCase):
    # P2: the total-gate Docling escalation output is no longer adopted
    # entirely ungated -- gibberish blocks and repeat artifacts (the two
    # content checks from evaluate_local_ocr_gate) reject it.
    def test_clean_output_is_acceptable(self):
        chunks = [
            ("c1", "An ordinary paragraph of extracted body text with real words.", {}),
            ("c2", "Another normal paragraph long enough to look like prose.", {}),
        ]
        ok, counts = module._docling_escalation_acceptable(chunks)
        self.assertTrue(ok)
        self.assertEqual(counts, {"gibberish_blocks": 0, "repeat_artifacts": 0})

    def test_gibberish_block_rejects(self):
        chunks = [
            ("c1", "An ordinary paragraph of extracted body text with real words.", {}),
            # >50 chars of mostly non-printable/symbol soup fails looks_like_gibberish.
            ("c2", "\x00\x01\x02\x03\x04\x05\x06\x07" * 10, {}),
        ]
        ok, counts = module._docling_escalation_acceptable(chunks)
        self.assertFalse(ok)
        self.assertGreaterEqual(counts["gibberish_blocks"], 1)

    def test_repeat_artifact_rejects(self):
        chunks = [("c1", "Prefix text " + ("=" * 40) + " suffix", {})]
        ok, counts = module._docling_escalation_acceptable(chunks)
        self.assertFalse(ok)
        self.assertGreaterEqual(counts["repeat_artifacts"], 1)


class SortChunksInReadingOrderTests(unittest.TestCase):
    # P5: page patches splice recovered chunks into (page, reading_order)
    # position instead of appending them after the final chapter.
    def test_patched_chunks_interleave_by_page(self):
        chunks = [
            ("body:p1", "page one", {"page": 1, "reading_order": 0}),
            ("body:p3", "page three", {"page": 3, "reading_order": 0}),
            # Patch output appended last but belonging to page 2.
            ("scanpatch:p2", "recovered page two", {"page": 2, "reading_order": 0}),
        ]
        ordered = module._sort_chunks_in_reading_order(chunks)
        self.assertEqual([cid for cid, _t, _m in ordered], ["body:p1", "scanpatch:p2", "body:p3"])

    def test_sort_is_stable_within_a_page(self):
        chunks = [
            ("a", "one", {"page": 2, "reading_order": 1}),
            ("b", "two", {"page": 2}),  # missing reading_order -> 0
            ("c", "three", {"page": 2, "reading_order": 1}),
        ]
        ordered = module._sort_chunks_in_reading_order(chunks)
        self.assertEqual([cid for cid, _t, _m in ordered], ["b", "a", "c"])


class ExactChunkDeduplicationTests(unittest.TestCase):
    def test_drops_only_equivalent_repair_stage_duplicates(self):
        chunks = [
            ("XGEQTPS3:scanpatch:b0:p1", "recovered", {"page": 1, "zone": "body"}),
            ("XGEQTPS3:scanpatch:b0:p1", "recovered", {"zone": "body", "page": 1}),
            ("XGEQTPS3:p2", "next page", {"page": 2}),
        ]

        deduplicated = module._deduplicate_exact_chunk_records(chunks)

        self.assertEqual([row[0] for row in deduplicated], ["XGEQTPS3:scanpatch:b0:p1", "XGEQTPS3:p2"])

    def test_rejects_conflicting_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "conflicting duplicate chunk id: XGEQTPS3:p1"):
            module._deduplicate_exact_chunk_records([
                ("XGEQTPS3:p1", "first extraction", {"page": 1}),
                ("XGEQTPS3:p1", "different extraction", {"page": 1}),
            ])


class AiTocReasonClassificationTests(unittest.TestCase):
    def test_reason_sets_are_disjoint(self):
        self.assertFalse(
            module.NO_STRUCTURE_AI_TOC_REASONS & module.AI_TOC_ALIGNMENT_FAILURE_REASONS,
        )


if __name__ == "__main__":
    unittest.main()


class StructureRecoveryDisabledTests(unittest.TestCase):
    """Choice (A) off: PDFs are indexed as plain text (note 80).

    No structuring engine is reachable, so the scan route has nothing to
    escalate to. Local OCR is unaffected -- turning a page image into text is
    extraction, not structure recovery, and choice (A) says OCR stays on
    RapidOCR/NDLOCR.
    """

    def _route(self, quality, pages):
        return module._scanned_pdf_ocr_route(
            quality, total_pages=pages, item_key="ITEM",
        )

    def test_scan_route_is_inert_when_recovery_is_off(self):
        with patch.object(module, "pdf_structure_recovery_enabled", return_value=False):
            for pages in (10, 300):
                self.assertEqual(
                    self._route({"source_class": "scanned_no_text"}, pages),
                    (None, "structure_recovery_disabled"),
                )

    def test_scan_route_is_active_when_recovery_is_on(self):
        with (
            patch.object(module, "pdf_structure_recovery_enabled", return_value=True),
            patch.object(module, "mistral_batch_queue_enabled", return_value=False),
        ):
            self.assertEqual(
                self._route({"source_class": "scanned_no_text"}, 10),
                ("docling", "structure_engine_docling"),
            )

    def test_a_degraded_ocr_layer_is_still_not_replaced_when_recovery_is_off(self):
        with patch.object(module, "pdf_structure_recovery_enabled", return_value=False):
            route, reason = self._route({
                "source_class": "scanned_ocr_layer", "ocr_layer_quality": "degraded",
            }, 300)
        self.assertIsNone(route)
        self.assertEqual(reason, "structure_recovery_disabled")


class StructureEngineDispatchTests(unittest.TestCase):
    """_structure_with_engine picks the worker and degrades safely (note 80)."""

    class _Worker:
        def __init__(self, result=None, error=None):
            self.result, self.error, self.calls = result, error, 0

        def extract(self, *_args, **_kwargs):
            self.calls += 1
            if self.error:
                raise self.error
            return self.result

    def _run(self, engine, docling, granite):
        return module._structure_with_engine(
            engine, Path("x.pdf"), "ATT", {},
            docling_worker=docling, granite_worker=granite,
        )

    def test_docling_engine_uses_the_docling_worker(self):
        docling = self._Worker(result=([("id", "t", {})], {"parser": "docling"}))
        granite = self._Worker(result=([], {}))
        _chunks, quality = self._run("docling", docling, granite)
        self.assertEqual(quality["parser"], "docling")
        self.assertEqual(granite.calls, 0)

    def test_granite_engine_uses_the_granite_worker(self):
        docling = self._Worker(result=([], {"parser": "docling"}))
        granite = self._Worker(result=([("id", "t", {})], {"parser": "granite_docling_mlx"}))
        _chunks, quality = self._run("granite", docling, granite)
        self.assertEqual(quality["parser"], "granite_docling_mlx")
        self.assertEqual(docling.calls, 0)

    def test_granite_failure_falls_back_to_docling(self):
        # Granite was chosen for quality, not as a requirement, so losing it
        # must not cost the document its structure entirely.
        docling = self._Worker(result=([("id", "t", {})], {"parser": "docling"}))
        granite = self._Worker(error=RuntimeError("mlx buffer exhausted"))
        _chunks, quality = self._run("granite", docling, granite)
        self.assertEqual(quality["parser"], "docling")
        self.assertEqual(docling.calls, 1)


class SourceContentUnchangedTests(unittest.TestCase):
    """mtime and size alone cannot tell a same-size replacement from no change.

    A file replaced at the same path with the same byte-count -- a corrected
    scan re-saved from the same source, a sync tool that does not always
    preserve mtime -- kept its stale text indefinitely, and nothing could
    detect it: the manifest recorded no information a replacement would
    change (2026-07-28).
    """

    def test_matching_mtime_and_size_with_no_stored_signature_is_unchanged(self):
        # An old row written before content signatures existed must not be
        # forced to re-parse solely for lacking one.
        prev = {"mtime": 100.0, "size": 2048}
        self.assertTrue(module._source_content_unchanged(
            prev, mtime=100.0, size=2048, signature=None))

    def test_a_stored_signature_that_disagrees_means_changed(self):
        prev = {"mtime": 100.0, "size": 2048, "content_signature": "sha256:old"}
        self.assertFalse(module._source_content_unchanged(
            prev, mtime=100.0, size=2048, signature="sha256:new"))

    def test_a_stored_signature_that_agrees_means_unchanged(self):
        prev = {"mtime": 100.0, "size": 2048, "content_signature": "sha256:same"}
        self.assertTrue(module._source_content_unchanged(
            prev, mtime=100.0, size=2048, signature="sha256:same"))

    def test_differing_mtime_or_size_means_changed_regardless_of_signature(self):
        prev = {"mtime": 100.0, "size": 2048, "content_signature": "sha256:same"}
        self.assertFalse(module._source_content_unchanged(
            prev, mtime=200.0, size=2048, signature="sha256:same"))
        self.assertFalse(module._source_content_unchanged(
            prev, mtime=100.0, size=4096, signature="sha256:same"))

    def test_no_prior_entry_means_changed(self):
        self.assertFalse(module._source_content_unchanged(
            None, mtime=100.0, size=2048, signature=None))
