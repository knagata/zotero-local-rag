"""Tests for the scan-derived OCR text-layer audit (note 79, stage 2).

The cases here encode the failure modes the 18-document calibration actually
exhibited -- fabricated evidence, an uncalibrated verdict label, an unstable
word estimate -- so a regression that reintroduces any of them fails here.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ocr_layer_audit as ola  # noqa: E402
from llm_client import InvalidLLMResponse  # noqa: E402
from ocr_layer_audit import (  # noqa: E402
    ACCEPTABLE, DEGRADED, UNVERIFIED, audit_ocr_text_layer, classify_rate,
    verify_reported_items,
)


class _FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def generate_json(self, prompt, schema=None, timeout=None, max_tokens=None):
        self.calls += 1
        self.prompt = prompt
        self.max_tokens = max_tokens
        return self.payload


SAMPLE = (
    "the settlement co the north retained its sin1ply built granary while "
    "the surrounding fields were tended communally throughout the season "
) * 12


class VerifyReportedItemsTests(unittest.TestCase):
    def test_items_absent_from_the_source_are_rejected(self):
        # The calibration's central failure: the model echoed the prompt's own
        # example strings for documents where it had little to report, and one
        # document was called severely degraded on entirely fabricated evidence.
        items = [
            {"as_printed": "co", "likely_intended": "to", "defect": "ocr_misrecognition"},
            {"as_printed": "righrs", "likely_intended": "rights", "defect": "ocr_misrecognition"},
        ]
        verified, rejected = verify_reported_items(items, SAMPLE)
        self.assertEqual([item["as_printed"] for item in verified], ["co"])
        self.assertEqual([item["as_printed"] for item in rejected], ["righrs"])

    def test_wholly_fabricated_report_verifies_to_nothing(self):
        items = [
            {"as_printed": f"bogus{n}", "likely_intended": "x", "defect": "ocr_misrecognition"}
            for n in range(40)
        ]
        verified, rejected = verify_reported_items(items, SAMPLE)
        self.assertEqual(verified, [])
        self.assertEqual(len(rejected), 40)

    def test_non_corrections_are_rejected(self):
        items = [{"as_printed": "the", "likely_intended": "the", "defect": "ocr_misrecognition"}]
        verified, _ = verify_reported_items(items, SAMPLE)
        self.assertEqual(verified, [])

    def test_only_ocr_misrecognition_counts(self):
        # Hyphenation and accent/quote normalisation showed up as "defects" on
        # healthy born-digital controls, so other classes must not count.
        items = [
            {"as_printed": "co", "likely_intended": "to", "defect": "word_boundary"},
            {"as_printed": "the", "likely_intended": "th e", "defect": "letter_spacing"},
        ]
        verified, rejected = verify_reported_items(items, SAMPLE)
        self.assertEqual(verified, [])
        self.assertEqual(len(rejected), 2)

    def test_malformed_payloads_do_not_raise(self):
        for payload in (None, "text", 5, [None, "x", {}]):
            verified, _ = verify_reported_items(payload, SAMPLE)
            self.assertEqual(verified, [])


class ClassifyRateTests(unittest.TestCase):
    def test_threshold_boundary(self):
        self.assertEqual(classify_rate(0.015), DEGRADED)
        self.assertEqual(classify_rate(0.0149), ACCEPTABLE)
        self.assertEqual(classify_rate(0.0), ACCEPTABLE)


class AuditOcrTextLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._sample = ola.sample_body_pages
        ola.sample_body_pages = lambda path, pages=3: (SAMPLE, [10, 11, 12])

    def tearDown(self) -> None:
        ola.sample_body_pages = self._sample

    def test_verified_errors_produce_a_rate_and_verdict(self):
        llm = _FakeLLM({"items": [
            {"as_printed": "co", "likely_intended": "to", "defect": "ocr_misrecognition"},
            {"as_printed": "sin1ply", "likely_intended": "simply", "defect": "ocr_misrecognition"},
        ]})
        result = audit_ocr_text_layer(Path("x.pdf"), "ITEM", client=llm)
        self.assertEqual(result["ocr_layer_verified_count"], 2)
        self.assertEqual(result["ocr_layer_audit_reason"], "measured")
        self.assertIsNotNone(result["ocr_layer_error_rate"])

    def test_denominator_is_counted_locally_not_taken_from_the_model(self):
        # The model's own word estimate varied between 2,000 and 6,452 for the
        # same text during calibration, so anything it claims is ignored.
        llm = _FakeLLM({"items": [], "approx_words_examined": 5})
        result = audit_ocr_text_layer(Path("x.pdf"), "ITEM", client=llm)
        self.assertGreater(result["ocr_layer_denominator"], 100)

    def test_model_verdict_field_is_ignored(self):
        # It did not track its own counts: severe at 1.0%, minor at 3.75%.
        llm = _FakeLLM({"items": [], "verdict": "severe_degradation"})
        result = audit_ocr_text_layer(Path("x.pdf"), "ITEM", client=llm)
        self.assertEqual(result["ocr_layer_quality"], ACCEPTABLE)

    def test_fabricated_report_yields_acceptable_not_degraded(self):
        llm = _FakeLLM({"items": [
            {"as_printed": f"ghost{n}", "likely_intended": "x", "defect": "ocr_misrecognition"}
            for n in range(40)
        ]})
        result = audit_ocr_text_layer(Path("x.pdf"), "ITEM", client=llm)
        self.assertEqual(result["ocr_layer_quality"], ACCEPTABLE)
        self.assertEqual(result["ocr_layer_verified_count"], 0)
        self.assertEqual(result["ocr_layer_rejected_count"], 40)

    def test_prompt_contains_no_literal_example_strings(self):
        # Those examples are exactly what the model echoed back as findings.
        for leaked in ("righrs", "srorage", "wirhout", "1ike", "sofware"):
            self.assertNotIn(leaked, ola.AUDIT_PROMPT)

    def test_no_cloud_policy_gate_remains(self):
        # Removed deliberately (2026-07-27): anything indexed is returned by
        # search and reaches the assistant anyway, so the gate protected
        # nothing it was still in -- while a momentarily unreachable Zotero
        # made it fail closed and re-OCR every scanned PDF.
        self.assertFalse(hasattr(ola, "cloud_text_allowed"))

    def test_llm_failure_leaves_the_layer_unverified_rather_than_degraded(self):
        class Boom:
            def generate_json(self, *a, **k):
                raise RuntimeError("connection reset")

        result = audit_ocr_text_layer(Path("x.pdf"), "ITEM", client=Boom())
        self.assertEqual(result["ocr_layer_quality"], UNVERIFIED)
        self.assertIn("llm_failed", result["ocr_layer_audit_reason"])

    def test_a_truncated_reply_is_retried_once(self):
        # A degraded CJK document filled the 4096-token default and came back
        # as unparseable JSON, so the audit gave up precisely on the documents
        # it exists to catch (XGEQTPS3, 2026-07-27).
        class FlakyThenFine:
            def __init__(self):
                self.calls = 0

            def generate_json(self, prompt, schema=None, timeout=None, max_tokens=None):
                self.calls += 1
                if self.calls == 1:
                    raise ola.LLMError("Invalid JSON response: Expecting ',' delimiter")
                return {"items": [{"as_printed": "co", "likely_intended": "to",
                                   "defect": "ocr_misrecognition"}]}

        llm = FlakyThenFine()
        result = audit_ocr_text_layer(Path("x.pdf"), "ITEM", client=llm)
        self.assertEqual(llm.calls, 2)
        self.assertEqual(result["ocr_layer_audit_reason"], "measured")

    def test_audit_asks_for_a_token_budget_large_enough_for_a_full_list(self):
        llm = _FakeLLM({"items": []})
        audit_ocr_text_layer(Path("x.pdf"), "ITEM", client=llm)
        self.assertGreaterEqual(llm.max_tokens, 8000)

    def test_too_small_a_sample_is_not_measured(self):
        ola.sample_body_pages = lambda path, pages=3: ("short text", [1])
        llm = _FakeLLM({"items": []})
        result = audit_ocr_text_layer(Path("x.pdf"), "ITEM", client=llm)
        self.assertEqual(result["ocr_layer_quality"], UNVERIFIED)
        self.assertEqual(result["ocr_layer_audit_reason"], "insufficient_sample")
        self.assertEqual(llm.calls, 0)


class SalvageFromTruncatedReplyTests(unittest.TestCase):
    """A truncated reply must still yield its complete items (XGEQTPS3).

    The salvage path existed but was fed the *parser's message*, which names a
    character offset into text the client had already dropped. So every
    truncated audit still recorded ``llm_unavailable`` and the document stayed
    ``unverified`` -- and truncation correlates with bad OCR, because the worse
    the text layer, the longer the item list. The undecodable body now travels
    on the exception.
    """

    def setUp(self) -> None:
        self._sample = ola.sample_body_pages
        ola.sample_body_pages = lambda path, pages=3: (SAMPLE, [10, 11, 12])

    def tearDown(self) -> None:
        ola.sample_body_pages = self._sample

    def test_complete_items_are_recovered_from_a_cut_off_reply(self):
        truncated = (
            '{"items": ['
            '{"as_printed": "co", "likely_intended": "to", "defect": "ocr_misrecognition"},'
            '{"as_printed": "sin1ply", "likely_intended": "simply", "defect": "ocr_misrec'
        )

        class _Truncating:
            def generate_json(self, prompt, schema=None, timeout=None, max_tokens=None):
                raise InvalidLLMResponse("Invalid JSON response: Unterminated string", raw=truncated)

        result = audit_ocr_text_layer(Path("x.pdf"), "ITEM", client=_Truncating())
        self.assertEqual(result["ocr_layer_audit_reason"], "measured")
        # The first item is whole and verifiable; the cut-off one is not.
        self.assertEqual(result["ocr_layer_verified_count"], 1)

    def test_a_reply_with_no_recoverable_item_still_reports_the_failure(self):
        class _Empty:
            def generate_json(self, prompt, schema=None, timeout=None, max_tokens=None):
                raise InvalidLLMResponse("Invalid JSON response: Unterminated string", raw='{"items": [')

        result = audit_ocr_text_layer(Path("x.pdf"), "ITEM", client=_Empty())
        self.assertTrue(result["ocr_layer_audit_reason"].startswith("llm_unavailable"))
        self.assertEqual(result["ocr_layer_quality"], UNVERIFIED)


class ReplacementRequiredTests(unittest.TestCase):
    """Which audit outcomes justify discarding the existing text (2026-07-27).

    Only a *measured* degraded verdict does. Every other unverified outcome is
    a statement about the audit, not about the text, and re-OCRing on that
    basis costs hours per document -- during a provider outage, for every
    scanned document in the run at once.
    """

    def _q(self, quality, reason):
        return {"ocr_layer_quality": quality, "ocr_layer_audit_reason": reason}

    def test_measured_degraded_is_replaced(self):
        self.assertTrue(ola.replacement_required(self._q("degraded", "measured")))

    def test_measured_acceptable_is_kept(self):
        self.assertFalse(ola.replacement_required(self._q("acceptable", "measured")))

    def test_transient_llm_failure_is_kept_for_a_later_retry(self):
        for reason in ("llm_unavailable:Invalid JSON response", "llm_failed:connection reset"):
            q = self._q("unverified", reason)
            self.assertFalse(ola.replacement_required(q), reason)
            self.assertTrue(ola.audit_was_transient_failure(q))

    def test_too_small_a_sample_is_kept_and_flagged(self):
        q = self._q("unverified", "insufficient_sample")
        self.assertFalse(ola.replacement_required(q))
        self.assertTrue(ola.audit_sample_too_small(q))

    def test_unreadable_file_is_kept_and_flagged_for_repair(self):
        q = self._q("unverified", "sampling_failed:cannot open broken document")
        self.assertFalse(ola.replacement_required(q))
        self.assertTrue(ola.audit_hit_file_problem(q))

    def test_an_audit_that_never_ran_still_triggers_replacement(self):
        # Disabling the audit must not silently promote unmeasured OCR text.
        self.assertTrue(ola.replacement_required({}))


class ShortDocumentSamplingTests(unittest.TestCase):
    def test_short_document_samples_every_page(self):
        # The middle-60% window left a two-page article with one page, which
        # happened to be its title page (VRGGXZ4Y, 2026-07-27).
        import fitz
        tmp = Path(__file__).resolve().parent / "_tmp_short.pdf"
        doc = fitz.open()
        for n in range(2):
            page = doc.new_page(width=400, height=600)
            page.insert_textbox(fitz.Rect(10, 10, 390, 590),
                                f"page {n} " + "body text here. " * 60, fontsize=8)
        doc.save(str(tmp)); doc.close()
        try:
            text, pages = ola.sample_body_pages(tmp)
            self.assertEqual(pages, [1, 2])
            self.assertIn("page 1", text)
        finally:
            tmp.unlink()


if __name__ == "__main__":
    unittest.main()


class AuditDisabledTests(unittest.TestCase):
    """Declining to measure must not be read as a finding (note 80, choice B).

    Turning the audit off is what happens whenever no paid LLM is configured.
    Treating "not measured" as "not trustworthy" there would send every scanned
    PDF to a full re-OCR, so switching the LLM off would *increase* the work
    done -- the opposite of what the operator asked for.
    """

    def _q(self, reason):
        return {"ocr_layer_quality": UNVERIFIED, "ocr_layer_audit_reason": reason}

    def test_disabled_audit_does_not_trigger_replacement(self):
        q = self._q(ola.DISABLED_REASON)
        self.assertTrue(ola.audit_was_disabled(q))
        self.assertFalse(ola.replacement_required(q))

    def test_disabled_is_distinct_from_a_failure(self):
        q = self._q(ola.DISABLED_REASON)
        self.assertFalse(ola.audit_was_transient_failure(q))
        self.assertFalse(ola.audit_hit_file_problem(q))
        self.assertFalse(ola.audit_sample_too_small(q))

    def test_an_unexplained_unverified_state_still_triggers_replacement(self):
        # No reason recorded at all: something went wrong that nobody labelled,
        # so the conservative reading applies.
        self.assertTrue(ola.replacement_required({"ocr_layer_quality": UNVERIFIED}))
