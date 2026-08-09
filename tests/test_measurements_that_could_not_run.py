"""A check that could not run must not read as a check that passed.

This is the failure mode with no symptom. A rollback that breaks leaves two
stores disagreeing and somebody eventually notices; a measurement that quietly
returns "nothing to report" leaves a library that looks fine and is not. The
project has shipped it before -- `verify_against_source` excluded a PDF it could
not read from its own audit and reported ``passed: true`` (F4, 2026-07-30).

The instance here was found by scanning for broad ``except`` handlers inside
functions that produce a verdict, then reading each one. Of eight, five were
already fail-closed and said so (``_rendered_page_is_visually_blank`` returns
False with "inability to render is not positive blank-page evidence"). One was
not, and it mattered:

``classify_pdf_source`` decides whether a PDF is a scan. On the reused-OCR path
its failure was swallowed with a bare ``pass``, so ``source_class`` was simply
absent -- and absent is indistinguishable from "born-digital" to the only thing
that reads it. ``reocr_quality`` tests ``source_class == "scanned_no_text"``,
which is False both when the document is not a scan and when nobody could tell,
so a scanned document whose classification failed would never be offered as a
re-OCR candidate again.

The tests below are about the shape rather than the site: a failed measurement
has to leave a mark, and the decision that consumes it has to treat the mark as
something to look at.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from src.reocr_quality import candidate_assessment  # noqa: E402


def _assessment(quality: dict) -> dict:
    return candidate_assessment(
        quality={"total_pages": 10, **quality},
        chunks=[{"text": "A perfectly ordinary page of readable text. " * 20,
                 "metadata": {"lang": "en"}}] * 10,
        structure_status="ok", summary_reports=[],
        current_engine="pymupdf", current_version="1",
        target_engine="docling", target_version="2",
        language="en",
    )


def test_a_document_that_classified_cleanly_is_not_a_candidate():
    # The control: without it, the test below could pass because everything is
    # a candidate.
    assessment = _assessment({"source_class": "born_digital"})
    assert "source_class_unknown" not in assessment["reasons"]
    assert not assessment["candidate"], assessment["reasons"]


def test_a_classification_that_failed_is_not_scored_as_a_clean_one():
    assessment = _assessment({"source_class_error": "cannot open PDF"})
    assert "source_class_unknown" in assessment["reasons"], (
        "a document whose source classification failed looks exactly like a "
        "born-digital one, so it will never be offered for re-OCR again"
    )
    assert assessment["candidate"]


def test_the_unknown_ranks_below_a_measured_scan():
    # Otherwise "we could not tell" outranks "we know it is a scan with no text
    # layer", and the queue is led by the documents there is least evidence for.
    unknown = _assessment({"source_class_error": "cannot open PDF"})
    measured = _assessment({"source_class": "scanned_no_text"})
    assert unknown["score"] < measured["score"]


def test_the_reuse_path_records_a_classification_failure_rather_than_dropping_it():
    # The audit's own docstring: reused text is OCR output, so accepting it
    # unmeasured carries old damage into V3. A swallowed classification is
    # exactly that, one measurement at a time.
    import index_from_zotero

    chunks = [("id-1", "Reused OCR text. " * 40, {"lang": "en"})]
    with (
        patch.object(index_from_zotero, "classify_pdf_source",
                     side_effect=RuntimeError("cannot open PDF")),
        patch.object(index_from_zotero, "ocr_layer_audit_enabled", return_value=False),
    ):
        _chunks, quality, reuse_ok = index_from_zotero._audit_reused_ocr_chunks(
            chunks=chunks, quality_info={}, pdf_path=Path("/nonexistent.pdf"),
            item_key="ITEM", prev={}, mtime=0.0, size=0, show_progress=False,
        )
    assert quality.get("source_class_error"), (
        "the classification failure left no trace, so nothing downstream can "
        "tell it apart from a document that is simply not a scan"
    )
    assert reuse_ok, "this failure is a missing measurement, not degraded text"
