"""Deterministic contracts around the PDF ingestion orchestrator.

The real ingestion baseline reaches only the ordinary embedded-text route and
cannot run in CI.  These tests replace extractor bodies with fakes, so route
dispatch and artifact-state decisions can be fixed before more of
``_extract_pdf_chunks`` is separated.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import index_from_zotero as module  # noqa: E402


def _extract_pdf(**overrides):
    values = {
        "a": SimpleNamespace(attachmentKey="ATT", title="Title"),
        "args": SimpleNamespace(use_docling=False),
        "col": object(),
        "docling_worker": Mock(),
        "file_path": Path("synthetic.pdf"),
        "files_manifest": {},
        "force_docling": False,
        "force_ndlocr": False,
        "granite_worker": Mock(),
        "manifest": {},
        "meta_base": {"itemKey": "ITEM"},
        "mtime": 100.0,
        "prev": None,
        "scope_item_key": "ITEM",
        "show_progress": False,
        "size": 2048,
        "source_metadata": {},
        "stored_signature": None,
        "structure_recovery": False,
        "v3_pipeline_fingerprint": "pipeline",
    }
    values.update(overrides)
    return module._extract_pdf_chunks(**values)


def test_explicit_ndlocr_dispatch_bypasses_docling_and_normal_pdf_extraction():
    worker = Mock()
    expected = ([('ndl:1', 'text', {'page': 1})], {'parser': 'ndlocr-lite'})
    with (
        patch.object(module, "extract_chunks_from_pdf_with_ndlocr", return_value=expected) as ndl,
        patch.object(module, "extract_chunks_from_pdf") as normal,
    ):
        result = _extract_pdf(
            force_ndlocr=True, force_docling=True, docling_worker=worker,
        )

    assert result.chunks == expected[0]
    assert result.quality == expected[1]
    ndl.assert_called_once()
    worker.extract.assert_not_called()
    normal.assert_not_called()


def test_explicit_docling_dispatch_returns_the_workers_result():
    expected = ([('docling:1', 'text', {'page': 1})], {'parser': 'docling'})
    worker = Mock()
    worker.extract.return_value = expected

    result = _extract_pdf(force_docling=True, docling_worker=worker)

    assert result.chunks == expected[0]
    assert result.quality == expected[1]
    worker.extract.assert_called_once()


def test_explicit_docling_runtime_failure_becomes_an_empty_extraction(capfd):
    worker = Mock()
    worker.extract.side_effect = RuntimeError("worker stopped")

    result = _extract_pdf(force_docling=True, docling_worker=worker)

    assert result.chunks == []
    assert result.quality == {}
    assert "attachment=ATT" in capfd.readouterr().err


def test_override_helper_leaves_the_normal_route_untouched():
    worker = Mock()
    assert module._extract_pdf_override(
        attachment_key="ATT",
        file_path=Path("synthetic.pdf"),
        meta_base={},
        force_ndlocr=False,
        use_docling=False,
        docling_worker=worker,
        show_progress=False,
    ) is None
    worker.extract.assert_not_called()


@pytest.mark.parametrize(
    "inputs,action,local_exhausted,policy_reason",
    [
        (
            {
                "structure_recovery": False,
                "scanned_batch_defer": True,
                "chunks_present": False,
                "attempted_local_ocr": False,
                "fast_path_accepted": False,
                "queue_enabled": True,
            },
            "disabled",
            False,
            "",
        ),
        (
            {
                "structure_recovery": True,
                "scanned_batch_defer": False,
                "chunks_present": True,
                "attempted_local_ocr": False,
                "fast_path_accepted": True,
                "queue_enabled": True,
            },
            "keep",
            False,
            "",
        ),
        (
            {
                "structure_recovery": True,
                "scanned_batch_defer": True,
                "chunks_present": False,
                "attempted_local_ocr": False,
                "fast_path_accepted": False,
                "queue_enabled": False,
                "initial_route_reason": "structure_engine_mistral",
            },
            "defer",
            False,
            "structure_engine_mistral",
        ),
        (
            {
                "structure_recovery": True,
                "scanned_batch_defer": False,
                "chunks_present": True,
                "attempted_local_ocr": False,
                "fast_path_accepted": False,
                "queue_enabled": True,
                "total_pages": 30,
                "minimum_pages": 30,
            },
            "defer",
            False,
            "mistral_batch_queue",
        ),
        (
            {
                "structure_recovery": True,
                "scanned_batch_defer": False,
                "chunks_present": False,
                "attempted_local_ocr": True,
                "fast_path_accepted": False,
                "queue_enabled": True,
                "total_pages": 1,
                "minimum_pages": 30,
            },
            "defer",
            True,
            "mistral_batch_queue",
        ),
        (
            {
                "structure_recovery": True,
                "scanned_batch_defer": False,
                "chunks_present": False,
                "attempted_local_ocr": True,
                "fast_path_accepted": False,
                "queue_enabled": False,
            },
            "local_ocr_exhausted",
            True,
            "",
        ),
        (
            {
                "structure_recovery": True,
                "scanned_batch_defer": False,
                "chunks_present": True,
                "attempted_local_ocr": False,
                "fast_path_accepted": False,
                "queue_enabled": False,
            },
            "docling_escalation",
            False,
            "",
        ),
    ],
)
def test_pdf_gate_plan_preserves_the_existing_route_truth_table(
    inputs, action, local_exhausted, policy_reason,
):
    inputs = dict(inputs)
    plan = module._pdf_gate_plan(
        total_pages=inputs.pop("total_pages", 10),
        minimum_pages=inputs.pop("minimum_pages", 30),
        initial_route_reason=inputs.pop("initial_route_reason", ""),
        **inputs,
    )

    assert plan.action == action
    assert plan.local_ocr_exhausted is local_exhausted
    assert plan.policy_reason == policy_reason


def test_gate_environment_is_not_consulted_when_structure_recovery_is_disabled():
    with (
        patch.object(module, "pymupdf_fast_path_passes") as fast_path,
        patch.object(module, "mistral_batch_queue_enabled") as queue_enabled,
    ):
        plan = module._pdf_gate_plan_for_extraction(
            structure_recovery=False,
            scanned_batch_defer=False,
            chunks=[("p1", "text", {})],
            attempted_local_ocr=False,
            quality={},
            total_pages=100,
            minimum_pages=30,
            initial_route_reason="",
        )

    assert plan.action == "disabled"
    fast_path.assert_not_called()
    queue_enabled.assert_not_called()


def test_a_fast_path_rejection_consults_the_queue_once():
    with (
        patch.object(module, "pymupdf_fast_path_passes", return_value=False) as fast_path,
        patch.object(module, "mistral_batch_queue_enabled", return_value=True) as queue_enabled,
    ):
        plan = module._pdf_gate_plan_for_extraction(
            structure_recovery=True,
            scanned_batch_defer=False,
            chunks=[("p1", "text", {})],
            attempted_local_ocr=False,
            quality={"parser": "pymupdf"},
            total_pages=30,
            minimum_pages=30,
            initial_route_reason="",
        )

    assert plan.action == "defer"
    fast_path.assert_called_once_with({"parser": "pymupdf"})
    queue_enabled.assert_called_once_with()


def _status(**overrides):
    values = {
        "attachment": SimpleNamespace(attachmentKey="ATT"),
        "scope_item_key": "ITEM",
        "source_type": "pdf",
        "chunks": [
            ("p1", "abc", {"page": 1}),
            ("p3", "defgh", {"page": 3}),
        ],
        "quality": {"parser": "pymupdf", "expected_pages": 3, "processed_pages": 2},
        "coverage_adopted": False,
        "coverage_gap": None,
        "truncated": False,
        "ai_toc_alignment_failed": False,
        "degraded_reason": None,
        "degraded_message": None,
    }
    values.update(overrides)
    return module._extraction_status(**values)


def test_success_status_accounts_for_output_without_marking_it_retryable():
    item_key, status, fields = _status()

    assert (item_key, status) == ("ITEM", "success")
    assert fields["retryable"] is False
    assert fields["processor_version"] == "pymupdf"
    assert fields["counts"] == {
        "chunks": 2,
        "source_type": "pdf",
        "processed_pages": 2,
        "expected_pages": 3,
        "pages_without_chunks": [2],
        "chars_out": 8,
    }


@pytest.mark.parametrize(
    "flags,retryable,extra",
    [
        ({"truncated": True}, True, {}),
        (
            {"coverage_adopted": True, "coverage_gap": {"unaccounted_units": [2]}},
            True,
            {"source_coverage_shortfall": {"unaccounted_units": [2]}},
        ),
        (
            {
                "ai_toc_alignment_failed": True,
                "quality": {
                    "parser": "pymupdf",
                    "expected_pages": 3,
                    "processed_pages": 2,
                    "ai_toc_recovery_status": "body_coverage_below_threshold",
                },
            },
            False,
            {"ai_toc_reason": "body_coverage_below_threshold"},
        ),
    ],
)
def test_partial_or_unstructured_success_is_degraded_with_precise_retryability(
    flags, retryable, extra,
):
    _item_key, status, fields = _status(
        degraded_reason="measured_reason",
        degraded_message="measured message",
        **flags,
    )

    assert status == "degraded"
    assert fields["retryable"] is retryable
    assert fields["reason_code"] == "measured_reason"
    assert fields["message"] == "measured message"
    assert fields["counts"] | extra == fields["counts"]
