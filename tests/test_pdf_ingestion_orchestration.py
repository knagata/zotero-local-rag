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


def _mistral_deferral(*, inflight=False):
    files_manifest = {}
    manifest = {"inflight_attachments": ["ATT"] if inflight else []}
    previous = {
        "mtime": 90.0,
        "size": 1024,
        "quality": {"parser": "previous_canonical"},
        "previous_field": "preserved",
    }
    pymupdf_result = (
        [("transient", "not canonical", {"page": 1})],
        {
            "parser": "pymupdf",
            "total_pages": 30,
            "source_class": "scanned_no_text",
            "pdf_producer": "synthetic scanner",
        },
    )
    with (
        patch.object(module, "extract_chunks_from_pdf", return_value=pymupdf_result),
        patch.object(
            module,
            "_initial_scanned_pdf_ocr_route",
            return_value=("mistral_batch", "structure_engine_mistral"),
        ),
        patch.object(module, "mark_artifact_status") as status,
        patch.object(module, "save_manifest") as save,
        patch.object(module, "_delete_by_attachment_keys") as delete,
        patch.object(
            module,
            "paths",
            return_value=SimpleNamespace(manifest_path=Path("manifest.json")),
        ),
    ):
        result = _extract_pdf(
            files_manifest=files_manifest,
            manifest=manifest,
            prev=previous,
            structure_recovery=True,
            stored_signature="sha256:source",
        )
    return result, files_manifest, manifest, status, save, delete


def test_mistral_deferral_records_status_and_preserves_existing_canonical_chunks():
    result, files_manifest, manifest, status, save, delete = _mistral_deferral()

    assert result.deferred is True
    assert result.chunks == []
    assert result.quality == {}
    delete.assert_not_called()
    status.assert_called_once_with(
        "ITEM",
        "extraction",
        "blocked",
        attachment_key="ATT",
        reason_code=module.MISTRAL_TOC_QUEUE_REASON,
        message="AI TOC/PyMuPDF gate failed: scanned_pdf_ocr_replacement",
        retryable=False,
        source_fingerprint="stat:100.0:2048",
        processor_version=module.MISTRAL_TOC_QUEUE_PROCESSOR_VERSION,
        counts={
            "source_mtime": 100.0,
            "source_size": 2048,
            "total_pages": 30,
            "ai_toc_reason": "scanned_pdf_ocr_replacement",
            "fast_path_reason": None,
            "ai_toc_rejection_reason": None,
            "local_ocr_exhausted": False,
            "ai_toc_diagnostics": {},
        },
        fallback_kind="mistral_ocr",
    )
    assert manifest["inflight_attachments"] == []
    assert files_manifest["ATT"]["previous_field"] == "preserved"
    assert files_manifest["ATT"]["quality"] == {
        "parser": "previous_canonical",
        "source_class": "scanned_no_text",
        "pdf_producer": "synthetic scanner",
    }
    assert files_manifest["ATT"]["content_signature"] == "sha256:source"
    save.assert_called_once_with(Path("manifest.json"), manifest)


def test_mistral_deferral_cleans_only_a_matching_inflight_partial_write():
    _result, _files, manifest, _status, _save, delete = _mistral_deferral(inflight=True)

    delete.assert_called_once()
    assert delete.call_args.args[1] == ["ATT"]
    assert delete.call_args.kwargs == {"strict": True}
    assert manifest["inflight_attachments"] == []


@pytest.mark.parametrize(
    "chunks,recovered,local_exhausted,expected_reason",
    [
        ([], None, True, "local_ocr_quality_gate_failed"),
        ([], None, False, "pymupdf_no_chunks"),
        (
            [("p1", "text", {})],
            SimpleNamespace(
                accepted=False,
                reason="body_coverage_below_threshold",
                diagnostics={"body_coverage": 0.5},
            ),
            False,
            "body_coverage_below_threshold",
        ),
        ([("p1", "text", {})], None, False, "fast_path_rejected"),
    ],
)
def test_mistral_deferral_reason_precedence_is_explicit(
    chunks, recovered, local_exhausted, expected_reason,
):
    manifest = {"inflight_attachments": []}
    with (
        patch.object(module, "mark_artifact_status") as status,
        patch.object(module, "save_manifest"),
        patch.object(
            module,
            "paths",
            return_value=SimpleNamespace(manifest_path=Path("manifest.json")),
        ),
        patch.object(
            module,
            "pymupdf_fast_path_rejection_reason",
            return_value="fast_path_rejected",
        ),
    ):
        module._defer_pdf_to_mistral(
            attachment=SimpleNamespace(attachmentKey="ATT", title="Title"),
            scope_item_key="ITEM",
            col=object(),
            manifest=manifest,
            files_manifest={},
            previous=None,
            file_path=Path("source.pdf"),
            quality={},
            chunks=chunks,
            recovered=recovered,
            scanned_batch_defer=False,
            local_ocr_exhausted=local_exhausted,
            total_pages=30,
            mtime=100.0,
            size=2048,
            stored_signature=None,
            pipeline_fingerprint="pipeline",
            show_progress=False,
        )

    fields = status.call_args.kwargs
    assert fields["message"] == f"AI TOC/PyMuPDF gate failed: {expected_reason}"
    assert fields["counts"]["ai_toc_reason"] == expected_reason
    assert fields["counts"]["ai_toc_diagnostics"] == (
        recovered.diagnostics if recovered is not None else {}
    )


def _audit_pdf(*, enabled=True, cached=None, audit=None):
    source = (
        [("p1", "scan-derived text", {"page": 1})],
        {
            "parser": "pymupdf",
            "total_pages": 1,
            "has_outline": True,
            "source_class": "scanned_ocr_layer",
        },
    )
    with (
        patch.object(module, "extract_chunks_from_pdf", return_value=source),
        patch.object(
            module,
            "_initial_scanned_pdf_ocr_route",
            return_value=(None, "awaiting_ocr_layer_audit"),
        ),
        patch.object(
            module,
            "_scanned_pdf_ocr_route",
            return_value=(None, "not_scan_ocr_replacement"),
        ),
        patch.object(module, "ocr_layer_audit_enabled", return_value=enabled),
        patch.object(module, "_cached_ocr_layer_audit", return_value=cached) as cache,
        patch.object(module, "audit_ocr_text_layer", return_value=audit or {}) as run_audit,
        patch.object(module, "mark_artifact_status") as status,
    ):
        result = _extract_pdf(structure_recovery=False)
    return result, cache, run_audit, status


def test_disabled_ocr_layer_audit_is_distinct_from_an_unverified_failure():
    result, cache, run_audit, status = _audit_pdf(enabled=False)

    assert result.quality["ocr_layer_audit_reason"] == module.OCR_LAYER_AUDIT_DISABLED
    cache.assert_not_called()
    run_audit.assert_not_called()
    status.assert_not_called()


def test_cached_ocr_layer_audit_avoids_a_new_paid_measurement():
    cached = {
        "ocr_layer_quality": "acceptable",
        "ocr_layer_error_rate": 0.001,
        "ocr_layer_audit_reason": "measured",
    }
    result, cache, run_audit, status = _audit_pdf(cached=cached)

    assert result.quality["ocr_layer_audit_cached"] is True
    assert result.quality["ocr_layer_quality"] == "acceptable"
    cache.assert_called_once_with(None, mtime=100.0, size=2048)
    run_audit.assert_not_called()
    status.assert_not_called()


def test_transient_ocr_audit_failure_is_retryable_and_not_cached():
    audit = {
        "ocr_layer_quality": "unverified",
        "ocr_layer_audit_reason": "llm_unavailable:no configured model",
    }
    result, _cache, run_audit, status = _audit_pdf(audit=audit)

    assert result.quality["ocr_layer_needs_reaudit"] is True
    run_audit.assert_called_once_with(Path("synthetic.pdf"), "ITEM")
    status.assert_called_once_with(
        "ITEM",
        "extraction",
        "degraded",
        attachment_key="ATT",
        reason_code="ocr_layer_audit_deferred",
        message="llm_unavailable:no configured model",
        retryable=True,
    )


def test_source_file_problem_is_nonretryable_ocr_audit_degradation(capfd):
    audit = {
        "ocr_layer_quality": "unverified",
        "ocr_layer_audit_reason": "sampling_failed:broken xref",
    }
    _result, _cache, _run_audit, status = _audit_pdf(audit=audit)

    assert "repair or replace the file" in capfd.readouterr().err
    assert status.call_args.kwargs["reason_code"] == "source_file_unreadable"
    assert status.call_args.kwargs["retryable"] is False


def test_too_small_ocr_audit_sample_keeps_searchable_text_with_a_warning_tag():
    audit = {
        "ocr_layer_quality": "unverified",
        "ocr_layer_audit_reason": "insufficient_sample",
    }
    result, _cache, _run_audit, status = _audit_pdf(audit=audit)

    assert result.chunks[0][2]["quality_uncertain"] is True
    assert result.chunks[0][2]["quality_uncertain_reason"] == "ocr_layer_sample_too_small"
    assert result.quality["quality_uncertain"] is True
    status.assert_not_called()


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
