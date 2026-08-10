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
import docling_extract  # noqa: E402


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


@pytest.mark.parametrize("route", ["docling", "granite"])
def test_initial_scan_local_replacement_attaches_provenance_and_skips_layer_audit(route):
    source = (
        [],
        {
            "parser": "pymupdf",
            "total_pages": 1,
            "source_class": module.SCANNED_NO_TEXT,
        },
    )
    replacement = (
        [("ocr-p1", "recovered " * 20, {"page": 1, "reading_order": 0})],
        {"parser": route, "total_pages": 1, "has_outline": True},
    )
    docling_worker = Mock()
    granite_worker = Mock()
    with (
        patch.object(module, "extract_chunks_from_pdf", return_value=source),
        patch.object(
            module,
            "_initial_scanned_pdf_ocr_route",
            return_value=(route, f"structure_engine_{route}"),
        ),
        patch.object(module, "_structure_with_engine", return_value=replacement) as run_engine,
        patch.object(
            module,
            "_audit_pdf_ocr_layer",
            side_effect=lambda **values: (values["chunks"], values["quality"]),
        ) as audit,
        patch.object(module, "_scanned_pdf_ocr_route") as post_audit_route,
        patch.object(module, "pymupdf_fast_path_passes", return_value=True),
    ):
        result = _extract_pdf(
            docling_worker=docling_worker,
            granite_worker=granite_worker,
            source_metadata={
                "source_class": module.SCANNED_NO_TEXT,
                "pdf_producer": "synthetic scanner",
            },
            structure_recovery=True,
        )

    assert [row[0] for row in result.chunks] == ["ocr-p1"]
    assert result.quality["parser"] == route
    assert result.quality["source_class"] == module.SCANNED_NO_TEXT
    assert result.quality["pdf_producer"] == "synthetic scanner"
    assert result.quality["ocr_layer_quality"] == "not_applicable"
    assert result.quality["ocr_layer_audit_reason"] == "not_applicable_no_ocr_layer"
    run_engine.assert_called_once_with(
        route,
        Path("synthetic.pdf"),
        "ATT",
        {"itemKey": "ITEM"},
        docling_worker=docling_worker,
        granite_worker=granite_worker,
    )
    assert audit.call_args.kwargs["replacement_attempted"] is True
    post_audit_route.assert_not_called()


def test_initial_scan_local_replacement_failure_keeps_an_empty_extraction(capfd):
    source = (
        [],
        {
            "parser": "pymupdf",
            "total_pages": 1,
            "source_class": module.SCANNED_NO_TEXT,
        },
    )
    with (
        patch.object(module, "extract_chunks_from_pdf", return_value=source),
        patch.object(
            module,
            "_initial_scanned_pdf_ocr_route",
            return_value=("docling", "structure_engine_docling"),
        ),
        patch.object(
            module,
            "_structure_with_engine",
            side_effect=RuntimeError("worker stopped"),
        ),
        patch.object(
            module,
            "_pdf_gate_plan_for_extraction",
            return_value=module.PdfGatePlan(
                action="local_ocr_exhausted",
                local_ocr_exhausted=True,
            ),
        ),
    ):
        result = _extract_pdf(structure_recovery=True)

    assert result.chunks == []
    assert result.quality == {}
    assert "attachment=ATT" in capfd.readouterr().err


def _empty_pdf_docling_fallback(*, total_pages=2, worker_error=None):
    source = (
        [],
        {
            "parser": "pymupdf",
            "total_pages": total_pages,
            "source_class": module.BORN_DIGITAL,
        },
    )
    replacement = (
        [("docling-p1", "recovered", {"page": 1, "reading_order": 0})],
        {"parser": "docling", "total_pages": total_pages, "has_outline": True},
    )
    worker = Mock()
    worker.extract.return_value = replacement
    worker.extract.side_effect = worker_error
    gate = module.PdfGatePlan(
        action="local_ocr_exhausted" if worker_error or total_pages == 0 else "keep",
        local_ocr_exhausted=bool(worker_error or total_pages == 0),
    )
    with (
        patch.object(module, "extract_chunks_from_pdf", return_value=source),
        patch.object(
            module,
            "_initial_scanned_pdf_ocr_route",
            return_value=(None, "not_scan_ocr_replacement"),
        ),
        patch.object(module, "_pdf_gate_plan_for_extraction", return_value=gate) as plan,
    ):
        result = _extract_pdf(
            docling_worker=worker,
            source_metadata={
                "source_class": module.BORN_DIGITAL,
                "pdf_producer": "synthetic producer",
            },
            structure_recovery=True,
        )
    return result, worker, plan


def test_empty_pdf_docling_fallback_attaches_provenance_and_marks_local_attempt():
    result, worker, plan = _empty_pdf_docling_fallback()

    assert [row[0] for row in result.chunks] == ["docling-p1"]
    assert result.quality["parser"] == "docling"
    assert result.quality["source_class"] == module.BORN_DIGITAL
    assert result.quality["pdf_producer"] == "synthetic producer"
    worker.extract.assert_called_once_with(
        Path("synthetic.pdf"), "ATT", {"itemKey": "ITEM"},
    )
    assert plan.call_args.kwargs["attempted_local_ocr"] is True


def test_empty_pdf_docling_fallback_failure_remains_empty(capfd):
    result, _worker, plan = _empty_pdf_docling_fallback(
        worker_error=RuntimeError("worker stopped"),
    )

    assert result.chunks == []
    assert result.quality == {}
    assert plan.call_args.kwargs["attempted_local_ocr"] is True
    assert "attachment=ATT" in capfd.readouterr().err


def test_empty_zero_page_pdf_does_not_start_docling_fallback():
    result, worker, plan = _empty_pdf_docling_fallback(total_pages=0)

    assert result.chunks == []
    assert result.quality["total_pages"] == 0
    worker.extract.assert_not_called()
    assert plan.call_args.kwargs["attempted_local_ocr"] is False


def _generic_docling_gate_escalation(*, acceptable=True, worker_error=None):
    source = (
        [(
            "pymupdf-p1",
            "original searchable text",
            {
                "page": 1,
                "reading_order": 0,
                "quality_uncertain_reason": "page_level_warning",
            },
        )],
        {
            "parser": "pymupdf",
            "total_pages": 1,
            "has_outline": True,
            "source_class": module.BORN_DIGITAL,
            "quality_uncertain_reason": "page_level_warning",
        },
    )
    replacement = (
        [("docling-p1", "replacement text", {"page": 1, "reading_order": 0})],
        {"parser": "docling", "total_pages": 1, "has_outline": True},
    )
    worker = Mock()
    worker.extract.return_value = replacement
    worker.extract.side_effect = worker_error
    gate_counts = {"gibberish_blocks": 1, "repeat_artifacts": 0}
    with (
        patch.object(module, "extract_chunks_from_pdf", return_value=source),
        patch.object(
            module,
            "_initial_scanned_pdf_ocr_route",
            return_value=(None, "not_scan_ocr_replacement"),
        ),
        patch.object(
            module,
            "_pdf_gate_plan_for_extraction",
            return_value=module.PdfGatePlan("docling_escalation"),
        ),
        patch.object(
            module,
            "_docling_escalation_acceptable",
            return_value=(acceptable, gate_counts),
        ) as content_gate,
    ):
        result = _extract_pdf(
            docling_worker=worker,
            source_metadata={
                "source_class": module.BORN_DIGITAL,
                "pdf_producer": "synthetic producer",
            },
            structure_recovery=True,
        )
    return result, worker, content_gate, gate_counts


def test_generic_docling_gate_escalation_adopts_acceptable_output_with_provenance():
    result, worker, content_gate, _counts = _generic_docling_gate_escalation()

    assert [row[0] for row in result.chunks] == ["docling-p1"]
    assert result.quality["parser"] == "docling"
    assert result.quality["source_class"] == module.BORN_DIGITAL
    assert result.quality["pdf_producer"] == "synthetic producer"
    worker.extract.assert_called_once_with(
        Path("synthetic.pdf"), "ATT", {"itemKey": "ITEM"},
    )
    content_gate.assert_called_once_with(result.chunks)


def test_generic_docling_gate_rejection_keeps_original_chunks_with_merged_warning():
    result, _worker, content_gate, counts = _generic_docling_gate_escalation(
        acceptable=False,
    )

    assert [row[0] for row in result.chunks] == ["pymupdf-p1"]
    assert result.chunks[0][2]["quality_uncertain"] is True
    assert result.chunks[0][2]["quality_uncertain_reason"] == (
        "page_level_warning,docling_escalation_rejected"
    )
    assert result.quality["parser"] == "pymupdf"
    assert result.quality["quality_uncertain"] is True
    assert result.quality["quality_uncertain_reason"] == (
        "page_level_warning,docling_escalation_rejected"
    )
    assert "docling_escalation_gate" not in result.quality
    assert content_gate.call_args.args[0][0][0] == "docling-p1"
    assert counts == {"gibberish_blocks": 1, "repeat_artifacts": 0}


def test_generic_docling_gate_worker_failure_keeps_original_chunks_as_unavailable(capfd):
    result, _worker, content_gate, _counts = _generic_docling_gate_escalation(
        worker_error=RuntimeError("worker stopped"),
    )

    assert [row[0] for row in result.chunks] == ["pymupdf-p1"]
    assert result.chunks[0][2]["quality_uncertain_reason"] == (
        "page_level_warning,docling_escalation_unavailable"
    )
    assert result.quality["parser"] == "pymupdf"
    assert result.quality["quality_uncertain_reason"] == (
        "page_level_warning,docling_escalation_unavailable"
    )
    content_gate.assert_not_called()
    assert "attachment=ATT" in capfd.readouterr().err


def _ai_toc_recovery(*, recovered=None, previous=None, total_pages=30):
    source = (
        [("pymupdf-p1", "unstructured text", {"page": 1, "reading_order": 0})],
        {
            "parser": "pymupdf",
            "total_pages": total_pages,
            "has_outline": False,
            "source_class": module.BORN_DIGITAL,
        },
    )
    with (
        patch.object(module, "extract_chunks_from_pdf", return_value=source),
        patch.object(
            module,
            "_initial_scanned_pdf_ocr_route",
            return_value=(None, "not_scan_ocr_replacement"),
        ),
        patch.object(
            module,
            "try_ai_toc_fast_path",
            return_value=recovered,
        ) as run_recovery,
        patch.object(
            module,
            "_pdf_gate_plan_for_extraction",
            return_value=module.PdfGatePlan("keep"),
        ),
    ):
        result = _extract_pdf(
            prev=previous,
            structure_recovery=True,
        )
    return result, run_recovery


def test_ai_toc_reuses_same_source_no_structure_verdict_without_a_new_call():
    previous = {
        "mtime": 100.0,
        "size": 2048,
        "quality": {"ai_toc_recovery_status": "insufficient_body_headings"},
    }
    result, run_recovery = _ai_toc_recovery(previous=previous)

    assert [row[0] for row in result.chunks] == ["pymupdf-p1"]
    assert result.quality["ai_toc_recovery_status"] == "insufficient_body_headings"
    assert result.quality["ai_toc_recovery_status_cached"] is True
    run_recovery.assert_not_called()


def test_ai_toc_accepted_result_replaces_chunks_and_records_diagnostics():
    recovered = SimpleNamespace(
        accepted=True,
        reason="accepted",
        chunks=[("structured-p1", "structured", {"page": 1})],
        diagnostics={"body_coverage": 0.9, "matched_count": 3, "anchors": 4},
    )
    result, run_recovery = _ai_toc_recovery(recovered=recovered)

    assert [row[0] for row in result.chunks] == ["structured-p1"]
    assert result.quality["ai_toc_recovery_status"] == "accepted"
    assert result.quality["ai_toc_body_coverage"] == 0.9
    assert result.quality["ai_toc_matched_count"] == 3
    assert result.quality["ai_toc_diagnostics"] == recovered.diagnostics
    run_recovery.assert_called_once()
    assert run_recovery.call_args.args[0:4] == (
        Path("synthetic.pdf"),
        "ITEM",
        [("pymupdf-p1", "unstructured text", {"page": 1, "reading_order": 0})],
        {
            "parser": "pymupdf",
            "total_pages": 30,
            "has_outline": False,
            "source_class": module.BORN_DIGITAL,
        },
    )


def test_ai_toc_rejected_result_keeps_chunks_and_records_the_reason():
    recovered = SimpleNamespace(
        accepted=False,
        reason="body_coverage_below_threshold",
        chunks=[("unused", "unused", {})],
        diagnostics={"body_coverage": 0.4, "matched_count": 1},
    )
    result, _run_recovery = _ai_toc_recovery(recovered=recovered)

    assert [row[0] for row in result.chunks] == ["pymupdf-p1"]
    assert result.quality["ai_toc_recovery_status"] == recovered.reason
    assert result.quality["ai_toc_body_coverage"] == 0.4
    assert result.quality["ai_toc_matched_count"] == 1
    assert result.quality["ai_toc_diagnostics"] == recovered.diagnostics


def test_ai_toc_does_not_run_below_the_page_threshold():
    result, run_recovery = _ai_toc_recovery(total_pages=29)

    assert [row[0] for row in result.chunks] == ["pymupdf-p1"]
    assert "ai_toc_recovery_status" not in result.quality
    run_recovery.assert_not_called()


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


def _post_audit_scan_replacement(*, route, worker_error=None):
    source_chunks = [
        ("old-p1", "measured OCR text " * 10, {"page": 1, "reading_order": 0}),
        ("old-p2", "more measured text " * 10, {"page": 2, "reading_order": 0}),
    ]
    source_quality = {
        "parser": "pymupdf",
        "total_pages": 2,
        "has_outline": True,
        "source_class": module.SCANNED_OCR_LAYER,
    }
    audited_quality = {
        **source_quality,
        "ocr_layer_quality": "degraded",
        "ocr_layer_error_rate": 0.25,
        "ocr_layer_verified_count": 3,
        "ocr_layer_rejected_count": 1,
        "ocr_layer_audit_reason": "measured",
        "pre_replacement_only": "do not carry",
    }
    replacement = (
        [("new-p1", "recovered", {"page": 1, "reading_order": 0})],
        {"parser": route, "total_pages": 2, "has_outline": True},
    )
    docling_worker = Mock()
    granite_worker = Mock()
    deferred = module.PdfExtraction([], {}, deferred=True)
    gate = module.PdfGatePlan("defer" if route == "mistral_batch" else (
        "local_ocr_exhausted" if worker_error else "keep"
    ), local_ocr_exhausted=bool(worker_error))
    with (
        patch.object(
            module,
            "extract_chunks_from_pdf",
            return_value=(source_chunks, source_quality),
        ),
        patch.object(
            module,
            "_initial_scanned_pdf_ocr_route",
            return_value=(None, "awaiting_ocr_layer_audit"),
        ),
        patch.object(
            module,
            "_audit_pdf_ocr_layer",
            return_value=(source_chunks, audited_quality),
        ),
        patch.object(
            module,
            "_scanned_pdf_ocr_route",
            return_value=(route, f"structure_engine_{route}"),
        ) as choose_route,
        patch.object(
            module,
            "_structure_with_engine",
            return_value=replacement,
            side_effect=worker_error,
        ) as run_engine,
        patch.object(module, "_pdf_gate_plan_for_extraction", return_value=gate) as plan,
        patch.object(module, "_defer_pdf_to_mistral", return_value=deferred) as defer,
    ):
        result = _extract_pdf(
            docling_worker=docling_worker,
            granite_worker=granite_worker,
            source_metadata={
                "source_class": module.SCANNED_OCR_LAYER,
                "pdf_producer": "synthetic scanner",
            },
            structure_recovery=True,
        )
    return result, choose_route, run_engine, plan, defer, docling_worker, granite_worker


@pytest.mark.parametrize("route", ["docling", "granite"])
def test_post_audit_scan_replacement_carries_the_measured_verdict(route):
    result, choose_route, run_engine, plan, defer, docling, granite = (
        _post_audit_scan_replacement(route=route)
    )

    assert [row[0] for row in result.chunks] == ["new-p1"]
    assert result.quality["parser"] == route
    assert result.quality["source_class"] == module.SCANNED_OCR_LAYER
    assert result.quality["pdf_producer"] == "synthetic scanner"
    assert result.quality["ocr_layer_quality"] == "degraded"
    assert result.quality["ocr_layer_error_rate"] == 0.25
    assert result.quality["ocr_layer_verified_count"] == 3
    assert result.quality["ocr_layer_rejected_count"] == 1
    assert result.quality["ocr_layer_audit_reason"] == "measured"
    assert "pre_replacement_only" not in result.quality
    choose_route.assert_called_once()
    run_engine.assert_called_once_with(
        route,
        Path("synthetic.pdf"),
        "ATT",
        {"itemKey": "ITEM"},
        docling_worker=docling,
        granite_worker=granite,
    )
    assert plan.call_args.kwargs["attempted_local_ocr"] is True
    assert plan.call_args.kwargs["scanned_batch_defer"] is False
    defer.assert_not_called()


def test_post_audit_scan_mistral_route_defers_without_running_a_local_engine():
    result, _choose, run_engine, plan, defer, _docling, _granite = (
        _post_audit_scan_replacement(route="mistral_batch")
    )

    assert result.deferred is True
    run_engine.assert_not_called()
    assert plan.call_args.kwargs["chunks"] == []
    assert plan.call_args.kwargs["attempted_local_ocr"] is False
    assert plan.call_args.kwargs["scanned_batch_defer"] is True
    assert defer.call_args.kwargs["scanned_batch_defer"] is True
    assert defer.call_args.kwargs["quality"]["ocr_layer_quality"] == "degraded"
    assert defer.call_args.kwargs["quality"]["ocr_layer_audit_reason"] == "measured"


def test_post_audit_scan_local_replacement_failure_remains_empty(capfd):
    result, _choose, _run_engine, plan, defer, _docling, _granite = (
        _post_audit_scan_replacement(
            route="docling",
            worker_error=RuntimeError("worker stopped"),
        )
    )

    assert result.chunks == []
    assert result.quality == {}
    assert plan.call_args.kwargs["attempted_local_ocr"] is True
    assert plan.call_args.kwargs["scanned_batch_defer"] is False
    defer.assert_not_called()
    assert "attachment=ATT" in capfd.readouterr().err


def _born_digital_page_patch(*, patch_result=None, patch_error=None):
    source = (
        [
            ("p1", "page one", {"page": 1, "reading_order": 0}),
            ("p3", "page three", {"page": 3, "reading_order": 0}),
        ],
        {
            "parser": "pymupdf",
            "total_pages": 3,
            "has_outline": True,
            "source_class": module.BORN_DIGITAL,
            "scanned_pages": [2],
        },
    )
    patched = (
        [("p2", "page two", {"page": 2, "reading_order": 0})]
        if patch_result is None else patch_result
    )
    with (
        patch.object(module, "extract_chunks_from_pdf", return_value=source),
        patch.object(
            docling_extract,
            "patch_scanned_pages_with_docling",
            return_value=(patched, {2}),
            side_effect=patch_error,
        ) as run_patch,
        patch.object(
            module,
            "recompute_scanned_quality_after_patch",
            return_value={**source[1], "scanned_pages": []},
        ) as recompute,
        patch.object(module, "pymupdf_fast_path_passes", return_value=True),
    ):
        result = _extract_pdf(structure_recovery=True)
    return result, run_patch, recompute


def test_born_digital_scanned_page_patch_splices_in_reading_order_and_recomputes_quality():
    result, run_patch, recompute = _born_digital_page_patch()

    assert [row[0] for row in result.chunks] == ["p1", "p2", "p3"]
    assert result.quality["scanned_pages"] == []
    assert run_patch.call_args.args[1] == [2]
    recompute.assert_called_once()
    assert recompute.call_args.args[1:] == ({2}, 3)


def test_born_digital_scanned_page_patch_failure_keeps_the_original_extraction():
    result, _run_patch, recompute = _born_digital_page_patch(
        patch_error=RuntimeError("page worker stopped"),
    )

    assert [row[0] for row in result.chunks] == ["p1", "p3"]
    assert result.quality["scanned_pages"] == [2]
    recompute.assert_not_called()


def test_born_digital_patch_attempt_with_no_text_still_resolves_a_figure_page():
    result, _run_patch, recompute = _born_digital_page_patch(patch_result=[])

    assert [row[0] for row in result.chunks] == ["p1", "p3"]
    assert result.quality["scanned_pages"] == []
    recompute.assert_called_once()


def _scan_derived_page_patch(*, patch_error=None):
    source = (
        [
            ("p1", "x" * 100, {"page": 1, "reading_order": 0}),
            ("old-p2", "broken", {"page": 2, "reading_order": 0}),
        ],
        {
            "parser": "pymupdf",
            "total_pages": 2,
            "has_outline": True,
            "source_class": "scanned_ocr_layer",
            "blank_pages": [2],
        },
    )
    patched = [("new-p2", "recovered", {"page": 2, "reading_order": 0})]
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
        patch.object(module, "ocr_layer_audit_enabled", return_value=False),
        patch.object(
            docling_extract,
            "patch_corrupted_pages_with_docling",
            return_value=(patched, {2}),
            side_effect=patch_error,
        ) as run_patch,
        patch.object(
            module,
            "recompute_blank_pages_after_patch",
            return_value={**source[1], "blank_pages": []},
        ) as recompute,
    ):
        result = _extract_pdf(structure_recovery=True)
    return result, run_patch, recompute


def test_scan_derived_page_repair_replaces_the_failed_page_and_recomputes_blanks():
    result, run_patch, recompute = _scan_derived_page_patch()

    assert [row[0] for row in result.chunks] == ["p1", "new-p2"]
    assert result.quality["blank_pages"] == []
    assert run_patch.call_args.args[1] == [2]
    assert run_patch.call_args.kwargs["chunk_namespace"] == "scanrepair"
    recompute.assert_called_once()
    assert recompute.call_args.args[0]["blank_pages"] == [2]
    assert recompute.call_args.args[1] == {2}


def test_scan_derived_page_repair_failure_keeps_the_original_failed_page():
    result, _run_patch, recompute = _scan_derived_page_patch(
        patch_error=RuntimeError("repair worker stopped"),
    )

    assert [row[0] for row in result.chunks] == ["p1", "old-p2"]
    assert result.quality["blank_pages"] == [2]
    recompute.assert_not_called()


def _corrupted_text_page_patch(*, patch_error=None):
    source = (
        [
            ("p1", "clean", {"page": 1, "reading_order": 0}),
            ("old-p2", "garbled", {"page": 2, "reading_order": 0}),
            ("p3", "clean", {"page": 3, "reading_order": 0}),
        ],
        {
            "parser": "pymupdf",
            "total_pages": 3,
            "has_outline": True,
            "source_class": module.BORN_DIGITAL,
            "corrupted_pages": [2],
            "corrupted_ratio": 0.333,
            "is_corrupted": False,
            "extraction_failure_pages": [2],
            "extraction_failure_ratio": 0.333,
            "content_corruption_pages": [],
        },
    )
    patched = [("new-p2", "recovered", {"page": 2, "reading_order": 0})]
    repaired_quality = {
        **source[1],
        "corrupted_pages": [],
        "corrupted_ratio": 0.0,
        "extraction_failure_pages": [],
        "extraction_failure_ratio": 0.0,
    }
    with (
        patch.dict("os.environ", {"PDF_CORRUPTED_PAGE_PATCH_ENABLE": "1"}),
        patch.object(module, "extract_chunks_from_pdf", return_value=source),
        patch.object(
            docling_extract,
            "patch_corrupted_pages_with_docling",
            return_value=(patched, {2}),
            side_effect=patch_error,
        ) as run_patch,
        patch.object(
            module,
            "recompute_corrupted_quality_after_patch",
            return_value=repaired_quality,
        ) as recompute,
        patch.object(module, "pymupdf_fast_path_passes", return_value=True),
    ):
        result = _extract_pdf(structure_recovery=True)
    return result, run_patch, recompute


def test_corrupted_text_page_patch_replaces_garbled_chunks_and_recomputes_quality():
    result, run_patch, recompute = _corrupted_text_page_patch()

    assert [row[0] for row in result.chunks] == ["p1", "new-p2", "p3"]
    assert result.quality["corrupted_pages"] == []
    assert result.quality["extraction_failure_pages"] == []
    assert run_patch.call_args.args[1] == [2]
    recompute.assert_called_once()
    assert recompute.call_args.args[0]["corrupted_pages"] == [2]
    assert recompute.call_args.args[1:] == ({2}, 3)


def test_corrupted_text_page_patch_failure_keeps_the_garbled_chunks():
    result, _run_patch, recompute = _corrupted_text_page_patch(
        patch_error=RuntimeError("repair worker stopped"),
    )

    assert [row[0] for row in result.chunks] == ["p1", "old-p2", "p3"]
    assert result.quality["corrupted_pages"] == [2]
    assert result.quality["extraction_failure_pages"] == [2]
    recompute.assert_not_called()


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
