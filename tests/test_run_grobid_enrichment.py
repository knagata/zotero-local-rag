from unittest import mock

from scripts.run_grobid_enrichment import PROCESSOR, already_processed


def test_already_processed_requires_attachment_fingerprint_and_version():
    row = {
        "artifact_type": "references", "attachment_key": "A1", "status": "success",
        "source_fingerprint": "sha256:abc", "processor_version": PROCESSOR,
    }
    with mock.patch("scripts.run_grobid_enrichment.get_item_processing_status", return_value=[row]):
        assert already_processed("I1", "A1", "sha256:abc")
        assert not already_processed("I1", "A2", "sha256:abc")
        assert not already_processed("I1", "A1", "sha256:def")


def test_failed_reference_enrichment_is_not_considered_complete():
    row = {
        "artifact_type": "references", "attachment_key": "A1", "status": "failed",
        "source_fingerprint": "sha256:abc", "processor_version": PROCESSOR,
    }
    with mock.patch("scripts.run_grobid_enrichment.get_item_processing_status", return_value=[row]):
        assert not already_processed("I1", "A1", "sha256:abc")


def test_the_worker_refuses_to_run_while_the_feature_is_off():
    """The flag used to be overwritten here, which made it decorative.

    Enabling GROBID is the operator's declaration that the service exists; the
    worker checks that declaration instead of assuming it (note 84).
    """
    from scripts import run_grobid_enrichment as worker

    with mock.patch.object(worker, "grobid_enrichment_enabled", return_value=False), \
            mock.patch.object(worker, "asyncio") as scheduler:
        assert worker.main([]) == 2
        scheduler.run.assert_not_called()


def test_an_enabled_feature_without_its_service_stops_before_any_work():
    from scripts import run_grobid_enrichment as worker

    with mock.patch.object(worker, "grobid_enrichment_enabled", return_value=True), \
            mock.patch.object(
                worker, "verify_enabled_features",
                return_value=["GROBID enrichment is enabled but no service answers."],
            ), \
            mock.patch.object(worker, "asyncio") as scheduler:
        assert worker.main([]) == 2
        scheduler.run.assert_not_called()


def test_a_coherent_configuration_reaches_the_run_loop():
    from scripts import run_grobid_enrichment as worker

    counts = {"eligible": 0, "processed": 0, "skipped": 0, "failed": 0, "references": 0}
    with mock.patch.object(worker, "grobid_enrichment_enabled", return_value=True), \
            mock.patch.object(worker, "verify_enabled_features", return_value=[]), \
            mock.patch.object(worker, "run", return_value=None), \
            mock.patch.object(worker.asyncio, "run", return_value=counts) as run:
        assert worker.main([]) == 0
        assert run.call_count == 1
