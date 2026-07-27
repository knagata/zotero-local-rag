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
