"""The real-corpus ingestion net must never inherit approval for hosted calls."""
from __future__ import annotations

import subprocess
from unittest.mock import patch

from scripts.build_ingestion_baseline import (
    HOSTED_INGESTION_FEATURE_FLAGS,
    ingest,
)


def test_ingestion_baseline_forces_hosted_features_off(tmp_path):
    inherited = {name: "1" for name in HOSTED_INGESTION_FEATURE_FLAGS}
    completed = subprocess.CompletedProcess([], 0, stdout="", stderr="")

    with patch.dict("os.environ", inherited), patch(
        "scripts.build_ingestion_baseline.subprocess.run",
        return_value=completed,
    ) as run:
        ingest("ITEM", tmp_path)

    child = run.call_args.kwargs["env"]
    assert child["PDF_CACHE_DIR"] == str(tmp_path / "pdf_cache")
    assert {
        name: child.get(name) for name in HOSTED_INGESTION_FEATURE_FLAGS
    } == {name: "0" for name in HOSTED_INGESTION_FEATURE_FLAGS}


def test_observer_reads_current_artifact_schema_and_does_not_hide_schema_errors(tmp_path):
    import sqlite3
    import pytest
    from scripts.build_ingestion_baseline import observe
    with sqlite3.connect(tmp_path / 'relations.db') as connection:
        connection.execute('CREATE TABLE artifact_processing_status(item_key, artifact_type, attachment_key, status, reason_code)')
        connection.execute("INSERT INTO artifact_processing_status VALUES ('ITEM','extraction','ATT','degraded','source_gap')")
    with patch('src.chunk_store.get_item_chunks', return_value=[]):
        assert observe('ITEM', tmp_path)['artifact_status'] == [{
            'artifact_type': 'extraction', 'attachment_key': 'ATT',
            'status': 'degraded', 'reason_code': 'source_gap'}]
        with sqlite3.connect(tmp_path / 'relations.db') as connection:
            connection.execute('ALTER TABLE artifact_processing_status RENAME COLUMN artifact_type TO wrong_column')
        with pytest.raises(sqlite3.OperationalError):
            observe('ITEM', tmp_path)


def test_failed_or_empty_measurement_cannot_become_baseline():
    import copy
    import pytest
    from scripts.build_ingestion_baseline import validate_measurement
    valid = {'item_key':'ITEM', 'exit_code':0, 'chunks':[{}], 'manifest':{'ATT':{}},
             'second_pass':{'exit_code':0}, 'quality_pass':{'exit_code':0}}
    validate_measurement([valid])
    with pytest.raises(ValueError):
        validate_measurement([])
    for phase in [None, 'second_pass', 'quality_pass']:
        invalid = copy.deepcopy(valid)
        (invalid if phase is None else invalid[phase])['exit_code'] = 1
        with pytest.raises(ValueError):
            validate_measurement([invalid])
    for field in ['chunks','manifest']:
        with pytest.raises(ValueError):
            validate_measurement([{**valid,field:[]}])
