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
    assert {
        name: child.get(name) for name in HOSTED_INGESTION_FEATURE_FLAGS
    } == {name: "0" for name in HOSTED_INGESTION_FEATURE_FLAGS}
