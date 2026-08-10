from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.embedding_scale_pilot import main, run_pilot
from tests.data_plane_fixture import temporary_data_plane
from tests.synthetic_library import deterministic_embedding_function


def test_fake_pilot_reports_scale_recovery_reopen_and_hnsw(tmp_path: Path):
    with temporary_data_plane(tmp_path / "fixture"):
        report = run_pilot(
            tmp_path / "fixture" / "pilot", deterministic_embedding_function(),
            batch_size=4, sync_threshold=2, chunk_count=12,
        )

    assert report["embedding_dimension"] == 32
    assert report["chunk_count"] == 12
    assert report["throughput_chunks_per_second"] > 0
    assert report["data_plane_bytes"] > 0
    assert report["baseline_data_plane_bytes"] > 0
    assert report["incremental_data_plane_bytes"] > 0
    assert report["incremental_bytes_per_chunk"] > 0
    assert report["count_before_close"] == report["count_after_reopen"] == 12
    assert report["ids_match_after_reopen"]
    assert report["hnsw_query"]["ids"]
    recovery = report["interruption_resume"]
    assert recovery["enabled"] and recovery["ids_complete"]
    compensation = report["compensation"]
    assert compensation["restored_ids_match"]
    assert compensation["stores_match"]


def test_pilot_cli_requires_an_explicit_embedding_mode(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, "scripts/embedding_scale_pilot.py", "--fake",
         "--data-plane", str(tmp_path / "plane"), "--chunks", "3", "--no-recovery"],
        check=True, capture_output=True, text=True,
    )
    report = json.loads(result.stdout)
    assert report["profile"] == "fake"
    assert report["isolated_data_plane"] == str(tmp_path / "plane")
    assert (tmp_path / "plane" / "chroma" / "chroma.sqlite3").exists()


def test_pilot_refuses_the_active_plane_and_restores_sync_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    with temporary_data_plane(tmp_path / "active") as plane:
        with pytest.raises(ValueError, match="overlaps the active V3 plane"):
            run_pilot(
                plane.chroma_dir, deterministic_embedding_function(),
                exercise_recovery=False,
            )

        monkeypatch.setenv("CHROMA_HNSW_SYNC_THRESHOLD", "73")
        run_pilot(
            tmp_path / "isolated", deterministic_embedding_function(),
            chunk_count=3, exercise_recovery=False,
        )
        assert os.environ["CHROMA_HNSW_SYNC_THRESHOLD"] == "73"


def test_recovery_exercise_requires_a_partial_first_batch(tmp_path: Path):
    with pytest.raises(ValueError, match="chunk_count greater than batch_size"):
        run_pilot(
            tmp_path / "pilot", deterministic_embedding_function(),
            batch_size=8, chunk_count=8,
        )


def test_pilot_refuses_to_measure_an_existing_pilot_plane(tmp_path: Path):
    root = tmp_path / "pilot"
    first = run_pilot(
        root, deterministic_embedding_function(),
        chunk_count=3, exercise_recovery=False,
    )

    with pytest.raises(ValueError, match="must be a new or empty directory"):
        run_pilot(
            root, deterministic_embedding_function(),
            chunk_count=3, exercise_recovery=False,
        )
    with patch(
        "scripts.embedding_scale_pilot._embedder",
        side_effect=AssertionError("embedder must not be created"),
    ) as create_embedder, pytest.raises(
        ValueError, match="must be a new or empty directory",
    ):
        main([
            "--fake", "--data-plane", str(root),
            "--chunks", "3", "--no-recovery",
        ])

    assert first["count_after_reopen"] == 3
    create_embedder.assert_not_called()


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
def test_pilot_report_refuses_active_relations_database_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    suffix: str,
):
    relations = tmp_path / "active-relations.db"
    output = Path(f"{relations}{suffix}")
    original = b"sqlite-data-must-survive"
    output.write_bytes(original)
    monkeypatch.setenv("RELATIONS_DB_PATH", str(relations))

    with pytest.raises(ValueError, match="report path overlaps the active V3 plane"):
        main([
            "--fake", "--data-plane", str(tmp_path / "pilot"),
            "--chunks", "3", "--no-recovery", "--output", str(output),
        ])

    assert output.read_bytes() == original
    assert not (tmp_path / "pilot").exists()
