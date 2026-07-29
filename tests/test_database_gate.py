from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from src.database_gate import database_state_fingerprint, validate_database_gate


def test_database_state_fingerprint_changes_with_chunk_identity():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = root / "manifest.json"
        pipeline = root / "pipeline.json"
        manifest.write_text("{}", encoding="utf-8")
        pipeline.write_text("{}", encoding="utf-8")
        first = database_state_fingerprint(
            manifest_path=manifest, pipeline_config_path=pipeline,
            chroma_rows=[{"id": "a", "text": "one", "metadata": {"itemKey": "I"}}],
            lexical_ids=["a"],
            structure_rows=[("structure", "I", "fp", "3", "exact")],
        )
        second = database_state_fingerprint(
            manifest_path=manifest, pipeline_config_path=pipeline,
            chroma_rows=[{"id": "a", "text": "changed", "metadata": {"itemKey": "I"}}],
            lexical_ids=["a"],
            structure_rows=[("structure", "I", "fp", "3", "exact")],
        )
        assert first != second


def test_database_gate_rejects_a_changed_database_generation():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "gate.json"
        path.write_text(json.dumps({
            "new_only": True,
            "new_collection": "v3",
            "gate": {"passed": True},
            "database_state": {
                "fingerprint": "sha256:audited",
                "manifest_path": "manifest.json",
                "lexical_db_path": "lexical.sqlite3",
                "pipeline_config_path": "pipeline.json",
            },
        }), encoding="utf-8")
        with patch(
            "src.database_gate.live_database_state_fingerprint",
            return_value="sha256:changed",
        ):
            with pytest.raises(RuntimeError, match="changed after its audit"):
                validate_database_gate(path, collection_name="v3")


def test_database_gate_requires_new_only_pass():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "gate.json"
        path.write_text(json.dumps({
            "new_only": False,
            "new_collection": "v3",
            "gate": {"passed": True},
            "database_state": {},
        }), encoding="utf-8")
        with pytest.raises(RuntimeError, match="--new-only"):
            validate_database_gate(path)
