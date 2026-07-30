from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "run_mistral_ocr_batch", ROOT / "scripts" / "run_mistral_ocr_batch.py",
)
MODULE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MODULE)


def _uploaded_state(input_path: Path) -> dict:
    return {
        "schema_version": "mistral-ocr-batch-v1",
        "phase": "uploaded",
        "model": "mistral-ocr-latest",
        "input_files": [{"input_path": str(input_path), "input_file_id": "file-123"}],
        "candidates": [],
        "candidate_count": 1,
    }


class SubmitIdempotencyTests(unittest.TestCase):
    def test_submit_checkpoints_phase_before_calling_create_job(self):
        # 2026-07-30 regression: create_job() (a billed call) used to run
        # before any state was persisted recording the attempt, so a crash
        # between create_job() succeeding and the final save left
        # phase="uploaded" with every input_file_id populated -- a retry saw
        # no pending uploads and called create_job() again, submitting a
        # second duplicate paid batch job.
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input-001.jsonl"
            input_path.write_text("{}\n", encoding="utf-8")
            state_path = Path(directory) / "state.json"
            state = _uploaded_state(input_path)

            fake_client = Mock()
            phases_at_create_job_time = []

            def create_job_side_effect(*args, **kwargs):
                phases_at_create_job_time.append(
                    json.loads(state_path.read_text(encoding="utf-8"))["phase"]
                )
                return {"id": "job-1"}

            fake_client.create_job.side_effect = create_job_side_effect

            args = argparse.Namespace(state=state_path, upload_workers=1, timeout_hours=24)
            with patch.object(MODULE, "client_from_env", return_value=fake_client):
                result = MODULE.submit(args, state)

            # phase="submitting" was already on disk (not "uploaded") by the
            # time the billed create_job() call happened.
            self.assertEqual(phases_at_create_job_time, ["submitting"])
            self.assertEqual(result["phase"], "submitted")
            self.assertEqual(result["job_id"], "job-1")
            fake_client.create_job.assert_called_once()

    def test_submit_refuses_to_retry_from_an_interrupted_submitting_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input-001.jsonl"
            input_path.write_text("{}\n", encoding="utf-8")
            state_path = Path(directory) / "state.json"
            state = _uploaded_state(input_path)
            state["phase"] = "submitting"

            fake_client = Mock()
            args = argparse.Namespace(state=state_path, upload_workers=1, timeout_hours=24)
            with patch.object(MODULE, "client_from_env", return_value=fake_client):
                with self.assertRaises(RuntimeError):
                    MODULE.submit(args, state)
            fake_client.create_job.assert_not_called()


if __name__ == "__main__":
    unittest.main()
