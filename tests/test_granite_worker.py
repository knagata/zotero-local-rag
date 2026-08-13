"""Tests for the isolated Granite worker (note 80, choice C).

Granite cannot share the main virtualenv -- mlx-vlm needs transformers>=5.14
while the Docling pipeline pins <5.9 on macOS -- so the boundary is a plain
subprocess speaking JSON, not the forked worker Docling uses. These tests pin
that protocol and its failure handling without launching the real model.
"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from granite_worker import GraniteUnavailable, GraniteWorker  # noqa: E402


def _completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(
        args=["python", "runner"], returncode=returncode, stdout=stdout, stderr=stderr,
    )


OK_RESPONSE = json.dumps({
    "status": "ok",
    "chunks": [["ATT:p1:para0", "body text", {"page": 1, "zone": "body"}]],
    "quality_info": {"parser": "granite_docling_mlx", "total_pages": 1},
})


class GraniteWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.worker = GraniteWorker(timeout_sec=5)
        self.interpreter = patch(
            "granite_worker.granite_python", return_value=Path(sys.executable),
        )
        self.interpreter.start()

    def tearDown(self) -> None:
        self.interpreter.stop()

    def test_successful_run_restores_chunk_tuples(self):
        # JSON has no tuples, so the parent must rebuild the shape the rest of
        # the pipeline expects from DoclingWorker.
        with patch("subprocess.run", return_value=_completed(OK_RESPONSE)):
            chunks, quality = self.worker.extract(Path("x.pdf"), "ATT", {})
        self.assertEqual(len(chunks), 1)
        self.assertIsInstance(chunks[0], tuple)
        self.assertEqual(chunks[0][1], "body text")
        self.assertEqual(quality["parser"], "granite_docling_mlx")

    def test_request_carries_the_document_identity(self):
        with patch("subprocess.run", return_value=_completed(OK_RESPONSE)) as run:
            self.worker.extract(Path("/tmp/doc.pdf"), "ATT", {"itemKey": "ITEM"})
        request = json.loads(run.call_args.kwargs["input"])
        self.assertEqual(request["pdf_path"], "/tmp/doc.pdf")
        self.assertEqual(request["attachment_key"], "ATT")
        self.assertEqual(request["meta_base"]["itemKey"], "ITEM")

    def test_reported_error_becomes_a_runtime_error(self):
        payload = json.dumps({
            "status": "error", "message": "Pipeline VlmPipeline failed",
            "traceback": "ValueError: Coordinate lower is less than upper",
        })
        with patch("subprocess.run", return_value=_completed(payload)):
            with self.assertRaises(RuntimeError) as caught:
                self.worker.extract(Path("x.pdf"), "ATT", {})
        self.assertIn("Pipeline VlmPipeline failed", str(caught.exception))
        self.assertIn("Coordinate lower is less than upper", str(caught.exception))

    def test_a_crash_with_no_output_reports_stderr(self):
        with patch("subprocess.run", return_value=_completed("", "Killed: 9", 137)):
            with self.assertRaises(RuntimeError) as caught:
                self.worker.extract(Path("x.pdf"), "ATT", {})
        self.assertIn("Killed", str(caught.exception))

    def test_unparseable_output_is_reported_rather_than_raising_valueerror(self):
        with patch("subprocess.run", return_value=_completed("not json at all")):
            with self.assertRaises(RuntimeError) as caught:
                self.worker.extract(Path("x.pdf"), "ATT", {})
        self.assertIn("unparseable", str(caught.exception))

    def test_timeout_is_reported_as_a_runtime_error(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 5)):
            with self.assertRaises(RuntimeError) as caught:
                self.worker.extract(Path("x.pdf"), "ATT", {})
        self.assertIn("timed out", str(caught.exception))

    def test_malformed_chunk_rows_are_dropped_rather_than_crashing(self):
        payload = json.dumps({
            "status": "ok",
            "chunks": [["only", "two"], "a string", ["id", "text", {"page": 1}]],
            "quality_info": {},
        })
        with patch("subprocess.run", return_value=_completed(payload)):
            chunks, _quality = self.worker.extract(Path("x.pdf"), "ATT", {})
        self.assertEqual(len(chunks), 1)


class GraniteUnavailableTests(unittest.TestCase):
    def test_a_missing_interpreter_names_the_path_and_the_remedy(self):
        worker = GraniteWorker()
        with patch("granite_worker.granite_python", return_value=Path("/nope/python")):
            self.assertFalse(worker.available())
            with self.assertRaises(GraniteUnavailable) as caught:
                worker.extract(Path("x.pdf"), "ATT", {})
        message = str(caught.exception)
        self.assertIn("/nope/python", message)
        self.assertIn("GRANITE_VENV_PYTHON", message)


if __name__ == "__main__":
    unittest.main()
