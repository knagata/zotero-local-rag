"""Regression checks for the restartable U5 orchestration state machine."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_u5_scanned_reingest", ROOT / "scripts" / "run_u5_scanned_reingest.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class U5QueueStateTests(unittest.TestCase):
    def test_prepare_migrates_old_schema_to_pending_and_preserves_prior_result(self):
        async def parents():
            return {"A": "ITEM"}

        source = type("Source", (), {
            "kind": "scanned_ocr_layer",
            "as_metadata": lambda self: {"source_class": "scanned_ocr_layer"},
        })()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "scan.pdf"
            pdf.write_bytes(b"placeholder")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"files": {"A": {
                "pdf_path": str(pdf), "title": "scan",
            }}}), encoding="utf-8")
            queue = root / "queue.json"
            queue.write_text(json.dumps({
                "schema_version": "u5-scanned-reingest-1",
                "items": [{
                    "attachment_key": "A", "source_fingerprint": f"stat:{pdf.stat().st_mtime}:{pdf.stat().st_size}",
                    "status": "completed", "attempts": 2, "last_result": {"log": "old.log"},
                }],
            }), encoding="utf-8")
            with (
                patch.object(MODULE, "MANIFEST", manifest),
                patch.object(MODULE, "_parent_keys", parents),
                patch.object(MODULE, "classify_pdf_source", return_value=source),
            ):
                payload = MODULE.prepare_queue(queue)
            row = payload["items"][0]
            self.assertEqual(payload["schema_version"], MODULE.QUEUE_SCHEMA_VERSION)
            self.assertEqual(row["status"], "pending")
            self.assertEqual(row["attempts"], 0)
            self.assertEqual(row["last_result"], {"log": "old.log"})
            self.assertEqual(row["migrated_from_schema"], "u5-scanned-reingest-1")

    def test_prepare_keeps_current_schema_state_but_recovers_running(self):
        async def parents():
            return {"A": "ITEM", "B": "ITEM"}

        source = type("Source", (), {
            "kind": "scanned_ocr_layer",
            "as_metadata": lambda self: {"source_class": "scanned_ocr_layer"},
        })()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {}
            rows = []
            for key in ("A", "B"):
                pdf = root / f"{key}.pdf"
                pdf.write_bytes(key.encode())
                files[key] = {"pdf_path": str(pdf)}
                rows.append({
                    "attachment_key": key,
                    "source_fingerprint": f"stat:{pdf.stat().st_mtime}:{pdf.stat().st_size}",
                    "status": "completed" if key == "A" else "running",
                    "attempts": 2,
                })
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"files": files}), encoding="utf-8")
            queue = root / "queue.json"
            queue.write_text(json.dumps({
                "schema_version": MODULE.QUEUE_SCHEMA_VERSION, "items": rows,
            }), encoding="utf-8")
            with (
                patch.object(MODULE, "MANIFEST", manifest),
                patch.object(MODULE, "_parent_keys", parents),
                patch.object(MODULE, "classify_pdf_source", return_value=source),
            ):
                payload = MODULE.prepare_queue(queue)
            by_key = {row["attachment_key"]: row for row in payload["items"]}
            self.assertEqual(by_key["A"]["status"], "completed")
            self.assertEqual(by_key["A"]["attempts"], 2)
            self.assertEqual(by_key["B"]["status"], "pending")
            self.assertEqual(by_key["B"]["attempts"], 2)
            self.assertIn("recovered_interrupted_at", by_key["B"])

    def test_runner_passes_exact_attachment_to_indexer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = root / "queue.json"
            queue.write_text(json.dumps({"items": [{
                "attachment_key": "PDF2", "item_key": "PARENT", "status": "pending",
                "attempts": 0, "source_fingerprint": "stat:1:2",
            }]}), encoding="utf-8")
            completed = SimpleNamespace(returncode=0, stdout='{"event":"index_batch_result","inflight_attachments":[]}\n')
            with (
                patch.object(MODULE.subprocess, "run", return_value=completed) as run,
                patch.object(MODULE, "_manifest_outcome", return_value={"verified": True}),
                patch.object(MODULE, "_audit_item", return_value={"item_passed": True}),
            ):
                MODULE.run_queue(queue, root / "logs", root / "audits", 3, 1)
            command = run.call_args.args[0]
            self.assertEqual(
                command[command.index("--attachment") + 1], "PDF2",
            )

    def test_retry_deferred_is_explicit_and_attachment_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory) / "queue.json"
            queue.write_text(json.dumps({"items": [
                {"attachment_key": "A", "status": "deferred"},
                {"attachment_key": "B", "status": "deferred"},
                {"attachment_key": "C", "status": "completed"},
            ]}), encoding="utf-8")
            self.assertEqual(MODULE.retry_deferred(queue, {"A"}), 1)
            rows = json.loads(queue.read_text(encoding="utf-8"))["items"]
            self.assertEqual([row["status"] for row in rows], ["pending", "deferred", "completed"])
            self.assertIn("retry_requested_at", rows[0])

    def test_result_event_uses_the_final_index_payload(self):
        output = '{"event":"other"}\n{"event":"index_batch_result","updated_pdf":1}\n'
        self.assertEqual(MODULE._result_event(output), {"event": "index_batch_result", "updated_pdf": 1})

    def test_recover_interrupted_only_resets_running_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory) / "queue.json"
            queue.write_text(json.dumps({"items": [
                {"attachment_key": "A", "status": "running"},
                {"attachment_key": "B", "status": "pending"},
            ]}), encoding="utf-8")
            self.assertEqual(MODULE.recover_interrupted(queue), 1)
            rows = json.loads(queue.read_text(encoding="utf-8"))["items"]
            self.assertEqual([row["status"] for row in rows], ["pending", "pending"])
            self.assertIn("recovered_interrupted_at", rows[0])


if __name__ == "__main__":
    unittest.main()


class ReconcileAdoptedTests(unittest.TestCase):
    """A deferral is a claim about the future and must be retired when met.

    Nothing closed a deferral once its Batch result was adopted, so the label
    outlived the work. Read back later, 31 documents looked unsent when their
    results had been adopted, and the next step would have been to pay to OCR
    them a second time (2026-07-28).
    """

    def _queue(self, tmp, rows):
        path = Path(tmp) / "queue.json"
        path.write_text(json.dumps({"schema_version": "x", "items": rows}), encoding="utf-8")
        return path

    def test_a_deferral_the_ledger_calls_adopted_is_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._queue(tmp, [{"attachment_key": "A", "status": "deferred"}])
            ledger = [{"attachment_key": "A", "status": "success", "processor_version": "mistral_ocr"}]
            with patch.object(MODULE, "get_artifact_processing_statuses", return_value=ledger):
                self.assertEqual(MODULE.reconcile_adopted(path), 1)
            row = json.loads(path.read_text())["items"][0]
            self.assertEqual(row["status"], "completed_via_mistral_batch")
            self.assertIn("reconciled_at", row)

    def test_a_deferral_with_no_ledger_record_is_left_alone(self):
        # "not yet adopted" and "adopted but unrecorded" must not be conflated
        # in the direction that silently drops the work.
        with tempfile.TemporaryDirectory() as tmp:
            path = self._queue(tmp, [{"attachment_key": "A", "status": "deferred"}])
            with patch.object(MODULE, "get_artifact_processing_statuses", return_value=[]):
                self.assertEqual(MODULE.reconcile_adopted(path), 0)
            self.assertEqual(json.loads(path.read_text())["items"][0]["status"], "deferred")

    def test_a_local_extraction_does_not_close_a_cloud_deferral(self):
        # The document was deferred *to Mistral*; a later Docling success is a
        # different outcome and must not be read as the Batch having landed.
        with tempfile.TemporaryDirectory() as tmp:
            path = self._queue(tmp, [{"attachment_key": "A", "status": "deferred"}])
            ledger = [{"attachment_key": "A", "status": "success", "processor_version": "docling"}]
            with patch.object(MODULE, "get_artifact_processing_statuses", return_value=ledger):
                self.assertEqual(MODULE.reconcile_adopted(path), 0)

    def test_rows_in_other_states_are_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._queue(tmp, [{"attachment_key": "A", "status": "completed"}])
            ledger = [{"attachment_key": "A", "status": "success", "processor_version": "mistral_ocr"}]
            with patch.object(MODULE, "get_artifact_processing_statuses", return_value=ledger):
                self.assertEqual(MODULE.reconcile_adopted(path), 0)
            self.assertEqual(json.loads(path.read_text())["items"][0]["status"], "completed")
