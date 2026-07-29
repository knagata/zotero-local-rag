from __future__ import annotations

import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from docling_worker import DoclingWorker, _relieve_worker_memory_pressure  # noqa: E402


# Fake stand-ins for docling_extract.extract_chunks_from_pdf_with_docling and
# index_from_zotero.relieve_memory_pressure, written to a temp directory that
# gets prepended to sys.path for the test class. multiprocessing's "spawn"
# context copies the parent's sys.path at process-start time, so the worker
# subprocess resolves these two module names to the fakes below instead of
# the real (heavy, side-effectful) modules -- keeping these tests fast and
# hermetic without ever invoking real Docling.
_FAKE_DOCLING_EXTRACT = '''
import os
import time


def extract_chunks_from_pdf_with_docling(pdf_path, attachment_key, meta_base):
    if attachment_key == "CRASH":
        os._exit(1)
    if attachment_key == "SLEEP":
        time.sleep(5)
    if attachment_key == "BOOM":
        raise ValueError("simulated extraction failure")
    chunks = [("c1", f"chunk-for-{attachment_key}", {"attachmentKey": attachment_key})]
    quality_info = {"parser": "fake-docling", "attachment_key": attachment_key}
    return chunks, quality_info
'''

_FAKE_INDEX_FROM_ZOTERO = '''
def relieve_memory_pressure():
    pass
'''


class DoclingWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(cls._tmpdir.name)
        (tmp_path / "docling_extract.py").write_text(_FAKE_DOCLING_EXTRACT)
        (tmp_path / "index_from_zotero.py").write_text(_FAKE_INDEX_FROM_ZOTERO)
        cls._sys_path_entry = str(tmp_path)
        sys.path.insert(0, cls._sys_path_entry)

    @classmethod
    def tearDownClass(cls):
        try:
            sys.path.remove(cls._sys_path_entry)
        except ValueError:
            pass
        cls._tmpdir.cleanup()

    def test_extract_success_across_process_boundary(self):
        worker = DoclingWorker(timeout_sec=10)
        try:
            chunks, quality_info = worker.extract(Path("/fake/doc.pdf"), "OK1", {})
            self.assertEqual(chunks, [("c1", "chunk-for-OK1", {"attachmentKey": "OK1"})])
            self.assertEqual(
                quality_info, {"parser": "fake-docling", "attachment_key": "OK1"}
            )
        finally:
            worker.shutdown()

    def test_respawns_after_hard_crash(self):
        worker = DoclingWorker(timeout_sec=10)
        try:
            with self.assertRaises(RuntimeError) as ctx:
                worker.extract(Path("/fake/doc.pdf"), "CRASH", {})
            self.assertIn("crashed while processing", str(ctx.exception))

            # A subsequent call must succeed -- proving the worker respawned
            # rather than hanging or raising forever.
            chunks, quality_info = worker.extract(Path("/fake/doc.pdf"), "OK2", {})
            self.assertEqual(chunks, [("c1", "chunk-for-OK2", {"attachmentKey": "OK2"})])
        finally:
            worker.shutdown()

    def test_timeout_then_recovers(self):
        worker = DoclingWorker(timeout_sec=1)
        try:
            start = time.monotonic()
            with self.assertRaises(RuntimeError) as ctx:
                worker.extract(Path("/fake/doc.pdf"), "SLEEP", {})
            elapsed = time.monotonic() - start
            self.assertIn("timed out after", str(ctx.exception))
            # Must not have waited out the full fake 5s sleep.
            self.assertLess(elapsed, 4)

            chunks, quality_info = worker.extract(Path("/fake/doc.pdf"), "OK3", {})
            self.assertEqual(chunks, [("c1", "chunk-for-OK3", {"attachmentKey": "OK3"})])
        finally:
            worker.shutdown()

    def test_in_worker_exception_does_not_respawn_and_worker_stays_usable(self):
        worker = DoclingWorker(timeout_sec=10)
        try:
            process_before = worker._process
            with self.assertRaises(RuntimeError) as ctx:
                worker.extract(Path("/fake/doc.pdf"), "BOOM", {})
            message = str(ctx.exception)
            self.assertIn("simulated extraction failure", message)
            self.assertNotIn("crashed", message)
            self.assertNotIn("timed out", message)
            # Same process instance: a normal in-worker exception must not
            # trigger a respawn -- the worker is still alive and fine.
            self.assertIs(worker._process, process_before)

            chunks, quality_info = worker.extract(Path("/fake/doc.pdf"), "OK4", {})
            self.assertEqual(chunks, [("c1", "chunk-for-OK4", {"attachmentKey": "OK4"})])
        finally:
            worker.shutdown()

    def test_shutdown_is_idempotent(self):
        worker = DoclingWorker(timeout_sec=10)
        worker.shutdown()
        worker.shutdown()  # must not raise

    def test_shutdown_after_crash_is_safe(self):
        worker = DoclingWorker(timeout_sec=10)
        with self.assertRaises(RuntimeError):
            worker.extract(Path("/fake/doc.pdf"), "CRASH", {})
        worker.shutdown()
        worker.shutdown()

    def test_worker_cleanup_never_calls_mps_empty_cache(self):
        calls: list[str] = []
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                is_available=lambda: True,
                empty_cache=lambda: calls.append("cuda"),
            ),
            mps=types.SimpleNamespace(
                empty_cache=lambda: calls.append("mps"),
            ),
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            _relieve_worker_memory_pressure()
        self.assertEqual(calls, ["cuda"])


if __name__ == "__main__":
    unittest.main()
