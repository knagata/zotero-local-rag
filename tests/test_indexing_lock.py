from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import index_from_zotero as module  # noqa: E402


class IndexingLockTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "indexing.lock"
        self.path_patch = patch.object(module, "INDEXING_LOCK_PATH", self.path)
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)
        self.addCleanup(self.tempdir.cleanup)

    def test_live_owner_prevents_a_second_acquisition(self):
        first = module._acquire_indexing_lock()
        self.addCleanup(module._release_indexing_lock, first)
        with self.assertRaisesRegex(SystemExit, "別のインデクサー"):
            module._acquire_indexing_lock()

    def test_release_does_not_delete_another_owners_lock(self):
        first = module._acquire_indexing_lock()
        replacement = {**first, "token": "replacement-owner"}
        self.path.write_text(json.dumps(replacement), encoding="utf-8")
        module._release_indexing_lock(first)
        self.assertTrue(self.path.exists())
        self.assertEqual(
            json.loads(self.path.read_text(encoding="utf-8"))["token"],
            "replacement-owner",
        )

    def test_lock_file_contains_private_owner_token(self):
        lock = module._acquire_indexing_lock()
        self.addCleanup(module._release_indexing_lock, lock)
        stored = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(stored["token"], lock["token"])
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_stale_recovery_cannot_remove_a_concurrently_published_lock(self):
        self.path.write_text(json.dumps({"pid": 999_999_999}), encoding="utf-8")
        stale_inspected = threading.Event()
        allow_recovery = threading.Event()
        original_read_text = Path.read_text
        reads = []

        def controlled_read(path, *args, **kwargs):
            reads.append(threading.current_thread().name)
            result = original_read_text(path, *args, **kwargs)
            if threading.current_thread().name == "recover-a" and not stale_inspected.is_set():
                stale_inspected.set()
                self.assertTrue(allow_recovery.wait(timeout=2))
            return result

        acquired = []
        conflicts = []

        def acquire():
            try:
                acquired.append(module._acquire_indexing_lock())
            except SystemExit:
                conflicts.append(True)

        with patch.object(Path, "read_text", controlled_read):
            first = threading.Thread(target=acquire, name="recover-a")
            second = threading.Thread(target=acquire, name="recover-b")
            first.start()
            self.assertTrue(stale_inspected.wait(timeout=2))
            second.start()
            time.sleep(0.05)
            # The second recovery cannot inspect/unlink while the first holds
            # the stable metadata guard around stale inspection and removal.
            self.assertNotIn("recover-b", reads)
            allow_recovery.set()
            first.join(timeout=2)
            second.join(timeout=2)

        self.assertEqual(len(acquired), 1)
        self.assertEqual(len(conflicts), 1)
        self.addCleanup(module._release_indexing_lock, acquired[0])
        stored = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(stored["token"], acquired[0]["token"])


if __name__ == "__main__":
    unittest.main()
