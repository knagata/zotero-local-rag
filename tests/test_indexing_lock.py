from __future__ import annotations

import contextlib
import json
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
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
        # Through the run's resolved paths rather than by patching a module
        # constant: the constant is what a test could reach, not what the code
        # is about, and reaching it meant every test here had to know the
        # module's spelling of its own configuration.
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(module.use_paths(
            replace(module.paths(), indexing_lock_path=self.path),
        ))
        self.addCleanup(self.tempdir.cleanup)

    def test_live_owner_prevents_a_second_acquisition(self):
        first = module._acquire_indexing_lock()
        self.addCleanup(module._release_indexing_lock, first)
        with self.assertRaisesRegex(SystemExit, "別のインデクサー"):
            module._acquire_indexing_lock()

    def test_unverifiable_owner_is_never_removed(self):
        self.path.write_text(json.dumps({"pid": 12345, "token": "owner"}))
        with patch("os.kill", side_effect=PermissionError("denied")):
            with self.assertRaisesRegex(SystemExit, "確認できません"):
                module._acquire_indexing_lock()
        self.assertEqual(json.loads(self.path.read_text())["token"], "owner")

    def test_corrupt_or_ownerless_lock_is_never_removed(self):
        for payload in ('not json', '[]', '{}', '{"pid": -1}', '{"pid": true}'):
            self.path.write_text(payload)
            with self.assertRaisesRegex(SystemExit, "停止"):
                module._acquire_indexing_lock()
            self.assertEqual(self.path.read_text(), payload)

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

    def test_stale_lock_removal_failure_stops_without_retry(self):
        self.path.write_text(json.dumps({"pid": 12345, "token": "stale"}))
        original_unlink = Path.unlink
        attempts = []

        def refuse_stale(path, *args, **kwargs):
            if path == self.path:
                attempts.append(path)
                if len(attempts) > 1:
                    raise AssertionError("lock deletion must not retry indefinitely")
                raise PermissionError("denied")
            return original_unlink(path, *args, **kwargs)

        with patch("os.kill", side_effect=ProcessLookupError()), \
             patch.object(Path, "unlink", refuse_stale):
            with self.assertRaisesRegex(SystemExit, "削除できません"):
                module._acquire_indexing_lock()
        self.assertEqual(len(attempts), 1)
        self.assertEqual(json.loads(self.path.read_text())["token"], "stale")
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_release_preserves_unrecognizable_replacement(self):
        for payload in ("[]", "null", '"replacement"'):
            if self.path.exists():
                self.path.unlink()
            lock = module._acquire_indexing_lock()
            self.path.write_text(payload)
            module._release_indexing_lock(lock)
            self.assertEqual(self.path.read_text(), payload)

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


def test_native_read_lease_excludes_writer_publication(tmp_path):
    from src import indexing_lock
    path = tmp_path / 'indexing.lock'
    started = threading.Event()
    acquired = threading.Event()
    def write():
        started.set()
        owner = indexing_lock.acquire(path)
        acquired.set()
        indexing_lock.release(owner, lock_path=path)
    thread = threading.Thread(target=write)
    with indexing_lock.native_read_access(path):
        thread.start()
        assert started.wait(1)
        assert not acquired.wait(0.1)
        assert not path.exists()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert acquired.is_set()
    assert not path.exists()
