"""The lock that says an indexer is writing, and who is allowed to say it.

Lifted out of ``index_from_zotero`` because it was reachable only from there.
Re-OCR adoption mutates the same four stores the indexer does -- Chroma, the
lexical index, the manifest and the structure database -- and took no lock at
all, so an adoption running beside an indexer could overwrite either
generation, and an MCP query in between could read half of one.

The lock is a file whose contents name the owner. Publication is a hard link
from a fully written temporary inode, so no reader ever sees a partial lock,
and every inspection-then-removal happens inside a guard with a stable inode of
its own: a pathname alone cannot give a compare-and-remove critical section,
because the path can be replaced between the two steps.
"""
from __future__ import annotations

import atexit
import fcntl
import json
import os
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def default_path(project_root: Path) -> Path:
    """Where the lock lives, written once so two callers cannot disagree.

    ``run_reocr_queue`` and ``index_from_zotero`` both need it, and a second
    copy of ``data/indexing.lock`` in either of them is the drift this project
    has already paid for twice with the manifest and Chroma paths.
    """
    return project_root / "data" / "indexing.lock"


@contextmanager
def _metadata_guard(lock_path: Path):
    """Serialize publication, stale recovery, and release of the lock path.

    The guard has a stable inode and is never replaced. This supplies the
    compare-and-remove critical section that a pathname alone cannot provide:
    no owner can publish or release a lock between stale inspection and unlink.
    """
    guard_path = lock_path.with_name(f".{lock_path.name}.guard")
    guard_fd = os.open(guard_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(guard_fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(guard_fd, fcntl.LOCK_UN)
        os.close(guard_fd)

def acquire(lock_path: Path) -> dict:
    """Create the indexing lock file.  Exits with an error if another indexer
    is currently running (lock file exists and the owning process is alive).

    Returns the lock metadata dict that should be passed to ``release``.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_data = {
        "pid": os.getpid(),
        "token": uuid.uuid4().hex,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "operation": "indexing",
        # The lock is released at the path it was taken at, not at whatever the
        # run resolves to later. Releasing by re-resolution reads and unlinks a
        # file the lock was never held on if the plane changed in between --
        # which is how a test against a temporary plane came to touch the real
        # data directory at interpreter exit (2026-08-09).
        "path": str(lock_path),
    }
    payload = json.dumps(lock_data, ensure_ascii=False).encode("utf-8")
    candidate_path = lock_path.with_name(
        f".{lock_path.name}.{lock_data['token']}.tmp"
    )
    candidate_fd = os.open(
        candidate_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
    )
    try:
        os.write(candidate_fd, payload)
        os.fsync(candidate_fd)
    finally:
        os.close(candidate_fd)

    try:
        while True:
            with _metadata_guard(lock_path):
                try:
                    # Publish a fully written inode atomically. The metadata
                    # guard also prevents another recovery from unlinking this
                    # inode after it has inspected an older lock.
                    os.link(candidate_path, lock_path)
                except FileExistsError:
                    pass
                else:
                    break

                # Inspection and removal are one critical section. Every
                # publisher/releaser takes the same stable-inode flock.
                try:
                    existing = json.loads(lock_path.read_text(encoding="utf-8"))
                except Exception:
                    existing = {}
                existing_pid = existing.get("pid")
                if existing_pid is not None:
                    try:
                        os.kill(existing_pid, 0)  # signal 0 = existence check
                        raise SystemExit(
                            f"別のインデクサーが実行中です (PID={existing_pid})。\n"
                            f"ロックファイル: {lock_path}\n"
                            "インデクサーの完了を待ってから再実行してください。\n"
                            "（プロセスが存在しないはずなのにロックが残っている場合は、"
                            "手動で削除してください）"
                        )
                    except OSError:
                        print(
                            f"[WARN] 古いロックファイルを削除します（PID={existing_pid} は存在しません）",
                            file=sys.__stderr__,
                        )
                else:
                    print(
                        "[WARN] PID情報のない古いロックファイルを削除します",
                        file=sys.__stderr__,
                    )
                try:
                    lock_path.unlink()
                except OSError:
                    continue
    finally:
        try:
            candidate_path.unlink()
        except FileNotFoundError:
            pass

    # Ensure the lock is released on normal exit, SystemExit, or KeyboardInterrupt.
    atexit.register(release, lock_data)

    return lock_data


def release(lock_data: dict[str, Any] | None = None, *, lock_path: Path | None = None) -> None:
    """Remove only a lock owned by this process/run (best-effort)."""
    held_at = Path(lock_data["path"]) if lock_data and lock_data.get("path") else lock_path
    if held_at is None:
        return
    try:
        with _metadata_guard(held_at):
            if lock_data is not None:
                try:
                    current = json.loads(held_at.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    return
                if current.get("token") != lock_data.get("token"):
                    return
            held_at.unlink()
    except FileNotFoundError:
        pass  # already gone — nothing to do
    except OSError as e:
        print(f"[WARN] ロックファイルの削除に失敗しました: {e}", file=sys.__stderr__)


@contextmanager
def held(lock_path: Path):
    """Hold the lock for the duration of a unit of work.

    The reason this exists rather than a pair of calls: ``main_async`` acquired
    the lock and released it on the last line of its happy path, so a run that
    raised left the lock behind and relied on the ``atexit`` handler. One run
    per process made that survivable; anything that runs an ingest and then
    keeps going was refused by a lock naming its own live process.
    """
    lock_data = acquire(lock_path)
    try:
        yield lock_data
    finally:
        release(lock_data)


__all__ = ["acquire", "default_path", "held", "release"]
