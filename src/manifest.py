# src/manifest.py
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

#: Bytes sampled from each end of a file for content_signature. A full-file
#: hash would need to read every tracked PDF on every startup check just to
#: confirm nothing changed, which does not scale; a replacement almost always
#: differs somewhere in its first or last megabyte (front matter, colophon,
#: trailer), so a bounded sample catches the practical case cheaply.
CONTENT_SIGNATURE_SAMPLE_BYTES = 1_048_576


def content_signature(path: Path, size: int) -> str:
    """A cheap fingerprint of a file's content, not just its size and mtime.

    The skip check that decides whether to re-parse a source compared only
    mtime and size, so a file replaced at the same path with the same
    byte-count -- a corrected scan re-saved from the same source, a sync tool
    that does not always preserve mtime -- kept its stale text indefinitely,
    and nothing could detect it: the manifest recorded no information a
    replacement would change (2026-07-28).
    """
    digest = hashlib.sha256()
    digest.update(str(int(size)).encode("ascii"))
    with open(path, "rb") as handle:
        digest.update(handle.read(CONTENT_SIGNATURE_SAMPLE_BYTES))
        if size > CONTENT_SIGNATURE_SAMPLE_BYTES:
            handle.seek(max(size - CONTENT_SIGNATURE_SAMPLE_BYTES, 0))
            digest.update(handle.read(CONTENT_SIGNATURE_SAMPLE_BYTES))
    return "sha256:" + digest.hexdigest()


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    """
    Manifest format:
    {
      "version": 1,
      "files": {
         "<attachmentKey>": {"mtime": <float>, "size": <int>, "pdf_path": "<str>"}
      },
      "notes": {
         "<noteKey>": {"version": <int|null>}
      }
    }
    """
    if not manifest_path.exists():
        return {"version": 1, "files": {}, "notes": {}}

    try:
        txt = manifest_path.read_text(encoding="utf-8").strip()
        if not txt:
            return {"version": 1, "files": {}, "notes": {}}

        obj = json.loads(txt)
        if not isinstance(obj, dict):
            return {"version": 1, "files": {}, "notes": {}}

        obj.setdefault("version", 1)
        obj.setdefault("files", {})
        obj.setdefault("notes", {})

        if not isinstance(obj["files"], dict):
            obj["files"] = {}
        if not isinstance(obj["notes"], dict):
            obj["notes"] = {}

        return obj

    except Exception:
        # Best-effort quarantine of a corrupt manifest.
        backup = manifest_path.with_suffix(".json.bak")
        try:
            manifest_path.replace(backup)
        except Exception:
            pass
        return {"version": 1, "files": {}, "notes": {}}


#: Set while this thread holds the writer lock, so a ``save_manifest`` inside an
#: ``updating`` block does not deadlock against the lock its own caller holds.
_HOLDING = threading.local()


@contextmanager
def _writer_lock(manifest_path: Path):
    """Serialize manifest writers on a guard file beside the manifest.

    The guard has a stable inode and is never replaced, for the same reason the
    indexing lock has one: a lock keyed on a path that gets replaced is not a
    lock. Re-entrant within a thread so ``updating`` can save through the same
    code path as everyone else.
    """
    if getattr(_HOLDING, "path", None) == str(manifest_path):
        yield
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    guard = manifest_path.with_name(f".{manifest_path.name}.writer")
    guard_fd = os.open(guard, os.O_RDWR | os.O_CREAT, 0o600)
    _HOLDING.path = str(manifest_path)
    try:
        fcntl.flock(guard_fd, fcntl.LOCK_EX)
        yield
    finally:
        _HOLDING.path = None
        fcntl.flock(guard_fd, fcntl.LOCK_UN)
        os.close(guard_fd)


def _write(manifest_path: Path, manifest: Dict[str, Any]) -> None:
    """Replace the manifest with a fully written file, or leave it alone.

    The temporary file gets an inode of its own. Every writer used to build the
    same ``.json.tmp`` path, so two of them wrote into one file and whichever
    renamed second published a mixture -- and a crash mid-write left that
    shared name behind for the next writer to append meaning to. ``fsync``
    before the rename, and on the directory after it, so a crash cannot leave
    the rename recorded and the bytes missing.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2)
    handle, temporary = tempfile.mkstemp(
        dir=str(manifest_path.parent), prefix=f".{manifest_path.name}.", suffix=".tmp",
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest_path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    directory_fd = os.open(str(manifest_path.parent), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def save_manifest(manifest_path: Path, manifest: Dict[str, Any]) -> None:
    """Write the manifest, serialized against other writers."""
    with _writer_lock(manifest_path):
        _write(manifest_path, manifest)


@contextmanager
def updating(manifest_path: Path):
    """Read, change and write the manifest without losing a concurrent change.

    ``save_manifest`` alone cannot prevent a lost update: two callers that each
    load, work for a minute and save leave only the second one's version, and
    nothing reports that the first one's entries are gone. Holding the lock
    across the whole cycle is the only thing that does, so the cycle is offered
    as one operation:

        with updating(path) as manifest:
            manifest["files"].pop(key, None)

    Nothing is written if the body raises.
    """
    with _writer_lock(manifest_path):
        manifest = load_manifest(manifest_path)
        yield manifest
        _write(manifest_path, manifest)