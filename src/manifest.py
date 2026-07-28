# src/manifest.py
from __future__ import annotations

import hashlib
import json
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


def save_manifest(manifest_path: Path, manifest: Dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(manifest_path)