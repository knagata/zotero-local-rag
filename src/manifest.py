# src/manifest.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


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