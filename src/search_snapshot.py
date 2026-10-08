"""Pin query processes to a separately published, physically validated copy."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

POINTER_NAME = "search_snapshot.json"
ORIGINAL_ENV = "_RAG_SEARCH_ORIGINAL_ENV"
ACTIVE_ROOT = "_RAG_SEARCH_SNAPSHOT_ROOT"
PATH_KEYS = ("CHROMA_DIR", "MANIFEST_PATH", "LEXICAL_DB_PATH", "RELATIONS_DB_PATH", "PIPELINE_CONFIG_PATH")


def pointer_path(root: Path) -> Path:
    return root / "data" / POINTER_NAME


def load_snapshot(root: Path) -> dict | None:
    pointer = pointer_path(root)
    if not pointer.exists():
        return None
    record = json.loads(pointer.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise TypeError("Search snapshot pointer must be an object")
    directory = Path(record["directory"]).resolve()
    home = (root / "data" / "search_snapshots").resolve()
    if directory == home or home not in directory.parents:
        raise ValueError("Search snapshot must be a separate directory inside data/search_snapshots")
    proof = json.loads((directory / "snapshot.json").read_text(encoding="utf-8"))
    if (not isinstance(proof, dict) or proof.get("ready") is not True
            or record.get("generation") != proof.get("generation")):
        raise ValueError("Search snapshot has no matching validation certificate")
    if (not isinstance(proof.get("attachments"), int) or proof["attachments"] < 0
            or not isinstance(proof.get("published_at"), str)
            or not isinstance(proof.get("audit_passed"), bool)):
        raise ValueError("Search snapshot certificate is incomplete")
    manifest = json.loads((directory / "manifest_v3.json").read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("hnsw_validated") is not True or manifest.get("inflight_attachments")
            or manifest.get("post_index_pending")):
        raise ValueError("Search snapshot manifest is not query-ready")
    for name in ("chroma/chroma.sqlite3", "chroma/embedder_config_v3.json", "lexical_v3.sqlite3", "relations.db"):
        path = directory / name
        if not path.is_file() or directory not in path.resolve().parents:
            raise ValueError(f"Search snapshot is missing {name}")
    return {**proof, "directory": str(directory)}


def writer_environment(base: Mapping[str, str]) -> dict[str, str]:
    """Remove reader-only paths before a maintenance child can inherit them."""
    environment = dict(base)
    original = environment.pop(ORIGINAL_ENV, None)
    active = environment.pop(ACTIVE_ROOT, None)
    if original is not None:
        values = json.loads(original)
        if not isinstance(values, dict) or set(values) != set(PATH_KEYS):
            raise ValueError("Invalid canonical environment for search snapshot")
        for key, value in values.items():
            if value is None:
                environment.pop(key, None)
            elif isinstance(value, str):
                environment[key] = value
            else:
                raise ValueError("Canonical environment path must be a string")
    elif active is not None:
        raise ValueError("Search reader cannot launch a writer without canonical paths")
    return environment


def apply_search_snapshot(root: Path) -> dict | None:
    """Called only by query entry points, never by an indexer or auditor."""
    snapshot = load_snapshot(root)
    if snapshot is None:
        if os.environ.get(ACTIVE_ROOT):
            raise ValueError("Pinned search snapshot pointer disappeared")
        return None
    canonical = writer_environment(os.environ)
    os.environ[ORIGINAL_ENV] = json.dumps({key: canonical.get(key) for key in PATH_KEYS})
    directory = Path(snapshot["directory"])
    os.environ.update({
        "CHROMA_DIR": str(directory / "chroma"),
        "MANIFEST_PATH": str(directory / "manifest_v3.json"),
        "LEXICAL_DB_PATH": str(directory / "lexical_v3.sqlite3"),
        "RELATIONS_DB_PATH": str(directory / "relations.db"),
        "PIPELINE_CONFIG_PATH": str(directory / "chroma" / "embedder_config_v3.json"),
        ACTIVE_ROOT: str(directory),
    })
    return snapshot


def read_lock_path(root: Path) -> Path:
    try:
        from .indexing_lock import default_path
    except ImportError:  # pragma: no cover - direct src/ entrypoints
        from indexing_lock import default_path
    directory = os.environ.get(ACTIVE_ROOT)
    if directory:
        configured = Path(os.environ.get("CHROMA_DIR", "")).resolve()
        if configured != (Path(directory) / "chroma").resolve():
            raise ValueError("Search snapshot paths changed after initialization")
        return Path(directory) / default_path(root).name
    return default_path(root)


def public_status(root: Path) -> dict:
    try:
        snapshot = load_snapshot(root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"available": False, "error": str(exc)}
    if snapshot is None:
        return {"available": False}
    return {"available": True, "generation": snapshot["generation"],
            "attachments": snapshot["attachments"], "published_at": snapshot["published_at"],
            "audit_passed": snapshot["audit_passed"]}
