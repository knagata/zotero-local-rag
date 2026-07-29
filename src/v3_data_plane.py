"""Fail-closed resolution of the only supported production data plane."""
from __future__ import annotations

import os
from pathlib import Path


V3_COLLECTION = "zotero_paragraphs_v3"
V3_MANIFEST_NAME = "manifest_v3.json"
V3_LEXICAL_NAME = "lexical_v3.sqlite3"
LEGACY_COLLECTION_PREFIX = "zotero_paragraphs"


def require_v3_enabled() -> None:
    raw = str(os.environ.get("INGEST_STRUCTURED_V3_ENABLE") or "1").strip().casefold()
    if raw not in {"1", "true", "yes", "on"}:
        raise RuntimeError(
            "The legacy data plane is retired. "
            "INGEST_STRUCTURED_V3_ENABLE must be 1 (or unset)."
        )


def collection_name() -> str:
    require_v3_enabled()
    configured = str(os.environ.get("CHROMA_COLLECTION") or V3_COLLECTION).strip()
    if configured != V3_COLLECTION:
        raise RuntimeError(
            f"Unsupported Chroma collection {configured!r}; "
            f"the only production collection is {V3_COLLECTION!r}."
        )
    return configured


def manifest_path(project_root: Path) -> Path:
    require_v3_enabled()
    configured = Path(os.environ.get(
        "MANIFEST_PATH", project_root / "data" / V3_MANIFEST_NAME,
    )).expanduser()
    if configured.name != V3_MANIFEST_NAME:
        raise RuntimeError(
            f"Unsupported manifest {configured}; the legacy manifest is retired "
            f"and the filename must be {V3_MANIFEST_NAME}."
        )
    return configured if configured.is_absolute() else project_root / configured


def lexical_path(project_root: Path) -> Path:
    require_v3_enabled()
    configured = Path(os.environ.get(
        "LEXICAL_DB_PATH", project_root / "data" / V3_LEXICAL_NAME,
    )).expanduser()
    if configured.name != V3_LEXICAL_NAME:
        raise RuntimeError(
            f"Unsupported lexical index {configured}; the legacy FTS is retired "
            f"and the filename must be {V3_LEXICAL_NAME}."
        )
    return configured if configured.is_absolute() else project_root / configured


def enforce_environment(project_root: Path) -> None:
    """Validate inherited values, then publish explicit V3 values to children."""
    require_v3_enabled()
    collection = collection_name()
    manifest = manifest_path(project_root)
    lexical = lexical_path(project_root)
    os.environ["INGEST_STRUCTURED_V3_ENABLE"] = "1"
    os.environ["CHROMA_COLLECTION"] = collection
    os.environ["MANIFEST_PATH"] = str(manifest)
    os.environ["LEXICAL_DB_PATH"] = str(lexical)
    os.environ["HIERARCHICAL_SEARCH_V2_ENABLE"] = "1"


__all__ = [
    "V3_COLLECTION", "V3_LEXICAL_NAME", "V3_MANIFEST_NAME", "collection_name",
    "enforce_environment", "lexical_path", "manifest_path", "require_v3_enabled",
]
