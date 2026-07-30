"""Fail-closed resolution of the only supported production data plane."""
from __future__ import annotations

import os
from pathlib import Path


V3_COLLECTION = "zotero_paragraphs_v3"
V3_MANIFEST_NAME = "manifest_v3.json"
V3_LEXICAL_NAME = "lexical_v3.sqlite3"
V3_PIPELINE_CONFIG_NAME = "embedder_config_v3.json"
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


def resolve_configured_path(project_root: Path, raw: str | Path) -> Path:
    """Expand ``~`` and resolve a relative path against ``project_root``.

    The one place this resolution rule is written. ``manifest_path()``/
    ``lexical_path()``/``chroma_dir()`` read their raw value from
    ``os.environ`` and delegate here; a caller working from a config dict
    that has not been exported into ``os.environ`` yet (Setup.command, which
    never loads its own ``.env`` back into itself) can call this directly
    with its own already-known raw value instead of reimplementing the same
    rule -- two independent copies of it previously drifted out of sync,
    with only one of them applying ``.expanduser()`` (2026-07-30).
    """
    configured = Path(raw).expanduser()
    return configured if configured.is_absolute() else project_root / configured


def manifest_path(project_root: Path) -> Path:
    require_v3_enabled()
    configured = resolve_configured_path(
        project_root, os.environ.get("MANIFEST_PATH", project_root / "data" / V3_MANIFEST_NAME),
    )
    if configured.name != V3_MANIFEST_NAME:
        raise RuntimeError(
            f"Unsupported manifest {configured}; the legacy manifest is retired "
            f"and the filename must be {V3_MANIFEST_NAME}."
        )
    return configured


def lexical_path(project_root: Path) -> Path:
    require_v3_enabled()
    configured = resolve_configured_path(
        project_root, os.environ.get("LEXICAL_DB_PATH", project_root / "data" / V3_LEXICAL_NAME),
    )
    if configured.name != V3_LEXICAL_NAME:
        raise RuntimeError(
            f"Unsupported lexical index {configured}; the legacy FTS is retired "
            f"and the filename must be {V3_LEXICAL_NAME}."
        )
    return configured


def chroma_dir(project_root: Path) -> Path:
    require_v3_enabled()
    return resolve_configured_path(
        project_root, os.environ.get("CHROMA_DIR", project_root / "data" / "chroma"),
    )


def pipeline_config_path(project_root: Path) -> Path:
    """The embedder config is pinned inside the collection's own directory.

    A config that lives outside the Chroma directory it describes can outlive
    or contradict that collection, so its location is an invariant rather than
    a preference. This used to be re-checked by hand in each caller; keeping it
    beside the collection/manifest/lexical guards means a future caller of
    ``enforce_environment`` gets all four for free (2026-07-30).
    """
    require_v3_enabled()
    expected = chroma_dir(project_root) / V3_PIPELINE_CONFIG_NAME
    configured = os.environ.get("PIPELINE_CONFIG_PATH")
    if configured is None:
        return expected
    resolved = resolve_configured_path(project_root, configured)
    if resolved != expected:
        raise RuntimeError(
            f"Unsupported pipeline config {resolved}; it is pinned to "
            f"{expected} inside the V3 Chroma directory."
        )
    return expected


def chroma_collection_populated(chroma_directory: Path) -> bool | None:
    """True/False for the V3 collection's actual contents, ``None`` if unknown.

    ``chroma.sqlite3`` is created the moment anything opens the persistent
    client -- including a Setup run that failed partway through a build, or
    simply configuring the Claude Desktop MCP server -- and a rebuild deletes
    collections without deleting that file. Its mere existence is therefore
    not evidence of data; only the named collection's row count is.

    Takes an already-resolved directory rather than reading ``CHROMA_DIR``
    itself, so a caller working from saved config (not yet exported into its
    own process environment) still gets an answer about the real target.

    Only ``chromadb.errors.NotFoundError`` -- confirmed to be exactly what
    ``get_collection`` raises for a genuinely absent collection -- is treated
    as "not populated". Anything else (a locked sqlite file, corruption, a
    permission error) is a state this function cannot vouch for and must
    report as unknown, not silently narrowed to "empty" (2026-07-30).
    """
    if not chroma_directory.exists():
        return False
    try:
        import chromadb
    except ImportError:
        return None
    client = None
    try:
        client = chromadb.PersistentClient(path=str(chroma_directory))
        try:
            collection = client.get_collection(V3_COLLECTION)
        except chromadb.errors.NotFoundError:
            return False
        return collection.count() > 0
    except Exception:
        return None
    finally:
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                close()


def enforce_environment(project_root: Path) -> None:
    """Validate inherited values, then publish explicit V3 values to children."""
    require_v3_enabled()
    collection = collection_name()
    manifest = manifest_path(project_root)
    lexical = lexical_path(project_root)
    chroma = chroma_dir(project_root)
    pipeline_config = pipeline_config_path(project_root)
    os.environ["INGEST_STRUCTURED_V3_ENABLE"] = "1"
    os.environ["CHROMA_COLLECTION"] = collection
    os.environ["MANIFEST_PATH"] = str(manifest)
    os.environ["LEXICAL_DB_PATH"] = str(lexical)
    os.environ["CHROMA_DIR"] = str(chroma)
    os.environ["PIPELINE_CONFIG_PATH"] = str(pipeline_config)
    os.environ["HIERARCHICAL_SEARCH_V2_ENABLE"] = "1"


__all__ = [
    "V3_COLLECTION", "V3_LEXICAL_NAME", "V3_MANIFEST_NAME",
    "V3_PIPELINE_CONFIG_NAME", "chroma_collection_populated", "chroma_dir",
    "collection_name", "enforce_environment", "lexical_path", "manifest_path",
    "pipeline_config_path", "require_v3_enabled", "resolve_configured_path",
]
