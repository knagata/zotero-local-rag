"""V3 database lifecycle: classify existing state, rebuild, and audit.

Non-interactive by design. Setup.command (scripts/setup_wizard.py) owns the
interactive confirmation UX (typed REBUILD, [Y/n] prompts) and calls these
primitives; it does not reimplement them. Kept separate from
src/v3_data_plane.py's environment-invariant guards -- that module is
imported at startup by every V3 entry point (index_from_zotero.py,
rag_mcp_server.py, ...), so its import surface stays free of the
subprocess-running, DB-lifecycle concerns that only Setup needs (2026-07-30).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.env_utils import environment_with_saved_dotenv
from src.v3_data_plane import chroma_collection_populated

#: A rebuild destroys the Chroma collections, the manifest, the pipeline
#: config and the lexical index (``_reset_rebuild_target`` in
#: src/index_from_zotero.py). Skipping its confirmation therefore requires
#: *positive proof* that none of that exists yet -- every other answer,
#: including "the manifest is unreadable", has to be treated as "there is
#: something to lose" (2026-07-30).
DB_STATE_EMPTY = "empty"
DB_STATE_POPULATED = "populated"
DB_STATE_UNKNOWN = "unknown"


def existing_database_state(chroma_directory: Path, manifest_file: Path) -> str:
    """Classify what a rebuild would destroy: empty, populated, or unknown.

    Takes already-resolved paths rather than reading ``CHROMA_DIR``/
    ``MANIFEST_PATH`` from the process environment: a caller working from
    saved config that has not been exported into its own process
    environment (Setup.command never loads its own ``.env`` back into
    itself) still gets a correct answer instead of silently checking the
    default location (2026-07-30).

    Only ``DB_STATE_EMPTY`` proves a rebuild is free. A corrupt manifest or a
    Chroma store whose population cannot be determined both mean the
    opposite of "nothing here" -- they mean we cannot see what is here, so
    the confirmation must stand.
    """
    chroma_populated = chroma_collection_populated(chroma_directory)
    if chroma_populated is None:
        return DB_STATE_UNKNOWN
    if chroma_populated:
        # The collection's contents are the thing a rebuild deletes; their
        # presence outranks whatever the manifest does or does not say.
        return DB_STATE_POPULATED
    if not manifest_file.exists():
        return DB_STATE_EMPTY
    try:
        data = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return DB_STATE_UNKNOWN
    if not isinstance(data, dict):
        return DB_STATE_UNKNOWN
    return DB_STATE_POPULATED if data.get("files") else DB_STATE_EMPTY


def run_rebuild(root_dir: Path) -> int:
    """The one implementation of a V3 rebuild.

    Only reached when there is either nothing to lose or the operator just
    typed REBUILD to confirm losing it -- that confirmation lives in the
    caller, not here.
    """
    child_environment = environment_with_saved_dotenv(root_dir)
    for args in (
        ["uv", "run", "src/index_from_zotero.py", "--rebuild", "--progress"],
        ["uv", "run", "python", "scripts/rebuild_document_structure.py", "--all"],
    ):
        result = subprocess.run(args, cwd=root_dir, env=child_environment)
        if result.returncode != 0:
            return result.returncode
    return 0


def run_audit(root_dir: Path) -> int:
    result = subprocess.run(
        ["uv", "run", "python", "scripts/run_db_audit.py"], cwd=root_dir,
        env=environment_with_saved_dotenv(root_dir),
    )
    return result.returncode


__all__ = [
    "DB_STATE_EMPTY", "DB_STATE_POPULATED", "DB_STATE_UNKNOWN",
    "existing_database_state", "run_audit", "run_rebuild",
]
