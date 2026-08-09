"""No test may write into the real library. Enforced, not intended.

On 2026-08-09 a new in-process test ran the real indexer against a synthetic
three-item library. It redirected Chroma, the lexical index, the manifest and
the PDF cache into a temporary directory, and missed the fifth store:
``RELATIONS_DB_PATH``. The run read the real ``data/relations.db``, concluded
that every item except the synthetic three had been deleted from Zotero, and
purged 574 items, 205,538 citations and 41,133 references.

Every layer that could have caught it was a promise. The fixture's own
docstring warned that a partial redirection is the dangerous shape. The test
that checked the redirection covered the fields of ``IngestPaths`` -- which do
not include the relations database -- so it passed while being vacuous about
the store that mattered. A checklist of "the paths a test must redirect" fails
the same way: it is a list someone has to remember to extend.

So the check here is not a list of what to redirect. It is a wall around the
real directory: any attempt to open anything under ``data/`` for writing, from
any test, fails immediately and says which file. Reads stay allowed, because
the corpus tests legitimately read the indexed library.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTECTED = (ROOT / "data").resolve()

#: Set by a test that genuinely means to write into the real data directory.
#: Nothing in the suite does today; the escape hatch exists so that the guard
#: can be argued with rather than disabled wholesale.
_OPT_OUT = "ALLOW_WRITES_TO_REAL_DATA_DIR"


class RealDataWriteAttempted(RuntimeError):
    """Raised in place of a write that would have hit the real library."""


def _as_text(path) -> str:
    """Audit arguments arrive as str or bytes depending on the event.

    ``sqlite3.connect`` passes bytes. Formatting those with ``str()`` yields
    ``b'/Users/...'``, which resolves to a relative path that is under nothing,
    so the guard silently allowed every database connection until this was
    found by a test that expected to be blocked and was not.
    """
    if isinstance(path, bytes):
        return path.decode("utf-8", "surrogateescape")
    return str(path)


def _is_protected(path) -> bool:
    text = _as_text(path)
    if not text:
        return False
    try:
        resolved = Path(text).resolve()
    except (OSError, ValueError):
        return False
    return PROTECTED == resolved or PROTECTED in resolved.parents


def _write_mode(mode: str, flags: int) -> bool:
    if any(character in mode for character in "wxa+"):
        return True
    return bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))


def _audit(event: str, arguments: tuple) -> None:
    if os.environ.get(_OPT_OUT) == "1":
        return
    if event == "open":
        path, mode, flags = arguments
        if isinstance(path, int):  # already-open descriptor
            return
        if _write_mode(_as_text(mode or ""), int(flags or 0)) and _is_protected(path):
            raise RealDataWriteAttempted(
                f"a test tried to open the real library for writing: {path}\n"
                "Point the run at a temporary plane (tests/data_plane_fixture.py). "
                "This is the guard that 2026-08-09 was missing."
            )
    elif event == "sqlite3.connect":
        (path,) = arguments
        target = _as_text(path)
        # A read-only URI is how the audits and corpus tests reach the library.
        if "mode=ro" in target:
            return
        if _is_protected(target.split("?", 1)[0].replace("file:", "")):
            raise RealDataWriteAttempted(
                f"a test opened a real database without mode=ro: {target}\n"
                "SQLite opens read-write by default, so this could delete rows. "
                "Use a temporary path, or append '?mode=ro' with uri=True."
            )


sys.addaudithook(_audit)


def _default_relations_database_to_a_scratch_file() -> None:
    """Every test gets its own relations database unless it asked for one.

    Set here, at conftest import, rather than in a fixture: ``db_relations``
    resolves ``DB_PATH`` from the environment when it is imported, and test
    modules are imported before any fixture runs. A fixture that sets the
    variable is therefore too late by exactly one import -- which is the same
    shape of defect as the one that made this file necessary, and it is why
    fifteen tests were still reaching the real directory after the audit hook
    was already in place.

    Making the safe thing the default matters more than the wall: the hook
    turns a mistake into a failure a developer has to understand, and this
    means there is usually no mistake to understand.
    """
    if os.environ.get("RELATIONS_DB_PATH"):
        return
    import tempfile

    scratch = Path(tempfile.mkdtemp(prefix="pytest-relations-")) / "relations.db"
    os.environ["RELATIONS_DB_PATH"] = str(scratch)


_default_relations_database_to_a_scratch_file()
