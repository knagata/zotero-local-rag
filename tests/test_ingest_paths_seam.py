"""Importing the indexer must not decide where the indexer writes.

``index_from_zotero`` used to compute its manifest, Chroma directory, cache and
lock path into module constants at import. Everything else in the data plane
(`v3_data_plane`, `chunk_store`) resolves per call and follows a change to the
environment; this module was the one that could not, and the cost was not
abstract: the ingestion net has to start a **child process** for each attachment
purely to get a different ``CHROMA_DIR`` in front of the import, which is why it
takes 140 seconds, why it cannot run in CI, and why it can only compare printed
output instead of calling anything.

So the paths are resolved on first use and swappable through ``use_paths``. The
tests here hold the three properties that arrangement rests on: nothing resolves
at import, a value set after import is the value used, and a swap is undone even
when the body raises. Without the last one a failing test leaves the next test
writing into a deleted temporary directory -- or, worse, does not.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "index_from_zotero.py"

#: Functions that answer "where does this run read and write".
_RESOLVERS = {
    "chroma_dir", "v3_manifest_path", "v3_collection_name", "lexical_path",
    "pipeline_config_path",
}

#: Environment variables that name a location rather than tune a behaviour.
#:
#: The distinction is the point. ``FLUSH_SIZE`` and ``UPSERT_BATCH_SIZE`` are
#: still read at import, and that is the same defect in a smaller size -- a
#: value set after import is ignored -- but getting one of them wrong makes a
#: run slower, while getting ``CHROMA_DIR`` wrong makes it write into the real
#: library. Only the second kind is checked here; widening this to every knob
#: is a separate change with its own verification.
_LOCATION_ENV = re.compile(r"(DIR|PATH|COLLECTION)$")


def _module_level_resolution() -> list[str]:
    """Location lookups in statements that run on import, not in a function."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            name = getattr(sub.func, "id", None) or getattr(sub.func, "attr", None)
            reads_environ = (
                isinstance(sub.func, ast.Attribute)
                and getattr(sub.func.value, "attr", None) == "environ"
            ) or name == "getenv"
            named = next(
                (a.value for a in sub.args if isinstance(a, ast.Constant)
                 and isinstance(a.value, str)),
                "",
            )
            if name in _RESOLVERS or (reads_environ and _LOCATION_ENV.search(named)):
                found.append(f"line {sub.lineno}: {ast.unparse(sub)[:70]}")
    return found


def test_nothing_resolves_a_location_at_import_time():
    resolved = _module_level_resolution()
    assert not resolved, (
        "index_from_zotero resolves a path or collection while being imported. "
        "That makes the environment at import the only environment the run can "
        "have, which is what forced the ingestion net into a child process:\n  "
        + "\n  ".join(resolved)
    )


def test_a_location_set_after_import_is_the_one_used():
    # The property the static check above exists to protect, asked of a real
    # interpreter: import first, configure second, and the run must follow the
    # configuration rather than the import.
    probe = (
        "import sys; sys.path.insert(0, 'src');"
        "import index_from_zotero as m;"
        "import os, tempfile;"
        "d = tempfile.mkdtemp();"
        "os.environ['CHROMA_DIR'] = d + '/chroma';"
        "os.environ['MANIFEST_PATH'] = d + '/manifest_v3.json';"
        "print(m.paths().chroma_dir == __import__('pathlib').Path(d + '/chroma'),"
        " m.paths().manifest_path.parent == __import__('pathlib').Path(d))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "True True", result.stdout


def test_a_swap_is_undone_when_the_body_raises():
    module = _indexer()
    before = module.paths()
    with pytest.raises(RuntimeError):
        with module.use_paths(replace(before, manifest_path=Path("/nowhere/manifest_v3.json"))):
            assert module.paths().manifest_path == Path("/nowhere/manifest_v3.json")
            raise RuntimeError("the body failed")
    assert module.paths() is before


def test_a_swap_covers_every_location_the_run_writes_to(tmp_path: Path):
    # A partial redirection is the dangerous shape: the run would write most of
    # itself into the temporary plane and one store into the real library.
    module = _indexer()
    from tests.data_plane_fixture import temporary_data_plane

    with temporary_data_plane(tmp_path) as plane:
        for field, value in vars(plane).items():
            if field in {"project_root", "collection_name", "zotero_data_dir"}:
                continue
            assert tmp_path in Path(value).parents or Path(value) == tmp_path, (
                f"{field} points outside the temporary plane: {value}"
            )
        assert module.paths() == plane


def _indexer():
    sys.path.insert(0, str(ROOT / "src"))
    import index_from_zotero

    return index_from_zotero
