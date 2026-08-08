"""A configured path is resolved in one place, not in each module that needs it.

``CHROMA_DIR`` names the database every entry point reads and writes. Reading it
straight out of the environment looks harmless and is not: ``~/chroma`` becomes
a directory literally named ``~``, and a relative value resolves against
whatever directory the process happened to start in, so one configuration names
different databases depending on how it was invoked.

``v3_data_plane.resolve_configured_path`` is where that rule is written, and its
own docstring records the rule drifting apart once already, in 2026-07-30, when
only one of two copies expanded ``~``. ``chunk_store`` carries a second note
from 2026-08-04, when the expansion had been added there but the relative case
had not. Both were repaired by hand, module by module, and seven modules were
still reading the variable directly when this test was written -- the same fault
regrowing because nothing stopped it.

So it is a rule now rather than a habit. A module that needs the configured
location asks for it; only the place that defines the resolution reads the raw
value.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Variables whose raw value must pass through the resolver before use.
CONFIGURED_PATH_VARIABLES = frozenset({
    "CHROMA_DIR", "MANIFEST_PATH", "LEXICAL_DB_PATH",
})

#: Where the resolution rule lives, and so the only module allowed to read the
#: raw value. ``enforce_environment`` also writes resolved values back into the
#: environment for child processes, which is the same module's business.
RESOLVER_MODULE = "src/v3_data_plane.py"

SEARCHED = ("src", "scripts", "citation_graph")


def _environment_reads(tree: ast.AST) -> set[str]:
    """Names read out of ``os.environ`` anywhere in this module."""
    found: set[str] = set()
    for node in ast.walk(tree):
        name = None
        # os.environ.get("X") / os.environ.get("X", default)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"get", "setdefault"}
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "environ" and node.args
                and isinstance(node.args[0], ast.Constant)):
            name = node.args[0].value
        # os.environ["X"]
        elif (isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "environ"
                and isinstance(node.slice, ast.Constant)):
            name = node.slice.value
        if isinstance(name, str):
            found.add(name)
    return found


def test_only_the_resolver_reads_a_configured_path_from_the_environment():
    offenders: list[str] = []
    for directory in SEARCHED:
        for path in sorted((ROOT / directory).rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            if relative == RESOLVER_MODULE or "attic" in relative:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:  # pragma: no cover -- compileall covers this
                continue
            for name in sorted(_environment_reads(tree) & CONFIGURED_PATH_VARIABLES):
                offenders.append(f"  {relative}: reads {name} directly")
    assert not offenders, (
        "a configured path is being read straight from the environment. Pass the "
        "raw value through v3_data_plane.resolve_configured_path, which expands "
        "~ and resolves a relative value against the project root -- both halves "
        "of that rule have been lost once already by a copy that looked "
        "harmless:\n" + "\n".join(offenders)
    )


def test_the_resolver_handles_both_halves_of_the_rule():
    # The two failures this guards against, stated as the behaviour rather than
    # as an absence of copies.
    from src.v3_data_plane import resolve_configured_path

    assert resolve_configured_path(ROOT, "~/chroma") == Path.home() / "chroma"
    assert resolve_configured_path(ROOT, "data/chroma") == ROOT / "data" / "chroma"
    assert resolve_configured_path(ROOT, "/tmp/chroma") == Path("/tmp/chroma")
