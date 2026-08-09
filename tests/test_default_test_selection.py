"""What the default ``pytest`` run leaves out, and whether it still can.

The suite deselects the ``slow`` mark by default, because 140 of its 205 seconds
were five tests that CI does not run at all: they need the library, Zotero, the
embedding model and a locally generated baseline. Waiting three minutes for a
check nobody else enforces is what was removed; the check itself stays, one
``-m slow`` away.

That arrangement is held together by a string in ``pyproject.toml`` and a
decorator, and both fail silently. A mark spelled ``@pytest.mark.slw`` is not
deselected -- it is simply not the mark being excluded -- so the test rejoins the
default run and the only symptom is that the run is slow again. A renamed mark
leaves the documented opt-in command pointing at nothing, and a run with no
marked tests at all makes the exclusion a no-op that still looks configured.

So the properties are asserted here rather than trusted: every mark a test
carries is one the configuration declares, the declared marks are the ones the
default selection excludes, something actually carries each of them, and the
documents name the command that gets them back. The last assertion is the one
that runs pytest for real, in a scratch directory with the project's own
``addopts``, because whether a command-line ``-m`` overrides ``addopts`` is a
fact about pytest and not about anything written here.

The configuration is read out of ``pyproject.toml`` rather than asked of pytest:
``getini("markers")`` answers with pytest's own marks and every plugin's as
well, and what is being checked here is what this project declared.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # 3.10, where pytest itself brings tomli
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]

#: Marks pytest defines itself; a test may carry these without declaring them.
_BUILTIN_MARKS = frozenset({
    "parametrize", "skip", "skipif", "xfail", "usefixtures", "filterwarnings",
})

#: ``@pytest.mark.name`` and ``pytestmark = pytest.mark.name`` alike.
_MARK = re.compile(r"pytest\.mark\.([A-Za-z_][A-Za-z0-9_]*)")

#: Documents that tell a reader how to run the tests.
_DOCUMENTS = ("CLAUDE.md", "docs/development.md")


def _configuration() -> dict:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return payload["tool"]["pytest"]["ini_options"]


def _declared_marks() -> set[str]:
    return {entry.split(":", 1)[0].strip() for entry in _configuration()["markers"]}


def _marks_in_use() -> dict[str, list[str]]:
    """Which marks the test files carry, by file.

    This file is left out of the scan: it names marks in prose and writes a
    probe test carrying one, and neither is a mark this suite runs under.
    """
    used: dict[str, list[str]] = {}
    for path in sorted(ROOT.glob("tests/test_*.py")):
        if path.name == Path(__file__).name:
            continue
        for name in _MARK.findall(path.read_text(encoding="utf-8")):
            if name not in _BUILTIN_MARKS:
                used.setdefault(name, []).append(path.name)
    return used


def test_every_mark_a_test_carries_is_declared():
    undeclared = {
        name: files for name, files in _marks_in_use().items()
        if name not in _declared_marks()
    }
    assert not undeclared, (
        "a test carries a mark that pyproject.toml does not declare. An "
        "undeclared mark is not the one being deselected, so the test runs by "
        "default and nothing says so:\n"
        + "\n".join(f"  {name}: {', '.join(files)}" for name, files in sorted(undeclared.items()))
    )


def test_every_mark_the_default_run_excludes_is_declared():
    """Excluding an undeclared mark deselects nothing and says nothing.

    Not the converse: a declared mark need not be excluded. ``corpus`` marks
    the tests that need the indexed library, which *should* run on a machine
    that has one -- what it is for is measuring the coverage budget under the
    same selection CI gets, so the floor is one both environments can reach.
    """
    declared = _declared_marks()
    expression = _configuration()["addopts"]
    excluded = set(re.findall(r"\bnot\s+([a-z_]+)", expression))
    assert excluded, f"addopts excludes nothing: {expression!r}"
    assert excluded <= declared, (
        f"addopts excludes {sorted(excluded - declared)}, which pyproject.toml "
        "does not declare, so it deselects nothing"
    )


def test_something_actually_carries_each_excluded_mark():
    # An exclusion with nothing behind it deselects zero tests while reading
    # exactly like one that works, and the next slow test is written without it.
    unused = _declared_marks() - set(_marks_in_use())
    assert not unused, (
        f"no test carries {sorted(unused)}, so excluding it does nothing. "
        "Either mark the tests it was meant for or drop the declaration."
    )


def test_the_documents_name_the_command_that_runs_the_excluded_tests():
    # A test that is excluded by default and documented nowhere is a test that
    # stops being run. CLAUDE.md is read at the start of every session.
    declared = sorted(_declared_marks())
    missing = [
        f"  {document}: " + " or ".join(f"uv run pytest -m {name}" for name in declared)
        for document in _DOCUMENTS
        if not any(
            f"pytest -m {name}" in (ROOT / document).read_text(encoding="utf-8")
            for name in declared
        )
    ]
    assert not missing, (
        "a document describes how to run the tests without naming how to run "
        "the excluded ones:\n" + "\n".join(missing)
    )


def test_an_explicit_marker_expression_beats_the_default_exclusion(tmp_path: Path):
    """The opt-in works, asked of pytest rather than assumed.

    Run against two throwaway tests in a scratch directory carrying the
    project's own ``addopts``, so this stays a claim about the configuration
    and costs no library, no model and no seconds.
    """
    configuration = _configuration()
    markers = "\n".join(f"    {json.dumps(entry)}," for entry in configuration["markers"])
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\n"
        f"addopts = {json.dumps(configuration['addopts'])}\n"
        f"markers = [\n{markers}\n]\n",
        encoding="utf-8",
    )
    marked = "slow"
    (tmp_path / "test_selection_probe.py").write_text(
        "import pytest\n\n\n"
        "def test_ordinary():\n    pass\n\n\n"
        f"@pytest.mark.{marked}\n"
        "def test_marked():\n    pass\n",
        encoding="utf-8",
    )

    def collected(*extra: str) -> set[str]:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", *extra],
            cwd=tmp_path, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return {
            line.split("::", 1)[1].strip()
            for line in result.stdout.splitlines() if "::" in line
        }

    assert collected() == {"test_ordinary"}
    assert collected("-m", marked) == {"test_marked"}
    assert collected("-m", "") == {"test_ordinary", "test_marked"}
