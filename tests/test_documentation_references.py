"""Every file a document names has to exist.

Documentation rots quietly. `.claude/STATE.md` spent two months telling readers
that the citation graph lived in ``scripts/show_citation_graph.py`` after it had
moved, and `docs/architecture.md` listed ``citation_insights.py`` under ``src/``
for as long as it had been ``citation_graph/insights.py`` -- found by the first
run of this test, not by anyone reading the file. Nothing contradicts a stale
sentence, so it survives every review that does not happen to look at it.

A file path is the part of a document that can be checked mechanically, so it is
checked here. This does not make a document true: it cannot tell that a
described behaviour has changed, and it deliberately does not require the module
tables to be exhaustive (``docs/architecture.md`` §7 lists 23 of 81 modules on
purpose). It catches the specific failure of naming something that is not there,
which is the one that has actually happened twice.

``TASKS.md`` is excluded: it is the durable record of what was done, so naming a
file that was since removed is correct there rather than wrong.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Documents that describe the code as it is now.
DOCUMENTS = ("CLAUDE.md", "SPEC.md", "README.md", *sorted(
    str(path.relative_to(ROOT)) for path in (ROOT / "docs").glob("*.md")
))

#: A path inside the repository, in backticks: `src/embedder.py`, `docs/x.md`.
_QUALIFIED = re.compile(
    r"`([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+\."
    r"(?:py|md|json|jsonl|toml|yml|yaml|js|css|sqlite3))`"
)
#: A bare module name in backticks, as the module tables write them.
_BARE_MODULE = re.compile(r"`([A-Za-z0-9_-]+\.py)`")

#: Trees whose contents are generated or deliberately untracked, so a reference
#: into them describes a runtime location rather than a file in the repository.
_GENERATED = ("data/", "dev-notes/", "evaluations/", "tests/baselines/")

#: Names that are not repository files despite matching the pattern.
_NOT_REPOSITORY_FILES = frozenset({
    "pyproject.toml",   # named as a file, lives at the root, matched as bare
})


def _referenced_paths(text: str) -> set[str]:
    return {
        reference for reference in _QUALIFIED.findall(text)
        if not reference.startswith(_GENERATED)
    }


def _referenced_modules(text: str) -> set[str]:
    return set(_BARE_MODULE.findall(text)) - _NOT_REPOSITORY_FILES


def _module_exists(name: str) -> bool:
    """A bare module name resolves anywhere a module of the project lives."""
    return any(
        (ROOT / package / name).exists()
        for package in ("src", "scripts", "tests", "citation_graph", ".")
    )


def test_documents_do_not_name_files_that_are_gone():
    missing: list[str] = []
    for document in DOCUMENTS:
        text = (ROOT / document).read_text(encoding="utf-8")
        for reference in sorted(_referenced_paths(text)):
            if not (ROOT / reference).exists():
                missing.append(f"  {document}: {reference}")
        for module in sorted(_referenced_modules(text)):
            if not _module_exists(module):
                missing.append(f"  {document}: {module}")
    assert not missing, (
        "a document names a file that does not exist. Either the file moved and "
        "the document was not updated with it, or the document is describing "
        "something that no longer happens:\n" + "\n".join(missing)
    )


def test_the_check_covers_the_file_read_at_the_start_of_every_session():
    # CLAUDE.md is loaded into every session, so a stale path in it is repeated
    # to every reader before anyone opens the file. It must not fall out of the
    # document list by accident.
    assert "CLAUDE.md" in DOCUMENTS
    assert _referenced_paths((ROOT / "CLAUDE.md").read_text(encoding="utf-8"))
