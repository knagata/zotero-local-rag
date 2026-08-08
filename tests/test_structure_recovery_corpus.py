"""The flat-PDF recovery rules, checked against the books they were written for.

The unit tests beside this one describe each rule with rows built by hand. They
are necessary and they are not sufficient: every rule here is a heuristic tuned
against a corpus, so the question that decides whether a change is an
improvement is what it does to real books, and a hand-built fixture cannot
answer it. Both times that question was answered by counting how many documents
came out non-flat, the count went the right way while the trees got worse -- a
contents region that swallowed 171 pages scored better than the code before it,
because a wrong tree counts the same as a right one.

So the trees themselves are recorded, in
``tests/baselines/structure_recovery.json``, and re-derived here.
A rule change now has to show what it did to 84 books before it can pass.

Regenerate with::

    uv run python scripts/build_structure_recovery_baseline.py            # diff
    uv run python scripts/build_structure_recovery_baseline.py --write    # adopt

This one test needs the indexed library, so it skips where the library is not
present -- CI, a fresh clone. That is a real limit and the reason the hand-built
tests stay: they are what CI enforces. Freezing the corpus into a fixture would
lift the limit at a cost of 2.5 MB, five times the largest file in the
repository, and shrinking it means dropping the ordinary body-text rows, which
is exactly where the last round of faults lived.
"""
from __future__ import annotations

import json

import pytest

from scripts.build_structure_recovery_baseline import (
    BASELINE_PATH, _attachment_rows, recovered_paths,
)


def _library_is_available() -> bool:
    from src.chunk_store import list_item_keys

    return bool(list_item_keys())


def _baseline() -> dict:
    if not BASELINE_PATH.exists():  # pragma: no cover -- committed alongside this test
        pytest.fail(f"{BASELINE_PATH} is missing; run the builder with --write")
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _render(paths: list[list[str]] | None) -> str:
    if paths is None:
        return "      (flat -- recovery declined)"
    return "\n".join(f"      {' / '.join(path)}" for path in paths)


@pytest.mark.skipif(
    not _library_is_available(),
    reason="needs the indexed Zotero library; the hand-built rule tests cover CI",
)
def test_the_recovered_trees_are_the_ones_that_were_reviewed():
    baseline = _baseline()
    drifted: list[str] = []
    for entry in baseline["items"]:
        rows = _attachment_rows(entry["item_key"], entry["attachment_key"])
        if not rows:
            # The book is no longer indexed. Not a rule change, and not this
            # test's business to fail over -- say so and move on.
            drifted.append(
                f"  ? {entry['item_key']} has no chunks; regenerate the baseline"
            )
            continue
        if len(rows) != entry["chunks"]:
            # Re-ingestion changed the input, so a different tree is expected and
            # says nothing about the rules. Distinguished from a rule change so
            # the two are never confused for one another.
            drifted.append(
                f"  ? {entry['item_key']} now has {len(rows)} chunks, "
                f"not {entry['chunks']}; regenerate the baseline"
            )
            continue
        found = recovered_paths(rows, entry["attachment_key"])
        if found != entry["paths"]:
            drifted.append(
                f"  ~ {entry['item_key']}  {entry['title'][:52]}\n"
                f"    recorded:\n{_render(entry['paths'])}\n"
                f"    now:\n{_render(found)}"
            )
    if drifted:
        # pytest.fail rather than assert: the whole point of this test is the
        # diff, and assertion rewriting truncates a message this long to its
        # first line -- which is the bare count, the very number that made two
        # regressions look like improvements.
        pytest.fail(
            f"{len(drifted)} of {len(baseline['items'])} books recover a "
            "different tree than the one recorded. Read each one before "
            "adopting it: a change that raises the recovery count can still be "
            "a book losing its chapters to its own notes index.\n\n"
            + "\n".join(drifted),
            pytrace=False,
        )


def test_no_recovered_tree_repeats_a_boundary():
    """A book opens each of its divisions once.

    Reads the recorded baseline rather than the library, so it runs everywhere
    and guards the adoption step: a tree with this fault must not become the
    thing every later change is compared against.

    A repeat means the same heading was read as a boundary twice -- a printed
    contents entry taken for the opener it lists, a running header taken for the
    chapter it names, or back-of-book notes taken for the chapters they serve.
    All three have happened, and each of them survived a review that only looked
    at how many books came out non-flat.
    """
    offenders = []
    for entry in _baseline()["items"]:
        paths = entry["paths"]
        if not paths:
            continue
        seen = [tuple(path) for path in paths]
        repeated = {path for path in seen if seen.count(path) > 1}
        if repeated:
            offenders.append(f"  {entry['item_key']}: {sorted(repeated)}")
    if offenders:
        pytest.fail("\n".join(["a boundary is claimed twice:"] + offenders),
                    pytrace=False)
