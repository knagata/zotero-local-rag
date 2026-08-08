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

That file is generated locally and is not in the repository: it names 84 of the
library's books, with their Zotero keys and headings taken from their text, and
this repository is public.

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
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


#: The baseline is not in the repository. It lists 84 of the library's books by
#: title, with their Zotero keys and the headings recovered from their text --
#: an inventory of what the user owns, and this repository is public. It is
#: ignored for the same reason ``evaluations/`` is, and rebuilt locally with
#: ``scripts/build_structure_recovery_baseline.py --write``.
_needs_baseline = pytest.mark.skipif(
    not BASELINE_PATH.exists(),
    reason=f"{BASELINE_PATH.name} is generated locally; run the builder with --write",
)


def _render(paths: list[list] | None) -> str:
    if paths is None:
        return "      (flat -- recovery declined)"
    return "\n".join(f"      {chunks:5}  {' / '.join(path)}" for path, chunks in paths)


@_needs_baseline
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


@_needs_baseline
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
        boundaries = entry["paths"]
        if not boundaries:
            continue
        seen = [tuple(path) for path, _chunks in boundaries]
        repeated = {path for path in seen if seen.count(path) > 1}
        if repeated:
            offenders.append(f"  {entry['item_key']}: {sorted(repeated)}")
    if offenders:
        pytest.fail("\n".join(["a boundary is claimed twice:"] + offenders),
                    pytrace=False)


#: A boundary holding this share of the document, while another holds almost
#: nothing, is the signature of divisions that were read off a contents page
#: instead of the body.
_DOMINANT_SHARE = 0.90
_NEGLIGIBLE_CHUNKS = 5


@_needs_baseline
def test_no_recovered_tree_hangs_the_whole_book_off_one_boundary():
    """A book's divisions divide it.

    Japan-ness in Architecture recovers four parts, and 99% of its 1,730 chunks
    land under the fourth while the other three hold two or three chunks each --
    the chunks of their own lines on the contents page, which is where all four
    were read from. The list of paths looks like a four-part book; only the
    weights show that it is one part wearing the name of the fourth.

    This is the check that would have refused it without anyone reading the
    tree, and the reason the baseline records how much of the document sits
    under each boundary rather than the shape alone.
    """
    offenders = []
    for entry in _baseline()["items"]:
        boundaries = entry["paths"]
        if not boundaries:
            continue
        # Weighed by top-level division, not by boundary. Two things otherwise
        # split a division's share and hide it: a path that recurs after a gap is
        # recorded twice, and a chunk under a subsection is recorded against the
        # subsection though it is just as much inside the chapter above it.
        # Japan-ness in Architecture came to 62% rather than 99% and slipped
        # through -- its dominant part was split between itself and its one
        # child.
        weights: dict[str, int] = {}
        for path, chunks in boundaries:
            weights[path[0]] = weights.get(path[0], 0) + chunks
        total = sum(weights.values())
        if not total or len(weights) < 2:
            continue
        largest = max(weights.values())
        starved = [
            division for division, chunks in weights.items()
            if chunks <= _NEGLIGIBLE_CHUNKS
        ]
        if largest / total >= _DOMINANT_SHARE and starved:
            offenders.append(
                f"  {entry['item_key']}  {entry['title'][:44]}\n"
                f"    {largest}/{total} chunks ({largest / total:.0%}) under one "
                f"boundary, while {len(starved)} hold "
                f"{_NEGLIGIBLE_CHUNKS} or fewer:\n"
                + "\n".join(f"      {division}" for division in starved[:6])
            )
    if offenders:
        pytest.fail(
            "a recovered tree does not divide its document; these divisions "
            "were most likely read off a printed contents page:\n"
            + "\n".join(offenders),
            pytrace=False,
        )
