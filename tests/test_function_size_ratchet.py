"""The long functions may get shorter. They may not get longer.

Splitting `main_async` has been an intention for weeks, and an intention is a
thing that competes with every other intention. What it lacked was a mechanism:
nothing objected when a function grew, so the register of oversized functions
was accurate only on the day someone counted. Twenty-three functions are over
150 lines and the largest is 1,113.

So the sizes are recorded in `tests/function_size_budget.json` and checked here.
A recorded function that grows fails, a new function born over the limit fails,
and a recorded function that shrinks fails too -- with a different message,
because the fix is to adopt the smaller number and let the ratchet hold there.
That last one is what makes the total fall rather than drift: the budget cannot
quietly keep describing a function that was already split.

What this does not do is judge the code. A 200-line function that reads as one
list of cases is not the same problem as a 200-line branch chain, and nothing
here can tell them apart. It bounds the growth and leaves the reading to the
person doing the splitting.

Adopt a change with::

    uv run python scripts/build_function_size_budget.py            # diff
    uv run python scripts/build_function_size_budget.py --write    # adopt
"""
from __future__ import annotations

from scripts.build_function_size_budget import (
    BUDGET_PATH, LIMIT, TREES, load_budget, measure,
)

_ADOPT = "uv run python scripts/build_function_size_budget.py --write"


def test_no_recorded_function_grew():
    recorded, measured = load_budget(), measure()
    grew = {
        name: (recorded[name], size) for name, size in measured.items()
        if name in recorded and size > recorded[name]
    }
    assert not grew, (
        "a function that was already too long got longer. The budget is a "
        "ceiling that comes down, so this has to be paid back rather than "
        "recorded:\n" + "\n".join(
            f"  {name}: {was} -> {now} lines (+{now - was})"
            for name, (was, now) in sorted(grew.items(), key=lambda kv: kv[0])
        )
    )


def test_no_new_function_is_born_over_the_limit():
    recorded, measured = load_budget(), measure()
    born = {name: size for name, size in measured.items() if name not in recorded}
    assert not born, (
        f"a function over {LIMIT} lines is not in the budget. If it is genuinely "
        "new, it is the thing this check exists to prevent; if it was renamed or "
        f"moved, adopt it with `{_ADOPT}` and it keeps its ceiling:\n"
        + "\n".join(f"  {name}: {size} lines" for name, size in sorted(born.items()))
    )


def test_a_function_that_shrank_is_recorded_at_its_new_size():
    # Otherwise the ratchet stops at the first split: the budget goes on
    # allowing the old size, and the next edit may spend it again.
    recorded, measured = load_budget(), measure()
    shrank = {
        name: (recorded[name], size) for name, size in measured.items()
        if name in recorded and size < recorded[name]
    }
    assert not shrank, (
        f"a function got shorter than its recorded size. Adopt it with `{_ADOPT}` "
        "so the ceiling comes down with it:\n" + "\n".join(
            f"  {name}: {was} -> {now} lines ({now - was})"
            for name, (was, now) in sorted(shrank.items(), key=lambda kv: kv[0])
        )
    )


def test_the_budget_names_no_function_that_is_gone():
    recorded, measured = load_budget(), measure()
    absent = {
        name: size for name, size in recorded.items()
        if name not in measured
    }
    assert not absent, (
        "the budget records a function that is no longer over the limit or no "
        f"longer exists. Adopt it with `{_ADOPT}`; a stale entry is a ceiling "
        "nothing is under:\n"
        + "\n".join(f"  {name}: was {size} lines" for name, size in sorted(absent.items()))
    )


def test_the_budget_covers_the_trees_it_claims_to():
    # A tree dropped from TREES takes its ceilings with it and nothing says so.
    recorded = load_budget()
    assert recorded, f"{BUDGET_PATH.name} is empty; generate it with `{_ADOPT}`"
    covered = {name.split("/", 1)[0] for name in recorded}
    assert covered <= set(TREES)
    assert "src" in covered, (
        "the budget records nothing under src/, which is where the largest "
        f"functions are. TREES is {TREES}."
    )


def test_every_recorded_size_is_over_the_limit():
    # A recorded function at or under the limit would be handed a ceiling it
    # does not need, and could then grow up to it unopposed.
    recorded = load_budget()
    too_small = {name: size for name, size in recorded.items() if size <= LIMIT}
    assert not too_small, (
        f"the budget records a function of {LIMIT} lines or fewer, which the "
        "limit already covers:\n"
        + "\n".join(f"  {name}: {size} lines" for name, size in sorted(too_small.items()))
    )
