"""The number of swallowed exceptions may go down. It may not go up.

Almost every defect fixed in this project on 2026-08-09 had the same shape: a
broad ``except`` that turned a failure into a value the caller could not tell
apart from a normal answer. The compensation that abandoned three stores and
the failure record because one store raised. The PDF classification whose
failure read as "not a scan", so a scanned book was never offered for re-OCR
again. The Chroma error reported to a reader as "that chunk id does not exist".
The citation contexts a broken database returned as "nothing was quoted".

There are 205 more blind excepts. Fixing them is a reading job -- each one has
to be looked at, because some are right and say so. What does not need reading
is the direction: a new one is a new instance of the defect class this project
keeps paying for, and it can be refused today without touching the existing
ones.

So the counts are frozen, in the same shape as the function-size ratchet: they
may fall, and when they fall they must be re-recorded, or the budget goes on
permitting a number nobody is under any more.

    uv run python scripts/build_lint_budget.py            # diff
    uv run python scripts/build_lint_budget.py --write    # adopt
"""
from __future__ import annotations

from scripts.build_lint_budget import RULES, TREES, load_budget, measure

_ADOPT = "uv run python scripts/build_lint_budget.py --write"


def test_no_rule_has_more_findings_than_recorded():
    recorded, measured = load_budget(), measure()
    grew = {
        rule: (recorded[rule], measured[rule]) for rule in measured
        if rule in recorded and measured[rule] > recorded[rule]
    }
    assert not grew, (
        "a new instance of a defect shape this project has already shipped:\n"
        + "\n".join(
            f"  {rule}: {was} -> {now} (+{now - was})"
            for rule, (was, now) in sorted(grew.items())
        )
        + "\nFix it rather than recording it. If the except is genuinely right, "
        "say why in a comment and add a targeted `# noqa` with that reason."
    )


def test_a_rule_that_improved_is_recorded_at_its_new_count():
    # Otherwise the first cleanup is the last one that counts: the budget goes
    # on permitting the old number and the next change can spend it again.
    recorded, measured = load_budget(), measure()
    shrank = {
        rule: (recorded[rule], measured[rule]) for rule in measured
        if rule in recorded and measured[rule] < recorded[rule]
    }
    assert not shrank, (
        f"fewer findings than recorded -- adopt them with `{_ADOPT}`:\n"
        + "\n".join(
            f"  {rule}: {was} -> {now} ({now - was})"
            for rule, (was, now) in sorted(shrank.items())
        )
    )


def test_the_budget_covers_every_rule_and_tree_it_claims():
    # A rule quietly dropped from RULES takes its ceiling with it.
    recorded = load_budget()
    assert recorded, "the budget is empty; generate it with the builder"
    assert set(recorded) == set(RULES)
    assert "src" in TREES and "tests" in TREES, (
        "tests are checked too: a test that swallows an exception is a test "
        "that passes for the wrong reason"
    )
