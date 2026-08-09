"""Record how many of each defect-shaped lint finding exist, so it can only fall.

The rules here are not style. Each one names a shape this project has actually
shipped a defect in:

* ``BLE001`` blind except -- the rollback that abandoned four stores because one
  raised, the classification failure that read as "not a scan", the Chroma error
  that came back as "chunk not found";
* ``S110`` / ``S112`` try-except-pass / -continue -- the same, with the evidence
  discarded rather than merely widened;
* ``PLW1510`` subprocess without ``check`` -- a child that failed reads as a
  child that succeeded;
* ``TRY004`` a type error raised as something else, which callers then catch
  for the wrong reason;
* ``F401`` unused import, the only one here that is merely tidy, kept because it
  is free and auto-fixable.

Turning all 308 into failures today would mean 308 suppressions, and a
suppression is a claim nobody checks. Freezing the counts means the existing
ones stay visible and a new one fails, which is the part that changes what
happens next.

Usage::

    uv run python scripts/build_lint_budget.py            # diff
    uv run python scripts/build_lint_budget.py --write    # adopt
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGET_PATH = ROOT / "tests" / "lint_budget.json"

#: Rules counted here. Deliberately not the whole default set: the point is a
#: budget somebody reads, not a wall of formatting.
RULES = ("BLE001", "S110", "S112", "PLW1510", "TRY004", "F401")

#: Trees checked. Tests are included: a test that swallows an exception is a
#: test that passes for the wrong reason.
TREES = ("src", "scripts", "citation_graph", "tests")

_STAT = re.compile(r"^\s*(\d+)\s+([A-Z]+\d+)")


def measure(root: Path = ROOT) -> dict[str, int]:
    """Current count per rule, as ruff reports it."""
    result = subprocess.run(
        ["uv", "run", "ruff", "check", "--statistics",
         "--select", ",".join(RULES), *TREES],
        cwd=root, capture_output=True, text=True, timeout=300,
    )
    counts = {rule: 0 for rule in RULES}
    for line in result.stdout.splitlines():
        matched = _STAT.match(line)
        if matched and matched.group(2) in counts:
            counts[matched.group(2)] = int(matched.group(1))
    return counts


def load_budget(path: Path = BUDGET_PATH) -> dict[str, int]:
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8"))["counts"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="adopt the measured counts")
    args = parser.parse_args(argv)

    measured, recorded = measure(), load_budget()
    for rule in RULES:
        now, was = measured.get(rule, 0), recorded.get(rule)
        if was is None:
            print(f"new     {rule:8s} {now:5d}")
        elif now > was:
            print(f"grew    {rule:8s} {was:5d} -> {now:5d}  (+{now - was})")
        elif now < was:
            print(f"shrank  {rule:8s} {was:5d} -> {now:5d}  ({now - was})")
    print(f"\n{sum(measured.values())} findings across {len(RULES)} rules.")

    if args.write:
        BUDGET_PATH.write_text(
            json.dumps({"rules": list(RULES), "trees": list(TREES),
                        "counts": measured}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {BUDGET_PATH.relative_to(ROOT)}")
    elif measured != recorded:
        print("run with --write to adopt.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
