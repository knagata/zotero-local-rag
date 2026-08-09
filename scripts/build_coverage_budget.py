"""Record how many statements no test reaches, as a ceiling that comes down.

Counted as *unreached statements*, not as a percentage. A percentage falls when
code is added even if the addition is well covered, so it punishes writing code
and rewards writing none; the count of statements nothing exercises rises only
when something untested is added or a test is removed, which is the thing worth
refusing.

The numbers this froze were not arbitrary. The three most-changed files in the
repository were also the least covered -- `index_from_zotero` 42%,
`citation_graph/server` 37%, `rag_mcp_server` 41% -- which is why touching them
kept producing surprises. The floor stops that getting worse while it is being
improved.

Reads an existing coverage data file rather than measuring: the suite is run
under coverage once, and both the test result and this check come from that run.

An increase fails; a decrease is reported and adopted deliberately. Unlike
function sizes, this number is not the same everywhere -- CI reaches slightly
less of four modules than macOS does, through branches guarded by platform or
by what is installed -- so a rule that failed on a decrease would leave the two
environments unable to both pass. CI is the one that enforces, so CI's numbers
are the ones recorded.

    uv run coverage run --source=src,citation_graph -m pytest -q
    uv run python scripts/build_coverage_budget.py --check
    uv run python scripts/build_coverage_budget.py --write   # adopt
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGET_PATH = ROOT / "tests" / "coverage_budget.json"

#: Below this many measured files the data is from a partial run -- a single
#: test file, or `-m slow` -- and comparing it to the budget would report every
#: module as a regression.
MINIMUM_MEASURED_FILES = 60


def measure(root: Path = ROOT) -> dict[str, int]:
    """Unreached statements per module, from the coverage data on disk."""
    import coverage

    data = coverage.Coverage(data_file=str(root / ".coverage"))
    data.load()
    measured = data.get_data().measured_files()
    if len(measured) < MINIMUM_MEASURED_FILES:
        raise SystemExit(
            f"the coverage data covers {len(measured)} files, which is fewer than "
            f"{MINIMUM_MEASURED_FILES}: it looks like a partial run. Re-run the "
            "whole suite under coverage before checking or adopting."
        )
    missing: dict[str, int] = {}
    for path in measured:
        analysis = data.analysis2(path)
        relative = Path(path).resolve().relative_to(root).as_posix()
        missing[relative] = len(analysis[3])  # statements no test executed
    return dict(sorted(missing.items()))


def load_budget(path: Path = BUDGET_PATH) -> dict[str, int]:
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8"))["unreached_statements"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="adopt the measured counts")
    parser.add_argument("--check", action="store_true", help="fail if any module got worse")
    args = parser.parse_args(argv)

    measured, recorded = measure(), load_budget()
    grew = {k: (recorded[k], v) for k, v in measured.items()
            if k in recorded and v > recorded[k]}
    shrank = {k: (recorded[k], v) for k, v in measured.items()
              if k in recorded and v < recorded[k]}
    added = {k: v for k, v in measured.items() if k not in recorded and v}
    gone = {k: v for k, v in recorded.items() if k not in measured}

    for name, (was, now) in sorted(grew.items()):
        print(f"worse   {was:5d} -> {now:5d}  (+{now - was})  {name}")
    for name, (was, now) in sorted(shrank.items()):
        print(f"better  {was:5d} -> {now:5d}  ({now - was})  {name}")
    for name, value in sorted(added.items()):
        print(f"new     {'':5s}    {value:5d}          {name}")
    for name, value in sorted(gone.items()):
        print(f"gone    {value:5d}                     {name}")
    print(f"\n{sum(measured.values())} statements unreached across {len(measured)} modules.")

    if args.write:
        BUDGET_PATH.write_text(
            json.dumps({
                "measured_on": f"{platform.system().lower()} python{sys.version_info.major}."
                               f"{sys.version_info.minor}",
                "enforced_by": "the CI workflow, which measures on linux",
                "unreached_statements": measured,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {BUDGET_PATH.relative_to(ROOT)}")
        if platform.system().lower() != "linux":
            print("  note: CI measures on linux and reaches slightly less of "
                  "citation_mapper, env_utils, pdf_extract and text_utils "
                  "(platform-guarded branches). Adopting here can make the next "
                  "CI run fail; raise those four to what CI reports.")
        return 0
    if args.check and (grew or added):
        print(
            "\nSomething was added that no test reaches, or a test that reached "
            "it was removed. Cover it, or adopt the new ceiling deliberately with "
            "--write and say why in the commit.",
            file=sys.stderr,
        )
        return 1
    if shrank:
        # Reported, not failed. Unlike function sizes, this number is not the
        # same in every environment: CI measures four modules as having more
        # unreached statements than macOS does, because branches guarded by
        # what is installed or which platform it is never run there. Failing on
        # a decrease would mean the two environments could never both pass, so
        # the ceiling is lowered deliberately with --write rather than by a red
        # test somebody has to work around.
        print("\nFewer unreached statements than recorded -- lower the ceiling "
              "with --write when you mean to.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
