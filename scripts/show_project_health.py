"""Where the codebase stands, measured rather than remembered.

Written because "how much is left?" was being answered from memory, and memory
answers it differently every time. Everything below is re-derived on each run
from the code, the budgets and the register, so a number here is either current
or the command failed.

    uv run python scripts/show_project_health.py
    uv run python scripts/show_project_health.py --coverage   # slower, runs the suite

The goal these numbers serve is not "no findings". It is that a change can be
made safely: that the mechanical layers refuse a regression, and that the parts
which can only be judged by reading are small enough to read. Where a target is
stated below, it is one that can actually be reached and stayed at.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_function_size_budget import LIMIT, load_budget as load_sizes  # noqa: E402
from scripts.build_coverage_budget import load_budget as load_coverage  # noqa: E402
from scripts.build_lint_budget import load_budget as load_lint  # noqa: E402

REGISTER = ROOT / "docs" / "post-refactor-followups.md"


def _section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def _tests() -> tuple[int, int]:
    """Collected tests, and how many the default run leaves out."""
    result = subprocess.run(
        ["uv", "run", "pytest", "-q", "--collect-only"],
        cwd=ROOT, capture_output=True, text=True, timeout=600, check=False,
    )
    matched = re.search(r"(\d+)/(\d+) tests collected \((\d+) deselected\)", result.stdout)
    if matched:
        return int(matched.group(2)), int(matched.group(3))
    matched = re.search(r"(\d+) tests collected", result.stdout)
    return (int(matched.group(1)), 0) if matched else (0, 0)


def _register_counts() -> tuple[int, int]:
    text = REGISTER.read_text(encoding="utf-8")
    open_section = text[text.index("## P"):text.index("## Longer-term")]
    closed_section = text[text.index("## Closed"):text.index("## Reassessment")]
    return (
        len(re.findall(r"^### ", open_section, flags=re.M)),
        len(re.findall(r"^- \*\*", closed_section, flags=re.M)),
    )


def _mechanical_layers() -> list[tuple[str, str]]:
    sizes = load_sizes()
    lint = load_lint()
    return [
        ("function sizes", f"{len(sizes)} over {LIMIT} lines, largest {max(sizes.values(), default=0)}"),
        ("swallowed exceptions", f"{sum(lint.values())} frozen across {len(lint)} rules"),
        ("unreached statements",
         f"{sum(load_coverage().values())} frozen across {len(load_coverage())} modules"),
        ("real-data writes", "refused by tests/conftest.py for every test"),
        ("slow tests", "excluded by default, opt in with -m slow"),
    ]


def _coverage() -> list[tuple[str, float]]:
    """Per-module coverage of the default suite. Needs the coverage package."""
    result = subprocess.run(
        ["uv", "run", "--with", "coverage", "python", "-m", "coverage", "run",
         "--source=src,citation_graph", "-m", "pytest", "-q"],
        cwd=ROOT, capture_output=True, text=True, timeout=1800, check=False,
    )
    if result.returncode not in (0, 1):
        print(f"  (coverage run failed: {result.stderr[-200:]})")
        return []
    report = subprocess.run(
        ["uv", "run", "--with", "coverage", "python", "-m", "coverage", "report"],
        cwd=ROOT, capture_output=True, text=True, timeout=600, check=False,
    )
    rows = []
    for line in report.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[-1].endswith("%") and parts[0].endswith(".py"):
            rows.append((parts[0], float(parts[-1].rstrip("%"))))
        elif parts[:1] == ["TOTAL"]:
            rows.append(("TOTAL", float(parts[-1].rstrip("%"))))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", action="store_true",
                        help="also run the suite under coverage (minutes, not seconds)")
    args = parser.parse_args(argv)

    _section("Checks that run on every push")
    collected, deselected = _tests()
    print(f"  tests collected     {collected} ({deselected} deselected by default)")
    for name, state in _mechanical_layers():
        print(f"  {name:19s} {state}")

    _section("Known defects, from the register")
    open_items, closed_items = _register_counts()
    print(f"  open                {open_items}")
    print(f"  closed              {closed_items}")
    print(f"  register            {REGISTER.relative_to(ROOT)}")

    if args.coverage:
        _section("Coverage of the default suite")
        rows = [row for row in _coverage() if row[0] == "TOTAL" or row[1] < 60]
        for name, percent in sorted(rows, key=lambda r: r[1]):
            print(f"  {percent:5.1f}%  {name}")
        print("  (modules under 60% only, plus the total)")

    print()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
