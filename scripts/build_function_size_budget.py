"""Record how long the long functions are, so the number can only go down.

The register of oversized functions was kept by hand and in prose, which means
it was true on the day it was written. This builds it from the source instead:
every function over ``LIMIT`` lines, with the size it currently has.
``tests/test_function_size_ratchet.py`` reads the result and refuses both a
function that grew past its record and a new one born over the limit, so
splitting becomes work with a visible end rather than a standing intention.

The budget is tracked in git, unlike ``tests/baselines/``: it holds file paths
and function names, nothing from the library.

Usage::

    uv run python scripts/build_function_size_budget.py            # diff
    uv run python scripts/build_function_size_budget.py --write    # adopt
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGET_PATH = ROOT / "tests" / "function_size_budget.json"

#: A function longer than this has to be recorded to exist.
#:
#: 150 is where the current distribution thins out -- 23 functions above it
#: against 1,400 below -- so it names the outliers without conscripting ordinary
#: code. It is a ceiling for new functions, not a target for old ones: the
#: recorded sizes come down as they are split, and the limit follows.
LIMIT = 150

#: Where the ratchet applies. Tests are excluded: their longest function is 67
#: lines, and a long test is usually a table of cases rather than a tangle.
TREES = ("src", "scripts", "citation_graph")


def _length(node: ast.AST) -> int:
    """Lines from ``def`` to the last line of the body, decorators excluded.

    A nested function counts once on its own and again inside its parent, which
    is the honest reading: the parent is that long to read.
    """
    return node.end_lineno - node.lineno + 1


def measure(root: Path = ROOT) -> dict[str, int]:
    """Every function over ``LIMIT``, as ``path::qualified.name`` -> lines."""
    sizes: dict[str, int] = {}
    for tree in TREES:
        for path in sorted((root / tree).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            try:
                parsed = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:  # pragma: no cover -- caught by compileall in CI
                continue
            relative = path.relative_to(root).as_posix()
            for qualified_name, node in _functions(parsed):
                if _length(node) > LIMIT:
                    sizes[f"{relative}::{qualified_name}"] = _length(node)
    return sizes


def _functions(node: ast.AST, prefix: str = ""):
    """Walk classes and functions, naming each by its qualified path."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = f"{prefix}{child.name}"
            yield name, child
            yield from _functions(child, prefix=f"{name}.")
        elif isinstance(child, ast.ClassDef):
            yield from _functions(child, prefix=f"{prefix}{child.name}.")
        else:
            yield from _functions(child, prefix=prefix)


def load_budget(path: Path = BUDGET_PATH) -> dict[str, int]:
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8"))["functions"])


def _render(sizes: dict[str, int]) -> str:
    payload = {
        "limit": LIMIT,
        "trees": list(TREES),
        "functions": dict(sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0]))),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="adopt the measured sizes")
    args = parser.parse_args(argv)

    measured = measure()
    recorded = load_budget()

    grew = {k: (recorded[k], v) for k, v in measured.items() if k in recorded and v > recorded[k]}
    shrank = {k: (recorded[k], v) for k, v in measured.items() if k in recorded and v < recorded[k]}
    added = {k: v for k, v in measured.items() if k not in recorded}
    gone = {k: v for k, v in recorded.items() if k not in measured}

    for label, rows in (("grew", grew), ("shrank", shrank)):
        for name, (was, now) in sorted(rows.items(), key=lambda kv: kv[1][1] - kv[1][0]):
            print(f"{label:7s} {was:5d} -> {now:5d}  {name}")
    for name, size in sorted(added.items(), key=lambda kv: -kv[1]):
        print(f"new     {'':5s}    {size:5d}  {name}")
    for name, size in sorted(gone.items(), key=lambda kv: -kv[1]):
        print(f"under   {size:5d} -> {LIMIT:5d}-  {name}")

    total = len(measured)
    largest = max(measured.values(), default=0)
    print(f"\n{total} functions over {LIMIT} lines, largest {largest}.")

    if args.write:
        BUDGET_PATH.write_text(_render(measured), encoding="utf-8")
        print(f"wrote {BUDGET_PATH.relative_to(ROOT)}")
        return 0
    if grew or shrank or added or gone:
        print("run with --write to adopt.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
