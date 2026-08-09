#!/usr/bin/env python3
"""Which parts of the ingest loop the ingestion baseline actually watches.

``tests/test_ingestion_baseline.py`` runs the real loop over a handful of
attachments, which makes it easy to say it "covers ingestion" and easy to be
wrong. Three attachments reached 21% of the loop, and three blocks had already
been lifted out of the parts it could not see before anyone measured. Six
attachments and three passes each reach 28%.

The number is the point of this script, not the refactor. Read a block's
coverage before moving it: below it, a green net means the check did not object,
not that it verified anything.

Run it before choosing what to lift next, and again after widening the corpus:

    uv run python scripts/measure_ingestion_net_coverage.py

Needs Zotero open and the attachments on disk, like the baseline itself. Uses
sys.settrace rather than a coverage package so it stays a dependency-free
measurement of one function.
"""
from __future__ import annotations

import ast
import json
import os
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_ingestion_baseline import (  # noqa: E402
    CORPUS, DERIVED_DATA_PLANE_VARIABLES, _zotero_is_reachable,
)

TARGET = ROOT / "src" / "index_from_zotero.py"


def loop_span() -> tuple[int, int]:
    """The line range of main_async's per-attachment loop."""
    tree = ast.parse(TARGET.read_text(encoding="utf-8"))
    function = next(
        node for node in ast.walk(tree)
        if getattr(node, "name", "") == "main_async"
    )
    loop = next(
        node for node in ast.walk(function)
        if isinstance(node, (ast.For, ast.AsyncFor))
        and node.end_lineno - node.lineno > 1000
    )
    return loop.lineno, loop.end_lineno


def _trace_one(item_key: str, plane: Path, low: int, high: int, *,
               force: bool = True, check_quality: bool = False) -> set[int]:
    """Ingest one attachment in this process, recording the loop lines reached."""
    hit: set[int] = set()
    target = str(TARGET)

    def tracer(frame, event, _arg):
        if frame.f_code.co_filename != target:
            return None
        if event == "line" and low <= frame.f_lineno <= high:
            hit.add(frame.f_lineno)
        return tracer

    for name in DERIVED_DATA_PLANE_VARIABLES:
        os.environ.pop(name, None)
    os.environ.update({
        "CHROMA_DIR": str(plane / "chroma"),
        "MANIFEST_PATH": str(plane / "manifest_v3.json"),
        "LEXICAL_DB_PATH": str(plane / "lexical_v3.sqlite3"),
        "RELATIONS_DB_PATH": str(plane / "relations.db"),
    })
    argv = sys.argv
    sys.argv = [
        "index_from_zotero.py", "--item", item_key,
        *(["--force-reparse"] if force else []),
        *(["--check-quality"] if check_quality else []),
    ]
    sys.path.insert(0, str(ROOT / "src"))
    sys.settrace(tracer)
    try:
        runpy.run_path(target, run_name="__main__")
    except SystemExit:
        pass
    finally:
        sys.settrace(None)
        sys.argv = argv
    return hit


def _executable_lines(low: int, high: int) -> set[int]:
    lines = TARGET.read_text(encoding="utf-8").split("\n")
    return {
        number for number in range(low, high + 1)
        if lines[number - 1].strip() and not lines[number - 1].strip().startswith("#")
    }


def main() -> None:
    if not _zotero_is_reachable():
        raise SystemExit("Zotero's local API is not answering. Open Zotero and try again.")
    low, high = loop_span()
    hit: set[int] = set()
    # One child per attachment: the loop reads module-level state settled at
    # import, so tracing them in one process would measure the second and third
    # against the first one's world.
    for item_key in CORPUS:
        with tempfile.TemporaryDirectory(prefix="net-coverage-") as raw:
            plane = Path(raw) / "plane"
            reached: set[int] = set()
            # The same three passes the baseline makes, against one plane: the
            # first indexes, the second finds the source unchanged, the third
            # asks for quality again. Tracing only the first measured a third of
            # what the net actually walks and reported it as the whole.
            for label, options in (
                ("index", {}),
                ("unchanged", {"force": False}),
                ("quality", {"force": False, "check_quality": True}),
            ):
                handoff = Path(raw) / f"hit-{label}.json"
                child = subprocess.run(
                    [sys.executable, "-c",
                     "import json,sys;from pathlib import Path;"
                     f"sys.path.insert(0, {str(ROOT)!r});"
                     "from scripts.measure_ingestion_net_coverage import _trace_one, loop_span;"
                     "low, high = loop_span();"
                     f"json.dump(sorted(_trace_one({item_key!r}, Path({str(plane)!r}),"
                     f" low, high, **{options!r})), open({str(handoff)!r}, 'w'))"],
                    cwd=ROOT, capture_output=True, text=True,
                )
                if not handoff.exists():
                    print(f"  {item_key} ({label}): no trace -- "
                          f"{(child.stderr.strip().splitlines() or ['?'])[-1][:110]}",
                          file=sys.stderr)
                    continue
                reached |= set(json.loads(handoff.read_text()))
            hit |= reached
            print(f"  {item_key}: {len(reached)} lines", file=sys.stderr)

    executable = _executable_lines(low, high)
    covered = hit & executable
    print(f"\nloop {low}-{high}: {len(covered)} of {len(executable)} executable "
          f"lines reached ({100 * len(covered) // max(1, len(executable))}%)")

    tree = ast.parse(TARGET.read_text(encoding="utf-8"))
    function = next(n for n in ast.walk(tree) if getattr(n, "name", "") == "main_async")
    loop = next(n for n in ast.walk(function)
                if isinstance(n, (ast.For, ast.AsyncFor)) and n.end_lineno - n.lineno > 1000)
    lines = TARGET.read_text(encoding="utf-8").split("\n")
    print("\nby block (only those worth lifting out):")
    for statement in loop.body:
        span = statement.end_lineno - statement.lineno + 1
        if span < 18:
            continue
        block = {n for n in range(statement.lineno, statement.end_lineno + 1)} & executable
        reached = len(block & hit)
        share = 100 * reached / max(1, len(block))
        watched = "watched" if share >= 60 else ("partly" if share >= 20 else "UNWATCHED")
        print(f"  L{statement.lineno:<6} {span:>4} lines  {share:>3.0f}%  {watched:<9} "
              f"{lines[statement.lineno - 1].strip()[:44]}")


if __name__ == "__main__":
    main()
