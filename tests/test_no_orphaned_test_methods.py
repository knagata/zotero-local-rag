"""No test method may be trapped inside an ``if __name__ == "__main__":`` guard,
or inside another function's body.

Appending a test with ``cat >>`` lands after that guard in a file that has one
in the middle rather than at the end. The result parses fine and even looks
right on a read-through -- correct indentation, a plausible docstring -- but
the method becomes a nested function inside the guard's body, invisible to
unittest's class-based discovery. It is never collected and can never fail,
which is exactly what happened twice in this session: three tests pinning the
single-child summary rule, and one pinning a re-parenting cleanup, ran zero
times for as long as they existed, silently discarding the property each was
written to guard (2026-07-28, found by manual inspection after a code-review
fix's regression test passed for a reason that turned out to be this).

The same shape recurs one level down: a module-level ``def test_...():``
placed after the guard is fine (pytest collects module-level test functions
too), but a ``def test_...(self):`` then appended *inside that function's
body* -- rather than back up in the class above it -- is dead code no
collector ever sees. Two more tests were found in exactly this state in
tests/test_pdf_extract_layout.py (2026-07-29): both had been cited as passing
evidence for the P3-1 zone fix and the P3-9 CJK fix, but had run zero times.
"""
from __future__ import annotations

import ast
import pathlib
import unittest

TESTS_DIR = pathlib.Path(__file__).resolve().parent


def _orphaned_defs(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            is_main_guard = (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
            )
            if is_main_guard:
                names.extend(
                    n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # A test def whose direct parent is another function (not a class,
            # not the module) is unreachable by any collector: pytest and
            # unittest both only look inside classes and at module level.
            names.extend(
                n.name for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test")
            )
    return names


class NoOrphanedTestMethodsTests(unittest.TestCase):
    def test_no_test_file_hides_a_method_inside_its_main_guard(self):
        offenders = {}
        for path in sorted(TESTS_DIR.glob("*.py")):
            if path.name == pathlib.Path(__file__).name:
                continue
            found = _orphaned_defs(path)
            if found:
                offenders[path.name] = found
        self.assertEqual(
            offenders, {},
            "these def(s) are nested inside `if __name__ == \"__main__\":` and are "
            "never collected by pytest -- move the guard to the end of the file, "
            "or move the def(s) into a class above it",
        )


if __name__ == "__main__":
    unittest.main()
