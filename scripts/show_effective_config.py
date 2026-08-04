#!/usr/bin/env python3
"""Print every setting this project reads, and the value in effect right now.

Written because there was no way to answer "why does the server behave
differently?" without reading code. On 2026-08-03 the server silently lost
real page content for a week: PDF_SCANNED_PAGE_PATCH_ENABLE was set on one
machine and unset on the other, and the code default happened to disable the
repair. Finding that took hours of comparing two machines by hand, because
nothing could show what either one was actually running with.

Run this on both machines and diff the output.

The list of settings is derived from the source with ``ast`` on every run,
not from a hand-maintained registry: a registry is one more thing to forget
to update, and the failure it would cause -- a setting missing from exactly
the report you are using to hunt a config difference -- is the failure this
script exists to prevent.

Values whose name looks like a credential are never printed; the report shows
only whether one is set and how long it is, so the output stays safe to paste
into an issue or a chat.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_utils import load_dotenv_native  # noqa: E402

#: Scanned for ``os.environ`` reads. Tests are excluded deliberately: a
#: setting only ever read by a test is not part of the runtime configuration.
SOURCE_DIRS = ("src", "scripts", "citation_graph")

#: Read from the environment but supplied by the OS or a third-party library,
#: not by this project's configuration. Reporting them adds noise to the diff.
NOT_OUR_CONFIG = frozenset({
    "APPDATA", "HOME", "PATH", "PWD", "TMPDIR", "USER",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM",
    "RAYON_RS_NUM_CPUS", "PYTORCH_ENABLE_MPS_FALLBACK", "VIRTUAL_ENV",
})

#: Substrings that mark a value as a credential. Matched case-insensitively
#: against the *name*; the value itself is never inspected or printed.
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def _is_secret(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in SECRET_MARKERS)


class _EnvReadVisitor(ast.NodeVisitor):
    """Collects ``os.environ`` *reads*, ignoring writes.

    ``os.environ["OMP_NUM_THREADS"] = "1"`` configures a third-party library;
    treating that assignment as a read would list a value this project never
    consumes. Only a Load-context subscript counts.
    """

    def __init__(self, path: str, found: dict[str, set[str]]):
        self.path = path
        self.found = found

    def _record(self, name: str) -> None:
        if isinstance(name, str) and name.isupper():
            self.found.setdefault(name, set()).add(self.path)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in {"get", "getenv"} and node.args:
            base = func.value
            reads_env = (
                (isinstance(base, ast.Attribute) and base.attr == "environ")
                or (isinstance(base, ast.Name) and base.id == "os")
            )
            first = node.args[0]
            if reads_env and isinstance(first, ast.Constant):
                self._record(first.value)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if (
            isinstance(node.value, ast.Attribute)
            and node.value.attr == "environ"
            and isinstance(node.ctx, ast.Load)
            and isinstance(node.slice, ast.Constant)
        ):
            self._record(node.slice.value)
        self.generic_visit(node)


def discover_settings(root: Path = ROOT) -> dict[str, set[str]]:
    """Every environment variable the runtime source reads, to the files."""
    found: dict[str, set[str]] = {}
    for directory in SOURCE_DIRS:
        for path in sorted((root / directory).rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                continue
            _EnvReadVisitor(str(path.relative_to(root)), found).visit(tree)
    return {k: v for k, v in found.items() if k not in NOT_OUR_CONFIG}


def dotenv_names(root: Path = ROOT) -> set[str]:
    """Names assigned in .env, to tell a file setting from a shell one."""
    path = root / ".env"
    if not path.is_file():
        return set()
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*([A-Za-z_][A-Za-z_0-9]*)\s*=", line)
        if match:
            names.add(match.group(1))
    return names


def build_report(root: Path = ROOT) -> dict:
    settings = discover_settings(root)
    from_dotenv = dotenv_names(root)
    rows = []
    for name in sorted(settings):
        raw = os.environ.get(name)
        if raw is None:
            value, state = None, "unset"
        elif _is_secret(name):
            value, state = f"<set, {len(raw)} chars>", "set"
        else:
            value, state = raw, "set"
        rows.append({
            "name": name,
            "state": state,
            "value": value,
            # Where it came from matters when diffing machines: "unset here,
            # set in .env there" is the shape of the drift that hid the
            # 2026-08-03 content loss.
            "source": (".env" if name in from_dotenv else "environment") if raw is not None else None,
            "read_by": sorted(settings[name]),
        })
    return {
        "project_root": str(root),
        "settings_count": len(rows),
        "set_count": sum(1 for row in rows if row["state"] == "set"),
        "settings": rows,
    }


def format_text(report: dict, *, verbose: bool = False) -> str:
    lines = [
        f"# effective configuration — {report['project_root']}",
        f"# {report['set_count']} of {report['settings_count']} settings have a value",
        "",
    ]
    width = max((len(row["name"]) for row in report["settings"]), default=0)
    for row in report["settings"]:
        if row["state"] == "unset":
            lines.append(f"{row['name']:<{width}} = (unset)")
        else:
            lines.append(
                f"{row['name']:<{width}} = {row['value']}    [{row['source']}]"
            )
        if verbose:
            lines.append(f"{'':<{width}}   read by: {', '.join(row['read_by'])}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Machine-readable output.")
    parser.add_argument(
        "--verbose", action="store_true", help="Also show which files read each setting.",
    )
    parser.add_argument(
        "--set-only", action="store_true", help="Omit settings with no value.",
    )
    args = parser.parse_args()

    load_dotenv_native(ROOT)
    report = build_report()
    if args.set_only:
        # settings_count has to follow the filter. Leaving the unfiltered
        # total next to a filtered body makes the report disagree with
        # itself, which is fatal for the one thing it is for: being diffed
        # and trusted. total_settings keeps the unfiltered figure visible.
        kept = [row for row in report["settings"] if row["state"] == "set"]
        report = {
            **report,
            "settings": kept,
            "settings_count": len(kept),
            "total_settings": report["settings_count"],
            "filtered": "set-only",
        }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_text(report, verbose=args.verbose))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
