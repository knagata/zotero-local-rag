#!/usr/bin/env python3
"""Launch a fully-detached LLM summary backfill that survives harness job cleanup.

Starts a new session with ``os.setsid()`` so the process is immune to
process-group kills, then execs ``caffeinate`` wrapping
``scripts/build_structure_summaries.py --all --mode llm --embed``.
Safe to interrupt and re-run: the differential-skip logic (R1) means already
generated summaries are skipped with zero LLM calls on the next run.

Usage: detached_summary_backfill.py --database-gate PATH [--workers N] [--limit N]
``--workers`` (default 10) controls concurrent items; ``--limit`` optionally
bounds this run to N items instead of the whole remaining backlog.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=20, help="Concurrent items (default 20).")
    parser.add_argument("--limit", type=int, default=0, help="Bound this run to N items (default: all remaining).")
    parser.add_argument(
        "--database-gate", type=Path, required=True,
        help="Passing server rebuild audit for the current DB generation.",
    )
    args = parser.parse_args()

    import os
    os.setsid()  # new session: detach from the harness process group
    python = str(ROOT / ".venv" / "bin" / "python")
    exec_args = [
        "caffeinate", "-i", "-s", python,
        str(ROOT / "scripts" / "build_structure_summaries.py"),
        "--all", "--mode", "llm", "--embed", "--workers", str(args.workers),
        "--database-gate", str(args.database_gate.resolve()),
    ]
    if args.limit > 0:
        exec_args += ["--limit", str(args.limit)]
    os.chdir(ROOT)
    os.execvp("caffeinate", exec_args)


if __name__ == "__main__":
    main()
