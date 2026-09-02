#!/usr/bin/env python3
"""Execute one fixed browser-admin maintenance job in an isolated process."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from citation_graph.admin_jobs import run_job, schedule_followup_update_check


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--job-type", required=True)
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    exit_code = run_job(args.job_id, args.job_type, args.token)
    if exit_code == 0:
        schedule_followup_update_check(args.job_type, ROOT)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
