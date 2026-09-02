#!/usr/bin/env python3
"""Start one read-only update check unless another admin job is active."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from citation_graph.admin_jobs import JobAlreadyRunningError, start_job


def main() -> int:
    try:
        record = start_job("update_check", "", "scheduled-launch-agent", ROOT)
    except JobAlreadyRunningError as exc:
        print(f"scheduled_update_check=deferred ({exc})")
        return 0
    print(f"scheduled_update_check=started job_id={record['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
