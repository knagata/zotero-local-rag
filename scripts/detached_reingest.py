#!/usr/bin/env python3
"""Launch a fully-detached EPUB re-ingestion that survives harness job cleanup.

Reads item keys from a file (one per line), starts a new session with
``os.setsid()`` so the process is immune to process-group kills, then execs
``caffeinate`` wrapping ``index_from_zotero.py --force-reparse`` for those items.
Run it nohup'd and disowned from a foreground shell; the parent returns while
this keeps running independently.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    items_file = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "quality" / "epub_reingest_remaining.txt"
    keys = [line.strip() for line in items_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not keys:
        raise SystemExit("no items to re-ingest")
    os.setsid()  # new session: detach from the harness process group
    os.environ.setdefault("EMB_DEVICE", "mps")  # ensure Apple-Silicon GPU embedding
    python = str(ROOT / ".venv" / "bin" / "python")
    args = ["caffeinate", "-i", "-s", python, str(ROOT / "src" / "index_from_zotero.py"),
            "--force-reparse", "--progress"]
    for key in keys:
        args += ["--item", key]
    os.chdir(ROOT)
    os.execvp("caffeinate", args)


if __name__ == "__main__":
    main()
