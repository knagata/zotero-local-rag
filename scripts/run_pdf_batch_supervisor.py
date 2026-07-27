#!/usr/bin/env python3
"""Run V3 PDF ingestion in restartable small batches until it is done or stalled."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = ROOT / "data" / "pdf_batch_supervisor_state.json"
DEFAULT_LOG = ROOT / "data" / "pdf_batch_supervisor.log"
DEFAULT_LOCK = ROOT / "data" / "pdf_batch_supervisor.lock"
RESULT_EVENT = "index_batch_result"


def parse_batch_result(output: str) -> dict[str, Any] | None:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("event") == RESULT_EVENT:
            return value
    return None


def write_state(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **values}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def build_command(batch_size: int) -> list[str]:
    return [
        "caffeinate", "-i", str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "src" / "index_from_zotero.py"),
        "--source-type", "pdf", "--limit", str(batch_size),
        "--require-data-dir", "--progress",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.batch_size > 3:
        parser.error("--batch-size must be between 1 and 3 (RapidOCR safety limit)")

    args.lock.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = args.lock.open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        write_state(args.state, status="duplicate_refused", message="supervisor lock is held")
        print("PDF batch supervisor is already running.", file=sys.stderr)
        return 2

    environment = os.environ.copy()
    environment.update({
        "PDF_OCR_FALLBACK": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": "src",
        "INGEST_STRUCTURED_V3_ENABLE": "1",
        "PDF_AI_TOC_FAST_PATH_ENABLE": "1",
        "PDF_AI_TOC_MIN_PAGES": "30",
        "PDF_MISTRAL_TOC_QUEUE_ENABLE": "1",
    })
    stopping = False
    child: subprocess.Popen[str] | None = None

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True
        write_state(args.state, status="stopping", signal=signum)
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    consecutive_failures = 0
    batch_number = 0

    with args.log.open("a", encoding="utf-8", buffering=1) as log:
        while not stopping:
            batch_number += 1
            command = build_command(args.batch_size)
            started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            write_state(args.state, status="running", batch=batch_number, started_at=started)
            log.write(f"\n[{started}] supervisor batch={batch_number} command={command!r}\n")
            child = subprocess.Popen(
                command, cwd=ROOT, env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            captured: list[str] = []
            assert child.stdout is not None
            for line in child.stdout:
                captured.append(line)
                log.write(line)
                sys.stdout.write(line)
                sys.stdout.flush()
            returncode = child.wait()
            child = None
            output = "".join(captured)
            result = parse_batch_result(output)

            if stopping:
                write_state(args.state, status="stopped", batch=batch_number, returncode=returncode)
                return 130
            if returncode == 0 and result is not None:
                consecutive_failures = 0
                if int(result.get("processed_parent_items", -1)) == 0:
                    write_state(args.state, status="complete", batch=batch_number, result=result)
                    return 0
                write_state(args.state, status="between_batches", batch=batch_number, result=result)
                continue

            consecutive_failures += 1
            write_state(
                args.state, status="retrying", batch=batch_number,
                returncode=returncode, consecutive_failures=consecutive_failures,
                message="batch failed or did not emit a result event",
            )
            if consecutive_failures >= args.max_consecutive_failures:
                write_state(
                    args.state, status="stalled", batch=batch_number,
                    returncode=returncode, consecutive_failures=consecutive_failures,
                )
                return 1
            time.sleep(args.retry_delay)

    write_state(args.state, status="stopped", batch=batch_number)
    return 130


if __name__ == "__main__":
    raise SystemExit(main())
