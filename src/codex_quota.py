"""Read Codex weekly usage and enforce a fail-closed remaining-quota floor."""
from __future__ import annotations

import argparse
import json
import selectors
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


WEEK_MINUTES = 7 * 24 * 60


class CodexQuotaError(RuntimeError):
    """Raised when the Codex weekly quota cannot be determined safely."""


class CodexQuotaFloorReached(CodexQuotaError):
    """Raised when remaining weekly quota is at or below the configured floor."""

    def __init__(self, quota: "WeeklyQuota", minimum_remaining_percent: int):
        self.quota = quota
        self.minimum_remaining_percent = minimum_remaining_percent
        super().__init__(
            f"weekly quota remaining {quota.remaining_percent}% "
            f"(floor {minimum_remaining_percent}%); reset={quota.resets_at_iso or 'unknown'}"
        )


@dataclass(frozen=True)
class WeeklyQuota:
    used_percent: int
    remaining_percent: int
    window_duration_mins: int
    resets_at: int | None

    @property
    def resets_at_iso(self) -> str | None:
        if self.resets_at is None:
            return None
        return datetime.fromtimestamp(self.resets_at).astimezone().isoformat()

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["resets_at_iso"] = self.resets_at_iso
        return result


def _weekly_window(payload: dict[str, Any]) -> WeeklyQuota:
    result = payload.get("result")
    if not isinstance(result, dict):
        raise CodexQuotaError("Codex returned no rate-limit result")
    by_id = result.get("rateLimitsByLimitId")
    snapshot = by_id.get("codex") if isinstance(by_id, dict) else None
    if not isinstance(snapshot, dict):
        snapshot = result.get("rateLimits")
    if not isinstance(snapshot, dict):
        raise CodexQuotaError("Codex returned no quota snapshot")

    windows = []
    for name in ("primary", "secondary"):
        window = snapshot.get(name)
        if not isinstance(window, dict):
            continue
        duration = window.get("windowDurationMins")
        used = window.get("usedPercent")
        if isinstance(duration, int) and duration >= WEEK_MINUTES and isinstance(used, int):
            windows.append((duration, used, window.get("resetsAt")))
    if not windows:
        raise CodexQuotaError("Codex returned no weekly-or-longer quota window")
    duration, used, resets_at = min(windows, key=lambda row: row[0])
    if not 0 <= used <= 100:
        raise CodexQuotaError(f"Codex returned an invalid used percentage: {used}")
    if resets_at is not None and not isinstance(resets_at, int):
        raise CodexQuotaError("Codex returned an invalid reset time")
    return WeeklyQuota(
        used_percent=used,
        remaining_percent=100 - used,
        window_duration_mins=duration,
        resets_at=resets_at,
    )


def read_weekly_quota(*, timeout: float = 10.0, codex_command: str = "codex") -> WeeklyQuota:
    """Query the local Codex app-server for the current weekly quota window."""
    messages = [
        {
            "method": "initialize", "id": 1,
            "params": {
                "clientInfo": {
                    "name": "zotero_local_rag", "title": "Zotero Local RAG", "version": "0.1.0",
                },
                "capabilities": {"experimentalApi": True},
            },
        },
        {"method": "initialized", "params": {}},
        {"method": "account/rateLimits/read", "id": 2, "params": {}},
    ]
    try:
        process = subprocess.Popen(
            [codex_command, "app-server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
    except OSError as exc:
        raise CodexQuotaError(f"Could not start Codex: {exc}") from exc
    try:
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write("".join(json.dumps(message) + "\n" for message in messages))
        process.stdin.flush()
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                events = selector.select(max(0.0, deadline - time.monotonic()))
                if not events:
                    break
                line = process.stdout.readline()
                if not line:
                    break
                try:
                    response = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if response.get("id") == 2:
                    if response.get("error"):
                        raise CodexQuotaError(f"Codex quota request failed: {response['error']}")
                    return _weekly_window(response)
        raise CodexQuotaError("Timed out while reading Codex weekly quota")
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def require_weekly_quota(minimum_remaining_percent: int) -> WeeklyQuota:
    """Return current quota or raise when it is unsafe to start another LLM request."""
    quota = read_weekly_quota()
    if quota.remaining_percent <= minimum_remaining_percent:
        raise CodexQuotaFloorReached(quota, minimum_remaining_percent)
    return quota


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-remaining-percent", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    if not 0 <= args.min_remaining_percent <= 100:
        parser.error("--min-remaining-percent must be between 0 and 100")
    try:
        quota = read_weekly_quota(timeout=args.timeout)
    except CodexQuotaError as exc:
        print(json.dumps({"allowed": False, "reason": "quota_unknown", "error": str(exc)}))
        return 4
    allowed = quota.remaining_percent > args.min_remaining_percent
    print(json.dumps({
        "allowed": allowed,
        "reason": "quota_available" if allowed else "weekly_quota_floor",
        "minimum_remaining_percent": args.min_remaining_percent,
        **quota.as_dict(),
    }, ensure_ascii=False))
    return 0 if allowed else 3


if __name__ == "__main__":
    raise SystemExit(main())
