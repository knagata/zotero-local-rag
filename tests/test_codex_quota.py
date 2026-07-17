from __future__ import annotations

import unittest

from unittest.mock import patch

from src.codex_quota import (
    CodexQuotaError, CodexQuotaFloorReached, WeeklyQuota, _weekly_window,
    require_weekly_quota,
)


class CodexQuotaTests(unittest.TestCase):
    def test_selects_weekly_window_and_calculates_remaining(self):
        quota = _weekly_window({"result": {"rateLimitsByLimitId": {"codex": {
            "primary": {"usedPercent": 81, "windowDurationMins": 10080, "resetsAt": 123},
            "secondary": {"usedPercent": 50, "windowDurationMins": 300, "resetsAt": 456},
        }}}})
        self.assertEqual(quota.used_percent, 81)
        self.assertEqual(quota.remaining_percent, 19)
        self.assertEqual(quota.resets_at, 123)

    def test_uses_backward_compatible_snapshot(self):
        quota = _weekly_window({"result": {"rateLimits": {
            "primary": {"usedPercent": 25, "windowDurationMins": 10080, "resetsAt": None},
        }}})
        self.assertEqual(quota.remaining_percent, 75)

    def test_fails_closed_without_weekly_window(self):
        with self.assertRaises(CodexQuotaError):
            _weekly_window({"result": {"rateLimits": {
                "primary": {"usedPercent": 10, "windowDurationMins": 300},
            }}})

    def test_rejects_invalid_percentage(self):
        with self.assertRaises(CodexQuotaError):
            _weekly_window({"result": {"rateLimits": {
                "primary": {"usedPercent": 101, "windowDurationMins": 10080},
            }}})

    def test_floor_is_inclusive(self):
        quota = WeeklyQuota(used_percent=80, remaining_percent=20,
                            window_duration_mins=10080, resets_at=None)
        with patch("src.codex_quota.read_weekly_quota", return_value=quota):
            with self.assertRaises(CodexQuotaFloorReached):
                require_weekly_quota(20)


if __name__ == "__main__":
    unittest.main()
