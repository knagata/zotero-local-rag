from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from citation_graph import server
from citation_graph.lifecycle import SingleFlight, StartOnce


class CitationGraphLifecycleTests(unittest.TestCase):
    def tearDown(self):
        server._initial_build_done.clear()
        server._rebuild_singleflight = SingleFlight(
            server._rebuild_graph_bg,
            lambda target: server._threading.Thread(target=target, daemon=False).start(),
            externally_busy=server._rebuild_lock.locked,
        )

    def test_close_watcher_is_started_explicitly_and_only_once(self):
        fake_thread = MagicMock()
        with patch.object(server._threading, "Thread", return_value=fake_thread) as thread:
            server._close_watcher_once = StartOnce(
                lambda: server._threading.Thread(
                    target=server._close_watcher, daemon=True,
                ).start()
            )
            server._start_close_watcher()
            server._start_close_watcher()
        thread.assert_called_once_with(target=server._close_watcher, daemon=True)
        fake_thread.start.assert_called_once()


class SingleFlightTests(unittest.TestCase):
    def test_task_completion_allows_a_later_schedule(self):
        queued = []
        calls = []
        flight = SingleFlight(lambda: calls.append("ran"), queued.append)
        self.assertTrue(flight.schedule())
        self.assertFalse(flight.schedule())
        queued.pop()()
        self.assertEqual(calls, ["ran"])
        self.assertTrue(flight.schedule())

    def test_start_failure_releases_the_scheduled_flag(self):
        def fail_start(_target):
            raise RuntimeError("thread unavailable")

        flight = SingleFlight(lambda: None, fail_start)
        with self.assertRaisesRegex(RuntimeError, "thread unavailable"):
            flight.schedule()
        self.assertFalse(flight.scheduled)

    def test_concurrent_index_requests_schedule_only_one_rebuild(self):
        server._initial_build_done.set()
        server._state["html"] = "<html></html>"
        fake_thread = MagicMock()
        with patch.object(server._threading, "Thread", return_value=fake_thread) as thread:
            for _ in range(20):
                self.assertEqual(server._route_index(), "<html></html>")
        thread.assert_called_once()
        self.assertFalse(thread.call_args.kwargs["daemon"])
        fake_thread.start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
