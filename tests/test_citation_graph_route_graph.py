from __future__ import annotations

import unittest
from unittest import mock

import citation_graph.server as server


class RouteGraphNotReadyTests(unittest.TestCase):
    """/api/graph must not crash while the graph hasn't been built yet.

    _rebuild_done.wait(timeout=120) can return without the event ever being
    set -- the initial build not finished (large library, or DB contention
    from a concurrent rebuild_document_structure.py run), or simply the
    120s timeout expiring. Traced from a real crash (2026-08-03):
    dict(_state["graph"]) with graph still None raised
    "TypeError: 'NoneType' object is not iterable" as an unhandled 500,
    unlike the sibling /api/semantic-layout route which already checks for
    None and returns a 503.
    """

    def setUp(self):
        self._original_graph = server._state.get("graph")
        self._original_event_state = server._rebuild_done.is_set()

    def tearDown(self):
        server._state["graph"] = self._original_graph
        if self._original_event_state:
            server._rebuild_done.set()
        else:
            server._rebuild_done.clear()

    def test_returns_503_instead_of_crashing_when_graph_is_none(self):
        server._state["graph"] = None
        # Simulate the real failure trigger -- wait() returning (timeout
        # expired, or the event just isn't set yet) without graph ever
        # becoming non-None -- without actually blocking for 120s.
        with mock.patch.object(server._rebuild_done, "wait", return_value=False):
            response = server._route_graph()

        self.assertEqual(response.status_code, 503)
        self.assertIn(b"no_graph", response.body)

    def test_returns_the_graph_once_built(self):
        server._state["graph"] = {"nodes": [{"id": "a"}], "edges": []}
        server._state["cache_hit"] = True
        with mock.patch.object(server._rebuild_done, "wait", return_value=True):
            response = server._route_graph()

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'"cache_hit":true', response.body)
        self.assertIn(b'"id":"a"', response.body)


if __name__ == "__main__":
    unittest.main()
