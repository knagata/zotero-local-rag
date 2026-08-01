from __future__ import annotations

import importlib
import os
import unittest
from unittest import mock


def _reload_with_env(env: dict) -> object:
    """Re-import update_citations with a controlled environment.

    API_BASE/API_PREFIX are module-level constants resolved at import time
    from os.environ, so exercising the default requires a fresh import with a
    clean environment rather than monkeypatching an already-imported module.
    """
    with mock.patch.dict(os.environ, env, clear=True), \
         mock.patch("src.env_utils.load_dotenv_native"):
        import src.update_citations as module
        return importlib.reload(module)


class ApiDefaultsTests(unittest.TestCase):
    """update_citations.py must default to Zotero's real local API port.

    Traced from a real run (2026-08-01, maintenance log) where this module's
    own hardcoded default (127.0.0.1:8080, empty prefix) silently pointed at
    nothing -- Zotero's local API is 23119, matching what
    zotero_source_localapi.py and work_identity.py already default to. With
    no ZOTERO_LOCAL_API_BASE override, the citation step connected to the
    wrong port, got a connection-refused, and reported 0 items as success.
    """

    def test_defaults_match_the_rest_of_the_codebase(self):
        module = _reload_with_env({})
        self.assertEqual(module.API_BASE, "http://127.0.0.1:23119/api")
        self.assertEqual(module.API_PREFIX, "users/0")

    def test_explicit_env_override_still_wins(self):
        module = _reload_with_env({
            "ZOTERO_LOCAL_API_BASE": "http://127.0.0.1:9999/api",
            "ZOTERO_LOCAL_API_PREFIX": "users/5",
        })
        self.assertEqual(module.API_BASE, "http://127.0.0.1:9999/api")
        self.assertEqual(module.API_PREFIX, "users/5")


class GetAllItemsConnectionFailureTests(unittest.TestCase):
    """A connection failure must not look identical to zero library items."""

    def setUp(self):
        self.module = _reload_with_env({})

    def test_a_failed_first_request_raises_instead_of_returning_empty(self):
        # Before this fix, _zotero_request returning None (connection
        # refused, timeout, etc.) on the very first page was indistinguishable
        # from a successful response with an empty item list: both broke the
        # loop and reported "Found 0 potential items" / completed as success.
        with mock.patch.object(self.module, "_zotero_request", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "Could not fetch items from the Zotero Local API"):
                self.module.get_all_items()

    def test_a_genuinely_empty_library_still_returns_an_empty_list(self):
        with mock.patch.object(self.module, "_zotero_request", return_value=[]):
            self.assertEqual(self.module.get_all_items(), [])

    def test_a_failure_on_a_later_page_stops_without_raising(self):
        # Only the first page failing is ambiguous with "nothing here"; a
        # later page failing after some items were already fetched is a
        # partial-result case the existing break-on-falsy behavior already
        # handles reasonably, not a mode this fix needs to change.
        calls = {"n": 0}
        item = {"key": "K1", "data": {"itemType": "journalArticle"}}

        def fake_request(_endpoint, params=None, **_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return [item] * 100
            return None

        with mock.patch.object(self.module, "_zotero_request", side_effect=fake_request):
            result = self.module.get_all_items()
        self.assertEqual(len(result), 100)


if __name__ == "__main__":
    unittest.main()
