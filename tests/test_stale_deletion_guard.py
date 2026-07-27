"""The stale-attachment deletion must not act on a short Zotero listing.

`index_from_zotero` derives what to delete as (manifest keys - enumerated
keys). The enumeration pages until a batch looks final, with no Total-Results
comparison and no floor, so a truncated listing makes live attachments look
retired -- and they were deleted from Chroma unconditionally, with no
confirmation against Zotero (2026-07-28).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # index_from_zotero imports its siblings flat

from src.index_from_zotero import STALE_DELETE_MAX_RATIO, STALE_DELETE_MIN_KEYS


def _would_delete(manifest_size: int, stale_count: int) -> bool:
    """Mirror of the guard, so the policy is pinned independently of the loop."""
    limit = max(STALE_DELETE_MIN_KEYS, int(manifest_size * STALE_DELETE_MAX_RATIO))
    return stale_count <= limit


class StaleDeletionGuardTests(unittest.TestCase):
    def test_an_ordinary_removal_still_proceeds(self):
        # Deleting a few items from a 586-attachment library is the normal case
        # and must not need an override.
        self.assertTrue(_would_delete(586, 3))

    def test_a_truncated_listing_cannot_empty_the_index(self):
        # One short page of an enumeration that returns 200 at a time.
        self.assertFalse(_would_delete(586, 386))

    def test_a_total_enumeration_failure_is_refused(self):
        self.assertFalse(_would_delete(586, 586))

    def test_a_small_library_keeps_an_absolute_floor(self):
        # 5% of 20 is one attachment; without the floor a tiny library could
        # not retire anything, and the operator would learn to bypass the guard.
        self.assertTrue(_would_delete(20, STALE_DELETE_MIN_KEYS))
        self.assertFalse(_would_delete(20, STALE_DELETE_MIN_KEYS + 1))

    def test_the_boundary_is_inclusive(self):
        limit = int(586 * STALE_DELETE_MAX_RATIO)
        self.assertTrue(_would_delete(586, limit))
        self.assertFalse(_would_delete(586, limit + 1))


if __name__ == "__main__":
    unittest.main()
