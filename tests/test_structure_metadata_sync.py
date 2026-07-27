"""Chunk metadata must follow the structure when the structure is rebuilt.

Node ids are derived from content, so a rebuild renames them. Only the ingest
path stamped the copy carried in chunk metadata, and retrieval reads that copy:
rag_mcp_server filters candidates with {"node_id": {"$in": ...}} using ids from
the database. After a rebuild those matched nothing, so the leaf route -- the
highest-weighted of the three fused routes -- returned empty and search
degraded in silence. 213,748 chunks (44.7%) were in that state on 2026-07-28.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from structure_metadata_sync import (  # noqa: E402
    desired_chunk_metadata, orphaned_chunk_ids, stale_chunk_updates,
)


NODES = [
    {"node_id": "dn:new", "zone": "endnote", "summary_policy": "exclude",
     "retrieval_policy": "normal", "citation_policy": "extract"},
]
MAPPING = {"dn:new": ["c1", "c2"]}


class SyncTests(unittest.TestCase):
    def test_a_renamed_node_is_restamped(self):
        current = {"c1": {"node_id": "dn:old", "zone": "endnote", "summary_policy": "exclude",
                          "retrieval_policy": "normal", "citation_policy": "extract"}}
        updates = stale_chunk_updates(current, desired_chunk_metadata(NODES, MAPPING))
        self.assertEqual(updates["c1"]["node_id"], "dn:new")

    def test_an_agreeing_chunk_is_left_alone(self):
        # Rewriting every chunk on every rebuild would cost more than the
        # rebuild and would hide how far things had actually drifted.
        current = {"c1": {"node_id": "dn:new", "zone": "endnote", "summary_policy": "exclude",
                          "retrieval_policy": "normal", "citation_policy": "extract"}}
        self.assertEqual(stale_chunk_updates(current, desired_chunk_metadata(NODES, MAPPING)), {})

    def test_policy_drift_alone_is_repaired(self):
        # A fresh node_id beside a stale retrieval_policy routes correctly and
        # is then filtered out, so the keys must move together.
        current = {"c1": {"node_id": "dn:new", "zone": "endnote", "summary_policy": "exclude",
                          "retrieval_policy": "explicit_only", "citation_policy": "extract"}}
        updates = stale_chunk_updates(current, desired_chunk_metadata(NODES, MAPPING))
        self.assertEqual(updates["c1"]["retrieval_policy"], "normal")

    def test_unrelated_metadata_is_preserved(self):
        current = {"c1": {"node_id": "dn:old", "title": "Chapter One", "page": 4,
                          "zone": "endnote", "summary_policy": "exclude",
                          "retrieval_policy": "normal", "citation_policy": "extract"}}
        updates = stale_chunk_updates(current, desired_chunk_metadata(NODES, MAPPING))
        self.assertEqual(updates["c1"]["title"], "Chapter One")
        self.assertEqual(updates["c1"]["page"], 4)

    def test_a_chunk_absent_from_the_index_is_not_invented(self):
        updates = stale_chunk_updates({}, desired_chunk_metadata(NODES, MAPPING))
        self.assertEqual(updates, {})

    def test_a_mapping_to_a_missing_node_is_skipped(self):
        desired = desired_chunk_metadata(NODES, {"dn:vanished": ["c9"]})
        self.assertEqual(desired, {})

    def test_chunks_the_structure_does_not_claim_are_reported(self):
        # Reported rather than cleared: the cause is upstream, and blanking the
        # metadata would hide that extraction produced chunks the structure
        # builder never saw.
        desired = desired_chunk_metadata(NODES, MAPPING)
        self.assertEqual(orphaned_chunk_ids(["c1", "c2", "c99"], desired), ["c99"])


if __name__ == "__main__":
    unittest.main()
