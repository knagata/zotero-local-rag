from __future__ import annotations

import sys
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import rag_mcp_server  # noqa: E402


class HierarchicalSearchTests(unittest.TestCase):
    def test_v2_is_the_default_when_the_flag_is_unset(self):
        paragraphs = Mock()
        expected = {"results": [], "candidate_items": []}
        with patch.dict(os.environ, {}, clear=True), patch.object(
            rag_mcp_server, "_col", return_value=paragraphs
        ), patch.object(
            rag_mcp_server, "_hierarchical_search_v2", return_value=expected
        ) as v2:
            response = rag_mcp_server.hierarchical_search("query", auto_expand=False)
        self.assertIs(response, expected)
        v2.assert_called_once()

    def test_rag_search_partitions_leaf_filter_into_chroma_safe_batches(self):
        paragraphs = Mock()
        paragraphs._embedding_function.return_value = [[0.1, 0.2]]
        paragraphs.count.return_value = 3
        responses = [
            {"ids": [[f"chunk-{index}"]], "documents": [["x" * 250]],
             "metadatas": [[{"node_id": f"n{index}", "retrieval_policy": "normal"}]],
             "distances": [[0.1]]}
            for index in range(3)
        ]
        paragraphs.query.side_effect = responses
        with patch.object(rag_mcp_server, "_col", return_value=paragraphs), \
             patch.object(rag_mcp_server, "_check_indexing_lock", return_value=(False, None)):
            result = rag_mcp_server.rag_search(
                "query", k=5, include_leaf_ids=[f"n{index}" for index in range(205)],
                auto_expand=False, hybrid=False,
            )
        self.assertEqual(paragraphs.query.call_count, 3)
        batch_sizes = [
            len(call.kwargs["where"]["$and"][-1]["node_id"]["$in"])
            for call in paragraphs.query.call_args_list
        ]
        self.assertEqual(batch_sizes, [100, 100, 5])
        self.assertEqual(len(result["results"]), 3)

    def test_rank_is_computed_across_merged_batches_not_within_each_one(self):
        """A batch-2 top hit must not tie a batch-1 top hit at rank 1.

        include_leaf_ids over 100 ids is split into multiple Chroma queries,
        one per batch (partition_leaf_ids caps each at 100). RRF rank used to
        come from each batch's own position, so the single best hit in every
        batch scored the same rank-1 contribution no matter how much worse it
        was than hits already ranked lower in another batch. 48 of 540 items
        in the library have more than 100 leaf nodes and so route through more
        than one batch (2026-07-28, found in code review).
        """
        paragraphs = Mock()
        paragraphs._embedding_function.return_value = [[0.1, 0.2]]
        # Batch 1: three hits, distances 0.05/0.06/0.07 (all close, all better
        # than anything in batch 2). Batch 2: one hit, distance 0.5 (far
        # worse) -- but it would have been "rank 1 of its batch" under the old
        # per-batch scheme.
        responses = [
            {
                "ids": [["good-1", "good-2", "good-3"]],
                "documents": [["a" * 250, "b" * 250, "c" * 250]],
                "metadatas": [[
                    {"node_id": "n1", "retrieval_policy": "normal"},
                    {"node_id": "n2", "retrieval_policy": "normal"},
                    {"node_id": "n3", "retrieval_policy": "normal"},
                ]],
                "distances": [[0.05, 0.06, 0.07]],
            },
            {
                "ids": [["far-1"]],
                "documents": [["d" * 250]],
                "metadatas": [[{"node_id": "n101", "retrieval_policy": "normal"}]],
                "distances": [[0.5]],
            },
        ]
        paragraphs.query.side_effect = responses
        with patch.object(rag_mcp_server, "_col", return_value=paragraphs), \
             patch.object(rag_mcp_server, "_check_indexing_lock", return_value=(False, None)):
            result = rag_mcp_server.rag_search(
                "query", k=4,
                include_leaf_ids=[f"n{index}" for index in range(101)],
                auto_expand=False, hybrid=False,
            )
        ids_in_order = [row["id"] for row in result["results"]]
        # The far batch-2 hit must sort behind every batch-1 hit, and the
        # batch-1 hits must keep their distance order -- neither is possible
        # if rank came from each batch's own h_idx.
        self.assertEqual(ids_in_order, ["good-1", "good-2", "good-3", "far-1"])

    def test_rag_search_deepens_once_when_post_filters_underfill_results(self):
        """Short/excluded nearest neighbours must not permanently consume k."""
        paragraphs = Mock()
        paragraphs._embedding_function.return_value = [[0.1, 0.2]]

        def response(prefix, text):
            return {
                "ids": [[f"{prefix}-{index}" for index in range(10)]],
                "documents": [[text for _ in range(10)]],
                "metadatas": [[{"retrieval_policy": "normal"} for _ in range(10)]],
                "distances": [[index / 100 for index in range(10)]],
            }

        # The fixed initial 5*k query contains only fragments.  The deeper
        # query reaches full paragraphs, so the public k=2 contract is met.
        paragraphs.query.side_effect = [
            response("fragment", "too short"),
            response("full", "evidence " * 40),
        ]
        with patch.object(rag_mcp_server, "_col", return_value=paragraphs), \
             patch.object(rag_mcp_server, "_check_indexing_lock", return_value=(False, None)):
            result = rag_mcp_server.rag_search(
                "query", k=2, auto_expand=False, hybrid=False,
            )

        self.assertEqual([row["id"] for row in result["results"]], ["full-0", "full-1"])
        self.assertEqual(
            [call.kwargs["n_results"] for call in paragraphs.query.call_args_list],
            [10, 20],
        )

    def test_retired_route_flag_cannot_disable_v3_routing(self):
        paragraphs = Mock()
        expected = {"results": [], "candidate_nodes": []}
        with patch.dict(os.environ, {"HIERARCHICAL_SEARCH_V2_ENABLE": "0"}), patch.object(
            rag_mcp_server, "_col", return_value=paragraphs
        ), patch.object(
            rag_mcp_server, "_hierarchical_search_v2", return_value=expected,
        ) as v3_search:
            response = rag_mcp_server.hierarchical_search("gift exchange", auto_expand=False)

        self.assertIs(response, expected)
        v3_search.assert_called_once()

    def test_v2_routes_node_hit_to_descendant_chunks_and_keeps_direct_search(self):
        summary_collection = Mock()
        summary_collection.query.return_value = {
            "metadatas": [[{
                "itemKey": "ITEM", "node_id": "dn:chapter", "title": "Chapter",
                "node_type": "chapter", "depth": 2,
            }]],
            "documents": [["Chapter summary"]],
        }
        client = Mock()
        client.get_collection.return_value = summary_collection
        paragraphs = Mock()
        paragraphs._embedding_function.return_value = [[0.1, 0.2]]
        paragraphs._chroma_client = client
        item_hit = {"id": "chunk-child", "text": "child",
                    "meta": {"itemKey": "ITEM", "title": "Real Book Title", "year": 1990}}
        direct_hit = {"id": "chunk-other", "text": "other", "meta": {"itemKey": "OTHER"}}
        with patch.dict(os.environ, {"HIERARCHICAL_SEARCH_V2_ENABLE": "1"}), patch.object(
            rag_mcp_server, "_col", return_value=paragraphs
        ), patch.object(
            rag_mcp_server, "get_node_descendant_chunks", return_value=["chunk-child"]
        ), patch.object(
            rag_mcp_server, "get_node_descendant_leaf_ids", return_value=["dn:leaf"]
        ), patch.object(
            rag_mcp_server, "rag_search", side_effect=[
                {"results": [item_hit]}, {"results": [item_hit]}, {"results": [direct_hit]},
            ]
        ) as mock_search, patch.object(
            rag_mcp_server, "get_searchable_document_node_ids",
            side_effect=lambda ids: set(ids),
        ), patch.object(
            rag_mcp_server, "get_item_root_summaries", return_value={},
        ):
            response = rag_mcp_server.hierarchical_search("query", auto_expand=False)
        self.assertEqual(response["candidate_nodes"][0]["node_id"], "dn:chapter")
        self.assertEqual(response["results"][0]["id"], "chunk-child")
        self.assertIn("leaf", response["results"][0]["retrieval_paths"])
        self.assertEqual(response["results"][0]["routed_by_node_ids"], ["dn:chapter"])
        self.assertEqual(mock_search.call_count, 3)
        self.assertEqual(mock_search.call_args_list[0].kwargs["include_leaf_ids"], ["dn:leaf"])
        # R8-1: candidate_items bibliography resolves from chunk metadata (real
        # item title/year), not the __sum_node node (chapter) heading.
        item_candidate = next(c for c in response["candidate_items"] if c["item_key"] == "ITEM")
        self.assertEqual(item_candidate["title"], "Real Book Title")
        self.assertEqual(item_candidate["year"], 1990)

    def test_item_summary_uses_v3_item_root(self):
        with patch.object(
            rag_mcp_server, "get_item_root_summary",
            return_value={"node_type": "item_root", "summary": "whole item"},
        ):
            result = rag_mcp_server.get_item_summary("ITEM")
        self.assertEqual(result["summary"]["summary"], "whole item")

    def test_disabled_stale_summary_embedding_cannot_route_results(self):
        summary_collection = Mock()
        summary_collection.query.return_value = {
            "metadatas": [[{
                "itemKey": "ITEM", "node_id": "disabled:node", "title": "Chapter",
            }]],
            "documents": [["disabled summary"]],
        }
        client = Mock()
        client.get_collection.return_value = summary_collection
        paragraphs = Mock()
        paragraphs._embedding_function.return_value = [[0.1, 0.2]]
        paragraphs._chroma_client = client
        with patch.object(
            rag_mcp_server, "get_searchable_document_node_ids", return_value=set(),
        ), patch.object(
            rag_mcp_server, "rag_search", return_value={"results": []},
        ) as search:
            response = rag_mcp_server._hierarchical_search_v2(
                ["query"], k=5, k_items=5, where=None, include_direct=False,
                return_summaries=False, paragraph_collection=paragraphs,
            )
        self.assertEqual(response["candidate_nodes"], [])
        self.assertEqual(response["candidate_items"], [])
        search.assert_not_called()

    def test_v2_routes_same_item_search_by_aggregated_node_rrf(self):
        """Two lower-ranked nodes for B beat A's first single node."""
        summary_collection = Mock()
        summary_collection.query.return_value = {
            "metadatas": [[
                {"itemKey": "A", "node_id": "a:1", "title": "A"},
                {"itemKey": "B", "node_id": "b:1", "title": "B one"},
                {"itemKey": "B", "node_id": "b:2", "title": "B two"},
            ]],
            "documents": [["A", "B one", "B two"]],
        }
        client = Mock()
        client.get_collection.return_value = summary_collection
        paragraphs = Mock()
        paragraphs._embedding_function.return_value = [[0.1, 0.2]]
        paragraphs._chroma_client = client
        with patch.object(
            rag_mcp_server, "get_node_descendant_chunks", return_value=[]
        ), patch.object(
            rag_mcp_server, "get_node_descendant_leaf_ids", return_value=[]
        ), patch.object(
            rag_mcp_server, "get_searchable_document_node_ids",
            side_effect=lambda ids: set(ids),
        ), patch.object(
            rag_mcp_server, "rag_search", return_value={"results": []}
        ) as mock_search:
            rag_mcp_server._hierarchical_search_v2(
                ["query"], k=1, k_items=1, where=None, include_direct=False,
                return_summaries=False, paragraph_collection=paragraphs,
            )

        self.assertEqual(mock_search.call_count, 1)
        self.assertEqual(mock_search.call_args.kwargs["include_item_keys"], ["B"])


if __name__ == "__main__":
    unittest.main()
