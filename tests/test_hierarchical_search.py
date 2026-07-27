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

    def test_summary_routing_only_queries_llm_summaries(self):
        summary_collection = Mock()
        summary_collection.query.return_value = {
            "metadatas": [[{"itemKey": "LLM", "title": "LLM summary"}]],
        }
        client = Mock()
        client.get_collection.return_value = summary_collection
        paragraphs = Mock()
        paragraphs._embedding_function.return_value = [[0.1, 0.2]]
        paragraphs._chroma_client = client
        # This exercises the legacy (non-v2) summary routing path, so pin the
        # flag off — the repo .env enables v2 globally after cutover.
        with patch.dict(os.environ, {"HIERARCHICAL_SEARCH_V2_ENABLE": "0"}), patch.object(
            rag_mcp_server, "_col", return_value=paragraphs
        ), patch.object(
            rag_mcp_server, "rag_search", return_value={"results": []}
        ), patch.object(
            rag_mcp_server, "load_item_summary", return_value=None
        ), patch.object(
            rag_mcp_server, "get_disabled_summary_keys", return_value=set()
        ):
            response = rag_mcp_server.hierarchical_search("gift exchange", auto_expand=False)

        self.assertEqual(response["candidate_items"][0]["item_key"], "LLM")
        self.assertEqual(summary_collection.query.call_count, 2)
        for call in summary_collection.query.call_args_list:
            self.assertEqual(call.kwargs["where"], {"summary_kind": "llm"})
        self.assertIn("report_summary_quality", response["reporting_obligation"])

    def test_disabled_summary_is_not_used_for_routing(self):
        item_collection = Mock()
        item_collection.query.return_value = {
            "metadatas": [[{"itemKey": "ITEM", "title": "item"}]],
        }
        section_collection = Mock()
        section_collection.query.return_value = {
            "metadatas": [[{
                "itemKey": "ITEM", "section_id": "w0", "title": "section",
            }]],
        }
        client = Mock()
        client.get_collection.side_effect = [item_collection, section_collection]
        paragraphs = Mock()
        paragraphs._embedding_function.return_value = [[0.1, 0.2]]
        paragraphs._chroma_client = client
        with patch.object(rag_mcp_server, "_col", return_value=paragraphs), patch.object(
            rag_mcp_server, "rag_search", return_value={"results": []}
        ), patch.object(
            rag_mcp_server, "get_disabled_summary_keys",
            return_value={("ITEM", ""), ("ITEM", "w0")},
        ):
            response = rag_mcp_server.hierarchical_search("query", auto_expand=False)
        self.assertEqual(response["candidate_items"], [])

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
            rag_mcp_server, "load_item_summary", return_value=None
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


if __name__ == "__main__":
    unittest.main()
