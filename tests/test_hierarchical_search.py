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

        with patch.object(rag_mcp_server, "_col", return_value=paragraphs), patch.object(
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
        item_hit = {"id": "chunk-child", "text": "child", "meta": {"itemKey": "ITEM"}}
        direct_hit = {"id": "chunk-other", "text": "other", "meta": {"itemKey": "OTHER"}}
        with patch.dict(os.environ, {"HIERARCHICAL_SEARCH_V2_ENABLE": "1"}), patch.object(
            rag_mcp_server, "_col", return_value=paragraphs
        ), patch.object(
            rag_mcp_server, "get_node_descendant_chunks", return_value=["chunk-child"]
        ), patch.object(
            rag_mcp_server, "rag_search", side_effect=[{"results": [item_hit]}, {"results": [direct_hit]}]
        ) as mock_search, patch.object(
            rag_mcp_server, "load_item_summary", return_value=None
        ):
            response = rag_mcp_server.hierarchical_search("query", auto_expand=False)
        self.assertEqual(response["candidate_nodes"][0]["node_id"], "dn:chapter")
        self.assertEqual(response["results"][0]["id"], "chunk-child")
        self.assertIn("summary_descendant", response["results"][0]["retrieval_paths"])
        self.assertEqual(mock_search.call_count, 2)


if __name__ == "__main__":
    unittest.main()
