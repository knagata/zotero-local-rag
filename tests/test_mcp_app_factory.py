import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import rag_mcp_server as server


def _tool_names(app):
    return {tool.name for tool in asyncio.run(app.list_tools())}


def test_create_mcp_preserves_tools_and_returns_distinct_servers():
    first = server.create_mcp()
    second = server.create_mcp()

    assert first is not second
    assert _tool_names(first) == _tool_names(server.mcp)
    assert _tool_names(second) == _tool_names(server.mcp)
    assert "rag_search" in _tool_names(first)
    assert "search_open_texts" in _tool_names(first)
    assert "inspect_open_text_candidate" in _tool_names(first)


def test_external_discovery_tools_are_read_only_and_check_duplicates(monkeypatch):
    async def fake_search(query, sources, limit, search_mode):
        return {
            "query": query, "sources": sources, "limit": limit,
            "search_mode": search_mode, "writes_performed": False,
        }

    class FakeZotero:
        async def _get_json(self, path, params):
            if params["q"] == "10.1234/example":
                return [{"data": {
                    "key": "DOI1", "itemType": "journalArticle", "title": "Other title",
                    "DOI": "10.1234/example",
                }}]
            return [
                {"data": {"key": "ATT1", "itemType": "attachment", "title": "Same title"}},
                {"data": {"key": "DUP1", "itemType": "book", "title": "Same title"}},
            ]

        @staticmethod
        def _unwrap_item(item):
            return item["data"]["key"], item["data"]

    monkeypatch.setattr(server, "search_external_texts", fake_search)
    monkeypatch.setattr(server, "_z_api", lambda: FakeZotero())

    searched = asyncio.run(server.search_open_texts("topic", ["ndl"], 3))
    inspected = asyncio.run(server.inspect_open_text_candidate({
        "source": "ndl", "title": "Same title", "landing_url": "https://example.test",
    }))
    identifier_match = asyncio.run(server.inspect_open_text_candidate({
        "source": "jstage", "title": "Different title", "landing_url": "https://example.test/2",
        "identifiers": {"doi": "10.1234/example", "uri": "https://ignored.example"},
    }))
    assert searched["writes_performed"] is False
    assert searched["search_mode"] == "auto"
    assert inspected["zotero_duplicates"][0]["key"] == "DUP1"
    assert inspected["writes_performed"] is False
    assert identifier_match["zotero_duplicates"][0]["match_reason"] == "identifier"
    try:
        asyncio.run(server.search_open_texts("   "))
    except ValueError as exc:
        assert "must not be empty" in str(exc)
    else:
        raise AssertionError("empty external query must be rejected")
