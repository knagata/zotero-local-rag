import asyncio
import sys
from pathlib import Path

import pytest

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
    assert "import_zotero_candidate" in _tool_names(first)
    import_tool = next(
        tool for tool in asyncio.run(first.list_tools())
        if tool.name == "import_zotero_candidate"
    )
    assert import_tool.annotations.readOnlyHint is False
    assert import_tool.annotations.destructiveHint is True


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


def test_zotero_import_uses_client_confirmation_without_second_phrase(monkeypatch):
    candidate = {
        "source": "ndl", "title": "New book",
        "landing_url": "https://ndlsearch.ndl.go.jp/books/example",
    }
    calls = []

    async def fake_inspect(value):
        return {"zotero_duplicates": [], "candidate": value, "writes_performed": False}

    def fake_create(value, duplicates, **kwargs):
        calls.append(("propose", value, duplicates, kwargs))
        return {"proposal_id": "P1", "approval_phrase": "IMPORT P1", "writes_performed": False}

    def fake_claim(proposal_id, phrase):
        calls.append(("claim", proposal_id, phrase))
        return {"candidate": candidate, "import_mode": "metadata", "allow_duplicate": False}

    async def fake_execute(row):
        calls.append(("write", row))
        return {"item_key": "ITEM1", "import_mode": "metadata", "writes_performed": True}

    monkeypatch.setattr(server, "inspect_open_text_candidate", fake_inspect)
    monkeypatch.setattr(server, "create_import_proposal", fake_create)
    monkeypatch.setattr(server, "claim_import_proposal", fake_claim)
    monkeypatch.setattr(server, "acquire_import_write_lock", lambda: 42)
    monkeypatch.setattr(server, "release_import_write_lock", lambda _descriptor: None)
    monkeypatch.setattr(server, "execute_import", fake_execute)
    monkeypatch.setattr(server, "finish_import_proposal", lambda *args: calls.append(("finish", args)))

    approved = asyncio.run(server.import_zotero_candidate(candidate))

    assert approved["writes_performed"] is True
    assert approved["indexing"]["status"] == "not_applicable"
    assert [call[0] for call in calls] == ["propose", "claim", "write", "finish"]


def test_exact_duplicate_cannot_be_overridden_and_write_lock_is_released(monkeypatch):
    candidate = {
        "source": "jstage", "title": "Same article",
        "landing_url": "https://example.test/article",
        "identifiers": {"doi": "10.1234/same"},
    }
    events = []

    async def fake_inspect(_value):
        return {"zotero_duplicates": [{
            "key": "EXISTING", "match_reason": "identifier",
            "url": "https://example.test/article",
        }]}

    async def forbidden_write(_row):
        raise AssertionError("an exact duplicate must not be written")

    monkeypatch.setattr(server, "inspect_open_text_candidate", fake_inspect)
    monkeypatch.setattr(server, "execute_import", forbidden_write)
    monkeypatch.setattr(server, "acquire_import_write_lock", lambda: events.append("acquire") or 7)
    monkeypatch.setattr(server, "release_import_write_lock", lambda fd: events.append(("release", fd)))
    monkeypatch.setattr(server, "finish_import_proposal", lambda *args: events.append(("finish", args[1])))

    with pytest.raises(ValueError, match="exact identifier"):
        asyncio.run(server._execute_claimed_import("P1", {
            "candidate": candidate, "import_mode": "metadata", "allow_duplicate": True,
        }))

    assert events == ["acquire", ("finish", "failed"), ("release", 7)]
