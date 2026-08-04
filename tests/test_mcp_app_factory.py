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
