import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp.shared.exceptions import McpError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import rag_mcp_http_server as remote
import rag_mcp_server as local


def _environment(**overrides):
    values = {
        "REMOTE_MCP_PUBLIC_URL": "https://zotero.example.ts.net",
        "REMOTE_MCP_GOOGLE_CLIENT_ID": "client-id",
        "REMOTE_MCP_GOOGLE_CLIENT_SECRET": "client-secret",
        "REMOTE_MCP_ALLOWED_GOOGLE_EMAILS": "Allowed@Example.com",
    }
    values.update(overrides)
    return values


def _tool_names(app):
    return {tool.name for tool in asyncio.run(app.list_tools())}


def test_remote_settings_are_loopback_only_and_hide_secret():
    settings = remote.RemoteMCPSettings.from_environment(_environment())

    assert settings.host == "127.0.0.1"
    assert settings.allowed_google_emails == frozenset({"allowed@example.com"})
    assert "client-secret" not in repr(settings)

    with pytest.raises(ValueError, match="loopback"):
        remote.RemoteMCPSettings.from_environment(_environment(REMOTE_MCP_HOST="0.0.0.0"))


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("REMOTE_MCP_PUBLIC_URL", "http://example.test", "https"),
        ("REMOTE_MCP_PUBLIC_URL", "https://example.test/mcp", "must not include"),
        ("REMOTE_MCP_ALLOWED_GOOGLE_EMAILS", "", "required"),
        ("REMOTE_MCP_ALLOWED_GOOGLE_EMAILS", ", ,", "at least one"),
        ("REMOTE_MCP_GOOGLE_CLIENT_SECRET", "", "required"),
        ("REMOTE_MCP_PORT", "not-a-port", "integer"),
        ("REMOTE_MCP_PORT", "70000", "between"),
    ],
)
def test_remote_settings_fail_closed(key, value, match):
    with pytest.raises(ValueError, match=match):
        remote.RemoteMCPSettings.from_environment(_environment(**{key: value}))


def test_remote_mcp_reuses_the_local_tool_registry(monkeypatch):
    provider_settings = {}

    def fake_provider(**kwargs):
        provider_settings.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(remote, "GoogleProvider", fake_provider)
    monkeypatch.setattr(
        remote,
        "get_access_token",
        lambda: SimpleNamespace(claims={"email": "allowed@example.com", "email_verified": True}),
    )
    app = remote.create_remote_mcp(remote.RemoteMCPSettings.from_environment(_environment()))

    assert _tool_names(app) == _tool_names(local.mcp)
    assert provider_settings["required_scopes"] == ["openid", "email"]


def test_google_allowlist_accepts_only_configured_verified_email(monkeypatch):
    middleware = remote.GoogleEmailAllowlistMiddleware(frozenset({"allowed@example.com"}))

    async def call(email, verified=True):
        monkeypatch.setattr(
            remote,
            "get_access_token",
            lambda: SimpleNamespace(claims={"email": email, "email_verified": verified})
            if email is not None else None,
        )

        async def next_call(context):
            return "accepted"

        return await middleware.on_request(object(), next_call)

    assert asyncio.run(call("Allowed@Example.com")) == "accepted"
    with pytest.raises(McpError, match="not authorized"):
        asyncio.run(call("somebody-else@example.com"))
    with pytest.raises(McpError, match="not authorized"):
        asyncio.run(call(None))
    with pytest.raises(McpError, match="not authorized"):
        asyncio.run(call("allowed@example.com", verified=False))


def test_main_runs_http_transport_without_changing_stdio_entrypoint(monkeypatch, capsys):
    settings = remote.RemoteMCPSettings.from_environment(_environment())
    calls = []
    fake_server = SimpleNamespace(run=lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["rag_mcp_http_server.py"])
    monkeypatch.setattr(remote.RemoteMCPSettings, "from_environment", lambda: settings)
    monkeypatch.setattr(remote, "create_remote_mcp", lambda value: fake_server)

    remote.main()

    assert calls == [{
        "transport": "http",
        "host": "127.0.0.1",
        "port": 8000,
        "path": "/mcp",
        "show_banner": False,
    }]
    assert "https://zotero.example.ts.net/mcp" in capsys.readouterr().out
