"""Authenticated HTTP entry point for the Zotero RAG MCP server.

The existing ``rag_mcp_server.py`` remains the stdio entry point.  This module
only supplies the network transport and authentication needed by a remote MCP
client; both entry points mount the same tool registry.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from fastmcp.server.auth.providers.google import GoogleProvider
from fastmcp.server.dependencies import get_access_token
from fastmcp.server.middleware import Middleware
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData, INVALID_REQUEST

from rag_mcp_server import create_mcp


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _required(environment: dict[str, str], name: str) -> str:
    value = environment.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required for the remote MCP server")
    return value


def _public_url(value: str) -> str:
    value = value.rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("REMOTE_MCP_PUBLIC_URL must be an absolute https:// URL")
    if parsed.path or parsed.query or parsed.fragment:
        raise ValueError("REMOTE_MCP_PUBLIC_URL must not include a path, query, or fragment")
    return value


@dataclass(frozen=True)
class RemoteMCPSettings:
    public_url: str
    google_client_id: str
    google_client_secret: str = field(repr=False)
    allowed_google_emails: frozenset[str]
    host: str = "127.0.0.1"
    port: int = 8000

    @classmethod
    def from_environment(cls, environment: dict[str, str] | None = None) -> "RemoteMCPSettings":
        env = dict(os.environ if environment is None else environment)
        users = frozenset(
            email.strip().casefold()
            for email in _required(env, "REMOTE_MCP_ALLOWED_GOOGLE_EMAILS").split(",")
            if email.strip()
        )
        if not users:
            raise ValueError("REMOTE_MCP_ALLOWED_GOOGLE_EMAILS must name at least one email address")
        host = env.get("REMOTE_MCP_HOST", "127.0.0.1").strip()
        if host not in _LOOPBACK_HOSTS:
            raise ValueError("REMOTE_MCP_HOST must be a loopback host; expose it through Tailscale Funnel")
        try:
            port = int(env.get("REMOTE_MCP_PORT", "8000"))
        except ValueError as exc:
            raise ValueError("REMOTE_MCP_PORT must be an integer") from exc
        if not 1 <= port <= 65535:
            raise ValueError("REMOTE_MCP_PORT must be between 1 and 65535")
        return cls(
            public_url=_public_url(_required(env, "REMOTE_MCP_PUBLIC_URL")),
            google_client_id=_required(env, "REMOTE_MCP_GOOGLE_CLIENT_ID"),
            google_client_secret=_required(env, "REMOTE_MCP_GOOGLE_CLIENT_SECRET"),
            allowed_google_emails=users,
            host=host,
            port=port,
        )


class GoogleEmailAllowlistMiddleware(Middleware):
    """Reject unverified Google addresses and users outside the allowlist."""

    def __init__(self, allowed_emails: frozenset[str]):
        self.allowed_emails = allowed_emails

    async def on_request(self, context, call_next):
        token = get_access_token()
        claims = token.claims if token else {}
        email = str(claims.get("email") or "").casefold()
        verified = claims.get("email_verified") in {True, "true", "True"}
        if not verified or not email or email not in self.allowed_emails:
            raise McpError(ErrorData(code=INVALID_REQUEST, message="Google account is not authorized"))
        return await call_next(context)


def create_remote_mcp(settings: RemoteMCPSettings):
    auth = GoogleProvider(
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        base_url=settings.public_url,
        required_scopes=["openid", "email"],
    )
    return create_mcp(
        auth=auth,
        middleware=[GoogleEmailAllowlistMiddleware(settings.allowed_google_emails)],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Zotero RAG as an authenticated HTTP MCP server")
    parser.parse_args()
    settings = RemoteMCPSettings.from_environment()
    server = create_remote_mcp(settings)
    print(
        f"[zotero-rag-remote] serving {settings.public_url}/mcp via "
        f"http://{settings.host}:{settings.port}/mcp",
        flush=True,
    )
    server.run(
        transport="http",
        host=settings.host,
        port=settings.port,
        path="/mcp",
        show_banner=False,
    )


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    main()
