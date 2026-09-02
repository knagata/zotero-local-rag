"""Google-authenticated browser proxy for the local Citation Graph.

The ordinary graph server remains an unauthenticated loopback-only application
at ``http://127.0.0.1:7234``.  This process is a separate loopback-only entry
point intended to sit behind Tailscale Funnel; it authenticates a browser with
Google and proxies accepted requests to the ordinary local server.
"""
from __future__ import annotations

import argparse
import hmac
import os
import secrets
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode, urlsplit

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from starlette.middleware.sessions import SessionMiddleware

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from citation_graph.admin_routes import register_admin_routes
from src.env_utils import environment_with_saved_dotenv

GOOGLE_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
})


def _required(environment: dict[str, str], name: str) -> str:
    value = environment.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required for the remote Citation Graph")
    return value


def _https_origin(value: str) -> str:
    value = value.rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("CITATION_GRAPH_PUBLIC_URL must be an absolute https:// URL")
    if parsed.path or parsed.query or parsed.fragment:
        raise ValueError("CITATION_GRAPH_PUBLIC_URL must not include a path, query, or fragment")
    return value


def _loopback_url(value: str) -> str:
    value = value.rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme != "http" or parsed.hostname not in _LOOPBACK_HOSTS:
        raise ValueError("CITATION_GRAPH_LOCAL_URL must be an http:// loopback URL")
    if parsed.path or parsed.query or parsed.fragment:
        raise ValueError("CITATION_GRAPH_LOCAL_URL must not include a path, query, or fragment")
    return value


@dataclass(frozen=True)
class CitationGraphRemoteSettings:
    public_url: str
    google_client_id: str
    google_client_secret: str = field(repr=False)
    session_secret: str = field(repr=False)
    allowed_google_emails: frozenset[str]
    local_url: str = "http://127.0.0.1:7234"
    host: str = "127.0.0.1"
    port: int = 7244

    @classmethod
    def from_environment(
        cls, environment: dict[str, str] | None = None,
    ) -> CitationGraphRemoteSettings:
        env = dict(os.environ if environment is None else environment)
        allowed = frozenset(
            email.strip().casefold()
            for email in _required(env, "REMOTE_MCP_ALLOWED_GOOGLE_EMAILS").split(",")
            if email.strip()
        )
        if not allowed:
            raise ValueError("REMOTE_MCP_ALLOWED_GOOGLE_EMAILS must name at least one email")
        secret = _required(env, "CITATION_GRAPH_SESSION_SECRET")
        if len(secret) < 32:
            raise ValueError("CITATION_GRAPH_SESSION_SECRET must contain at least 32 characters")
        host = env.get("CITATION_GRAPH_REMOTE_HOST", "127.0.0.1").strip()
        if host not in _LOOPBACK_HOSTS:
            raise ValueError("CITATION_GRAPH_REMOTE_HOST must be a loopback host")
        try:
            port = int(env.get("CITATION_GRAPH_REMOTE_PORT", "7244"))
        except ValueError as exc:
            raise ValueError("CITATION_GRAPH_REMOTE_PORT must be an integer") from exc
        if not 1 <= port <= 65535:
            raise ValueError("CITATION_GRAPH_REMOTE_PORT must be between 1 and 65535")
        return cls(
            public_url=_https_origin(_required(env, "CITATION_GRAPH_PUBLIC_URL")),
            google_client_id=_required(env, "REMOTE_MCP_GOOGLE_CLIENT_ID"),
            google_client_secret=_required(env, "REMOTE_MCP_GOOGLE_CLIENT_SECRET"),
            session_secret=secret,
            allowed_google_emails=allowed,
            local_url=_loopback_url(
                env.get("CITATION_GRAPH_LOCAL_URL", "http://127.0.0.1:7234")
            ),
            host=host,
            port=port,
        )

    @property
    def callback_url(self) -> str:
        return f"{self.public_url}/auth/callback"


def _safe_next(value: str | None) -> str:
    return value if value and value.startswith("/") and not value.startswith("//") else "/"


def _authenticated_email(request: Request, allowed: frozenset[str]) -> str | None:
    email = str(request.session.get("email") or "").casefold()
    expires_at = request.session.get("expires_at")
    if not email or email not in allowed or not isinstance(expires_at, (int, float)):
        return None
    if expires_at <= time.time():
        request.session.clear()
        return None
    return email


async def _google_identity(code: str, verifier: str, settings: CitationGraphRemoteSettings) -> dict:
    async with httpx.AsyncClient(timeout=15) as client:
        token_response = await client.post(GOOGLE_TOKEN_URL, data={
            "code": code,
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "redirect_uri": settings.callback_url,
            "grant_type": "authorization_code",
            "code_verifier": verifier,
        })
        token_response.raise_for_status()
        id_token = str(token_response.json().get("id_token") or "")
        if not id_token:
            raise ValueError("Google did not return an ID token")
        identity_response = await client.get(GOOGLE_TOKENINFO_URL, params={"id_token": id_token})
        identity_response.raise_for_status()
        identity = identity_response.json()
    if str(identity.get("aud") or "") != settings.google_client_id:
        raise ValueError("Google ID token audience does not match")
    if str(identity.get("iss") or "") not in {
        "accounts.google.com", "https://accounts.google.com",
    }:
        raise ValueError("Google ID token issuer does not match")
    if identity.get("email_verified") not in {True, "true", "True"}:
        raise ValueError("Google email is not verified")
    return identity


def create_remote_app(settings: CitationGraphRemoteSettings) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret,
        session_cookie="citation_graph_session",
        same_site="lax",
        https_only=True,
        max_age=12 * 60 * 60,
    )

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/auth/login")
    async def login(request: Request, next: str = "/") -> RedirectResponse:
        state = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)
        request.session.clear()
        request.session.update({"oauth_state": state, "pkce_verifier": verifier, "next": _safe_next(next)})
        import base64
        import hashlib
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode()).digest()
        ).rstrip(b"=").decode()
        query = urlencode({
            "client_id": settings.google_client_id,
            "redirect_uri": settings.callback_url,
            "response_type": "code",
            "scope": "openid email",
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "access_type": "online",
            "prompt": "select_account",
        })
        return RedirectResponse(f"{GOOGLE_AUTHORIZE_URL}?{query}", status_code=302)

    @app.get("/auth/callback")
    async def callback(request: Request, code: str = "", state: str = "") -> Response:
        expected = str(request.session.get("oauth_state") or "")
        verifier = str(request.session.get("pkce_verifier") or "")
        next_path = _safe_next(request.session.get("next"))
        if not code or not expected or not hmac.compare_digest(state, expected) or not verifier:
            request.session.clear()
            return JSONResponse({"error": "invalid OAuth callback state"}, status_code=400)
        try:
            identity = await _google_identity(code, verifier, settings)
        except (httpx.HTTPError, ValueError):
            request.session.clear()
            return JSONResponse({"error": "Google authentication failed"}, status_code=401)
        email = str(identity.get("email") or "").casefold()
        if email not in settings.allowed_google_emails:
            request.session.clear()
            return JSONResponse({"error": "Google account is not authorized"}, status_code=403)
        request.session.clear()
        request.session.update({"email": email, "expires_at": int(time.time()) + 12 * 60 * 60})
        return RedirectResponse(next_path, status_code=302)

    @app.get("/auth/logout")
    async def logout(request: Request) -> RedirectResponse:
        request.session.clear()
        response = RedirectResponse("/auth/login", status_code=302)
        response.delete_cookie("citation_graph_session")
        return response

    register_admin_routes(app, settings, _authenticated_email)

    @app.api_route("/{path:path}", methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def proxy(request: Request, path: str) -> Response:
        if not _authenticated_email(request, settings.allowed_google_emails):
            if request.method in {"GET", "HEAD"} and "text/html" in request.headers.get("accept", ""):
                next_path = request.url.path
                if request.url.query:
                    next_path += f"?{request.url.query}"
                return RedirectResponse(f"/auth/login?{urlencode({'next': next_path})}", status_code=302)
            return JSONResponse({"error": "authentication required"}, status_code=401)
        upstream_url = f"{settings.local_url}/{path}"
        if request.url.query:
            upstream_url += f"?{request.url.query}"
        headers = {
            key: value for key, value in request.headers.items()
            if key.lower() not in _HOP_BY_HOP | {"host", "content-length", "cookie"}
        }
        async with httpx.AsyncClient(timeout=120, follow_redirects=False) as client:
            upstream = await client.request(
                request.method, upstream_url, headers=headers, content=await request.body(),
            )
        response_headers = {
            key: value for key, value in upstream.headers.items()
            if key.lower() not in _HOP_BY_HOP | {"content-length", "content-encoding"}
        }
        return Response(upstream.content, status_code=upstream.status_code, headers=response_headers)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    settings = CitationGraphRemoteSettings.from_environment(
        environment_with_saved_dotenv(_PROJECT_ROOT)
    )
    print(
        f"[citation-graph-remote] {settings.public_url} -> {settings.local_url} "
        f"via http://{settings.host}:{settings.port}",
        flush=True,
    )
    uvicorn.run(create_remote_app(settings), host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":  # pragma: no cover
    main()
