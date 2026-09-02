from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlsplit

from fastapi.testclient import TestClient

from citation_graph.remote_proxy import (
    CitationGraphRemoteSettings,
    _authenticated_email,
    create_remote_app,
)


def settings(**changes) -> CitationGraphRemoteSettings:
    values = {
        "public_url": "https://graph.example.test:8443",
        "google_client_id": "client-id",
        "google_client_secret": "client-secret",
        "session_secret": "s" * 32,
        "allowed_google_emails": frozenset({"allowed@example.com"}),
        "local_url": "http://127.0.0.1:7234",
    }
    values.update(changes)
    return CitationGraphRemoteSettings(**values)


def test_settings_reuse_remote_mcp_google_credentials_and_require_loopback():
    configured = CitationGraphRemoteSettings.from_environment({
        "CITATION_GRAPH_PUBLIC_URL": "https://graph.example.test:8443",
        "CITATION_GRAPH_SESSION_SECRET": "x" * 32,
        "REMOTE_MCP_GOOGLE_CLIENT_ID": "id",
        "REMOTE_MCP_GOOGLE_CLIENT_SECRET": "secret",
        "REMOTE_MCP_ALLOWED_GOOGLE_EMAILS": "One@Example.com, two@example.com",
    })
    assert configured.allowed_google_emails == frozenset({"one@example.com", "two@example.com"})
    assert configured.callback_url == "https://graph.example.test:8443/auth/callback"


def test_browser_redirects_to_google_but_api_gets_401():
    client = TestClient(create_remote_app(settings()), base_url="https://graph.example.test:8443")
    browser = client.get("/", headers={"accept": "text/html"}, follow_redirects=False)
    assert browser.status_code == 302
    login = client.get(browser.headers["location"], follow_redirects=False)
    assert login.status_code == 302
    query = parse_qs(urlsplit(login.headers["location"]).query)
    assert query["redirect_uri"] == ["https://graph.example.test:8443/auth/callback"]
    assert query["code_challenge_method"] == ["S256"]
    assert client.get("/api/graph").status_code == 401


def test_invalid_callback_state_is_rejected():
    client = TestClient(create_remote_app(settings()), base_url="https://graph.example.test:8443")
    client.get("/auth/login")
    response = client.get("/auth/callback?code=code&state=wrong")
    assert response.status_code == 400


def test_allowed_google_identity_creates_session_and_proxies():
    app = create_remote_app(settings())
    client = TestClient(app, base_url="https://graph.example.test:8443")
    login = client.get("/auth/login", follow_redirects=False)
    state = parse_qs(urlsplit(login.headers["location"]).query)["state"][0]
    identity = {
        "aud": "client-id", "iss": "https://accounts.google.com",
        "email": "allowed@example.com", "email_verified": "true",
    }
    fake_upstream = AsyncMock()
    fake_upstream.status_code = 200
    fake_upstream.content = b"graph"
    fake_upstream.headers = {"content-type": "text/plain"}
    with patch("citation_graph.remote_proxy._google_identity", AsyncMock(return_value=identity)):
        callback = client.get(f"/auth/callback?code=code&state={state}", follow_redirects=False)
    assert callback.status_code == 302
    with patch("httpx.AsyncClient.request", AsyncMock(return_value=fake_upstream)) as request:
        response = client.get("/api/graph?x=1")
    assert response.status_code == 200
    assert response.text == "graph"
    assert request.await_args.args[1] == "http://127.0.0.1:7234/api/graph?x=1"


def test_disallowed_google_identity_is_forbidden():
    client = TestClient(create_remote_app(settings()), base_url="https://graph.example.test:8443")
    login = client.get("/auth/login", follow_redirects=False)
    state = parse_qs(urlsplit(login.headers["location"]).query)["state"][0]
    identity = {"email": "other@example.com", "email_verified": True}
    with patch("citation_graph.remote_proxy._google_identity", AsyncMock(return_value=identity)):
        response = client.get(f"/auth/callback?code=code&state={state}")
    assert response.status_code == 403


def test_expired_session_cannot_reach_upstream():
    request = SimpleNamespace(session={
        "email": "allowed@example.com",
        "expires_at": time.time() - 1,
    })
    assert _authenticated_email(request, frozenset({"allowed@example.com"})) is None
    assert request.session == {}


def _authenticated_client():
    configured = settings()
    client = TestClient(create_remote_app(configured), base_url="https://graph.example.test:8443")
    login = client.get("/auth/login", follow_redirects=False)
    state = parse_qs(urlsplit(login.headers["location"]).query)["state"][0]
    identity = {
        "aud": "client-id", "iss": "https://accounts.google.com",
        "email": "allowed@example.com", "email_verified": "true",
    }
    with patch("citation_graph.remote_proxy._google_identity", AsyncMock(return_value=identity)):
        response = client.get(f"/auth/callback?code=code&state={state}", follow_redirects=False)
    assert response.status_code == 302
    return client


def test_admin_page_and_api_require_google_session():
    client = TestClient(create_remote_app(settings()), base_url="https://graph.example.test:8443")
    assert client.get("/admin/", follow_redirects=False).status_code == 302
    assert client.get("/admin/api/status").status_code == 401


def test_authenticated_admin_can_read_status_and_request_fixed_job():
    client = _authenticated_client()
    page = client.get("/admin/")
    assert page.status_code == 200
    assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
    asset = client.get("/admin/assets/admin.js")
    assert asset.status_code == 200
    assert asset.headers["cache-control"] == "no-store"
    fake_status = {"manifest": {}, "definitions": [], "jobs": [], "active_job": None}
    fake_record = {"id": "job-id", "status": "queued", "token": "never-return"}
    with patch("citation_graph.admin_routes.system_status", return_value=fake_status):
        response = client.get("/admin/api/status")
    assert response.status_code == 200
    with patch("citation_graph.admin_routes.start_job", return_value=fake_record) as start:
        response = client.post("/admin/api/jobs", json={
            "job_type": "database_audit", "confirmation": "",
        }, headers={"Origin": "https://graph.example.test:8443"})
    assert response.status_code == 202
    assert "token" not in response.json()
    start.assert_called_once_with(
        "database_audit", "", "allowed@example.com", Path(__file__).resolve().parents[1],
    )


def test_admin_mutation_rejects_cross_origin_request():
    client = _authenticated_client()
    response = client.post(
        "/admin/api/jobs",
        json={"job_type": "database_audit", "confirmation": ""},
        headers={"Origin": "https://evil.example"},
    )
    assert response.status_code == 403
