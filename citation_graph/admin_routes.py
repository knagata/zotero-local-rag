"""OAuth-protected browser administration routes for Citation Graph."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel

from citation_graph.admin_jobs import (
    log_tail,
    public_record,
    read_record,
    start_job,
    stop_job,
    system_status,
)

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "citation_graph" / "static"


class StartJobRequest(BaseModel):
    job_type: str
    confirmation: str = ""


class StopJobRequest(BaseModel):
    confirmation: str = ""


def register_admin_routes(app: FastAPI, settings: Any, authenticated_email) -> None:
    def require_admin(request: Request) -> JSONResponse | None:
        if authenticated_email(request, settings.allowed_google_emails):
            return None
        return JSONResponse({"error": "authentication required"}, status_code=401)

    def require_same_origin(request: Request) -> JSONResponse | None:
        if request.headers.get("origin") == settings.public_url:
            return None
        return JSONResponse({"error": "request origin is not authorized"}, status_code=403)

    @app.get("/admin")
    @app.get("/admin/")
    async def admin_page(request: Request) -> Response:
        if not authenticated_email(request, settings.allowed_google_emails):
            return RedirectResponse("/auth/login?next=%2Fadmin%2F", status_code=302)
        return FileResponse(
            STATIC_DIR / "admin.html",
            media_type="text/html",
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": (
                    "default-src 'self'; script-src 'self'; style-src 'self'; "
                    "connect-src 'self'; img-src 'self'; frame-ancestors 'none'; "
                    "base-uri 'none'; form-action 'self'"
                ),
                "X-Frame-Options": "DENY",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/admin/assets/{name}")
    async def admin_asset(request: Request, name: str) -> Response:
        unauthorized = require_admin(request)
        if unauthorized:
            return unauthorized
        if name not in {"admin.css", "admin.js"}:
            return JSONResponse({"error": "not found"}, status_code=404)
        media_type = "text/css" if name.endswith(".css") else "text/javascript"
        return FileResponse(
            STATIC_DIR / name,
            media_type=media_type,
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/admin/api/status")
    async def admin_status(request: Request) -> Response:
        unauthorized = require_admin(request)
        if unauthorized:
            return unauthorized
        return JSONResponse(system_status(ROOT), headers={"Cache-Control": "no-store"})

    @app.get("/admin/api/jobs/{job_id}/log")
    async def admin_log(request: Request, job_id: str) -> Response:
        unauthorized = require_admin(request)
        if unauthorized:
            return unauthorized
        try:
            read_record(job_id, ROOT)
            value = log_tail(job_id, ROOT)
        except (OSError, ValueError):
            return JSONResponse({"error": "job not found"}, status_code=404)
        return Response(
            value, media_type="text/plain; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/admin/api/jobs")
    async def admin_start(request: Request, body: StartJobRequest) -> Response:
        unauthorized = require_admin(request)
        if unauthorized:
            return unauthorized
        wrong_origin = require_same_origin(request)
        if wrong_origin:
            return wrong_origin
        email = authenticated_email(request, settings.allowed_google_emails)
        try:
            record = start_job(body.job_type, body.confirmation, str(email), ROOT)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
        return JSONResponse(public_record(record), status_code=202)

    @app.post("/admin/api/jobs/{job_id}/stop")
    async def admin_stop(request: Request, job_id: str, body: StopJobRequest) -> Response:
        unauthorized = require_admin(request)
        if unauthorized:
            return unauthorized
        wrong_origin = require_same_origin(request)
        if wrong_origin:
            return wrong_origin
        email = authenticated_email(request, settings.allowed_google_emails)
        try:
            record = stop_job(job_id, body.confirmation, str(email), ROOT)
        except (OSError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
        return JSONResponse(public_record(record), status_code=202)
