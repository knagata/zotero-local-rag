"""Authenticated routes for the separate source review workspace."""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictBool

from citation_graph.debug_reader import DebugReader
from citation_graph.debug_store import ReviewConflict
from src.source_page import render_source_page


class DraftRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    revision: StrictInt = Field(ge=0)
    version: str = Field(min_length=64, max_length=64)
    edits: dict[str, str] = Field(default_factory=dict, max_length=500)
    toc: list[dict] = Field(default_factory=list, max_length=500)
    toc_changed: StrictBool = False
    zones: dict[str, str] = Field(default_factory=dict, max_length=500)


def failure(exc):
    code = 409 if isinstance(exc, ReviewConflict) else 404 if isinstance(exc, FileNotFoundError) else 400
    return JSONResponse({'error':str(exc)}, status_code=code, headers={'Cache-Control':'no-store'})


def register_debug_routes(app, root: Path, require_admin, require_same_origin, actor):
    reader = DebugReader(root)
    static = root / 'citation_graph/static'

    async def document(key, page):
        path, item = await reader.resolve(key)
        return await asyncio.to_thread(reader.inspect, key, path, item, page), path

    @app.get('/admin/debug/')
    async def debug_page(request: Request):
        denied = require_admin(request)
        if denied:
            return denied
        return FileResponse(static/'debug.html', headers={'Cache-Control':'no-store',
            'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
            'X-Content-Type-Options':'nosniff', 'X-Frame-Options':'DENY'})

    @app.get('/admin/debug/api/documents')
    async def inventory(request: Request):
        denied = require_admin(request)
        if denied:
            return denied
        return JSONResponse(await asyncio.to_thread(reader.inventory), headers={'Cache-Control':'no-store'})

    @app.get('/admin/debug/api/documents/{key}/pages/{page}')
    async def page_data(request: Request, key: str, page: int):
        denied = require_admin(request)
        if denied:
            return denied
        try:
            data, _ = await document(key, page)
            data.pop('all_chunks')
            return JSONResponse(data, headers={'Cache-Control':'no-store'})
        except (ValueError, OSError, RuntimeError) as exc:
            return failure(exc)

    @app.get('/admin/debug/api/documents/{key}/pages/{page}/image')
    async def page_image(request: Request, key: str, page: int, version: str):
        denied = require_admin(request)
        if denied:
            return denied
        try:
            data, path = await document(key, page)
            if version != data['version']:
                raise ReviewConflict('原本・索引が変更されました。ページを再読込してください。')
            _, jpeg = await asyncio.to_thread(render_source_page, path, page)
            latest, _ = await document(key, page)
            if latest["version"] != version:
                raise ReviewConflict("画像の生成中に原本・索引が変更されました。再読込してください。")
            return Response(jpeg, media_type='image/jpeg', headers={'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'})
        except (ValueError, OSError, RuntimeError) as exc:
            return failure(exc)

    @app.post('/admin/debug/api/documents/{key}/draft')
    async def save_draft(request: Request, key: str, body: DraftRequest):
        denied = require_admin(request) or require_same_origin(request)
        if denied:
            return denied
        try:
            data, _ = await document(key, 1)
            saved = await asyncio.to_thread(reader.save, data, body.revision, body.version, body.edits, body.toc, actor(request), body.toc_changed, body.zones)
            return JSONResponse(saved, headers={'Cache-Control':'no-store'})
        except (ValueError, OSError, RuntimeError) as exc:
            return failure(exc)

    @app.get('/admin/debug/api/documents/{key}/draft/export')
    async def export_draft(request: Request, key: str):
        denied = require_admin(request)
        if denied:
            return denied
        try:
            await reader.resolve(key)
            return JSONResponse(await asyncio.to_thread(reader.store.read, key), headers={
                'Cache-Control':'no-store','Content-Disposition':'attachment; filename="review-draft.json"'})
        except (ValueError, OSError, RuntimeError) as exc:
            return failure(exc)

    @app.post('/admin/debug/api/documents/{key}/apply')
    async def apply_draft(request: Request, key: str, body: DraftRequest):
        denied = require_admin(request) or require_same_origin(request)
        if denied:
            return denied
        try:
            data, _ = await document(key, 1)
            saved = data['draft']
            if (saved['revision'] != body.revision or data['version'] != body.version
                    or saved.get('version') != body.version):
                raise ReviewConflict('原本・索引・保存済み下書きが変わっています。差分を再確認してください。')
            if not saved['edits'] and not saved['toc'] and not saved.get('toc_changed') and not saved.get('zones'):
                raise ValueError('反映する修正がありません。本文・目次・分類を編集してください。')
            from citation_graph.debug_changes import corrected_rows
            corrected_rows(data['all_chunks'],saved['edits'],saved['toc'],zones=saved.get('zones'))
            from citation_graph.admin_jobs import start_job, public_record
            record = start_job('debug_apply', '', actor(request), root,
                               debug_selection={'attachment_key':key,'revision':body.revision})
            return JSONResponse(public_record(record), status_code=202)
        except (ValueError, OSError, RuntimeError) as exc:
            return failure(exc)

    @app.get('/admin/debug/api/documents/{key}/diff')
    async def draft_diff(request: Request, key: str, revision: int):
        denied = require_admin(request)
        if denied:
            return denied
        try:
            data, _ = await document(key, 1)
            draft = data['draft']
            if revision != draft['revision'] or data['version'] != draft.get('version'):
                raise ReviewConflict('保存済み下書きが現在の原本・索引と一致しません。再確認してください。')
            indexed = {row['id']:row['text'] for row in data['all_chunks']}
            return JSONResponse({'revision':revision,'version':data['version'],
                'text_changes':[{'chunk_id':key,'before':indexed[key],'after':text} for key,text in draft['edits'].items()],
                'zone_changes':[{'chunk_id':key,'before':next(row['metadata'].get('zone') or 'body' for row in data['all_chunks'] if row['id']==key),'after':zone} for key,zone in draft.get('zones',{}).items()],
                'toc_before':data['current_toc'],'toc_after':draft['toc'] if draft.get('toc_changed') or draft['toc'] else data['current_toc']}, headers={'Cache-Control':'no-store'})
        except (ValueError, OSError, RuntimeError) as exc:
            return failure(exc)

    @app.get('/admin/debug/api/documents/{key}/history')
    async def history(request: Request, key: str, before: int = 2**63-1):
        denied = require_admin(request)
        if denied:
            return denied
        return JSONResponse(await asyncio.to_thread(reader.store.history, key, before), headers={'Cache-Control':'no-store'})

    @app.get('/admin/debug/api/documents/{key}/history/{revision}')
    async def history_revision(request: Request, key: str, revision: int):
        denied = require_admin(request)
        if denied:
            return denied
        try:
            return JSONResponse(await asyncio.to_thread(reader.store.revision, key, revision), headers={'Cache-Control':'no-store'})
        except FileNotFoundError as exc:
            return failure(exc)

    @app.post('/admin/debug/api/documents/{key}/history/{revision}/restore')
    async def restore_revision(request: Request, key: str, revision: int, body: DraftRequest, before: bool = True):
        denied = require_admin(request) or require_same_origin(request)
        if denied:
            return denied
        try:
            data, _ = await document(key, 1)
            historical = await asyncio.to_thread(reader.store.revision,key,revision)
            if historical['source_sha256'] != data['source_sha256']:
                raise ReviewConflict('原本が異なる版の修正は自動復元できません。紙面と履歴を確認してください。')
            edits = historical.get('originals',{}) if before else historical['edits']
            toc = historical.get('toc_before',[]) if before else historical['toc']
            saved = await asyncio.to_thread(reader.save,data,body.revision,body.version,edits,toc,actor(request),historical.get("toc_changed",bool(historical["toc"])),historical.get("zones_before" if before else "zones",{}))
            return JSONResponse(saved,headers={'Cache-Control':'no-store'})
        except (ValueError,OSError,RuntimeError) as exc:
            return failure(exc)
