"""Authenticated routes for the separate source review workspace."""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictBool

from citation_graph.debug_reader import DebugReader
from citation_graph.debug_store import ReviewConflict
from citation_graph.debug_pdf import register_pdf_routes


class DraftRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    revision: StrictInt = Field(ge=0)
    version: str = Field(min_length=64, max_length=64)
    edits: dict[str, str] = Field(default_factory=dict, max_length=500)
    toc: list[dict] = Field(default_factory=list, max_length=500)
    toc_changed: StrictBool = False
    zones: dict[str, str] = Field(default_factory=dict, max_length=500)
    merges: list[list[str]] = Field(default_factory=list, max_length=500)
    cuts: dict[str,list[StrictInt]] = Field(default_factory=dict, max_length=500)


def failure(exc):
    code = 409 if isinstance(exc, ReviewConflict) else 404 if isinstance(exc, FileNotFoundError) else 400
    return JSONResponse({'error':str(exc)}, status_code=code, headers={'Cache-Control':'no-store'})


def register_debug_routes(app, root: Path, require_admin, require_same_origin, actor):
    reader = DebugReader(root)
    static = root / 'citation_graph/static'

    documents = register_pdf_routes(app,root,reader,require_admin,failure)
    document = documents.inspect

    @app.get('/admin/debug/')
    async def debug_page(request: Request):
        denied = require_admin(request)
        if denied:
            return denied
        return FileResponse(static/'debug.html', headers={'Cache-Control':'no-store',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self'; style-src 'self' 'unsafe-inline'; font-src 'self' blob:; connect-src 'self'; img-src 'self' blob: data:; object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
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

    @app.post('/admin/debug/api/documents/{key}/draft')
    async def save_draft(request: Request, key: str, body: DraftRequest):
        denied = require_admin(request) or require_same_origin(request)
        if denied:
            return denied
        try:
            data, _ = await document(key, 1)
            saved = await asyncio.to_thread(reader.save, data, body.revision, body.version, body.edits, body.toc, actor(request), body.toc_changed, body.zones, body.merges, body.cuts)
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
            if not saved['edits'] and not saved['toc'] and not saved.get('toc_changed') and not saved.get('zones') and not saved.get('merges') and not saved.get('cuts'):
                raise ValueError('反映する修正がありません。本文・目次・分類を編集してください。')
            from citation_graph.debug_changes import corrected_rows
            corrected_rows(data['all_chunks'],saved['edits'],saved['toc'],zones=saved.get('zones'),merges=saved.get('merges'),cuts=saved.get('cuts'))
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
            from citation_graph.debug_segments import merge_rows, split_rows
            source = merge_rows(data['all_chunks'],draft.get('merges'))
            indexed = {row['id']:row['text'] for row in source}
            preview, _, groups = split_rows(source,draft['edits'],draft['toc'],draft.get('cuts'))
            return JSONResponse({'revision':revision,'version':data['version'],
                'split_changes':[{'chunk_id':group[0],'parts':[row['text'] for row in preview if row['id'] in group]} for group in groups],
                'merge_changes':draft.get('merges',[]),
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
            from citation_graph.debug_segments import history_plan
            edits,toc,merges,cuts = history_plan(data['all_chunks'],historical,before)
            saved = await asyncio.to_thread(reader.save,data,body.revision,body.version,edits,toc,actor(request),historical.get("toc_changed",bool(historical["toc"])),historical.get("zones_before" if before else "zones",{}),merges,cuts)
            return JSONResponse(saved,headers={'Cache-Control':'no-store'})
        except (ValueError,OSError,RuntimeError) as exc:
            return failure(exc)
