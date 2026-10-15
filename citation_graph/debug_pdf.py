"""Authenticated PDF bytes and cheap generation guards for browser-side rendering."""
from __future__ import annotations

import asyncio
import json
import os
from collections import OrderedDict
from pathlib import Path

from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from citation_graph.debug_reader import source_stamp
from citation_graph.debug_store import ReviewConflict
from src.source_page import render_source_page
from src.v3_data_plane import chroma_dir, manifest_path


def index_stamp(root: Path) -> tuple:
    db = chroma_dir(root) / 'chroma.sqlite3'
    paths = (db, Path(str(db)+'-wal'), manifest_path(root))
    return tuple(source_stamp(path) if path.exists() else None for path in paths)


class ReviewDocuments:
    def __init__(self, root: Path, reader):
        self.root, self.reader = root, reader
        self.sessions: OrderedDict[str, dict] = OrderedDict()

    async def inspect(self, key: str, page: int):
        path, item = await self.reader.resolve(key)
        before = source_stamp(path), index_stamp(self.root)
        data = await asyncio.to_thread(self.reader.inspect, key, path, item, page)
        if before != (source_stamp(path), index_stamp(self.root)):
            raise ReviewConflict('読込み中に原本・索引が更新されました。資料を開き直してください。')
        self.sessions[key] = {'version':data['version'],'path':path,'stamp':before}
        self.sessions.move_to_end(key)
        while len(self.sessions) > 4:
            self.sessions.popitem(last=False)
        return data, path

    async def validate(self, key: str, version: str):
        saved = self.sessions.get(key)
        if saved is None:
            await self.inspect(key, 1)
            saved = self.sessions[key]
        path, _ = await self.reader.resolve(key)
        current = source_stamp(path), index_stamp(self.root)
        if saved['version'] != version or saved['path'] != path or saved['stamp'] != current:
            raise ReviewConflict('原本または索引が更新されました。再索引の完了後に資料を開き直してください。')
        self.sessions.move_to_end(key)
        return path, current[0]


def byte_range(value: str | None, size: int) -> tuple[int, int]:
    if value is None:
        return 0, size-1
    if not value.startswith('bytes=') or ',' in value:
        raise ValueError('single byte range required')
    first, separator, last = value[6:].partition('-')
    if not separator or not first and not last:
        raise ValueError('invalid range')
    if first:
        start, end = int(first), int(last) if last else size-1
    else:
        start, end = max(0,size-int(last)), size-1
    if start < 0 or start >= size or end < start or not first and int(last) <= 0:
        raise ValueError('unsatisfiable range')
    return start, min(end,size-1)


async def pdf_response(path: Path, stamp: tuple, request: Request):
    size = stamp[2]
    headers = {'Cache-Control':'no-store','Accept-Ranges':'bytes','Content-Type':'application/pdf',
               'X-Content-Type-Options':'nosniff','Content-Disposition':'inline; filename="source.pdf"',
               'ETag':'"'+str(stamp[1])+'-'+str(stamp[3])+'"'}
    try:
        start, end = byte_range(request.headers.get('range'),size)
    except ValueError:
        return Response(status_code=416,headers={**headers,'Content-Range':f'bytes */{size}'})
    status = 206 if request.headers.get('range') else 200
    headers['Content-Length'] = str(end-start+1)
    if status == 206:
        headers['Content-Range'] = f'bytes {start}-{end}/{size}'
    stream = path.open('rb')
    transferred = False
    try:
        stat = os.fstat(stream.fileno())
        if (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns) != stamp:
            raise ReviewConflict('送信前にPDFが変更されました。資料を開き直してください。')
        if request.method == 'HEAD':
            return Response(status_code=status,headers=headers)
        stream.seek(start)
        async def blocks():
            remaining = end-start+1
            try:
                while remaining:
                    if source_stamp(path) != stamp:
                        raise ReviewConflict('PDFが送信中に変更されました。再読込してください。')
                    block = await asyncio.to_thread(stream.read,min(256*1024,remaining))
                    if not block or source_stamp(path) != stamp:
                        raise ReviewConflict('PDFの送信を中断しました。再読込してください。')
                    remaining -= len(block)
                    yield block
            finally:
                stream.close()
        transferred = True
        return StreamingResponse(blocks(),status_code=status,headers=headers)
    finally:
        if not transferred:
            stream.close()


def register_pdf_routes(app, root: Path, reader, require_admin, failure):
    documents = ReviewDocuments(root,reader)
    static = root/'citation_graph/static/pdfjs'

    @app.get('/admin/pdfjs/{asset:path}')
    async def asset_file(request: Request, asset: str):
        denied = require_admin(request)
        if denied:
            return denied
        manifest = json.loads((static/'assets.json').read_text())
        if asset not in manifest['files']:
            return JSONResponse({'error':'not found'},status_code=404)
        media = 'text/javascript' if asset.endswith('.mjs') else 'text/css' if asset.endswith('.css') else 'application/wasm' if asset.endswith('.wasm') else 'application/octet-stream'
        return FileResponse(static/asset,media_type=media,headers={'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'})

    @app.get('/admin/debug/api/documents/{key}/state')
    async def current_state(request: Request, key: str, version: str):
        denied = require_admin(request)
        if denied:
            return denied
        try:
            await documents.validate(key,version)
            return JSONResponse({'version':version},headers={'Cache-Control':'no-store'})
        except (ValueError,OSError,RuntimeError) as exc:
            return failure(exc)

    @app.api_route('/admin/debug/api/documents/{key}/pdf',methods=['GET','HEAD'])
    async def source_pdf(request: Request, key: str, version: str):
        denied = require_admin(request)
        if denied:
            return denied
        try:
            path, stamp = await documents.validate(key,version)
            return await pdf_response(path,stamp,request)
        except (ValueError,OSError,RuntimeError) as exc:
            return failure(exc)

    @app.get('/admin/debug/api/documents/{key}/pages/{page}/image')
    async def legacy_image(request: Request, key: str, page: int, version: str):
        denied = require_admin(request)
        if denied:
            return denied
        try:
            data, path = await documents.inspect(key,page)
            if data['version'] != version:
                raise ReviewConflict('原本・索引が変更されました。再読込してください。')
            _, jpeg = await asyncio.to_thread(render_source_page,path,page)
            latest, _ = await documents.inspect(key,page)
            if latest['version'] != version:
                raise ReviewConflict('画像の生成中に原本・索引が変更されました。再読込してください。')
            return Response(jpeg,media_type='image/jpeg',headers={'Cache-Control':'no-store'})
        except (ValueError,OSError,RuntimeError) as exc:
            return failure(exc)
    return documents
