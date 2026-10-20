"""Request schema and read-only structured review preview route."""
from __future__ import annotations
import asyncio
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel,ConfigDict,Field,StrictInt,StrictBool
from citation_graph.debug_store import validate_draft,ReviewConflict

class DraftRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    revision: StrictInt = Field(ge=0)
    version: str = Field(min_length=64, max_length=64)
    edits: dict[str, str] = Field(default_factory=dict, max_length=500)
    toc: list[dict] = Field(default_factory=list, max_length=500)
    toc_changed: StrictBool = False
    rechunk: StrictBool = False
    zones: dict[str, str] = Field(default_factory=dict, max_length=500)
    merges: list[list[str]] = Field(default_factory=list, max_length=500)
    cuts: dict[str,list[StrictInt]] = Field(default_factory=dict, max_length=500)



def register_preview(app,document,require_admin,require_same_origin,failure):
    @app.post('/admin/debug/api/documents/{key}/preview')
    async def preview(request: Request, key: str, body: DraftRequest):
        denied = require_admin(request) or require_same_origin(request)
        if denied:
            return denied
        try:
            data, _ = await document(key,1)
            if body.version != data['version']:
                raise ReviewConflict('原本・索引が変わりました。再読込してください。')
            from citation_graph.debug_annotations import resolve_toc,validate_zones
            from citation_graph.review_plan import build_review_plan,public_plan
            payload = body.model_dump()
            payload['toc'] = resolve_toc(data['all_chunks'],body.toc,total_pages=data['total_pages'])
            validate_draft(data['all_chunks'],body.edits,payload['toc'])
            validate_zones(data['all_chunks'],body.zones)
            plan = await asyncio.to_thread(build_review_plan,data['all_chunks'],payload)
            return JSONResponse(public_plan(plan,data['all_chunks']),headers={'Cache-Control':'no-store'})
        except (ValueError,OSError,RuntimeError) as exc:
            return failure(exc)

