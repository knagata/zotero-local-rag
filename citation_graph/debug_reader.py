"""Canonical PDF inspection without native Chroma or hosted services."""
from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from contextlib import closing
from pathlib import Path

import fitz
import httpx
from src.chunk_store import get_item_chunks
from src.db_relations import get_artifact_processing_statuses, get_chunk_quality_reports, get_db_connection
from src.manifest import load_manifest, content_signature
from src.v3_data_plane import chroma_dir, manifest_path
from src.zotero_source_localapi import ZoteroLocalAPI
from citation_graph.debug_annotations import resolve_toc, validate_zones
from citation_graph.debug_store import DraftStore, ReviewConflict, validate_draft


def source_stamp(path: Path) -> tuple:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=32)
def source_hash(path: Path, stamp: tuple) -> str:
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    if source_stamp(path) != stamp:
        raise ReviewConflict('PDFが変更中です。同期完了後に再読込してください。')
    return digest.hexdigest()


@lru_cache(maxsize=32)
def indexed_signature(path: Path, stamp: tuple) -> str:
    return content_signature(path, stamp[2])


def verify_indexed_source(path: Path, key: str, manifest: dict, rows: list[dict]) -> None:
    if key in manifest.get('inflight_attachments',[]):
        raise ReviewConflict('この資料は索引を更新中です。完了後に開き直してください。')
    entry = (manifest.get('files') or {}).get(key) or {}
    if rows and 'size' in entry and 'mtime' in entry:
        stat = path.stat()
        if (stat.st_size != entry['size'] or stat.st_mtime != entry['mtime']
                or entry.get('content_signature') and indexed_signature(path,source_stamp(path)) != entry['content_signature']):
            raise ReviewConflict('PDFが索引作成後に変更されています。再索引の完了後に開き直してください。')


def current_toc(rows: list[dict]) -> list[dict]:
    result, previous = [], []
    for row in rows:
        raw = row['metadata'].get('structure_path') or []
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                raw = [raw]
        path = raw if isinstance(raw, list) else []
        if path and path != previous:
            result.append({'chunk_id':row['id'],'page':row['metadata'].get('page'),
                           'level':len(path),'title':str(path[-1]),'path':path})
        previous = path
    return result


def document_version(digest: str, rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps([digest, rows], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class DebugReader:
    def __init__(self, root: Path):
        self.root = root
        self.store = DraftStore(root / 'data/debug_review.sqlite3')

    def inventory(self) -> list[dict]:
        files = load_manifest(manifest_path(self.root)).get('files') or {}
        reasons: dict[str, set] = {}
        for row in get_artifact_processing_statuses():
            if row.get('artifact_type') in {'extraction','structure'} and row.get('status') in {'failed','blocked','degraded'}:
                reasons.setdefault(row.get('attachment_key') or row.get('item_key'), set()).add(row.get('reason_code') or row['status'])
        for row in get_chunk_quality_reports():
            reasons.setdefault(row.get('attachment_key'), set()).add('利用者の品質報告')
        parents = {}
        with closing(get_db_connection()) as connection:
            for item, attachment in connection.execute("SELECT DISTINCT item_key,attachment_key FROM document_nodes WHERE node_type='attachment_root'"):
                parents[attachment] = item
        result = []
        for key, entry in files.items():
            if Path(entry.get('pdf_path') or '').suffix.lower() != '.pdf':
                continue
            quality = entry.get('quality') or {}
            issues = reasons.get(key, set()) | reasons.get(parents.get(key), set())
            if quality.get('ocr_layer_needs_review') or quality.get('source_coverage_adopted'):
                issues.add('文字起こし・抽出範囲の要確認')
            result.append({'attachment_key': key, 'title': entry.get('title') or key, 'issues': sorted(issues)})
        return sorted(result, key=lambda row: (not bool(row['issues']), row['title']))

    async def resolve(self, key: str) -> tuple[Path, str]:
        if not re.fullmatch(r'[A-Z0-9]{8}', key):
            raise ValueError('添付IDが不正です。')
        api = ZoteroLocalAPI()
        try:
            raw = await api.get_item(key)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise FileNotFoundError("Zoteroに添付が見つかりません。同期・削除状態を確認してください。") from exc
            raise RuntimeError("Zotero APIが応答エラーを返しました。状態を確認してください。") from exc
        except httpx.RequestError as exc:
            raise RuntimeError("Zoteroに接続できません。起動・ローカル通信設定を確認してください。") from exc
        identity, data = api._unwrap_item(raw)
        if identity != key or data.get('itemType') != 'attachment' or data.get('deleted'):
            raise ValueError('現在のZotero添付ではありません。')
        import os
        path = api.resolve_pdf_path_from_attachment(key, data, os.environ.get('ZOTERO_DATA_DIR', str(Path.home()/'Zotero')))
        if not path:
            raise FileNotFoundError('原本が未取得です。Zoteroで添付を同期してください。')
        return Path(path), str(data.get('parentItem') or key)

    def inspect(self, key: str, path: Path, item: str, page: int) -> dict:
        with fitz.open(path) as pdf:
            if not pdf.is_pdf or pdf.needs_pass:
                raise ValueError('暗号化されていないPDFを指定してください。')
            if not 1 <= page <= len(pdf):
                raise ValueError('物理PDFページ番号が範囲外です。')
            total, label = len(pdf), pdf[page-1].get_label()
        rows = [row for row in get_item_chunks(item, chroma_dir=chroma_dir(self.root))
                if (row.get('metadata') or {}).get('attachmentKey') == key]
        manifest = load_manifest(manifest_path(self.root))
        verify_indexed_source(path,key,manifest,rows)
        digest = source_hash(path, source_stamp(path))
        version = document_version(digest, rows)
        draft = self.store.read(key)
        manual = ((manifest.get("files") or {}).get(key) or {}).get("manual_review") or {}
        applied = manual.get("revision") == draft["revision"] and manual.get("version") == version
        return {'attachment_key':key,'item_key':item,'title':(rows[0]['metadata'].get('title') if rows else path.stem),
                'page':page,'page_label':label,'total_pages':total,'version':version,
                'source_sha256':digest,'draft':draft,'draft_applied':applied,'current_toc':current_toc(rows),
                'draft_stale':bool(not applied and draft.get('version') and draft['version'] != version),
                'chunks':[row for row in rows if str(row['metadata'].get('page')) == str(page)],
                'anchors':[{'id':row['id'],'page':row['metadata'].get('page'),'zone':row['metadata'].get('zone') or 'body',
                            'preview':re.sub(r'\s+', ' ', row['text']).strip()[:160]} for row in rows],
                'all_chunks':rows}

    def save(self, document: dict, revision: int, version: str, edits: dict, toc: list, actor: str, toc_changed: bool = False, zones: dict | None = None) -> dict:
        if document['version'] != version:
            raise ReviewConflict('原本または索引が変更されました。再読込して修正内容を確認してください。')
        toc = resolve_toc(document['all_chunks'], toc, allow_unresolved=True, total_pages=document['total_pages'])
        zones = zones or {}
        validate_draft(document['all_chunks'], edits, toc, allow_unresolved=True)
        validate_zones(document['all_chunks'], zones)
        return self.store.save(document['attachment_key'], revision,
                               {'version':version,'source_sha256':document['source_sha256'],'edits':edits,'toc':toc,'toc_changed':toc_changed,
                                'originals':{row['id']:row['text'] for row in document['all_chunks'] if row['id'] in edits},
                                'zones':zones,'zones_before':{row['id']:row['metadata'].get('zone') or 'body' for row in document['all_chunks'] if row['id'] in zones},
                                'toc_before':current_toc(document['all_chunks'])}, actor)
