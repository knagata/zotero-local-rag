"""Apply a reviewed draft with local embeddings and a bounded rollback journal."""
from __future__ import annotations

import asyncio
import copy
from contextlib import closing
from pathlib import Path

from citation_graph.debug_changes import corrected_rows
from citation_graph.debug_journal import snapshot_relations, restore_relations, write_journal
from citation_graph.debug_reader import DebugReader
from citation_graph.debug_store import ReviewConflict


def apply_with_runtime(runtime, document: dict, draft: dict, root: Path) -> None:
    """All writes occur under the caller's canonical writer lock."""
    item, key = document['item_key'], document['attachment_key']
    old = runtime.rows(item)
    if len(old) > 50000 or not document['all_chunks']:
        raise ValueError('索引済みの資料を選び、対象が大きい場合は添付を分けてください。')
    changed = corrected_rows(document['all_chunks'], draft['edits'], draft['toc'], toc_changed=draft.get('toc_changed',False))
    by_id = {row['id']:row for row in changed}
    candidate = [by_id.get(row['id'], copy.deepcopy(row)) for row in old]
    source = [row for row in candidate if row['metadata'].get('source_type') != 'note']
    built = runtime.build(item, source)
    if not built['diagnostics'].get('valid'):
        raise ValueError('修正後の文書構造が不正です。下書きを確認してください。')
    candidate = runtime.attach(source, built['nodes'])
    manifest = runtime.manifest()
    journal = {'item_key':item,'attachment_key':key,'revision':draft['revision'],
               'rows':old,'manifest':manifest,'relations':runtime.snapshot(item),
               'vectors':runtime.vectors(list(draft['edits']))}
    write_journal(root/'data/debug_rollback/current.json.gz', journal)
    print('[修正反映] 復旧用の差分を保存・検証しました。', flush=True)
    committed = False
    try:
        runtime.invalidate()
        pending = copy.deepcopy(manifest)
        pending['hnsw_validated'] = False
        pending['inflight_attachments'] = sorted(set(pending.get('inflight_attachments',[])) | {key})
        runtime.save_manifest(pending)
        runtime.write(candidate, set(draft['edits']))
        runtime.structure(item, built)
        runtime.verify(candidate[0]['id'])
        runtime.verify_source()
        pending['inflight_attachments'] = [value for value in pending['inflight_attachments'] if value != key]
        pending['hnsw_validated'] = True
        from citation_graph.debug_reader import document_version
        canonical = [row for row in runtime.rows(item) if row['metadata'].get('attachmentKey') == key]
        pending['files'][key]['manual_review'] = {'revision':draft['revision'],'source_sha256':document['source_sha256'],
                                                'version':document_version(document['source_sha256'],canonical)}
        runtime.save_manifest(pending)
        committed = True
    finally:
        if not committed:
            runtime.restore(journal)


class Runtime:
    def __init__(self, root: Path):
        from src import db_relations, lexical_index, document_structure, manifest
        from src.chunk_store import get_item_chunks
        from src.embedder import get_collection
        from src.v3_data_plane import chroma_dir, manifest_path, V3_COLLECTION
        from scripts.run_db_audit import _validated_report_paths
        self.gate_path = _validated_report_paths(root)[0]
        self.root, self.db, self.lexical = root, db_relations, lexical_index
        self.manifest_path = manifest_path(root)
        self.load, self.save = manifest.load_manifest, manifest.save_manifest
        self.build, self.attach = document_structure.build_document_structure, document_structure.attach_structure_metadata
        self.rows = lambda item: get_item_chunks(item, chroma_dir=chroma_dir(root), collection_name=V3_COLLECTION)
        self.collection = get_collection(chroma_dir=chroma_dir(root), project_root=root,
                                         chroma_collection_env=V3_COLLECTION, chroma_collection_default=V3_COLLECTION,
                                         persist_active_config=False)
        from src.v3_runtime import pipeline_payload
        from src.v3_data_plane import pipeline_config_path
        import json
        stored = json.loads(pipeline_config_path(root).read_text())
        expected = pipeline_payload(self.collection._zotero_embedder_config, collection=V3_COLLECTION,
                                    run_code_fingerprint='manual-review')
        if expected['pipeline_fingerprint'] != stored.get('pipeline_fingerprint'):
            self.close()
            raise RuntimeError('埋め込み設定が既存索引と一致しません。設定を戻してください。')

    def manifest(self):
        return self.load(self.manifest_path)

    def save_manifest(self, value):
        self.save(self.manifest_path, value)

    def snapshot(self, item):
        with closing(self.db.get_db_connection()) as connection:
            return snapshot_relations(connection, item)

    def vectors(self, ids):
        if not ids:
            return {}
        values = self.collection.get(ids=ids, include=['embeddings'])
        if set(values["ids"]) != set(ids):
            raise RuntimeError("復旧用ベクトルをすべて取得できません。DBは変更しません。")
        return {key:vector.tolist() for key,vector in zip(values['ids'],values['embeddings'],strict=True)}

    def invalidate(self):
        self.gate_path.unlink(missing_ok=True)

    def write(self, rows, edited):
        for start in range(0,len(rows),128):
            batch=rows[start:start+128]
            self.collection.update(ids=[row['id'] for row in batch],metadatas=[row['metadata'] for row in batch])
            texts=[row for row in batch if row['id'] in edited]
            if texts:
                self.collection.update(ids=[row['id'] for row in texts], documents=[row['text'] for row in texts])
            self.lexical.upsert_chunks([row['id'] for row in batch],[row['text'] for row in batch],[row['metadata'] for row in batch])

    def structure(self, item, built):
        self.db.replace_document_structure(item, source_fingerprint=built['source_fingerprint'],
            structure_version=built['structure_version'],status=built['status'],nodes=built['nodes'],
            diagnostics=built['diagnostics'],confidence=built['confidence'])
        self.db.mark_artifact_status(item,'structure','success' if built['status'] in {'exact','recovered'} else 'degraded',
                                     reason_code=None if built['status'] in {'exact','recovered'} else 'flat_fallback')
        self.db.mark_artifact_status(item,'embeddings','success',source_fingerprint=built['source_fingerprint'])
        for artifact in ('summary','summary_index'):
            self.db.mark_artifact_status(item,artifact,'stale',reason_code='manual_review_applied')

    def verify(self, sample):
        from src.index_from_zotero import _flush_and_verify_hnsw
        _flush_and_verify_hnsw(self.collection,sample)

    def restore(self, journal):
        old=journal['rows']
        self.write(old,set())
        for start in range(0,len(old),128):
            batch=old[start:start+128]
            live=self.collection.get(ids=[row['id'] for row in batch],include=['metadatas'])
            current=dict(zip(live['ids'],live['metadatas'],strict=True))
            restored=[{**{key:None for key in current.get(row['id'],{}) if key not in row['metadata']},
                       **row['metadata']} for row in batch]
            self.collection.update(ids=[row['id'] for row in batch],metadatas=restored)
        for key,vector in journal['vectors'].items():
            row=next(row for row in old if row['id']==key)
            self.collection.update(ids=[key],documents=[row['text']],embeddings=[vector])
        with closing(self.db.get_db_connection()) as connection:
            restore_relations(connection,journal['item_key'],journal['relations'])
        self.verify(old[0]['id'])
        self.save_manifest(journal['manifest'])

    def close(self):
        self.collection._chroma_client.close()


def apply_review(root: Path, key: str, revision: int) -> None:
    from src.indexing_lock import held, default_path
    from src.database_gate import validate_database_gate
    from src.v3_data_plane import enforce_environment
    enforce_environment(root)
    with held(default_path(root)):
        reader=DebugReader(root)
        path,item=asyncio.run(reader.resolve(key))
        document=reader.inspect(key,path,item,1)
        draft=reader.store.read(key)
        if draft['revision'] != revision or draft.get('version') != document['version']:
            raise ReviewConflict('下書き・原本・索引が変更されました。差分を再確認してください。')
        if not draft['edits'] and not draft['toc'] and not draft.get('toc_changed'):
            raise ValueError('反映する修正がありません。')
        from scripts.run_db_audit import _validated_report_paths
        validate_database_gate(_validated_report_paths(root)[0],allow_pending_ocr=True)
        runtime=Runtime(root)
        def verify_source():
            from citation_graph.debug_reader import source_hash, source_stamp
            if source_hash(path, source_stamp(path)) != document['source_sha256']:
                raise ReviewConflict('反映中に原本が変更されました。同期完了後に再確認してください。')
        runtime.verify_source = verify_source
        try:
            verify_source()
            apply_with_runtime(runtime,document,draft,root)
        finally:
            runtime.close()
