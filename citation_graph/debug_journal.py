"""One bounded rollback journal; no accumulating complete database copies."""
from __future__ import annotations

import gzip
import json
import os
import tempfile
from pathlib import Path

MAX_JOURNAL_BYTES = 128 * 1024**2

TABLES = ('document_structures','document_nodes','document_node_chunks','document_node_summaries',
          'document_node_summary_parts','document_node_summary_reuse_cache',
          'artifact_processing_status','artifact_processing_events')


def condition(columns: list[str]) -> str:
    return 'item_key=?' if 'item_key' in columns else 'node_id IN (SELECT node_id FROM document_nodes WHERE item_key=?)'


def snapshot_relations(connection, item: str) -> dict:
    result = {}
    for table in TABLES:
        columns = [row[1] for row in connection.execute(f'PRAGMA table_info({table})')]
        if columns:
            rows = connection.execute(f'SELECT * FROM {table} WHERE {condition(columns)}', (item,)).fetchall()
            result[table] = {'columns':columns,'rows':[list(row) for row in rows]}
    return result


def restore_relations(connection, item: str, snapshot: dict) -> None:
    with connection:
        connection.execute('PRAGMA defer_foreign_keys=ON')
        for table in reversed(TABLES):
            if table in snapshot:
                connection.execute(f'DELETE FROM {table} WHERE {condition(snapshot[table]["columns"])}', (item,))
        for table in TABLES:
            if table in snapshot:
                rows = snapshot[table]['rows']
                marks = ','.join('?' for _ in snapshot[table]['columns'])
                connection.executemany(f'INSERT INTO {table} VALUES({marks})', rows)


def write_journal(path: Path, payload: dict) -> None:
    raw = json.dumps(payload, ensure_ascii=False).encode()
    if len(raw) > MAX_JOURNAL_BYTES:
        raise ValueError('復旧記録の上限を超えています。対象を分けて修正してください。')
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(gzip.compress(raw))
            stream.flush()
            os.fsync(stream.fileno())
        if gzip.decompress(temporary.read_bytes()) != raw:
            raise RuntimeError('復旧記録の検証に失敗しました。DBは変更しません。')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
