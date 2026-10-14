"""Versioned review drafts, kept separate from the canonical index."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class ReviewConflict(ValueError):
    pass


class DraftStore:
    def __init__(self, path: Path):
        self.path = path

    def read(self, attachment: str) -> dict:
        if not self.path.exists():
            return {'revision': 0, 'edits': {}, 'toc': []}
        with sqlite3.connect(self.path) as connection:
            if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='revisions'").fetchone():
                return {'revision': 0, 'edits': {}, 'toc': []}
            row = connection.execute('SELECT revision,payload FROM revisions WHERE attachment=? ORDER BY revision DESC LIMIT 1', (attachment,)).fetchone()
        return {'revision': row[0], **json.loads(row[1])} if row else {'revision': 0, 'edits': {}, 'toc': []}

    def save(self, attachment: str, expected: int, payload: dict, actor: str) -> dict:
        encoded = json.dumps(payload, ensure_ascii=False)
        if len(encoded.encode()) > 2 * 1024**2:
            raise ValueError('下書きが大きすぎます。修正を分けて保存してください。')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=10) as connection:
            connection.execute('CREATE TABLE IF NOT EXISTS revisions(attachment TEXT, revision INTEGER, payload TEXT, actor TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY(attachment,revision))')
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute('SELECT max(revision) FROM revisions WHERE attachment=?', (attachment,)).fetchone()
            revision = row[0] or 0
            if revision != expected:
                raise ReviewConflict('他の画面で下書きが更新されました。再読込してください。')
            connection.execute('INSERT INTO revisions(attachment,revision,payload,actor) VALUES(?,?,?,?)', (attachment, revision+1, encoded, actor))
        return {'revision': revision+1, **payload}

    def history(self, attachment: str, before: int = 2**63-1) -> list[dict]:
        if not self.path.exists():
            return []
        with sqlite3.connect(self.path) as connection:
            if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='revisions'").fetchone():
                return []
            rows = connection.execute('SELECT revision,actor,created_at,payload FROM revisions WHERE attachment=? AND revision<? ORDER BY revision DESC LIMIT 50', (attachment,before)).fetchall()
        return [{'revision':revision,'actor':actor,'created_at':created,
                 'text_changes':len(json.loads(payload)['edits']),'headings':len(json.loads(payload)['toc'])}
                for revision,actor,created,payload in rows]

    def revision(self, attachment: str, revision: int) -> dict:
        if not self.path.exists():
            raise FileNotFoundError('編集履歴がありません。')
        with sqlite3.connect(self.path) as connection:
            if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='revisions'").fetchone():
                raise FileNotFoundError('編集履歴がありません。')
            row = connection.execute('SELECT payload,actor,created_at FROM revisions WHERE attachment=? AND revision=?', (attachment,revision)).fetchone()
        if row is None:
            raise FileNotFoundError('指定された版がありません。')
        return {'revision':revision,'actor':row[1],'created_at':row[2],**json.loads(row[0])}


def validate_draft(rows: list[dict], edits: dict, toc: list[dict]) -> None:
    indexed = {row['id']: row for row in rows}
    if len(edits) > 500 or len(toc) > 500:
        raise ValueError('一度に保存できる修正・見出しは500件までです。')
    for key, text in edits.items():
        if key not in indexed or not isinstance(text, str) or not text.strip() or len(text) > 30000:
            raise ValueError('修正対象のチャンク、または修正本文が不正です。')
    order = {row['id']: n for n, row in enumerate(rows)}
    last, level = -1, 0
    for entry in toc:
        anchor, depth, title = entry.get('chunk_id'), entry.get('level'), entry.get('title')
        if (anchor not in indexed or order[anchor] <= last or isinstance(depth, bool)
                or not isinstance(depth, int) or not 1 <= depth <= min(8, level+1)
                or not isinstance(title, str) or not title.strip() or len(title) > 300):
            raise ValueError('目次は本文順・重複なしで指定し、階層は1段ずつ深くしてください。')
        last, level = order[anchor], depth
