import sqlite3
import pytest
from src import lexical_index as lexical


def test_upsert_does_not_rescan_the_corpus_for_every_id(tmp_path, monkeypatch):
    path = tmp_path / 'fts.db'
    ids = [f'keep-{n}' for n in range(2048)]
    lexical.upsert_chunks(ids, ['unrelated body'] * len(ids), [{}] * len(ids), path=path, replace=False)
    connect = lexical._connect
    work = [0]
    def counted(path=None):
        db = connect(path)
        def tick():
            work[0] += 1000
            return 0
        db.set_progress_handler(tick, 1000)
        return db
    monkeypatch.setattr(lexical, '_connect', counted)
    lexical.upsert_chunks([f'new-{n}' for n in range(32)], ['new body'] * 32, [{}] * 32, path=path)
    assert work[0] < 200_000, 'Batch replacement must not perform one full FTS scan per ID'
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT count(*) FROM chunks_fts').fetchone()[0] == 2080
        assert db.execute("SELECT count(*) FROM chunks_fts WHERE body='unrelated body'").fetchone()[0] == 2048


def test_large_upsert_replaces_duplicate_rows_and_preserves_unrelated_text(tmp_path):
    path = tmp_path / 'fts.db'
    ids = [f'id-{n}' for n in range(1100)] + ["quoted ' ? 日本語"]
    lexical.upsert_chunks(ids + ['keep'], ['old'] * len(ids) + ['untouched'], [{}] * (len(ids) + 1), path=path, replace=False)
    lexical.upsert_chunks(['id-42'], ['duplicate'], [{}], path=path, replace=False)
    lexical.upsert_chunks(ids, ['replacement'] * len(ids), [{}] * len(ids), path=path)
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT count(*) FROM chunks_fts').fetchone()[0] == len(ids) + 1
        assert db.execute("SELECT body FROM chunks_fts WHERE chunk_id='keep'").fetchone()[0] == 'untouched'
        assert db.execute("SELECT count(*) FROM chunks_fts WHERE body='replacement'").fetchone()[0] == len(ids)
        assert db.execute("SELECT count(*) FROM chunks_fts WHERE chunk_id='id-42'").fetchone()[0] == 1


def test_upsert_failure_rolls_back_deletion_and_partial_insertion(tmp_path):
    path = tmp_path / 'fts.db'
    lexical.upsert_chunks(['a', 'keep'], ['original', 'untouched'], [{}, {}], path=path)
    with pytest.raises(sqlite3.Error):
        lexical.upsert_chunks(['a', 'b'], ['replacement', {'invalid': 'body'}], [{}, {}], path=path)
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT chunk_id,body FROM chunks_fts ORDER BY chunk_id').fetchall() == [('a', 'original'), ('keep', 'untouched')]
