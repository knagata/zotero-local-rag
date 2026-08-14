from __future__ import annotations

import sqlite3

from citation_graph import server


def _metadata_database(path, rows):
    with sqlite3.connect(path) as connection:
        connection.execute("""
            CREATE TABLE embedding_metadata (
                id INTEGER NOT NULL, key TEXT NOT NULL, string_value TEXT
            )
        """)
        connection.executemany(
            "INSERT INTO embedding_metadata (id, key, string_value) VALUES (?, ?, ?)",
            rows,
        )


def test_item_title_uses_the_repeated_bibliographic_value_not_a_heading(
    tmp_path, monkeypatch,
):
    database = tmp_path / "chroma.sqlite3"
    rows = []
    titles = ["Bodies that matter"] * 3 + ["Preface"]
    for identifier, title in enumerate(titles, start=1):
        rows.extend([
            (identifier, "itemKey", "ITEM"),
            (identifier, "title", title),
            (identifier, "creators", "Butler, Judith"),
        ])
    _metadata_database(database, rows)
    monkeypatch.setattr(server, "CHROMA_DB", str(database))

    assert server.get_item_meta(["ITEM"]) == {
        "ITEM": {
            "title": "Bodies that matter",
            "creators": "Butler, Judith",
            "year": "",
        },
    }


def test_item_metadata_ties_are_deterministic(tmp_path, monkeypatch):
    database = tmp_path / "chroma.sqlite3"
    _metadata_database(database, [
        (1, "itemKey", "ITEM"), (1, "title", "Short"),
        (2, "itemKey", "ITEM"), (2, "title", "Longer title"),
    ])
    monkeypatch.setattr(server, "CHROMA_DB", str(database))

    assert server.get_item_meta(["ITEM"])["ITEM"]["title"] == "Longer title"
