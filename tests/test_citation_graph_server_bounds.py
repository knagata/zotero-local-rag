"""What the graph server does with a large request, a broken read, and two counts.

The graph is the far-reading half of this project: the user is looking across
the library for something they did not know to ask for. That makes its failure
mode specific -- not corruption, but a screen that looks complete and is not.
An edge whose contexts could not be read draws as an edge with nothing quoted,
which is a reason to stop looking; a node whose S2 total sits under the same
label as the count actually drawn invites a comparison between two different
numbers.

Three things are checked, each of which was wrong:

* the translation route bounded neither how many texts nor how long they were,
  on a paid, per-character API reachable from the page;
* a database failure in ``get_contexts_for_edge`` returned ``[]``;
* ``被引用数`` was S2's total for the work when S2 had one, and the graph's own
  count otherwise -- one label, two meanings, differing by as much as 4,188
  against 191.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from citation_graph import server  # noqa: E402


@pytest.fixture()
def client():
    return TestClient(server.create_app())


def test_too_many_texts_is_refused_before_the_paid_call(client):
    with patch.dict(server._state, {"translator": {"key": "k", "region": "r"}}):
        with patch("requests.post") as upstream:
            response = client.post("/api/translate/batch", json={
                "texts": ["x"] * (server.TRANSLATE_MAX_TEXTS + 1),
            })
    assert response.status_code == 422, response.text
    assert not upstream.called, "the request reached Azure before being rejected"


def test_one_enormous_text_is_refused(client):
    with patch.dict(server._state, {"translator": {"key": "k", "region": "r"}}):
        with patch("requests.post") as upstream:
            response = client.post("/api/translate/batch", json={
                "texts": ["x" * (server.TRANSLATE_MAX_TEXT_CHARS + 1)],
            })
    assert response.status_code == 413, response.text
    assert not upstream.called


def test_many_medium_texts_are_refused_by_the_total(client):
    # Each one is allowed; together they are not. Without the total, the
    # per-text limit is a suggestion.
    count = (server.TRANSLATE_MAX_TOTAL_CHARS // server.TRANSLATE_MAX_TEXT_CHARS) + 2
    assert count <= server.TRANSLATE_MAX_TEXTS
    with patch.dict(server._state, {"translator": {"key": "k", "region": "r"}}):
        with patch("requests.post") as upstream:
            response = client.post("/api/translate/batch", json={
                "texts": ["x" * server.TRANSLATE_MAX_TEXT_CHARS] * count,
            })
    assert response.status_code == 413, response.text
    assert not upstream.called


def test_an_upstream_failure_does_not_come_back_verbatim(client):
    # The upstream error can carry the request URL, and the subscription key
    # travels in that request's headers.
    with patch.dict(server._state, {"translator": {"key": "SECRET-KEY", "region": "r"}}):
        with patch("requests.post", side_effect=RuntimeError(
            "401 for https://api.cognitive.microsofttranslator.com/translate?key=SECRET-KEY"
        )):
            response = client.post("/api/translate/batch", json={"texts": ["hello"]})
    assert response.status_code == 502
    assert "SECRET-KEY" not in response.text


def test_contexts_that_could_not_be_read_are_not_reported_as_none(client):
    with patch.object(server, "get_contexts_for_edge", side_effect=RuntimeError("db is gone")):
        response = client.get("/api/edge/contexts", params={"src": "A", "tgt": "B"})
    assert response.status_code == 500
    payload = response.json()
    assert payload["contexts"] == []
    assert "db is gone" in payload["error"], (
        "the reader is shown an edge with nothing quoted, which is a reason to "
        "stop looking, when the truth is that nothing could be read"
    )


def test_the_context_reader_itself_raises_rather_than_returning_nothing(tmp_path):
    # The route test above patches the reader, so it says nothing about what
    # the reader does. This is the half that was wrong: a database it could not
    # read came back as an edge with no quoted context.
    not_a_database = tmp_path / "graph.db"
    not_a_database.write_bytes(b"this is not a sqlite file")
    # Real node ids: an edge shape the reader actually queries for. With
    # anything else it returns [] before touching the database, which is how
    # the first version of this test passed while proving nothing.
    with pytest.raises(Exception) as caught:
        server.get_contexts_for_edge(str(not_a_database), "item:AAAA1111", "paper:S2-1")
    assert "could not read citation contexts" in str(caught.value)


def test_the_migration_only_ignores_a_duplicate_column(tmp_path):
    # It used to retry five ALTER TABLEs under except Exception: pass, which
    # covers a locked database and a read-only file just as quietly.
    database = tmp_path / "graph.db"
    with sqlite3.connect(database) as connection:
        server._ensure_override_table(connection)
        server._ensure_override_table(connection)  # second run is the duplicate case
        columns = {row[1] for row in connection.execute(
            "PRAGMA table_info(node_identifier_overrides)"
        )}
    assert {"doi", "isbn", "title", "year", "authors", "citations"} <= columns

    class Refusing:
        def execute(self, statement, *args):
            if statement.strip().upper().startswith("ALTER"):
                raise sqlite3.OperationalError("attempt to write a readonly database")
            return None

    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        server._ensure_override_table(Refusing())


def test_a_node_reports_the_graph_count_separately_from_the_s2_total():
    # Both numbers are legitimate; sharing a label is what made them wrong.
    source = (ROOT / "citation_graph" / "server.py").read_text(encoding="utf-8")
    assert '"citationsInGraph"' in source
    assert '"citationsSource"' in source
    app_js = (ROOT / "citation_graph" / "static" / "app.js").read_text(encoding="utf-8")
    assert "被引用数（S2全体）" in app_js, (
        "the tooltip still labels S2's total for the work as if it were the "
        "number of citing papers the graph drew"
    )
