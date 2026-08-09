from __future__ import annotations

import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from citation_graph.graph_service import GraphBuildService, GraphSnapshot


def _args(**overrides):
    values = {
        "item": None,
        "top": 10,
        "citers": 20,
        "refs": 30,
        "min_cc": 2,
        "no_refs": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _service(*, item_row=None, top_items=None, citers=None, refs=None):
    get_item_row = Mock(return_value=item_row)
    get_top_items = Mock(return_value=top_items or [])
    get_citers = Mock(return_value=citers or [])
    get_refs = Mock(return_value=refs or [])
    get_item_meta = Mock(return_value={"ITEM": {"title": "A book"}})
    get_item_ref_counts = Mock(return_value={"ITEM": 3})
    build_graph_data = Mock(return_value={
        "nodes": [{"id": "ITEM"}],
        "edges": [{"id": "edge-1"}],
        "meta": {
            "n_items": 1,
            "n_nodes": 2,
            "n_edges": 1,
            "n_citer": 1,
            "n_ref": 0,
            "palette": {"node": "#fff"},
            "css_root": "root",
            "js_theme": "light",
        },
        "cache_hit": True,
    })
    build_html = Mock(return_value="<html>graph</html>")
    service = GraphBuildService(
        get_item_row=get_item_row,
        get_top_items=get_top_items,
        get_citers=get_citers,
        get_refs=get_refs,
        get_item_meta=get_item_meta,
        get_item_ref_counts=get_item_ref_counts,
        build_graph_data=build_graph_data,
        build_html=build_html,
    )
    return service, {
        "get_item_row": get_item_row,
        "get_top_items": get_top_items,
        "get_citers": get_citers,
        "get_refs": get_refs,
        "get_item_meta": get_item_meta,
        "get_item_ref_counts": get_item_ref_counts,
        "build_graph_data": build_graph_data,
        "build_html": build_html,
    }


def test_rebuild_item_selection_builds_a_complete_snapshot(tmp_path):
    database = tmp_path / "relations.db"
    database.touch()
    service, calls = _service(
        item_row={"item_key": "ITEM"},
        citers=[{"citing_paper_id": "CITER"}],
        refs=[{"cited_paper_id": "REF"}],
    )

    snapshot = service.rebuild(str(database), _args(item="ITEM"))

    assert isinstance(snapshot, GraphSnapshot)
    assert snapshot.graph == {
        "nodes": [{"id": "ITEM"}],
        "edges": [{"id": "edge-1"}],
    }
    assert snapshot.html == "<html>graph</html>"
    assert snapshot.cache_hit is True
    calls["get_item_row"].assert_called_once()
    calls["get_top_items"].assert_not_called()
    calls["get_citers"].assert_called_once()
    calls["get_refs"].assert_called_once()
    calls["build_graph_data"].assert_called_once()
    calls["build_html"].assert_called_once_with(
        n_items=1,
        n_nodes=2,
        n_edges=1,
        n_citer=1,
        n_ref=0,
        palette={"node": "#fff"},
        css_root="root",
        js_theme="light",
    )


def test_rebuild_top_selection_can_disable_references(tmp_path):
    database = tmp_path / "relations.db"
    database.touch()
    service, calls = _service(
        top_items=[{"item_key": "ITEM"}, {"item_key": "ITEM2"}],
        citers=[{"citing_paper_id": "CITER"}],
    )

    snapshot = service.rebuild(str(database), _args(top=2, no_refs=True))

    assert isinstance(snapshot, GraphSnapshot)
    calls["get_top_items"].assert_called_once()
    assert calls["get_top_items"].call_args.args[1] == 2
    calls["get_item_row"].assert_not_called()
    calls["get_refs"].assert_not_called()
    calls["get_citers"].assert_called_once()
    build_args = calls["build_graph_data"].call_args
    assert build_args.args[1] == [{"citing_paper_id": "CITER"}]
    assert build_args.args[2] == []


def test_rebuild_returns_none_without_items_and_does_not_render(tmp_path):
    database = tmp_path / "relations.db"
    database.touch()
    service, calls = _service(item_row=None)

    result = service.rebuild(str(database), _args(item="MISSING"))

    assert result is None
    calls["get_citers"].assert_not_called()
    calls["get_refs"].assert_not_called()
    calls["build_graph_data"].assert_not_called()
    calls["build_html"].assert_not_called()


def test_rebuild_closes_connection_when_citer_read_fails(tmp_path, monkeypatch):
    database = tmp_path / "relations.db"
    database.touch()
    connection = sqlite3.connect(":memory:")
    service, calls = _service(item_row={"item_key": "ITEM"})
    calls["get_citers"].side_effect = RuntimeError("database read failed")
    monkeypatch.setattr("citation_graph.graph_service.sqlite3.connect", lambda _path: connection)

    with pytest.raises(RuntimeError, match="database read failed"):
        service.rebuild(str(database), _args(item="ITEM"))

    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")
