from __future__ import annotations

import json
from unittest.mock import Mock, patch

from citation_graph import server


def _rows():
    items = [{"item_key": "ITEM", "citer_count": 3}]
    citers = [{
        "cited_item_key": "ITEM",
        "citing_paper_id": "BOTH",
        "citing_citation_count": 7,
    }]
    refs = [
        {
            "citing_item_key": "ITEM",
            "cited_paper_id": "BOTH",
            "cited_citation_count": 9,
        },
        {
            "citing_item_key": "ITEM",
            "cited_paper_id": "REF_ONLY",
            "cited_citation_count": 5,
        },
    ]
    return items, citers, refs


def test_layout_cache_ids_match_the_node_ids_used_by_layout():
    items, citers, refs = _rows()

    assert server._layout_node_ids(items, citers, refs) == [
        "item:ITEM",
        "paper:BOTH",
        "ref:REF_ONLY",
    ]


def test_exact_layout_cache_hit_avoids_vectors_and_layout(tmp_path):
    items, citers, refs = _rows()
    cache_path = tmp_path / "layout.json"
    expected = {
        "item:ITEM": (1.0, 2.0),
        "paper:BOTH": (3.0, 4.0),
        "ref:REF_ONLY": (5.0, 6.0),
    }
    with patch.object(server, "get_item_vectors", return_value={}), patch.object(
        server, "compute_layout", return_value=expected,
    ) as compute:
        first, first_hit = server._load_or_compute_layout(
            items, citers, refs, cache_path=cache_path,
        )
    assert first == expected
    assert first_hit is False
    compute.assert_called_once()

    with patch.object(
        server, "get_item_vectors", side_effect=AssertionError("vectors loaded on cache hit"),
    ), patch.object(
        server, "compute_layout", side_effect=AssertionError("layout ran on cache hit"),
    ):
        second, second_hit = server._load_or_compute_layout(
            items, citers, refs, cache_path=cache_path,
        )

    assert second == expected
    assert second_hit is True


def test_stale_cache_is_a_warm_start_and_is_replaced(tmp_path):
    items, citers, refs = _rows()
    cache_path = tmp_path / "layout.json"
    stale = {"item:ITEM": [10.0, 20.0]}
    cache_path.write_text(json.dumps({"key": "old", "positions": stale}))
    computed = {"item:ITEM": (11.0, 21.0)}
    layout = Mock(return_value=computed)

    with patch.object(server, "get_item_vectors", return_value={"ITEM": [1.0]}), patch.object(
        server, "compute_layout", layout,
    ):
        positions, cache_hit = server._load_or_compute_layout(
            items, citers, refs, cache_path=cache_path,
        )

    assert positions == computed
    assert cache_hit is False
    assert layout.call_args.kwargs["warm_start"] == {"item:ITEM": (10.0, 20.0)}
    assert layout.call_args.kwargs["semantic_vectors"] == {"ITEM": [1.0]}
    assert set(layout.call_args.kwargs["node_sizes"]) == {
        "item:ITEM", "paper:BOTH", "ref:REF_ONLY",
    }
    saved = json.loads(cache_path.read_text())
    assert saved["key"] != "old"
    assert saved["positions"] == {"item:ITEM": [11.0, 21.0]}


def test_corrupt_cache_is_nonfatal_and_recomputed(tmp_path):
    items, citers, refs = _rows()
    cache_path = tmp_path / "layout.json"
    cache_path.write_text("not json")

    with patch.object(server, "get_item_vectors", return_value={}), patch.object(
        server, "compute_layout", return_value={"item:ITEM": (1.0, 1.0)},
    ) as layout:
        positions, cache_hit = server._load_or_compute_layout(
            items, citers, refs, cache_path=cache_path,
        )

    assert positions == {"item:ITEM": (1.0, 1.0)}
    assert cache_hit is False
    layout.assert_called_once()
