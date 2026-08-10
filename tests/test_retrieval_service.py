from types import SimpleNamespace
from unittest.mock import Mock

from src.retrieval_service import (
    HierarchicalRetrievalDependencies,
    RetrievalService,
)


def test_direct_retrieval_survives_unavailable_summary_collection():
    search = Mock(return_value={
        "results": [{
            "id": "chunk-1", "rrf_score": 0.5,
            "meta": {"itemKey": "ITEM", "title": "Book", "year": 2020},
        }],
    })
    dependencies = HierarchicalRetrievalDependencies(
        search=search,
        searchable_node_ids=lambda _ids: set(),
        descendant_chunks=lambda _ids: [],
        descendant_leaf_ids=lambda _ids: [],
        item_root_summaries=lambda _keys, **_kwargs: {},
        fuse_paths=lambda paths, **_kwargs: paths[-1][1],
    )
    service = RetrievalService(
        dependencies, summary_collection_name="summaries", rrf_k=60,
    )
    paragraph_collection = SimpleNamespace(
        _chroma_client=SimpleNamespace(
            get_collection=Mock(side_effect=RuntimeError("missing")),
        ),
    )

    response = service.hierarchical_search(
        ["query"], k=5, k_items=5, where=None,
        include_direct=True, return_summaries=False,
        paragraph_collection=paragraph_collection,
    )

    assert [row["id"] for row in response["results"]] == ["chunk-1"]
    assert response["results"][0]["hierarchical_rrf_score"] == 0.5
    assert "sum_node collection unavailable" in response["warnings"][0]
    search.assert_called_once_with(
        ["query"], k=100, where=None, auto_expand=False, hybrid=True,
    )


def test_summary_candidates_route_leaf_and_item_evidence_and_attach_root_summary():
    search = Mock(side_effect=[
        {"results": [{
            "id": "leaf-hit", "rrf_score": 0.7,
            "meta": {"itemKey": "ITEM", "title": "Book", "year": 2020},
        }]},
        {"results": [{
            "id": "item-hit", "rrf_score": 0.4,
            "meta": {"itemKey": "ITEM", "title": "Book", "year": 2020},
        }]},
    ])
    fuse = Mock(return_value=[{
        "id": "leaf-hit", "rrf_score": 0.9,
        "meta": {"itemKey": "ITEM", "title": "Book", "year": 2020},
    }])
    dependencies = HierarchicalRetrievalDependencies(
        search=search,
        searchable_node_ids=lambda ids: set(ids),
        descendant_chunks=lambda _ids: ["leaf-hit"],
        descendant_leaf_ids=lambda _ids: ["LEAF", "LEAF"],
        item_root_summaries=lambda _keys, **_kwargs: {"ITEM": {"summary": "Root summary"}},
        fuse_paths=fuse,
    )
    service = RetrievalService(
        dependencies, summary_collection_name="summaries", rrf_k=60,
    )
    summary = SimpleNamespace(
        query=Mock(return_value={
            "metadatas": [[{
                "itemKey": "ITEM", "node_id": "NODE", "title": "Summary title",
                "node_type": "chapter", "depth": 1,
            }]],
            "documents": [["A summary used for routing"]],
        }),
    )
    collection = SimpleNamespace(get_collection=Mock(return_value=summary))
    paragraph_collection = SimpleNamespace(
        _chroma_client=collection,
        _embedding_function=Mock(return_value=[[0.1, 0.2]]),
    )

    response = service.hierarchical_search(
        ["query"], k=5, k_items=2, where={"source_type": "pdf"},
        include_direct=False, return_summaries=True,
        paragraph_collection=paragraph_collection,
    )

    assert response["candidate_nodes"][0]["node_id"] == "NODE"
    assert response["candidate_nodes"][0]["summary_snippet"] == "A summary used for routing"
    assert response["candidate_items"] == [{
        "item_key": "ITEM", "title": "Book", "year": 2020,
    }]
    assert response["results"][0]["hierarchical_rrf_score"] == 0.9
    assert response["results"][0]["item_summary_snippet"] == "Root summary"
    assert response["results"][0]["item_summary_provenance"] == "v3_item_root"
    assert search.call_count == 2
    assert search.call_args_list[0].kwargs["include_leaf_ids"] == ["LEAF"]
    assert search.call_args_list[1].kwargs["include_item_keys"] == ["ITEM"]
    fuse.assert_called_once()


def test_descendant_failures_are_warnings_and_do_not_abort_direct_search():
    search = Mock(return_value={"results": [{
        "id": "direct-hit", "rrf_score": 0.2,
        "meta": {"itemKey": "ITEM", "title": "Book", "year": 2020},
    }]})
    dependencies = HierarchicalRetrievalDependencies(
        search=search,
        searchable_node_ids=lambda ids: set(ids),
        descendant_chunks=lambda _ids: (_ for _ in ()).throw(RuntimeError("tree missing")),
        descendant_leaf_ids=lambda _ids: (_ for _ in ()).throw(RuntimeError("leaf missing")),
        item_root_summaries=lambda _keys, **_kwargs: {},
        fuse_paths=lambda paths, **_kwargs: paths[-1][1],
    )
    service = RetrievalService(
        dependencies, summary_collection_name="summaries", rrf_k=60,
    )
    summary = SimpleNamespace(
        query=Mock(return_value={
            "metadatas": [[{"itemKey": "ITEM", "node_id": "NODE"}]],
            "documents": [[]],
        }),
    )
    paragraph_collection = SimpleNamespace(
        _chroma_client=SimpleNamespace(get_collection=Mock(return_value=summary)),
        _embedding_function=lambda _queries: [[0.1]],
    )

    response = service.hierarchical_search(
        ["query"], k=2, k_items=1, where=None,
        include_direct=True, return_summaries=False,
        paragraph_collection=paragraph_collection,
    )

    assert [row["id"] for row in response["results"]] == ["direct-hit"]
    assert any("descendant lookup failed" in warning for warning in response["warnings"])
    assert any("leaf node lookup failed" in warning for warning in response["warnings"])
    assert search.call_count == 2
    assert "include_leaf_ids" not in search.call_args_list[0].kwargs
    assert search.call_args_list[0].kwargs["include_item_keys"] == ["ITEM"]
    assert "include_item_keys" not in search.call_args_list[1].kwargs
