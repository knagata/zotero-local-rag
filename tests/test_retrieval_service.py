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
