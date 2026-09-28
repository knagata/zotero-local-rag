from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rag_mcp_server


class FakeCollection:
    def get(self, *, ids, include):
        assert ids == ["ATT:p3:para4:part0"]
        return {
            "ids": ids,
            "documents": ["永井威三郎の愚痴"],
            "metadatas": [{"itemKey": "ITEM", "retrieval_policy": "normal"}],
        }


def test_exact_lexical_hit_survives_short_fragment_filter():
    with patch(
        "lexical_index.search_chunks",
        return_value=[{"chunk_id": "ATT:p3:para4:part0"}],
    ):
        fused = rag_mcp_server._fuse_lexical_results(
            {}, ["永井威三郎の愚痴"], internal_k=5,
            include_notes=False, include_item_keys=["ITEM"], col=FakeCollection(),
        )

    ranked = rag_mcp_server._rank_rag_hits(
        fused, k=5, exclude_set=set(), min_return_chars=200,
        allow_explicit=False, include_corrupted=False, language_balance=False,
    )

    assert [chunk_id for chunk_id, _ in ranked] == ["ATT:p3:para4:part0"]
    assert ranked[0][1]["lexical_match"] is True


def test_japanese_semantic_hit_uses_compact_source_floor():
    hits = {
        "ATT:p97:para9:part0": {
            "distance": 0.1,
            "rrf_score": 1.0,
            "document": "永井の文章には、ここまで自分を卑下する、あるいは愚痴のような表現が見られる。",
            "metadata": {"lang": "ja", "retrieval_policy": "normal"},
        },
    }

    ranked = rag_mcp_server._rank_rag_hits(
        hits, k=5, exclude_set=set(), min_return_chars=200,
        allow_explicit=False, include_corrupted=False, language_balance=False,
    )

    assert [chunk_id for chunk_id, _ in ranked] == ["ATT:p97:para9:part0"]
