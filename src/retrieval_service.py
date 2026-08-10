"""Framework-independent orchestration for hierarchical evidence retrieval."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class HierarchicalRetrievalDependencies:
    search: Callable[..., dict[str, Any]]
    searchable_node_ids: Callable[[list[str]], set[str]]
    descendant_chunks: Callable[[list[str]], list[str]]
    descendant_leaf_ids: Callable[[list[str]], list[str]]
    item_root_summaries: Callable[..., dict[str, dict[str, Any]]]
    fuse_paths: Callable[..., list[dict[str, Any]]]


class RetrievalService:
    def __init__(
        self, dependencies: HierarchicalRetrievalDependencies, *,
        summary_collection_name: str, rrf_k: int,
    ) -> None:
        self._deps = dependencies
        self._summary_collection_name = summary_collection_name
        self._rrf_k = rrf_k

    def _collect_summary_candidates(
        self, queries: list[str], k_items: int, paragraph_collection: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, dict[str, Any]], list[str]]:
        """Query summary nodes, retaining only candidates with usable identity."""
        warnings: list[str] = []
        candidate_nodes: list[dict[str, Any]] = []
        candidate_scores: dict[str, float] = {}
        candidate_items: dict[str, dict[str, Any]] = {}
        try:
            client = getattr(paragraph_collection, "_chroma_client", None)
            collection = client.get_collection(self._summary_collection_name)
            embeddings = paragraph_collection._embedding_function(queries)
            response = collection.query(
                query_embeddings=embeddings, n_results=max(k_items * 3, 30),
                where={"summary_kind": "llm"},
                include=["metadatas", "documents", "distances"],
            )
            documents_by_query = response.get("documents") or []
            for query_index, metadata_rows in enumerate(response.get("metadatas") or []):
                documents = (
                    documents_by_query[query_index]
                    if query_index < len(documents_by_query) else []
                )
                for rank, metadata in enumerate(metadata_rows or [], start=1):
                    if not isinstance(metadata, dict):
                        continue
                    if not metadata.get("itemKey") or not metadata.get("node_id"):
                        continue
                    node_id = str(metadata["node_id"])
                    item_key = str(metadata["itemKey"])
                    score = 1.0 / (self._rrf_k + rank)
                    candidate_scores[node_id] = candidate_scores.get(node_id, 0.0) + score
                    candidate_items.setdefault(item_key, metadata)
                    if not any(row["node_id"] == node_id for row in candidate_nodes):
                        document = documents[rank - 1] if rank - 1 < len(documents) else ""
                        candidate_nodes.append({
                            "node_id": node_id, "item_key": item_key,
                            "title": metadata.get("title"),
                            "node_type": metadata.get("node_type"),
                            "depth": metadata.get("depth"), "score": score,
                            "summary_snippet": str(document or "")[:180],
                        })
        except Exception as exc:
            warnings.append(f"sum_node collection unavailable: {exc}")
        return candidate_nodes, candidate_scores, candidate_items, warnings

    def _prepare_hierarchical_routes(
        self,
        candidate_nodes: list[dict[str, Any]],
        candidate_scores: dict[str, float],
        candidate_items: dict[str, dict[str, Any]],
        k_items: int,
        warnings: list[str],
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str], list[str], dict[str, list[str]]]:
        """Filter summary candidates and resolve their descendant routes."""
        candidate_nodes.sort(
            key=lambda row: (-candidate_scores[row["node_id"]], row["node_id"]),
        )
        searchable = self._deps.searchable_node_ids([
            str(node["node_id"]) for node in candidate_nodes
        ])
        candidate_nodes = [
            node for node in candidate_nodes if str(node["node_id"]) in searchable
        ]
        candidate_item_scores: dict[str, float] = {}
        for node in candidate_nodes:
            item_key = str(node["item_key"])
            candidate_item_scores[item_key] = (
                candidate_item_scores.get(item_key, 0.0)
                + candidate_scores[str(node["node_id"])]
            )
        candidate_items = {
            key: metadata for key, metadata in candidate_items.items()
            if key in candidate_item_scores
        }
        candidate_nodes = candidate_nodes[:max(k_items * 2, k_items)]

        descendant_by_node: dict[str, set[str]] = {}
        for node in candidate_nodes:
            try:
                descendant_by_node[node["node_id"]] = set(
                    self._deps.descendant_chunks([node["node_id"]]),
                )
            except Exception as exc:
                warnings.append(f"descendant lookup failed for {node['node_id']}: {exc}")
                descendant_by_node[node["node_id"]] = set()
        routed_ids = set().union(*descendant_by_node.values()) if descendant_by_node else set()
        routed_nodes_by_chunk = {
            chunk_id: [
                node_id for node_id, chunk_ids in descendant_by_node.items()
                if chunk_id in chunk_ids
            ]
            for chunk_id in routed_ids
        }
        leaf_ids: list[str] = []
        if candidate_nodes:
            try:
                leaf_ids = self._deps.descendant_leaf_ids([
                    node["node_id"] for node in candidate_nodes
                ])
            except Exception as exc:
                warnings.append(f"leaf node lookup failed: {exc}")
        candidate_item_keys = sorted(
            candidate_item_scores,
            key=lambda key: (-candidate_item_scores[key], key),
        )[:k_items]
        return candidate_nodes, candidate_items, candidate_item_keys, leaf_ids, routed_nodes_by_chunk

    def _search_hierarchical_evidence(
        self, queries: list[str], *, k: int, where: Any,
        candidate_item_keys: list[str], leaf_ids: list[str], include_direct: bool,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Run leaf, same-item, and optional direct evidence searches."""
        leaf_response = self._deps.search(
            queries, k=max(k * 12, 60), where=where, include_leaf_ids=leaf_ids,
            auto_expand=False, hybrid=False,
        ) if leaf_ids else {"results": []}
        item_response = self._deps.search(
            queries, k=max(k * 12, 60), where=where,
            include_item_keys=candidate_item_keys or None,
            auto_expand=False, hybrid=True,
        ) if candidate_item_keys else {"results": []}
        direct_response = self._deps.search(
            queries, k=max(k * 20, 100), where=where,
            auto_expand=False, hybrid=True,
        ) if include_direct else {"results": []}
        return leaf_response, item_response, direct_response

    def hierarchical_search(
        self, queries: list[str], *, k: int, k_items: int, where: Any,
        include_direct: bool, return_summaries: bool, paragraph_collection: Any,
    ) -> dict[str, Any]:
        """Route summary-node hits to descendants and fuse direct evidence."""
        candidate_nodes, candidate_scores, candidate_items, warnings = (
            self._collect_summary_candidates(queries, k_items, paragraph_collection)
        )
        (
            candidate_nodes, candidate_items, candidate_item_keys, leaf_ids,
            routed_nodes_by_chunk,
        ) = self._prepare_hierarchical_routes(
            candidate_nodes, candidate_scores, candidate_items, k_items, warnings,
        )
        leaf_ids = list(dict.fromkeys(leaf_ids))
        leaf_response, item_response, direct_response = self._search_hierarchical_evidence(
            queries, k=k, where=where, candidate_item_keys=candidate_item_keys,
            leaf_ids=leaf_ids, include_direct=include_direct,
        )

        bib_by_item: dict[str, dict[str, Any]] = {}
        for rows in (
            leaf_response.get("results") or [], item_response.get("results") or [],
            direct_response.get("results") or [],
        ):
            for row in rows:
                metadata = row.get("meta") or {}
                key = metadata.get("itemKey")
                if key and key not in bib_by_item:
                    bib_by_item[key] = {
                        "title": metadata.get("title"), "year": metadata.get("year"),
                    }

        fused_rows = self._deps.fuse_paths([
            ("leaf", leaf_response.get("results") or []),
            ("same_item", item_response.get("results") or []),
            ("direct", direct_response.get("results") or []),
        ], routed_nodes_by_chunk=routed_nodes_by_chunk)
        root_summaries = self._deps.item_root_summaries(
            list(dict.fromkeys(
                str((row.get("meta") or {}).get("itemKey") or "")
                for row in fused_rows[:k] if (row.get("meta") or {}).get("itemKey")
            )),
            searchable_only=True,
        ) if return_summaries else {}
        results = []
        for row in fused_rows[:k]:
            hit = dict(row)
            hit["hierarchical_rrf_score"] = hit.pop("rrf_score")
            if return_summaries:
                item_key = (hit.get("meta") or {}).get("itemKey")
                summary = root_summaries.get(str(item_key)) if item_key else None
                hit["item_summary_snippet"] = (
                    (summary.get("summary") or "")[:120] if summary else ""
                )
                hit["item_summary_provenance"] = "v3_item_root" if summary else "none"
            results.append(hit)

        result: dict[str, Any] = {
            "results": results, "candidate_nodes": candidate_nodes,
            "candidate_items": [{
                "item_key": key,
                "title": (bib_by_item.get(key) or {}).get("title") or metadata.get("title"),
                "year": (bib_by_item.get(key) or {}).get("year"),
            } for key, metadata in candidate_items.items()],
            "reporting_obligation": (
                "Verify claims against result chunks. If a summary concretely contradicts "
                "them, call report_summary_quality before completing the answer."
            ),
        }
        if warnings:
            result["warnings"] = warnings
        return result
