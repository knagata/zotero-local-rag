"""Framework-independent citation graph rebuild orchestration."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class GraphSnapshot:
    graph: dict[str, Any]
    html: str
    cache_hit: bool


class GraphBuildService:
    """Coordinate graph reads and construction without FastAPI dependencies."""

    def __init__(
        self, *, get_item_row: Callable[..., Any], get_top_items: Callable[..., Any],
        get_citers: Callable[..., Any], get_refs: Callable[..., Any],
        get_item_meta: Callable[..., Any], get_item_ref_counts: Callable[..., Any],
        build_graph_data: Callable[..., dict[str, Any]],
        build_html: Callable[..., str],
    ) -> None:
        self._get_item_row = get_item_row
        self._get_top_items = get_top_items
        self._get_citers = get_citers
        self._get_refs = get_refs
        self._get_item_meta = get_item_meta
        self._get_item_ref_counts = get_item_ref_counts
        self._build_graph_data = build_graph_data
        self._build_html = build_html

    def rebuild(self, db_path: str, args: Any) -> GraphSnapshot | None:
        connection = sqlite3.connect(db_path)
        connection.row_factory = sqlite3.Row
        try:
            if args.item:
                item_row = self._get_item_row(connection, args.item)
                items = [item_row] if item_row else []
            else:
                items = self._get_top_items(connection, args.top)
            if not items:
                return None
            item_keys = [item["item_key"] for item in items]
            citers = self._get_citers(
                connection, item_keys, args.citers, min_cc=args.min_cc,
            )
            refs = [] if args.no_refs else self._get_refs(
                connection, item_keys, args.refs, min_cc=args.min_cc,
            )
            item_meta = self._get_item_meta(item_keys)
            item_ref_counts = self._get_item_ref_counts(connection, item_keys)
        finally:
            connection.close()

        result = self._build_graph_data(
            items, citers, refs, item_meta=item_meta,
            item_ref_counts=item_ref_counts, db_path=db_path,
        )
        meta = result["meta"]
        html = self._build_html(
            n_items=meta["n_items"], n_nodes=meta["n_nodes"],
            n_edges=meta["n_edges"], n_citer=meta["n_citer"],
            n_ref=meta["n_ref"], palette=meta["palette"],
            css_root=meta["css_root"], js_theme=meta["js_theme"],
        )
        return GraphSnapshot(
            graph={"nodes": result["nodes"], "edges": result["edges"]},
            html=html, cache_hit=bool(result.get("cache_hit", False)),
        )
