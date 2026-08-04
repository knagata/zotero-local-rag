"""Canonical, source-order-preserving document structure v2.

The builder deliberately does not infer relationships from semantic similarity.
It consumes extractor metadata in document order, preserves every contiguous
heading run, and uses deterministic contiguous segments only when no structure
is available.  This keeps the tree safe to use as a retrieval routing map.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Sequence

try:
    from .chunk_store import natural_chunk_key
except ImportError:  # pragma: no cover - direct src entrypoint
    from chunk_store import natural_chunk_key


STRUCTURE_VERSION = "4"
TARGET_SEGMENT_CHARS = 18_000
MAX_SEGMENT_CHARS = 30_000
THIN_CONTAINER_MAX_DIRECT_CHARS = 300
THIN_CONTAINER_MIN_DESCENDANT_CHARS = 1_000

ZONE_POLICIES = {
    "body": ("include", "normal", "none"),
    "footnote": ("exclude", "explicit_only", "extract"),
    # Endnotes are searchable like body text (user decision, 2026-07-28). In the
    # humanities the substantive argument often lives in them -- one essay here
    # carries 12,717 characters in note [1] alone -- so gating them behind a
    # query that happens to contain the word "note" loses exactly the buried
    # material this index exists to surface. Footnotes stay explicit_only:
    # they are predominantly bibliographic. The zone travels with each result
    # so a reader can still see that a passage came from the notes.
    # summary_policy stays "exclude": notes must not feed a section's summary.
    "endnote": ("exclude", "normal", "extract"),
    "bibliography": ("exclude", "exclude", "extract"),
    "toc": ("exclude", "exclude", "none"),
    "index": ("exclude", "exclude", "none"),
    "colophon": ("exclude", "exclude", "none"),
    "front_matter": ("include", "normal", "none"),
    "back_matter": ("include", "normal", "none"),
    "other_paratext": ("exclude", "exclude", "none"),
    # Text known to be garbage -- OCR noise from a figure, a page that resisted
    # repair. Kept rather than discarded (note 79, U3) so it stays inspectable
    # and can be searched deliberately, but excluded from ordinary retrieval and
    # from summary input so it cannot pollute either.
    "corrupted": ("exclude", "exclude", "none"),
}


def normalise_zone(value: Any) -> str:
    zone = str(value or "body").strip().casefold().replace("-", "_")
    return zone if zone in ZONE_POLICIES else "body"


_PARATEXT_TITLE_ZONES = (
    (re.compile(r"^(?:contents|table of contents|目次)$", re.IGNORECASE), "toc"),
    (re.compile(r"^(?:cover|title page|half[ -]?title|扉)$", re.IGNORECASE), "other_paratext"),
    (re.compile(r"^(?:copyright(?: page)?|colophon|奥付)$", re.IGNORECASE), "colophon"),
    (re.compile(r"^(?:bibliography|references|参考文献|文献一覧)$", re.IGNORECASE), "bibliography"),
    (re.compile(r"^(?:index|索引)$", re.IGNORECASE), "index"),
    (re.compile(r"^(?:notes|endnotes|注)$", re.IGNORECASE), "endnote"),
)


def _heading_zone(path: Sequence[str], supplied_zone: Any) -> str:
    """Use an exact heading title to repair only an otherwise-body zone."""
    zone = normalise_zone(supplied_zone)
    if zone != "body" or not path:
        return zone
    title = _normalise_title(path[-1])
    for pattern, inferred in _PARATEXT_TITLE_ZONES:
        if pattern.fullmatch(title):
            return inferred
    return zone


def source_fingerprint(chunks: Sequence[Dict[str, Any]]) -> str:
    """Fingerprint text plus every normalized input that can change the tree."""
    digest = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda value: natural_chunk_key(str(value.get("id") or ""))):
        metadata = chunk.get("metadata") or {}
        path = _metadata_path(metadata)
        payload = {
            "id": str(chunk.get("id") or ""),
            "attachment": _attachment_key(chunk),
            "attachment_title": _normalise_title(
                metadata.get("filename") or metadata.get("title")
            ),
            "locator": str(metadata.get("locator") or ""),
            "path": path,
            "roles": _metadata_roles(metadata, len(path)),
            "zone": _heading_zone(path, metadata.get("zone")),
            "text": str(chunk.get("text") or ""),
        }
        digest.update(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"


def _normalise_title(value: Any) -> str:
    return " ".join(str(value or "").split())


def _node_id(*, item_key: str, kind: str, locator: Dict[str, Any]) -> str:
    payload = json.dumps(
        {"item_key": item_key, "kind": kind, "locator": locator},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return "dn:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _metadata_path(metadata: Dict[str, Any]) -> List[str]:
    """Read the richest available heading path without guessing hierarchy."""
    raw_path = metadata.get("structure_path") or metadata.get("heading_path")
    if isinstance(raw_path, str):
        try:
            raw_path = json.loads(raw_path)
        except (TypeError, ValueError):
            raw_path = [raw_path]
    if isinstance(raw_path, (list, tuple)):
        values: List[str] = []
        for value in raw_path:
            if isinstance(value, dict):
                value = value.get("title") or value.get("text")
            title = _normalise_title(value)
            if title:
                values.append(title)
        if values:
            return values
    values = [_normalise_title(metadata.get("chapter")), _normalise_title(metadata.get("section"))]
    return [value for value in values if value]


def _metadata_roles(metadata: Dict[str, Any], path_length: int) -> List[str]:
    raw = metadata.get("structure_roles")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            raw = []
    if not isinstance(raw, (list, tuple)):
        return []
    roles = [str(value).strip() for value in raw[:path_length]]
    return roles if len(roles) == path_length else []


def _heading_type(depth: int) -> str:
    if depth == 1:
        return "chapter"
    if depth == 2:
        return "section"
    return "subsection"


_ZOTERO_KEY_RE = re.compile(r"^[A-Za-z0-9]{8}$")


def _attachment_key(chunk: Dict[str, Any]) -> str:
    """Return the best attachment identity available for one source chunk.

    Older enrichment rows can lack ``attachmentKey`` even though their chunk
    IDs retain the canonical ``<attachment-key>:...`` namespace.  Prefer
    explicit metadata, but recover that namespace when it is unambiguous.
    """
    metadata = chunk.get("metadata") or {}
    explicit = str(metadata.get("attachmentKey") or metadata.get("attachment_key") or "").strip()
    if explicit:
        return explicit
    prefix = str(chunk.get("id") or "").split(":", 1)[0]
    return prefix if _ZOTERO_KEY_RE.fullmatch(prefix) else ""


def _attachment_groups(chunks: Sequence[Dict[str, Any]]) -> List[tuple[str, List[Dict[str, Any]]]]:
    """Return document-order attachment runs, never non-contiguous groups.

    One item's rows may contain attachments interleaved by their natural chunk
    IDs, or old rows with no recoverable attachment identity.  Grouping all
    matching keys globally made a leaf span intervening chunks, which violates
    the structure's contiguous-range invariant.  Splitting at every identity
    boundary preserves source order even for unknown identities; repeated
    attachment roots remain distinguishable through their first chunk ID.
    """
    groups: List[tuple[str, List[Dict[str, Any]]]] = []
    for chunk in sorted(chunks, key=lambda value: natural_chunk_key(str(value.get("id") or ""))):
        key = _attachment_key(chunk)
        if not groups or groups[-1][0] != key:
            groups.append((key, [chunk]))
        else:
            groups[-1][1].append(chunk)
    return groups


def _contiguous_segments(chunks: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Deterministically split a flat run while never reordering or clustering it."""
    segments: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_chars = 0
    for chunk in chunks:
        text_len = len(str(chunk.get("text") or ""))
        if current and (
            current_chars + text_len > MAX_SEGMENT_CHARS
            or current_chars >= TARGET_SEGMENT_CHARS
        ):
            segments.append(current)
            current = []
            current_chars = 0
        current.append(chunk)
        current_chars += text_len
    if current:
        segments.append(current)
    return segments


def _node_payload(
    *, item_key: str, parent_node_id: str | None, node_type: str, depth: int, ordinal: int,
    title: str | None, source_kind: str, locator: Dict[str, Any], confidence: float,
) -> Dict[str, Any]:
    node_id = _node_id(item_key=item_key, kind=node_type, locator=locator)
    zone = normalise_zone(locator.get("zone"))
    summary_policy, retrieval_policy, citation_policy = ZONE_POLICIES[zone]
    return {
        "node_id": node_id,
        "item_key": item_key,
        "attachment_key": locator.get("attachment_key"),
        "parent_node_id": parent_node_id,
        "node_type": node_type,
        "depth": depth,
        "ordinal": ordinal,
        "title": title,
        "normalized_title": _normalise_title(title).casefold() if title else None,
        "source_kind": source_kind,
        "source_locator": locator,
        "confidence": confidence,
        "content_chars": 0,
        "zone": zone,
        "summary_policy": summary_policy,
        "retrieval_policy": retrieval_policy,
        "citation_policy": citation_policy,
        "extraction_engine": locator.get("extraction_engine"),
        "extraction_version": locator.get("extraction_version"),
        "first_chunk_id": None,
        "last_chunk_id": None,
        "chunks": [],
    }


def _set_leaf_chunks(node: Dict[str, Any], chunks: Sequence[Dict[str, Any]]) -> None:
    ids = [str(chunk.get("id") or "") for chunk in chunks]
    if not ids or any(not value for value in ids):
        raise ValueError("every source chunk needs a non-empty id")
    node["chunks"] = [{"chunk_id": value, "ordinal": index} for index, value in enumerate(ids)]
    node["first_chunk_id"] = ids[0]
    node["last_chunk_id"] = ids[-1]
    node["content_chars"] = sum(len(str(chunk.get("text") or "")) for chunk in chunks)


def _roll_up_content(nodes: List[Dict[str, Any]]) -> None:
    by_id = {node["node_id"]: node for node in nodes}
    for node in reversed(nodes):
        parent_id = node.get("parent_node_id")
        if not parent_id:
            continue
        parent = by_id[parent_id]
        parent["content_chars"] += int(node.get("content_chars") or 0)
        child_first = node.get("first_chunk_id")
        parent_first = parent.get("first_chunk_id")
        if child_first and (
            parent_first is None or natural_chunk_key(str(child_first)) < natural_chunk_key(str(parent_first))
        ):
            parent["first_chunk_id"] = child_first
        child_last = node.get("last_chunk_id")
        parent_last = parent.get("last_chunk_id")
        if child_last and (
            parent_last is None or natural_chunk_key(str(child_last)) > natural_chunk_key(str(parent_last))
        ):
            parent["last_chunk_id"] = child_last


def _thin_container_diagnostics(nodes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Report heading-only containers without flattening their children."""
    children: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        if node.get("parent_node_id"):
            children[str(node["parent_node_id"])].append(node)
    output: List[Dict[str, Any]] = []
    for node in nodes:
        if node.get("node_type") not in {"chapter", "section", "subsection"}:
            continue
        if _heading_zone([str(node.get("title") or "")], "body") != "body":
            continue
        node_children = children.get(str(node["node_id"]), [])
        heading_children = [child for child in node_children if child.get("title")]
        if not heading_children:
            continue
        direct_chars = sum(
            int(child.get("content_chars") or 0) for child in node_children if child.get("chunks")
        )
        descendant_chars = int(node.get("content_chars") or 0) - direct_chars
        if (
            direct_chars <= THIN_CONTAINER_MAX_DIRECT_CHARS
            and descendant_chars >= THIN_CONTAINER_MIN_DESCENDANT_CHARS
        ):
            output.append({
                "node_id": str(node["node_id"]),
                "title": node.get("title"),
                "direct_content_chars": direct_chars,
                "descendant_content_chars": descendant_chars,
                "heading_child_count": len(heading_children),
                "classification": "heading_container",
            })
    return output


def validate_structure(nodes: Sequence[Dict[str, Any]], chunks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate V3 coverage, hierarchy, order, ranges, zones, and policies."""
    expected = [str(chunk.get("id") or "") for chunk in chunks]
    expected_set = set(expected)
    position = {chunk_id: index for index, chunk_id in enumerate(expected)}
    seen_nodes: set[str] = set()
    node_by_id: Dict[str, Dict[str, Any]] = {}
    children: Dict[str, List[str]] = defaultdict(list)
    assigned: List[str] = []
    errors: List[str] = []
    sibling_ordinals: set[tuple[str | None, int]] = set()
    for node in nodes:
        node_id = str(node.get("node_id") or "")
        if not node_id or node_id in seen_nodes:
            errors.append("duplicate_or_missing_node_id")
            continue
        parent = node.get("parent_node_id")
        if parent and parent not in seen_nodes:
            errors.append(f"parent_not_before_child:{node_id}")
        if parent:
            parent_node = node_by_id.get(str(parent))
            if parent_node is not None and int(node.get("depth") or 0) != int(parent_node.get("depth") or 0) + 1:
                errors.append(f"depth_mismatch:{node_id}")
            children[str(parent)].append(node_id)
        sibling_key = (str(parent) if parent else None, int(node.get("ordinal") or 0))
        if sibling_key in sibling_ordinals:
            errors.append(f"duplicate_sibling_ordinal:{node_id}")
        sibling_ordinals.add(sibling_key)
        zone = normalise_zone(node.get("zone"))
        if zone != str(node.get("zone") or "body"):
            errors.append(f"invalid_zone:{node_id}")
        expected_policies = ZONE_POLICIES[zone]
        actual_policies = (
            node.get("summary_policy", "include"), node.get("retrieval_policy", "normal"),
            node.get("citation_policy", "none"),
        )
        if actual_policies != expected_policies:
            errors.append(f"zone_policy_mismatch:{node_id}")
        seen_nodes.add(node_id)
        node_by_id[node_id] = dict(node)
        leaf_chunks = node.get("chunks") or []
        if leaf_chunks and any(child.get("parent_node_id") == node_id for child in nodes):
            errors.append(f"non_leaf_has_chunks:{node_id}")
        leaf_ids = [str(value.get("chunk_id") if isinstance(value, dict) else value) for value in leaf_chunks]
        leaf_positions = [position[value] for value in leaf_ids if value in position]
        if leaf_positions and leaf_positions != list(range(min(leaf_positions), max(leaf_positions) + 1)):
            errors.append(f"noncontiguous_leaf:{node_id}")
        assigned.extend(leaf_ids)
    ranges: Dict[str, tuple[int, int, int] | None] = {}
    for node in reversed(nodes):
        node_id = str(node.get("node_id") or "")
        direct_ids = [
            str(value.get("chunk_id") if isinstance(value, dict) else value)
            for value in node.get("chunks") or []
            if str(value.get("chunk_id") if isinstance(value, dict) else value) in position
        ]
        child_ranges = [ranges[child_id] for child_id in children.get(node_id, []) if ranges.get(child_id)]
        positions = [position[value] for value in direct_ids]
        for child_range in child_ranges:
            if child_range is not None:
                positions.extend((child_range[0], child_range[1]))
        count = len(direct_ids) + sum(child_range[2] for child_range in child_ranges if child_range is not None)
        if positions:
            first, last = min(positions), max(positions)
            ranges[node_id] = (first, last, count)
            if count != last - first + 1:
                errors.append(f"noncontiguous_node_range:{node_id}")
            if node.get("first_chunk_id") and str(node["first_chunk_id"]) != expected[first]:
                errors.append(f"first_chunk_mismatch:{node_id}")
            if node.get("last_chunk_id") and str(node["last_chunk_id"]) != expected[last]:
                errors.append(f"last_chunk_mismatch:{node_id}")
        else:
            ranges[node_id] = None
    if len(expected) != len(expected_set):
        errors.append("duplicate_input_chunk_id")
    if len(assigned) != len(set(assigned)):
        errors.append("duplicate_chunk_assignment")
    if set(assigned) != expected_set:
        errors.append("chunk_coverage_mismatch")
    if expected != sorted(expected, key=natural_chunk_key):
        errors.append("input_chunks_not_in_document_order")
    return {
        "valid": not errors,
        "errors": errors,
        "expected_chunk_count": len(expected),
        "assigned_chunk_count": len(assigned),
        "node_count": len(nodes),
        "leaf_count": sum(1 for node in nodes if node.get("chunks")),
        "zone_counts": dict(sorted({
            zone: sum(1 for node in nodes if node.get("chunks") and normalise_zone(node.get("zone")) == zone)
            for zone in ZONE_POLICIES
        }.items())),
    }


def attach_structure_metadata(
    chunks: Sequence[Dict[str, Any]], nodes: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return copied chunks annotated with their canonical V3 leaf and policies."""
    leaf_by_chunk: Dict[str, Dict[str, Any]] = {}
    for node in nodes:
        for value in node.get("chunks") or []:
            chunk_id = str(value.get("chunk_id") if isinstance(value, dict) else value)
            if not chunk_id or chunk_id in leaf_by_chunk:
                raise ValueError("each chunk must map to exactly one leaf")
            leaf_by_chunk[chunk_id] = dict(node)
    output: List[Dict[str, Any]] = []
    for chunk in chunks:
        chunk_id = str(chunk.get("id") or "")
        leaf = leaf_by_chunk.get(chunk_id)
        if leaf is None:
            raise ValueError(f"chunk has no canonical leaf: {chunk_id}")
        metadata = dict(chunk.get("metadata") or {})
        locator = leaf.get("source_locator") or {}
        metadata.update({
            "node_id": leaf["node_id"], "zone": leaf.get("zone", "body"),
            "depth": int(leaf.get("depth") or 0),
            "summary_policy": leaf.get("summary_policy", "include"),
            "retrieval_policy": leaf.get("retrieval_policy", "normal"),
            "citation_policy": leaf.get("citation_policy", "none"),
            "extraction_engine": leaf.get("extraction_engine") or metadata.get("extraction_engine") or "unknown",
            "extraction_version": leaf.get("extraction_version") or metadata.get("extraction_version") or "unknown",
            "chunk_scheme": 3,
        })
        if locator.get("path") and not metadata.get("structure_path"):
            metadata["structure_path"] = list(locator["path"])
        output.append({**chunk, "metadata": metadata})
    return output


#: Zones whose policy removes content from ordinary retrieval. "corrupted" is
#: deliberately absent: text known to be garbage is *meant* to be excluded, and
#: a document that is entirely corrupted has not been misclassified.
_EXCLUDING_ZONES = {"index", "toc", "bibliography", "colophon", "other_paratext"}


def _reclaim_fully_excluded_document(nodes: List[Dict[str, Any]]) -> str:
    """Restore a document whose every content-bearing leaf was excluded.

    A zone heuristic is a guess about *part* of a document. Applied to all of
    it, the guess has stopped describing structure and simply erases the work.
    No real document consists solely of its own index or table of contents, so
    this outcome is a misclassification by construction, whatever produced it.
    Left alone it fails silently and invisibly -- the document is not ranked
    badly in search, it is absent from it.

    Only leaves are consulted: intermediate heading nodes take their zone from
    the heading itself and stay ``body``, so a whole-document check that
    included them would never fire. Found via ``<main class="Index">`` on a
    Squarespace page, which marked a 65,000-character essay a book index
    (TA2PTL9B, 2026-07-28). The heuristic is fixed in ``html_extract``; this
    guard is here because the next such token will come from somewhere else.

    Returns the zone that was reclaimed, or "" when nothing was changed.
    """
    parents = {str(node.get("parent_node_id") or "") for node in nodes}
    leaves = [
        node for node in nodes
        if str(node.get("node_id")) not in parents
        and int(node.get("content_chars") or 0) > 0
    ]
    if not leaves:
        return ""
    zones = {str(node.get("zone") or "") for node in leaves}
    if len(zones) != 1:
        return ""
    zone = zones.pop()
    if zone not in _EXCLUDING_ZONES:
        return ""
    summary_policy, retrieval_policy, citation_policy = ZONE_POLICIES["body"]
    for node in leaves:
        node["zone"] = "body"
        node["summary_policy"] = summary_policy
        node["retrieval_policy"] = retrieval_policy
        node["citation_policy"] = citation_policy
    return zone


def build_document_structure(item_key: str, chunks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a canonical tree from ordered extractor chunks.

    The function is pure and intentionally accepts Chroma-style ``{"id",
    "text", "metadata"}`` rows, making it safe for audit dry-runs and tests.
    """
    if not item_key:
        raise ValueError("item_key is required")
    ordered_chunks = sorted(chunks, key=lambda value: natural_chunk_key(str(value.get("id") or "")))
    fingerprint = source_fingerprint(ordered_chunks)
    if not ordered_chunks:
        return {
            "item_key": item_key, "source_fingerprint": fingerprint,
            "structure_version": STRUCTURE_VERSION, "status": "unavailable", "confidence": 0.0,
            "nodes": [], "diagnostics": {"reason": "no_chunks", "valid": True},
        }

    nodes: List[Dict[str, Any]] = []
    sibling_ordinals: Dict[str | None, int] = defaultdict(int)

    def add_node(**kwargs: Any) -> Dict[str, Any]:
        parent = kwargs.get("parent_node_id")
        ordinal = sibling_ordinals[parent]
        sibling_ordinals[parent] += 1
        node = _node_payload(ordinal=ordinal, **kwargs)
        nodes.append(node)
        return node

    item_root = add_node(
        item_key=item_key, parent_node_id=None, node_type="item_root", depth=0,
        title=None, source_kind="item", locator={"item_key": item_key}, confidence=1.0,
    )
    explicit_runs = 0
    fallback_runs = 0
    attachment_count = 0

    for attachment_key, attachment_chunks in _attachment_groups(ordered_chunks):
        attachment_count += 1
        first_meta = attachment_chunks[0].get("metadata") or {}
        attachment_title = _normalise_title(first_meta.get("filename") or first_meta.get("title")) or None
        attachment_root = add_node(
            item_key=item_key, parent_node_id=item_root["node_id"], node_type="attachment_root", depth=1,
            title=attachment_title, source_kind="attachment",
            locator={"attachment_key": attachment_key, "first_chunk_id": attachment_chunks[0]["id"]},
            confidence=1.0,
        )
        active_path: List[str] = []
        active_nodes: List[Dict[str, Any]] = []
        index = 0
        while index < len(attachment_chunks):
            chunk = attachment_chunks[index]
            metadata = chunk.get("metadata") or {}
            path = _metadata_path(metadata)
            roles = _metadata_roles(metadata, len(path))
            zone = _heading_zone(path, metadata.get("zone"))
            run = [chunk]
            index += 1
            while index < len(attachment_chunks):
                next_metadata = attachment_chunks[index].get("metadata") or {}
                next_path = _metadata_path(next_metadata)
                if next_path != path or _heading_zone(next_path, next_metadata.get("zone")) != zone:
                    break
                run.append(attachment_chunks[index])
                index += 1

            if path:
                explicit_runs += 1
                common = 0
                while common < len(path) and common < len(active_path) and path[common] == active_path[common]:
                    common += 1
                active_path = active_path[:common]
                active_nodes = active_nodes[:common]
                parent = active_nodes[-1] if active_nodes else attachment_root
                for path_index in range(common, len(path)):
                    title = path[path_index]
                    heading = add_node(
                        item_key=item_key, parent_node_id=parent["node_id"],
                        node_type=_heading_type(path_index + 1), depth=parent["depth"] + 1,
                        title=title, source_kind="metadata_heading",
                        locator={
                            "attachment_key": attachment_key, "path": path[: path_index + 1],
                            "first_chunk_id": run[0]["id"],
                            **({"heading_role": roles[path_index]} if roles else {}),
                        }, confidence=0.85,
                    )
                    active_path.append(title)
                    active_nodes.append(heading)
                    parent = heading
                parent = active_nodes[-1]
                leaf = add_node(
                    item_key=item_key, parent_node_id=parent["node_id"], node_type="semantic_segment",
                    depth=parent["depth"] + 1, title=None, source_kind="metadata_heading",
                    locator={"attachment_key": attachment_key, "first_chunk_id": run[0]["id"], "path": path, "zone": zone,
                             "extraction_engine": metadata.get("extraction_engine"),
                             "extraction_version": metadata.get("extraction_version")},
                    confidence=0.85,
                )
                _set_leaf_chunks(leaf, run)
            else:
                fallback_runs += 1
                parent = active_nodes[-1] if active_nodes else attachment_root
                for segment_index, segment in enumerate(_contiguous_segments(run)):
                    leaf = add_node(
                        item_key=item_key, parent_node_id=parent["node_id"], node_type="semantic_segment",
                        depth=parent["depth"] + 1, title=None, source_kind="semantic_fallback",
                        locator={
                            "attachment_key": attachment_key, "first_chunk_id": segment[0]["id"],
                            "segment_index": segment_index, "zone": zone,
                            "extraction_engine": metadata.get("extraction_engine"),
                            "extraction_version": metadata.get("extraction_version"),
                        }, confidence=0.5,
                    )
                    _set_leaf_chunks(leaf, segment)

    _roll_up_content(nodes)
    diagnostics = validate_structure(nodes, ordered_chunks)
    diagnostics.update({
        "attachment_count": attachment_count,
        "explicit_runs": explicit_runs,
        "fallback_runs": fallback_runs,
    })
    thin_containers = _thin_container_diagnostics(nodes)
    diagnostics["thin_container_count"] = len(thin_containers)
    diagnostics["thin_containers"] = thin_containers
    if not diagnostics["valid"]:
        raise ValueError(f"invalid generated document structure: {diagnostics['errors']}")
    reclaimed = _reclaim_fully_excluded_document(nodes)
    if reclaimed:
        diagnostics["reclaimed_fully_excluded_zone"] = reclaimed
    if explicit_runs and not fallback_runs:
        status, confidence = "exact", 0.85
    elif explicit_runs:
        status, confidence = "recovered", 0.7
    else:
        status, confidence = "flat_fallback", 0.5
    return {
        "item_key": item_key, "source_fingerprint": fingerprint,
        "structure_version": STRUCTURE_VERSION, "status": status, "confidence": confidence,
        "nodes": nodes, "diagnostics": diagnostics,
    }


__all__ = [
    "MAX_SEGMENT_CHARS", "STRUCTURE_VERSION", "TARGET_SEGMENT_CHARS",
    "THIN_CONTAINER_MAX_DIRECT_CHARS", "THIN_CONTAINER_MIN_DESCENDANT_CHARS",
    "ZONE_POLICIES", "attach_structure_metadata", "build_document_structure", "normalise_zone",
    "source_fingerprint", "validate_structure",
]
