"""Bottom-up summaries for canonical document-structure nodes.

This is intentionally separate from the legacy item/section pipeline.  It can
be backfilled and evaluated behind a feature flag before changing retrieval.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List
from pathlib import Path
import hashlib
import json
import os

try:
    from .summary_core import (
        _extractive_section, _llm_summary_only_item, _llm_summary_only_section,
        classify_section_content, is_meta_summary,
    )
    from .chunk_store import get_item_chunks
    from .db_relations import (
        get_all_document_node_summaries, get_document_node_summary_reuse_cache, get_document_nodes, get_document_structure,
        get_item_processing_status, mark_artifact_status,
        replace_document_node_summary_parts, save_document_node_summary,
    )
    from .embedder import create_embedding_function, open_chroma_collection, resolve_embedder_settings
    from .chunk_store import active_collection_name
    from .llm_client import LLMError, RateLimitReached, get_llm_spec
    from .env_utils import load_dotenv_native
    from .orphan_cleanup import note_only_item
except ImportError:  # pragma: no cover
    from summary_core import _extractive_section, _llm_summary_only_item, _llm_summary_only_section, classify_section_content, is_meta_summary
    from chunk_store import get_item_chunks
    from db_relations import get_all_document_node_summaries, get_document_node_summary_reuse_cache, get_document_nodes, get_document_structure, get_item_processing_status, mark_artifact_status, replace_document_node_summary_parts, save_document_node_summary
    from embedder import create_embedding_function, open_chroma_collection, resolve_embedder_settings
    from chunk_store import active_collection_name
    from llm_client import LLMError, RateLimitReached, get_llm_spec
    from env_utils import load_dotenv_native
    from orphan_cleanup import note_only_item

# Load .env so the LLM roles are available when this module is used from the
# CLI or widget.
load_dotenv_native(Path(__file__).resolve().parents[1])


PROMPT_VERSION = "structure-v3-1"
MAX_PARENT_INPUT_CHARS = 30_000
#: Below this, a summary costs more than it saves. Summary length is nearly
#: independent of source length (median 248 chars for a 400-1,000 char leaf,
#: 448 for one over 20,000), so the compression a summary buys is decided
#: entirely by the source: 0.35 in the 400-1,000 band against 0.009 above
#: 20,000. Standing in for a 716-character passage with a 248-character
#: summary saves a few sentences of reading and loses the wording -- and a hit
#: on it sends the reader to the source anyway, so the layer adds a hop
#: instead of removing one. Raised from 400 after measuring 25,858 summaries
#: (2026-07-28); the leaf's own chunks stay searchable either way, since
#: chunks and summaries are separate collections.
MIN_LEAF_CHARS = 1_000
ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", ROOT / "data" / "chroma"))


def _input_fingerprint(payload: Dict[str, Any]) -> str:
    """Fingerprint the actual text-shaped input to a node-summary prompt."""
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _leaf_input_fingerprint(title: str, source_chunks: List[Dict[str, Any]]) -> str:
    # _llm_summary_only_section receives each bounded segment independently;
    # retain that segmentation and its 30k prompt truncation in the fingerprint.
    segments = []
    for group in _chunk_groups(source_chunks):
        segments.append("\n\n".join(
            str(row.get("text") or "") for row in group if row.get("text")
        )[:MAX_PARENT_INPUT_CHARS])
    return _input_fingerprint({"schema": "node-summary-input-v1", "kind": "leaf", "title": title, "segments": segments})


def _parent_input_fingerprint(title: str, children: List[Dict[str, Any]]) -> str:
    # Node IDs are deliberately excluded: a structure-version migration can
    # rename deterministic nodes without changing the text the LLM receives.
    return _input_fingerprint({
        "schema": "node-summary-input-v1", "kind": "parent", "title": title,
        "child_summaries": [str(row.get("summary") or "") for row in children],
    })


def _reuse_lookup(item_key: str, structure: Dict[str, Any], chunks: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index safe LLM summaries retained before a structure replacement.

    Older rows did not persist an input fingerprint.  They are eligible only
    when the *whole* source fingerprint is unchanged, which proves the current
    chunk text is identical before deriving the equivalent v1 fingerprint.
    Newer rows always use their stored per-node fingerprint, so a changed leaf
    does not invalidate unrelated summaries.
    """
    rows = [
        row for row in get_document_node_summary_reuse_cache(item_key)
        if str(row.get("summary_kind")) == "llm"
        and str(row.get("prompt_version")) == PROMPT_VERSION
        and str(row.get("quality_status")) in {"accepted", "candidate"}
    ]
    current_source = str(structure.get("source_fingerprint") or "")
    by_id = {str(row.get("node_id")): row for row in rows}
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        scope = row.get("input_scope") or {}
        fingerprint = str(scope.get("input_content_fingerprint") or "")
        if not fingerprint and str(row.get("source_fingerprint") or "") == current_source:
            title = str(row.get("title") or "")
            chunk_ids = scope.get("included_chunk_ids")
            child_ids = scope.get("included_child_ids")
            if isinstance(chunk_ids, list) and all(str(value) in chunks for value in chunk_ids):
                fingerprint = _leaf_input_fingerprint(title, [chunks[str(value)] for value in chunk_ids])
            elif isinstance(child_ids, list) and all(str(value) in by_id for value in child_ids):
                fingerprint = _parent_input_fingerprint(title, [by_id[str(value)] for value in child_ids])
        if fingerprint:
            lookup.setdefault(fingerprint, row)
    return lookup


def _nearest_title(node: Dict[str, Any], by_id: Dict[str, Dict[str, Any]]) -> str:
    current = node
    while current:
        if current.get("title"):
            return str(current["title"])
        parent_id = current.get("parent_node_id")
        current = by_id.get(str(parent_id)) if parent_id else None
    return "資料"


def _groups(rows: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    size = 0
    for row in rows:
        row_size = len(str(row.get("summary") or "")) + len(str(row.get("title") or "")) + 8
        if current and size + row_size > MAX_PARENT_INPUT_CHARS:
            groups.append(current)
            current, size = [], 0
        current.append(row)
        size += row_size
    if current:
        groups.append(current)
    return groups


def _chunk_groups(rows: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Keep source chunks in order while bounding each leaf LLM input."""
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    size = 0
    for row in rows:
        row_size = len(str(row.get("text") or ""))
        if current and size + row_size > MAX_PARENT_INPUT_CHARS:
            groups.append(current)
            current, size = [], 0
        current.append(row)
        size += row_size
    if current:
        groups.append(current)
    return groups


def _is_searchable(row: Dict[str, Any]) -> bool:
    """Mirrors document_node_summaries.searchable (db_relations.save_document_node_summary).

    A row only counts as a real, indexable child summary when it is LLM-kind
    and not degraded/disabled -- an extractive fallback is saved but never
    embedded. Both call sites of adds_nothing_over_its_children must count
    children by this same rule, not by "was anything generated for it": one
    counted an extractive-fallback sibling as a second child (so the parent
    was paid for), the other filtered it out first via
    get_all_document_node_summaries(searchable_only=True) (so the same parent
    was then suppressed as a one-child duplicate at embed time) -- a chapter
    with one LLM child and one extractive-fallback child was generated and
    billed as a two-child parent, then discarded as a one-child parent
    (2026-07-28, found in code review).
    """
    return str(row.get("kind") or row.get("summary_kind") or "") == "llm" and str(
        row.get("quality") or row.get("quality_status") or ""
    ) in {"accepted", "candidate"}


def adds_nothing_over_its_children(child_count: int) -> bool:
    """Whether a parent's summary would only restate its children's.

    THE SINGLE DEFINITION. Generation calls it to decide whether to *pay* for a
    summary; the search index calls it to decide whether to *keep* one. Those
    were once separate judgements in separate places, and the split had exactly
    the consequence the arrangement invites: the index suppressed 9,766 parents
    as duplicates that generation had already paid an LLM to produce -- 42% of
    a full rebuild's calls, bought and discarded (2026-07-28).

    Keep the rule here. A copy at either site can drift silently, and the
    direction it drifts in is spending.

    A parent with one child covers exactly that child's text, so reducing it
    yields a summary of a summary while the child says the same thing more
    precisely.
    """
    return child_count == 1


def inherited_title(
    node_id: str, titles: Dict[str, str], parents: Dict[str, str],
) -> str:
    """The nearest title at or above a node.

    Leaf nodes are ``semantic_segment``s and carry no title of their own, and
    the titled parent above them is usually dropped as a one-child duplicate.
    The embedded document was title-plus-summary, so the chapter heading ended
    up in the index nowhere at all: 11,262 of 12,280 summary rows had an empty
    title, and for 9,160 the heading text existed in no searchable field
    (2026-07-28). A reader searching for a chapter by name could not find it.

    Resolution walks the *unsuppressed* tree, because the node holding the
    title is precisely the one suppression removes.
    """
    seen: set[str] = set()
    current = str(node_id or "")
    while current and current not in seen:
        seen.add(current)
        title = str(titles.get(current) or "").strip()
        if title:
            return title
        current = str(parents.get(current) or "")
    return ""


def _select_searchable_summary_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop parents that only restate a single child, keeping the child."""
    searchable_ids = {str(row["node_id"]) for row in rows}
    children: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        parent_id = str(row.get("parent_node_id") or "")
        if parent_id in searchable_ids:
            children[parent_id].append(str(row["node_id"]))
    suppressed = {
        parent_id for parent_id, child_ids in children.items()
        if adds_nothing_over_its_children(len(child_ids))
    }
    return [row for row in rows if str(row["node_id"]) not in suppressed]


def _deepseek_model(role: str) -> str:
    """Resolve the configured role while keeping this API-only pipeline explicit."""
    spec = get_llm_spec(role)
    provider, separator, model = spec.partition(":")
    if provider != "deepseek" or not separator or not model:
        raise LLMError(f"Structure summaries require a DeepSeek API role, got {spec!r}.")
    return model


def _parent_summary(
    title: str, children: List[Dict[str, Any]], *, use_llm: bool,
    reduce_role: str = "standard",
) -> tuple[str, str, str, List[Dict[str, Any]]]:
    """Return summary, kind, model.  Child order is always document order.

    ``reduce_role`` picks the DeepSeek role for the reduction that produces the
    returned summary.  Spec §5.1 assigns leaf summaries to the cheap role and
    the parent (章/item root) reduction to the standard role, so parent-node
    callers use ``"standard"`` while the leaf-segment combiner passes ``"cheap"``.
    When children exceed the input bound they are first chunked with cheap
    intermediate passes and only the final synthesis uses ``reduce_role``.
    """
    if not children:
        return "", "extractive", "", []
    if not use_llm:
        return "\n\n".join(str(row["summary"]) for row in children)[:4000], "extractive", "extractive", []
    try:
        groups = _groups(children)
        # A single group is reduced directly, so the node summary itself is
        # produced by ``reduce_role`` (standard for parents; cheap for leaves).
        if len(groups) == 1:
            result, model = _llm_summary_only_item(title, [
                {"section_id": row["node_id"], "summary": row["summary"]} for row in groups[0]
            ], model=_deepseek_model(reduce_role), reasoning="disabled")
            summary = str(result.get("summary") or "").strip()
            if not summary or is_meta_summary(summary):
                raise LLMError("node summary was empty or meta")
            parts = [{"child_node_ids": [row["node_id"] for row in groups[0]], "summary": summary, "model": model}]
            return summary, "llm", model, parts
        reduced: List[Dict[str, Any]] = []
        parts: List[Dict[str, Any]] = []
        for index, group in enumerate(groups):
            result, model = _llm_summary_only_item(title, [
                {"section_id": row["node_id"], "summary": row["summary"]} for row in group
            ], model=_deepseek_model("cheap"), reasoning="disabled")
            summary = str(result.get("summary") or "").strip()
            if not summary or is_meta_summary(summary):
                raise LLMError("node summary was empty or meta")
            reduced.append({"node_id": f"part-{index}", "summary": summary})
            parts.append({"child_node_ids": [row["node_id"] for row in group], "summary": summary, "model": model})
        result, model = _llm_summary_only_item(title, [
            {"section_id": row["node_id"], "summary": row["summary"]} for row in reduced
        ], model=_deepseek_model(reduce_role), reasoning="disabled")
        summary = str(result.get("summary") or "").strip()
        if not summary or is_meta_summary(summary):
            raise LLMError("node summary was empty or meta")
        return summary, "llm", model, parts
    except RateLimitReached:
        raise
    except LLMError:
        return "\n\n".join(str(row["summary"]) for row in children)[:4000], "extractive", "extractive", []


def _summaries_are_current(item_key: str, structure: Dict[str, Any], mode: str) -> bool:
    """True when existing node summaries already match the current source and prompt.

    Skips regeneration (zero LLM calls) when every stored summary shares the
    current ``source_fingerprint`` and ``PROMPT_VERSION``, an llm run finds at
    least one llm summary, and the summary artifact is not stale/failed.
    """
    existing = get_all_document_node_summaries(item_key=item_key)
    if not existing:
        return False
    current_fingerprint = str(structure["source_fingerprint"])
    if any(str(row.get("source_fingerprint")) != current_fingerprint for row in existing):
        return False
    if any(str(row.get("prompt_version")) != PROMPT_VERSION for row in existing):
        return False
    if mode == "llm" and not any(str(row.get("summary_kind")) == "llm" for row in existing):
        return False
    summary_status = next(
        (row for row in get_item_processing_status(item_key)
         if row.get("artifact_type") == "summary" and not row.get("attachment_key")),
        None,
    )
    if summary_status and summary_status.get("status") in {"stale", "failed"}:
        return False
    return True


def build_structure_summaries(
    item_key: str, *, mode: str = "extractive", force: bool = False,
    collection_name: str | None = None,
) -> Dict[str, Any]:
    """Build leaf then parent summaries from a persisted v2 document tree."""
    structure = get_document_structure(item_key)
    nodes = get_document_nodes(item_key, include_chunks=True)
    if structure is None or not nodes:
        # An item whose only content is notes has no document tree by design
        # (notes are searchable but excluded from the canonical structure), so
        # "structure missing" is the expected state rather than a blockage.
        # Reporting it as blocked left a permanent unresolved entry alongside
        # the structure builder's (FSIXT5VE, 2026-07-27).
        if note_only_item(get_item_chunks(item_key, collection_name=collection_name)):
            mark_artifact_status(
                item_key, "summary", "excluded", reason_code="note_only_item",
                message="Item has only Zotero notes, which are excluded from the "
                        "canonical document tree by design.",
            )
            return {"item_key": item_key, "status": "excluded", "reason": "note_only_item"}
        mark_artifact_status(
            item_key, "summary", "blocked", reason_code="structure_missing",
            message="Build document structure v2 before building node summaries.",
        )
        return {"item_key": item_key, "status": "blocked", "reason": "structure_missing"}
    if not force and _summaries_are_current(item_key, structure, mode):
        return {"item_key": item_key, "status": "skipped_current"}
    chunks = {
        str(chunk.get("id") or ""): chunk
        for chunk in get_item_chunks(item_key, collection_name=collection_name)
    }
    if not chunks:
        mark_artifact_status(item_key, "summary", "blocked", reason_code="no_chunks")
        return {"item_key": item_key, "status": "blocked", "reason": "no_chunks"}
    reusable = _reuse_lookup(item_key, structure, chunks) if mode == "llm" and not force else {}
    requested_llm = mode == "llm"
    use_llm = requested_llm
    mark_artifact_status(
        item_key, "summary", "running", source_fingerprint=structure["source_fingerprint"],
        processor_version=PROMPT_VERSION,
    )
    by_id = {str(node["node_id"]): node for node in nodes}
    children: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        if node.get("parent_node_id"):
            children[str(node["parent_node_id"])].append(node)
    generated: Dict[str, Dict[str, Any]] = {}
    excluded_by_node: Dict[str, List[Dict[str, Any]]] = {}
    skipped = 0
    reused = 0
    try:
        for node in sorted(nodes, key=lambda row: (int(row["depth"]), str(row.get("first_chunk_id") or "")), reverse=True):
            node_id = str(node["node_id"])
            direct = node.get("chunks") or []
            title = _nearest_title(node, by_id)
            if direct:
                if node.get("summary_policy", "include") != "include":
                    # Notes and bibliographies remain source/search artifacts,
                    # but must not leak into a chapter's bottom-up summary.
                    skipped += 1
                    excluded_by_node[node_id] = [{
                        "node_id": node_id, "zone": node.get("zone", "body"),
                        "reason": "summary_policy_exclude",
                    }]
                    continue
                source_chunks = [chunks[row["chunk_id"]] for row in direct if row["chunk_id"] in chunks]
                if not source_chunks:
                    continue
                source_chars = sum(len(str(row.get("text") or "")) for row in source_chunks)
                if source_chars < MIN_LEAF_CHARS:
                    skipped += 1
                    excluded_by_node[node_id] = [{
                        "node_id": node_id, "zone": node.get("zone", "body"),
                        "reason": "below_min_chars",
                    }]
                    continue
                section = {"section_id": node_id, "chapter": title, "chunks": source_chunks}
                if classify_section_content(section) == "non_content":
                    skipped += 1
                    excluded_by_node[node_id] = [{
                        "node_id": node_id, "zone": node.get("zone", "body"),
                        "reason": "classified_non_content",
                    }]
                    continue
                input_fingerprint = _leaf_input_fingerprint(title, source_chunks)
                cached = reusable.get(input_fingerprint) if use_llm else None
                if cached:
                    summary = str(cached["summary"])
                    model_name = str(cached.get("model") or "")
                    kind, quality, leaf_parts = "llm", str(cached.get("quality_status") or "accepted"), []
                    reused += 1
                elif use_llm:
                    try:
                        pieces = []
                        for index, group in enumerate(_chunk_groups(source_chunks)):
                            piece = {"section_id": f"{node_id}:segment:{index}", "chapter": title, "chunks": group}
                            result, model_name = _llm_summary_only_section(
                                piece, model=_deepseek_model("cheap"), reasoning="disabled",
                            )
                            piece_summary = str(result.get("summary") or "").strip()
                            if not piece_summary or is_meta_summary(piece_summary):
                                raise LLMError("leaf summary was empty or meta")
                            pieces.append({"node_id": piece["section_id"], "summary": piece_summary, "kind": "llm",
                                           "quality": "accepted", "source_chunk_count": len(group),
                                           "source_chars": sum(len(str(row.get("text") or "")) for row in group)})
                        if len(pieces) == 1:
                            summary, model_name, leaf_parts = pieces[0]["summary"], model_name, []
                        else:
                            # Leaf segments combine at the cheap role (spec §5.1: leaf=cheap).
                            summary, reduced_kind, model_name, leaf_parts = _parent_summary(
                                title, pieces, use_llm=True, reduce_role="cheap",
                            )
                            if reduced_kind != "llm":
                                raise LLMError("leaf segment reduction failed")
                        kind, quality = "llm", "accepted"
                    except RateLimitReached:
                        raise
                    except LLMError:
                        result, model_name = _extractive_section(section), "extractive"
                        summary, kind, quality, leaf_parts = str(result.get("summary") or ""), "extractive", "degraded", []
                else:
                    result, model_name = _extractive_section(section), "extractive"
                    summary, kind, quality, leaf_parts = str(result.get("summary") or ""), "extractive", "degraded", []
                if not summary:
                    continue
                generated[node_id] = {
                    "node_id": node_id, "summary": summary, "kind": kind, "model": model_name,
                    "quality": quality, "source_chunk_count": len(source_chunks), "source_chars": source_chars,
                }
                excluded_by_node[node_id] = []
                save_document_node_summary(
                    node_id, item_key, summary, summary_kind=kind, model=model_name,
                    prompt_version=PROMPT_VERSION, source_fingerprint=structure["source_fingerprint"],
                    source_chunk_count=len(source_chunks), source_chars=source_chars,
                    quality_status=quality,
                    input_scope={
                        "included_chunk_ids": [str(row["id"]) for row in source_chunks],
                        "excluded": [], "input_content_fingerprint": input_fingerprint,
                        **({"reused_from_node_id": str(cached.get("node_id"))} if cached else {}),
                    },
                )
                replace_document_node_summary_parts(
                    node_id, leaf_parts, prompt_version=PROMPT_VERSION,
                    source_fingerprint=structure["source_fingerprint"],
                )
                continue
            all_children = children.get(node_id, [])
            child_rows = [
                generated[str(child["node_id"])] for child in all_children
                if str(child["node_id"]) in generated
            ]
            # Only searchable children count toward the single-child rule --
            # the same population the embed-time selector uses.
            searchable_child_rows = [row for row in child_rows if _is_searchable(row)]
            exclusions = [
                excluded for child in all_children
                for excluded in excluded_by_node.get(str(child["node_id"]), [])
            ]
            excluded_by_node[node_id] = exclusions
            if not child_rows:
                continue
            input_fingerprint = _parent_input_fingerprint(title, child_rows)
            cached = reusable.get(input_fingerprint) if use_llm else None
            # A genuine single child (searchable or not) is adopted regardless
            # -- there is nothing to gain from paying for a reduction over the
            # only thing there is to reduce. Only when there is more than one
            # child does searchability decide it: a parent with two children,
            # one LLM and one extractive-fallback, must be judged by the count
            # the embed-time selector will see (1 searchable child), not by
            # the raw count of everything that was generated (2, which is why
            # it was billed for a reduction and then suppressed anyway).
            if adds_nothing_over_its_children(len(child_rows)):
                only_child = child_rows[0]
            elif len(child_rows) > 1 and adds_nothing_over_its_children(len(searchable_child_rows)):
                only_child = searchable_child_rows[0]
            else:
                only_child = None
            if only_child is not None:
                summary = str(only_child.get("summary") or "")
                kind = str(only_child.get("kind") or "llm")
                model_name, parts = str(only_child.get("model") or ""), []
            elif cached:
                summary = str(cached["summary"])
                kind, model_name, parts = "llm", str(cached.get("model") or ""), []
                reused += 1
            else:
                summary, kind, model_name, parts = _parent_summary(title, child_rows, use_llm=use_llm)
            if not summary:
                continue
            quality = (
                "accepted" if kind == "llm" and all(row.get("quality") == "accepted" for row in child_rows)
                else ("candidate" if kind == "llm" else "degraded")
            )
            source_chunk_count = sum(int(row.get("source_chunk_count") or 0) for row in child_rows)
            source_chars = sum(int(row.get("source_chars") or 0) for row in child_rows)
            generated[node_id] = {
                "node_id": node_id, "summary": summary, "kind": kind, "model": model_name,
                "quality": quality, "source_chunk_count": source_chunk_count, "source_chars": source_chars,
            }
            save_document_node_summary(
                node_id, item_key, summary, summary_kind=kind, model=model_name,
                prompt_version=PROMPT_VERSION, source_fingerprint=structure["source_fingerprint"],
                source_chunk_count=source_chunk_count, source_chars=source_chars,
                quality_status=quality,
                input_scope={
                    "included_child_ids": [row["node_id"] for row in child_rows],
                    "excluded": exclusions, "input_content_fingerprint": input_fingerprint,
                    **({"reused_from_node_id": str(cached.get("node_id"))} if cached else {}),
                },
            )
            replace_document_node_summary_parts(
                node_id, parts, prompt_version=PROMPT_VERSION,
                source_fingerprint=structure["source_fingerprint"],
            )
        llm_count = sum(1 for row in generated.values() if row["kind"] == "llm")
        artifact_status = "success" if llm_count else ("empty" if not generated else "degraded")
        mark_artifact_status(
            item_key, "summary", artifact_status,
            reason_code=(
                "no_summary_content" if not generated else "no_llm_summary"
            ) if not llm_count else None,
            source_fingerprint=structure["source_fingerprint"], processor_version=PROMPT_VERSION,
            counts={"nodes": len(generated), "llm": llm_count, "extractive": len(generated) - llm_count, "skipped": skipped, "reused": reused},
            fallback_kind="extractive_outline" if not llm_count else None,
        )
        return {"item_key": item_key, "status": artifact_status, "nodes": len(generated), "llm": llm_count, "skipped": skipped, "reused": reused}
    except RateLimitReached:
        mark_artifact_status(item_key, "summary", "failed", reason_code="rate_limited", retryable=True)
        raise
    except Exception as exc:
        mark_artifact_status(item_key, "summary", "failed", reason_code="summary_build_failed", message=str(exc)[:1000], retryable=True)
        raise


def embed_structure_summaries(
    *, item_keys: set[str] | None = None, base_collection_name: str | None = None,
) -> Dict[str, int]:
    """Build the searchable LLM-only node-summary collection."""
    base = base_collection_name or active_collection_name(chroma_dir=CHROMA_DIR)
    if not base:
        raise RuntimeError("No active paragraph collection was found.")
    cfg = resolve_embedder_settings(ROOT)
    embedding_function = create_embedding_function(cfg)
    collection = open_chroma_collection(CHROMA_DIR, f"{base}__sum_node", embedding_function)
    all_rows = [
        row for row in get_all_document_node_summaries(searchable_only=True)
        if item_keys is None or row["item_key"] in item_keys
    ]
    # Built before suppression: the titled parent is the node suppression drops.
    titles = {str(row["node_id"]): str(row.get("title") or "") for row in all_rows}
    parents = {str(row["node_id"]): str(row.get("parent_node_id") or "") for row in all_rows}
    rows = _select_searchable_summary_rows(all_rows)
    if item_keys is None:
        existing_ids = set(collection.get(include=[]).get("ids") or [])
        expected_ids = {f"sum:node:{row['node_id']}" for row in rows}
        stale = sorted(existing_ids - expected_ids)
        for start in range(0, len(stale), 1000):
            collection.delete(ids=stale[start:start + 1000])
    else:
        for item_key in item_keys:
            try:
                collection.delete(where={"itemKey": item_key})
            except Exception:
                pass
    documents = []
    metadatas = []
    ids = []
    for row in rows:
        ids.append(f"sum:node:{row['node_id']}")
        heading = inherited_title(str(row["node_id"]), titles, parents)
        documents.append("\n".join(filter(None, [heading, str(row["summary"])]))[:4000])
        metadatas.append({
            "itemKey": str(row["item_key"]), "attachmentKey": str(row.get("attachment_key") or ""),
            "node_id": str(row["node_id"]), "parent_node_id": str(row.get("parent_node_id") or ""),
            "node_type": str(row.get("node_type") or ""), "depth": int(row.get("depth") or 0),
            "title": heading, "summary_kind": "llm",
            "structure_version": "3", "source_fingerprint": str(row.get("source_fingerprint") or ""),
        })
    batch_size = max(1, int(os.environ.get("SUMMARY_EMBED_BATCH_SIZE", "16")))
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start:start + batch_size]
        batch_docs = documents[start:start + batch_size]
        collection.upsert(
            ids=batch_ids, documents=batch_docs, metadatas=metadatas[start:start + batch_size],
            embeddings=embedding_function(batch_docs),
        )
    client = getattr(collection, "_chroma_client", None)
    if client:
        client.close()
    return {"nodes": len(ids)}


__all__ = ["PROMPT_VERSION", "build_structure_summaries", "embed_structure_summaries"]
