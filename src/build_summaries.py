"""Read-only compatibility helpers for summary experiments.

The legacy item/section summary builders and embedders were retired with the
V3 data plane.  Canonical writes belong to ``build_structure_summaries.py``.
This module deliberately exposes only grouping and prompt helpers used by
offline comparison scripts; importing or executing it cannot write summaries.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

try:
    from . import summary_core as _summary_core
except ImportError:  # pragma: no cover
    import summary_core as _summary_core


SUMMARY_ONLY_SCHEMA = _summary_core.SUMMARY_ONLY_SCHEMA
_extractive_section = _summary_core._extractive_section
_llm_summary_only_item = _summary_core._llm_summary_only_item
_llm_summary_only_section = _summary_core._llm_summary_only_section
_section_evidence_units = _summary_core._section_evidence_units
_section_source_text = _summary_core._section_source_text
_verify_summary_only_result = _summary_core._verify_summary_only_result
classify_section_content = _summary_core.classify_section_content
is_meta_summary = _summary_core.is_meta_summary


SECTION_WINDOW = 40


def split_sections(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group by chapter; use deterministic 40-chunk windows when absent."""
    chapter_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ungrouped: list[dict[str, Any]] = []
    for chunk in chunks:
        chapter = str(chunk.get("metadata", {}).get("chapter") or "").strip()
        (chapter_groups[chapter] if chapter else ungrouped).append(chunk)
    sections: list[dict[str, Any]] = []
    for index, (chapter, grouped) in enumerate(chapter_groups.items()):
        sections.append({
            "section_id": f"c{index}",
            "chapter": chapter,
            "chunks": grouped,
        })
    for start in range(0, len(ungrouped), SECTION_WINDOW):
        sections.append({
            "section_id": f"w{start // SECTION_WINDOW}",
            "chapter": "",
            "chunks": ungrouped[start : start + SECTION_WINDOW],
        })
    return sections


def _llm_section(section: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Compatibility wrapper used by non-writing model comparison scripts."""
    result, model = _llm_summary_only_section(section)
    result.setdefault("chapter_authors", [])
    result.setdefault("first_publication_note", None)
    return result, model
