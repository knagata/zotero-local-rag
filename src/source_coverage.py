"""Format-neutral source coverage contract for canonical ingestion.

An extractor returning some chunks is not evidence that it covered the whole
source.  This module gives PDF pages, EPUB spine documents, and bounded HTML
reads one small common vocabulary and a fail-closed validator.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping


COVERAGE_SCHEMA_VERSION = "source-coverage-1"


def _positive_units(values: Iterable[Any] | None) -> list[int]:
    units: set[int] = set()
    for value in values or ():
        if isinstance(value, bool):
            continue
        try:
            unit = int(value)
        except (TypeError, ValueError):
            continue
        if unit > 0:
            units.add(unit)
    return sorted(units)


def make_source_coverage(
    *,
    unit_kind: str,
    expected_units: Iterable[Any] | None,
    attempted_units: Iterable[Any] | None,
    text_units: Iterable[Any] | None,
    blank_units: Iterable[Any] | None = None,
    failed_units: Iterable[Any] | None = None,
    truncated: bool = False,
    expected_known: bool = True,
) -> dict[str, Any]:
    """Build a JSON-serializable coverage record without claiming success."""
    return {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "unit_kind": str(unit_kind or "unit"),
        "expected_known": bool(expected_known),
        "expected_units": _positive_units(expected_units),
        "attempted_units": _positive_units(attempted_units),
        "text_units": _positive_units(text_units),
        "blank_units": _positive_units(blank_units),
        "failed_units": _positive_units(failed_units),
        "truncated": bool(truncated),
    }


def validate_source_coverage(coverage: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a deterministic verdict; incomplete evidence never passes.

    A unit is accounted for only when it produced usable text or the extractor
    positively identified it as blank/non-text. Merely attempting a unit is not
    enough. Unknown expected coverage is retained as a distinct, non-passing
    state so callers must opt into a format-specific exception explicitly.
    """
    if not isinstance(coverage, Mapping):
        return {
            "passed": False,
            "reasons": ["missing_source_coverage"],
            "missing_attempts": [],
            "unaccounted_units": [],
        }

    expected_known = bool(coverage.get("expected_known"))
    expected = set(_positive_units(coverage.get("expected_units")))
    attempted = set(_positive_units(coverage.get("attempted_units")))
    text = set(_positive_units(coverage.get("text_units")))
    blank = set(_positive_units(coverage.get("blank_units")))
    failed = set(_positive_units(coverage.get("failed_units")))

    reasons: list[str] = []
    if not expected_known:
        reasons.append("expected_coverage_unknown")
    elif not expected:
        reasons.append("expected_units_empty")

    missing_attempts = sorted(expected - attempted) if expected_known else []
    unaccounted = sorted(expected - text - blank) if expected_known else []
    outside_expected = sorted((attempted | text | blank | failed) - expected) if expected_known else []

    if missing_attempts:
        reasons.append("source_units_not_attempted")
    if unaccounted:
        reasons.append("source_units_unaccounted")
    if failed:
        reasons.append("source_unit_failures")
    if bool(coverage.get("truncated")):
        reasons.append("source_truncated")
    if outside_expected:
        reasons.append("source_units_outside_expected")
    if text & blank:
        reasons.append("source_unit_text_blank_conflict")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "missing_attempts": missing_attempts,
        "unaccounted_units": unaccounted,
        "outside_expected_units": outside_expected,
    }


#: Coverage verdicts that describe a *degraded source*, not a broken extractor.
#:
#: A missing, failed, unattempted or unmeasurable unit means the document was
#: only partially recovered -- the remaining text is still real text, and
#: dropping the whole document over one bad page removes far more evidence than
#: it protects (user decision 2026-07-30).  Callers may adopt such a document
#: as quality-uncertain instead of failing it.
ADOPTABLE_COVERAGE_REASONS = frozenset({
    "expected_coverage_unknown",
    "expected_units_empty",
    "source_units_not_attempted",
    "source_units_unaccounted",
    "source_unit_failures",
    "source_truncated",
})

#: The remaining verdicts (``missing_source_coverage``,
#: ``source_units_outside_expected``, ``source_unit_text_blank_conflict``) say
#: the extractor contradicted itself: its unit numbering cannot be trusted, so
#: neither can the page/spine labels on its chunks.  Those stay fail-closed --
#: they are code defects to fix, not damaged sources to tag.


def coverage_gap_is_adoptable(verdict: Mapping[str, Any] | None) -> bool:
    """True when a failing coverage verdict is only a partial-recovery gap."""
    if not isinstance(verdict, Mapping) or verdict.get("passed"):
        return False
    reasons = {str(reason) for reason in verdict.get("reasons") or ()}
    return bool(reasons) and reasons <= ADOPTABLE_COVERAGE_REASONS


def coverage_shortfall(
    coverage: Mapping[str, Any] | None, verdict: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Summarize how much of the source is missing, for triage and reingest.

    Kept small and scalar-friendly on purpose: this is what a later, better
    engine queries to find the documents worth reprocessing first.
    """
    coverage = coverage if isinstance(coverage, Mapping) else {}
    verdict = verdict if isinstance(verdict, Mapping) else {}
    expected = _positive_units(coverage.get("expected_units"))
    unaccounted = _positive_units(verdict.get("unaccounted_units"))
    accounted = max(0, len(expected) - len(unaccounted))
    return {
        "unit_kind": str(coverage.get("unit_kind") or "unit"),
        "expected_units": len(expected),
        "accounted_units": accounted,
        "unaccounted_units": len(unaccounted),
        "covered_ratio": (
            round(accounted / len(expected), 4) if expected else None
        ),
        "reasons": [str(reason) for reason in verdict.get("reasons") or ()],
        "unaccounted_sample": unaccounted[:50],
        "missing_attempts_sample": _positive_units(verdict.get("missing_attempts"))[:50],
    }


def coverage_from_extraction(
    source_type: str,
    chunks: Iterable[tuple[str, str, Mapping[str, Any]]],
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize current extractor output into the common contract.

    Extractors may populate ``source_coverage`` directly.  The compatibility
    adapter below keeps older PDF/HTML routes honest while they migrate to that
    explicit contract; it never invents an expected EPUB spine count.
    """
    explicit = quality.get("source_coverage")
    if isinstance(explicit, Mapping):
        return dict(explicit)

    rows = list(chunks)
    kind = str(source_type or "").casefold()
    if kind == "html":
        failed = [1] if quality.get("failure_reason") else []
        return make_source_coverage(
            unit_kind="document",
            expected_units=[1],
            attempted_units=[1],
            text_units=[1] if rows else [],
            failed_units=failed,
            truncated=bool(
                quality.get("truncated_by_size")
                or quality.get("truncated_by_timeout")
                or quality.get("source_read_complete") is False
            ),
        )

    if kind == "epub":
        expected_source = quality.get("expected_spines")
        if isinstance(expected_source, (list, tuple, set, range)):
            expected = _positive_units(expected_source)
        else:
            expected_count = (
                expected_source
                or quality.get("spine_document_count")
                or (quality.get("epub_profile") or {}).get("spine_document_count")
                # Fixed-layout OCR reports PDF page count; after remapping it
                # is exactly the OPF spine count and remains valid EPUB
                # coverage evidence.
                or quality.get("total_pages")
            )
            try:
                expected_count = int(expected_count)
            except (TypeError, ValueError):
                expected_count = 0
            expected = list(range(1, expected_count + 1))
        text_units = {
            int(metadata.get("chapter_index")) + 1
            for _chunk_id, text, metadata in rows
            if str(text or "").strip()
            and isinstance(metadata.get("chapter_index"), int)
            and int(metadata["chapter_index"]) >= 0
        }
        return make_source_coverage(
            unit_kind="spine",
            expected_units=expected,
            attempted_units=quality.get("attempted_spines") or [],
            text_units=text_units,
            blank_units=quality.get("blank_spines") or [],
            failed_units=quality.get("failed_spines") or [],
            truncated=bool(quality.get("truncated_by_timeout")),
            expected_known=bool(expected),
        )

    expected_count = quality.get("expected_pages") or quality.get("total_pages")
    try:
        expected_count = int(expected_count)
    except (TypeError, ValueError):
        expected_count = 0
    expected = set(range(1, expected_count + 1))
    text_units: set[int] = set()
    for _chunk_id, text, metadata in rows:
        if not str(text or "").strip():
            continue
        page = metadata.get("page")
        if isinstance(page, bool):
            continue
        try:
            page = int(page)
        except (TypeError, ValueError):
            continue
        if page > 0:
            try:
                page_end = int(metadata.get("page_end") or page)
            except (TypeError, ValueError):
                page_end = page
            text_units.update(range(page, max(page, page_end) + 1))

    attempted_source = quality.get("attempted_pages") or quality.get("ocr_pages")
    if isinstance(attempted_source, int):
        attempted = set(range(1, max(0, attempted_source) + 1))
    elif isinstance(attempted_source, (list, tuple, set, range)):
        attempted = set(_positive_units(attempted_source))
    else:
        processed = quality.get("processed_pages")
        if isinstance(processed, int) and processed >= 0:
            attempted = set(range(1, processed + 1))
        else:
            # PyMuPDF and the page-based adapters iterate the complete source;
            # per-page failures are represented separately below.
            attempted = set(expected)

    failed: set[int] = set()
    for field in (
        "extraction_failure_pages", "missing_pages",
        "repeated_header_dropped_pages", "invalid_json_pages",
        "unresolved_nontext_pages",
    ):
        failed.update(_positive_units(quality.get(field)))

    return make_source_coverage(
        unit_kind="page",
        expected_units=expected,
        attempted_units=attempted,
        text_units=text_units,
        blank_units=quality.get("empty_pages") or quality.get("blank_pages") or [],
        failed_units=failed,
        truncated=bool(quality.get("truncated_by_timeout")),
        expected_known=expected_count > 0,
    )


__all__ = [
    "ADOPTABLE_COVERAGE_REASONS", "COVERAGE_SCHEMA_VERSION",
    "coverage_from_extraction", "coverage_gap_is_adoptable", "coverage_shortfall",
    "make_source_coverage", "validate_source_coverage",
]
