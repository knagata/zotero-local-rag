from src.source_coverage import (
    coverage_from_extraction, coverage_gap_is_adoptable, coverage_shortfall,
    make_source_coverage, validate_source_coverage,
)


def test_complete_text_and_confirmed_blank_units_pass() -> None:
    coverage = make_source_coverage(
        unit_kind="page",
        expected_units=[1, 2, 3],
        attempted_units=[1, 2, 3],
        text_units=[1, 3],
        blank_units=[2],
    )
    assert validate_source_coverage(coverage)["passed"] is True


def test_attempted_unit_without_text_or_blank_evidence_fails() -> None:
    coverage = make_source_coverage(
        unit_kind="page",
        expected_units=[1, 2],
        attempted_units=[1, 2],
        text_units=[1],
    )
    verdict = validate_source_coverage(coverage)
    assert verdict["passed"] is False
    assert verdict["unaccounted_units"] == [2]
    assert "source_units_unaccounted" in verdict["reasons"]


def test_missing_output_and_truncation_both_fail() -> None:
    coverage = make_source_coverage(
        unit_kind="spine",
        expected_units=[1, 2, 3],
        attempted_units=[1, 2],
        text_units=[1],
        blank_units=[2],
        failed_units=[3],
        truncated=True,
    )
    verdict = validate_source_coverage(coverage)
    assert verdict["missing_attempts"] == [3]
    assert "source_unit_failures" in verdict["reasons"]
    assert "source_truncated" in verdict["reasons"]


def test_unknown_expected_set_is_not_silently_complete() -> None:
    coverage = make_source_coverage(
        unit_kind="document",
        expected_units=[],
        attempted_units=[1],
        text_units=[1],
        expected_known=False,
    )
    assert validate_source_coverage(coverage)["reasons"] == ["expected_coverage_unknown"]


def test_conflicting_blank_and_text_evidence_fails() -> None:
    coverage = make_source_coverage(
        unit_kind="page",
        expected_units=[1],
        attempted_units=[1],
        text_units=[1],
        blank_units=[1],
    )
    assert "source_unit_text_blank_conflict" in validate_source_coverage(coverage)["reasons"]


def test_pdf_adapter_does_not_treat_attempted_missing_page_as_complete() -> None:
    coverage = coverage_from_extraction(
        "pdf",
        [("a:p1", "body", {"page": 1})],
        {"total_pages": 2, "extraction_failure_pages": [2]},
    )
    verdict = validate_source_coverage(coverage)
    assert verdict["passed"] is False
    assert verdict["unaccounted_units"] == [2]
    assert "source_unit_failures" in verdict["reasons"]


def test_pdf_adapter_accepts_confirmed_empty_page() -> None:
    coverage = coverage_from_extraction(
        "pdf",
        [("a:p1", "body", {"page": 1})],
        {"total_pages": 2, "empty_pages": [2]},
    )
    assert validate_source_coverage(coverage)["passed"] is True


def test_pdf_adapter_rejects_unresolved_nontext_page() -> None:
    coverage = coverage_from_extraction(
        "pdf",
        [("a:p1", "body", {"page": 1})],
        {"total_pages": 2, "unresolved_nontext_pages": [2]},
    )
    verdict = validate_source_coverage(coverage)
    assert verdict["passed"] is False
    assert "source_unit_failures" in verdict["reasons"]


def test_pdf_adapter_accounts_for_docling_page_spans() -> None:
    coverage = coverage_from_extraction(
        "pdf",
        [("a:p1", "spanning table", {"page": 1, "page_end": 2})],
        {"expected_pages": 2, "processed_pages": 2},
    )
    assert validate_source_coverage(coverage)["passed"] is True


def test_epub_adapter_requires_an_expected_spine_count() -> None:
    coverage = coverage_from_extraction(
        "epub",
        [("a:e1", "body", {"chapter_index": 0})],
        {"attempted_spines": [1]},
    )
    assert validate_source_coverage(coverage)["reasons"] == ["expected_coverage_unknown"]


def test_epub_adapter_accepts_explicit_expected_spine_units() -> None:
    coverage = coverage_from_extraction(
        "epub",
        [("ATT:epub:spine2", "Extracted text", {"chapter_index": 1})],
        {
            "expected_spines": [1, 2],
            "attempted_spines": [1, 2],
            "blank_spines": [1],
        },
    )
    assert validate_source_coverage(coverage)["passed"] is True


def test_partial_recovery_gaps_are_adoptable() -> None:
    # U5 (2026-07-30): one unreadable page must not cost the document its
    # embeddings -- these verdicts describe a damaged source, not a bug.
    for coverage in (
        make_source_coverage(
            unit_kind="page", expected_units=[1, 2], attempted_units=[1, 2],
            text_units=[1],
        ),
        make_source_coverage(
            unit_kind="page", expected_units=[1, 2], attempted_units=[1],
            text_units=[1], failed_units=[2],
        ),
        make_source_coverage(
            unit_kind="page", expected_units=[1, 2], attempted_units=[1, 2],
            text_units=[1, 2], truncated=True,
        ),
        make_source_coverage(
            unit_kind="spine", expected_units=[], attempted_units=[1],
            text_units=[1], expected_known=False,
        ),
    ):
        verdict = validate_source_coverage(coverage)
        assert verdict["passed"] is False
        assert coverage_gap_is_adoptable(verdict) is True


def test_self_contradicting_coverage_is_not_adoptable() -> None:
    # Wrong unit numbering would attach wrong page citations to real text.
    conflict = validate_source_coverage(make_source_coverage(
        unit_kind="page", expected_units=[1], attempted_units=[1],
        text_units=[1], blank_units=[1],
    ))
    assert coverage_gap_is_adoptable(conflict) is False
    outside = validate_source_coverage(make_source_coverage(
        unit_kind="page", expected_units=[1], attempted_units=[1, 7],
        text_units=[1],
    ))
    assert coverage_gap_is_adoptable(outside) is False
    assert coverage_gap_is_adoptable(None) is False


def test_passing_coverage_is_never_reported_as_an_adoptable_gap() -> None:
    verdict = validate_source_coverage(make_source_coverage(
        unit_kind="page", expected_units=[1], attempted_units=[1], text_units=[1],
    ))
    assert verdict["passed"] is True
    assert coverage_gap_is_adoptable(verdict) is False


def test_shortfall_summarizes_what_a_later_engine_should_reprocess() -> None:
    coverage = make_source_coverage(
        unit_kind="page", expected_units=[1, 2, 3, 4],
        attempted_units=[1, 2, 3, 4], text_units=[1, 2, 3],
    )
    gap = coverage_shortfall(coverage, validate_source_coverage(coverage))
    assert gap["unit_kind"] == "page"
    assert gap["expected_units"] == 4
    assert gap["accounted_units"] == 3
    assert gap["unaccounted_units"] == 1
    assert gap["covered_ratio"] == 0.75
    assert gap["unaccounted_sample"] == [4]
    assert gap["reasons"] == ["source_units_unaccounted"]


def test_shortfall_of_unknown_expected_coverage_reports_no_ratio() -> None:
    coverage = make_source_coverage(
        unit_kind="spine", expected_units=[], attempted_units=[1],
        text_units=[1], expected_known=False,
    )
    gap = coverage_shortfall(coverage, validate_source_coverage(coverage))
    assert gap["expected_units"] == 0
    assert gap["covered_ratio"] is None
    assert gap["reasons"] == ["expected_coverage_unknown"]


def test_html_adapter_rejects_size_truncation() -> None:
    coverage = coverage_from_extraction(
        "html",
        [("a:h1", "prefix", {})],
        {"source_read_complete": False},
    )
    assert "source_truncated" in validate_source_coverage(coverage)["reasons"]
