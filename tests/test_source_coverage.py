from src.source_coverage import (
    coverage_from_extraction, make_source_coverage, validate_source_coverage,
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


def test_html_adapter_rejects_size_truncation() -> None:
    coverage = coverage_from_extraction(
        "html",
        [("a:h1", "prefix", {})],
        {"source_read_complete": False},
    )
    assert "source_truncated" in validate_source_coverage(coverage)["reasons"]
