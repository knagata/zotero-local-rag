from __future__ import annotations

from pathlib import Path

from src.local_ocr_pipeline import (
    choose_local_ocr_engine, evaluate_local_ocr_gate, run_local_ocr,
)


def test_japanese_routes_to_one_lightweight_engine():
    ordinary = choose_local_ocr_engine("Japanese")
    assert ordinary.engine == "ndlocr_lite"
    assert ordinary.fallbacks == ("docling",)
    vertical = choose_local_ocr_engine("ja", vertical=True)
    assert vertical.engine == "ndlocr_lite"


def test_english_and_unknown_route_to_rapidocr_then_docling():
    for language in ("en", None):
        decision = choose_local_ocr_engine(language)
        assert decision.engine == "rapidocr"
        assert decision.fallbacks == ("docling",)


def test_gate_requires_attempt_coverage_and_clean_text():
    chunks = [
        ("A:p1", "Readable document text. " * 5, {"page": 1}),
        ("A:p2", "Second page of readable text. " * 5, {"page": 2}),
    ]
    gate = evaluate_local_ocr_gate(
        chunks, {"parser": "ndlocr_lite", "ocr_pages": [1, 2]},
        expected_pages=2,
    )
    assert gate["passed"] is True
    incomplete = evaluate_local_ocr_gate(
        chunks[:1], {"parser": "ndlocr_lite", "ocr_pages": [1]},
        expected_pages=2,
    )
    assert "incomplete_ocr_attempt_coverage" in incomplete["reasons"]


def test_router_falls_back_only_after_gate_rejection():
    calls: list[str] = []

    def bad(_path: Path, _key: str, _metadata: dict):
        calls.append("ndlocr_lite")
        return [], {"parser": "ndlocr_lite", "ocr_pages": [1]}

    def good(_path: Path, _key: str, _metadata: dict):
        calls.append("docling")
        return [
            ("A:p1", "十分に長い日本語の本文です。" * 10, {"page": 1}),
        ], {"parser": "yomitoku", "ocr_pages": [1]}

    chunks, quality, gate = run_local_ocr(
        Path("input.pdf"), "A", {}, language="ja", expected_pages=1,
        extractors={"ndlocr_lite": bad, "docling": good},
    )
    assert calls == ["ndlocr_lite", "docling"]
    assert chunks and gate["passed"]
    assert len(quality["local_ocr"]["attempts"]) == 2


def test_a_page_count_does_not_crash_the_gate():
    """Docling reports processed_pages as a count, not a list of page numbers.

    Iterating the count raised TypeError, so every Docling escalation through
    this gate failed outright -- the local fallback stage was dead rather than
    merely unhelpful (2026-07-28).
    """
    from src.local_ocr_pipeline import evaluate_local_ocr_gate

    chunks = [("A:p%d" % n, "substantive body text " * 30, {"page": n}) for n in (1, 2, 3)]
    verdict = evaluate_local_ocr_gate(chunks, {"processed_pages": 3}, expected_pages=3)
    assert "incomplete_ocr_attempt_coverage" not in verdict["reasons"]


def test_a_page_list_is_still_read_as_page_numbers():
    from src.local_ocr_pipeline import evaluate_local_ocr_gate

    chunks = [("A:p1", "substantive body text " * 30, {"page": 1})]
    verdict = evaluate_local_ocr_gate(chunks, {"ocr_pages": [1]}, expected_pages=3)
    assert "incomplete_ocr_attempt_coverage" in verdict["reasons"]


def test_a_string_is_not_read_as_a_sequence_of_characters():
    # "12" must not become pages {1, 2}; an unusable value means unknown.
    from src.local_ocr_pipeline import evaluate_local_ocr_gate

    chunks = [("A:p1", "substantive body text " * 30, {"page": 1})]
    verdict = evaluate_local_ocr_gate(chunks, {"processed_pages": "12"}, expected_pages=1)
    assert "incomplete_ocr_attempt_coverage" not in verdict["reasons"]
