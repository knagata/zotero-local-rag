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


def test_gate_rejects_high_confidence_latin_ocr_with_fused_words():
    fused = (
        "EverymorningwhenIopenmyemailsandreadthenewsIencounteranother"
        "longsequenceofEnglishwordsthattheOCRenginefailedtoseparate"
    ) * 4
    gate = evaluate_local_ocr_gate(
        [("A:p1", fused, {"page": 1})],
        {"parser": "rapidocr", "ocr_pages": [1], "mean_confidence": 0.99},
        expected_pages=1,
        require_text_for_every_page=True,
    )
    assert gate["passed"] is False
    assert gate["missing_word_boundaries"] == 1
    assert "missing_word_boundaries" in gate["reasons"]


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


def test_fixed_layout_router_rejects_partial_output_from_every_local_engine():
    calls: list[str] = []

    def partial(engine: str):
        def extract(_path: Path, _key: str, _metadata: dict):
            calls.append(engine)
            return [
                (f"A:{engine}:p1", "substantive body text " * 30, {"page": 1}),
            ], {"parser": engine, "ocr_pages": [1, 2], "processed_pages": 2}
        return extract

    chunks, quality, gate = run_local_ocr(
        Path("input.pdf"), "A", {}, language="en", expected_pages=2,
        require_text_for_every_page=True,
        extractors={
            "rapidocr": partial("rapidocr"),
            "docling": partial("docling"),
        },
    )

    assert calls == ["rapidocr", "docling"]
    assert chunks == []
    assert gate["reasons"] == ["all_local_ocr_engines_failed"]
    assert [row["status"] for row in quality["local_ocr"]["attempts"]] == [
        "rejected", "rejected",
    ]


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


def test_gate_rejects_ndlocr_missing_json_output():
    chunks = [("A:p1", "substantive body text " * 30, {"page": 1})]
    verdict = evaluate_local_ocr_gate(
        chunks,
        {
            "parser": "ndlocr-lite",
            "ocr_pages": [1],
            "pages_with_text": [1],
            "missing_pages": [2],
        },
        expected_pages=2,
    )
    assert verdict["passed"] is False
    assert "incomplete_ocr_attempt_coverage" in verdict["reasons"]
    assert "missing_ocr_output_pages" in verdict["reasons"]


def test_gate_rejects_rapidocr_pages_without_text():
    chunks = [("A:p1", "substantive body text " * 30, {"page": 1})]
    verdict = evaluate_local_ocr_gate(
        chunks,
        {
            "parser": "rapidocr",
            "ocr_pages": [1, 2],
            "pages_with_text": [1],
            "missing_pages": [2],
        },
        expected_pages=2,
    )
    assert verdict["passed"] is False
    assert "missing_ocr_output_pages" in verdict["reasons"]


def test_fixed_layout_gate_requires_chunk_text_for_every_page():
    chunks = [("A:p1", "substantive body text " * 30, {"page": 1})]
    verdict = evaluate_local_ocr_gate(
        chunks,
        {"parser": "docling", "processed_pages": 2},
        expected_pages=2,
        require_text_for_every_page=True,
    )
    assert verdict["passed"] is False
    assert "incomplete_ocr_text_coverage" in verdict["reasons"]


def test_gate_rejects_ndlocr_empty_json_output():
    chunks = [("A:p1", "substantive body text " * 30, {"page": 1})]
    verdict = evaluate_local_ocr_gate(
        chunks,
        {
            "parser": "ndlocr-lite",
            "ocr_pages": [1, 2],
            "pages_with_text": [1],
            "missing_pages": [2],
        },
        expected_pages=2,
    )
    assert verdict["passed"] is False
    assert "incomplete_ocr_text_coverage" in verdict["reasons"]


def test_a_string_is_not_read_as_a_sequence_of_characters():
    # "12" must not become pages {1, 2}; an unusable value means unknown.
    from src.local_ocr_pipeline import evaluate_local_ocr_gate

    chunks = [("A:p1", "substantive body text " * 30, {"page": 1})]
    verdict = evaluate_local_ocr_gate(chunks, {"processed_pages": "12"}, expected_pages=1)
    assert "incomplete_ocr_attempt_coverage" not in verdict["reasons"]
