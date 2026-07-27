from __future__ import annotations

import json
from pathlib import Path

import fitz

from src.mistral_ocr_batch import (
    build_batch_request, evaluate_ocr_result, parse_batch_output, source_matches,
)
from scripts.run_mistral_ocr_batch import select_input_batch, split_input_batches


def make_pdf(path: Path, pages: int = 2) -> None:
    doc = fitz.open()
    for index in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"page {index + 1}")
    doc.save(path)
    doc.close()


def test_build_batch_request_uses_custom_id_and_inline_pdf(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    make_pdf(pdf, 1)
    request = build_batch_request({"attachment_key": "ATT1"}, pdf)
    assert request["custom_id"] == "ATT1"
    assert request["body"]["document"]["document_url"].startswith("data:application/pdf;base64,")
    assert "model" not in request["body"]


def test_parse_batch_output_accepts_success_and_preserves_failure() -> None:
    text = "\n".join([
        json.dumps({"custom_id": "A", "response": {"status_code": 200, "body": {"pages": []}}}),
        json.dumps({"custom_id": "B", "response": {"status_code": 422, "body": {"message": "bad"}}}),
    ])
    parsed = parse_batch_output(text)
    assert parsed["A"]["ok"] is True
    assert parsed["B"]["ok"] is False
    assert parsed["B"]["status_code"] == 422


def test_quality_gate_requires_exact_page_coverage(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    make_pdf(pdf, 2)
    good = {"pages": [
        {"index": 0, "markdown": "This is ordinary extracted prose with enough useful text. " * 2},
        {"index": 1, "markdown": "Another page contains different readable document content. " * 2},
    ]}
    assert evaluate_ocr_result(good, pdf)["passed"] is True
    missing = {"pages": [{"index": 0, "markdown": "Readable but incomplete content. " * 4}]}
    report = evaluate_ocr_result(missing, pdf)
    assert report["passed"] is False
    assert "incomplete_page_coverage" in report["reasons"]


def test_quality_gate_keeps_complete_document_and_marks_problem_page(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    make_pdf(pdf, 2)
    result = {"pages": [
        {"index": 0, "markdown": "Readable document content. " * 10},
        {"index": 1, "markdown": "@" * 60},
    ]}
    report = evaluate_ocr_result(result, pdf)
    assert report["passed"] is True
    assert report["problem_pages"] == {2: ["gibberish_detected", "repeat_artifacts"]}


def test_source_fingerprint_rejects_changed_pdf(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    make_pdf(pdf, 1)
    stat = pdf.stat()
    row = {"source_size": stat.st_size, "source_mtime": stat.st_mtime}
    assert source_matches(row, pdf)[0] is True
    row["source_size"] += 1
    assert source_matches(row, pdf) == (False, "source_size_changed")


def test_input_batch_is_bounded_but_includes_one_oversize_pdf(tmp_path: Path) -> None:
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(b"a" * 90)
    second.write_bytes(b"b" * 90)
    resolved = [({"attachment_key": "A"}, first), ({"attachment_key": "B"}, second)]
    assert [row["attachment_key"] for row, _ in select_input_batch(resolved, 200)] == ["A"]
    assert [row["attachment_key"] for row, _ in select_input_batch(resolved, 1)] == ["A"]


def test_input_batches_cover_candidates_once_in_order(tmp_path: Path) -> None:
    paths = []
    for key in ("A", "B", "C"):
        path = tmp_path / f"{key}.pdf"
        path.write_bytes(b"x" * 100)
        paths.append(({"attachment_key": key}, path))
    batches = split_input_batches(paths, 200)
    assert [[row["attachment_key"] for row, _ in batch] for batch in batches] == [["A"], ["B"], ["C"]]
