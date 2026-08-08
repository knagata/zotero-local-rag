from src.flat_structure_diagnostics import classify_flat_attachment, diagnose_flat_item


def _rows(source_type="pdf", *, count=3, page=True, texts=None):
    texts = texts or ["ordinary body paragraph"] * count
    return [
        {
            "id": f"ATT:{index}", "text": text,
            "metadata": {
                "attachmentKey": "ATT", "itemKey": "ITEM",
                "source_type": source_type,
                **({"page": index + 1} if page else {}),
                **({"locator": f"epub:spine0:block{index}"} if source_type == "epub" else {}),
            },
        }
        for index, text in enumerate(texts)
    ]


def test_pdf_outline_is_the_highest_priority_refresh_candidate():
    result = classify_flat_attachment(
        "ATT", _rows(),
        source_inspector=lambda *_: {"source_available": True, "toc_entries": 8},
    )
    assert result["reason_code"] == "pdf_outline_refresh_candidate"
    assert result["priority"] == 100
    assert result["gold_recommended"] is False


def test_corrupt_epub_with_heading_evidence_is_a_gold_candidate():
    rows = _rows(
        "epub", texts=["Chapter 1 Introduction", "body", "Chapter 2 Results"],
    )
    result = classify_flat_attachment(
        "ATT", rows,
        source_inspector=lambda *_: {"source_available": False, "toc_entries": 0},
    )
    assert result["reason_code"] == "epub_body_heading_recovery_candidate"
    assert result["gold_recommended"] is True
    assert result["numbered_short_blocks"] == 2
    assert result["heading_evidence_blocks"] == 2


def test_one_explicit_numbered_heading_is_counted_only_once():
    rows = _rows(texts=["Chapter 1 Introduction", "ordinary body paragraph"])
    rows[0]["metadata"]["block_type"] = "heading"
    result = classify_flat_attachment(
        "ATT", rows,
        source_inspector=lambda *_: {"source_available": True, "toc_entries": 0},
    )
    assert result["explicit_heading_blocks"] == 1
    assert result["numbered_short_blocks"] == 1
    assert result["heading_evidence_blocks"] == 1
    assert result["reason_code"] == "flat_likely_appropriate_short_document"


def test_epub_toc_that_cannot_map_to_old_chunks_is_a_repair_candidate():
    result = classify_flat_attachment(
        "ATT", _rows("epub"),
        source_inspector=lambda *_: {
            "source_available": True, "toc_entries": 8,
            "refresh_mappable": False, "refresh_error": "no locator match",
        },
    )
    assert result["reason_code"] == "epub_toc_mapping_repair_candidate"
    assert result["gold_recommended"] is True


def test_short_unstructured_pdf_is_classified_as_likely_appropriate():
    result = classify_flat_attachment(
        "ATT", _rows(count=2),
        source_inspector=lambda *_: {"source_available": True, "toc_entries": 0},
    )
    assert result["reason_code"] == "flat_likely_appropriate_short_document"
    assert result["gold_recommended"] is False


def test_short_unstructured_source_is_not_accepted_as_flat_when_unavailable():
    result = classify_flat_attachment(
        "ATT", _rows(count=2),
        source_inspector=lambda *_: {
            "source_available": False, "toc_entries": 0,
            "source_error": "FileNotFoundError: source missing",
        },
    )
    assert result["reason_code"] == "source_unavailable_for_diagnosis"
    assert result["priority"] == 45
    assert result["gold_recommended"] is False


def test_footnote_numbers_and_repeated_ocr_digits_are_not_headings():
    rows = _rows(texts=[
        "1 This is a footnote sentence explaining the source in detail.",
        "1999 年9月21 日に大きな地震が発生した。",
        "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1",
    ])
    result = classify_flat_attachment(
        "ATT", rows,
        source_inspector=lambda *_: {"source_available": True, "toc_entries": 0},
    )
    assert result["numbered_short_blocks"] == 0
    assert result["reason_code"] == "flat_likely_appropriate_short_document"


def test_ocr_record_numbers_are_not_counted_as_numbered_headings():
    rows = _rows(texts=[
        "1.150131 ]Jf-51061 Internees Hut at Barmera",
        "1.150483 Indonesia KANG, Ping Hoo May 07, 1942 1308-10 (IOmths)",
        "1.150774 Indonesia LIE, In Dai August 11, 1944 1302-04 (65)",
    ])
    result = classify_flat_attachment(
        "ATT", rows,
        source_inspector=lambda *_: {"source_available": True, "toc_entries": 0},
    )
    assert result["numbered_short_blocks"] == 0
    assert result["reason_code"] == "flat_likely_appropriate_short_document"
    assert result["gold_recommended"] is False


def test_genuine_multilevel_numbered_headings_are_still_detected():
    rows = _rows(texts=[
        "1.2 Introduction", "1.2.1 Background", "2.1 Method",
    ])
    result = classify_flat_attachment(
        "ATT", rows,
        source_inspector=lambda *_: {"source_available": True, "toc_entries": 0},
    )
    assert result["numbered_short_blocks"] == 3


def test_single_epub_title_toc_is_not_treated_as_a_recovered_hierarchy():
    result = classify_flat_attachment(
        "ATT", _rows("epub"),
        source_inspector=lambda *_: {"source_available": True, "toc_entries": 1},
    )
    assert result["reason_code"] == "flat_likely_appropriate_short_document"


def test_long_pdf_without_outline_is_ranked_for_layout_or_printed_toc_review():
    rows = _rows(count=35)
    result = classify_flat_attachment(
        "ATT", rows,
        source_inspector=lambda *_: {"source_available": True, "toc_entries": 0},
    )
    assert result["reason_code"] == "pdf_printed_toc_or_layout_candidate"
    assert result["gold_recommended"] is True


def test_item_diagnosis_ignores_notes_and_sorts_attachments_by_priority():
    rows = _rows("epub", count=2)
    rows.append({
        "id": "NOTE", "text": "note",
        "metadata": {"source_type": "note", "attachmentKey": ""},
    })
    result = diagnose_flat_item(
        "ITEM", rows,
        source_inspector=lambda *_: {
            "source_available": True, "toc_entries": 3, "refresh_mappable": True,
        },
    )
    assert len(result["attachments"]) == 1
    assert result["title"] == ""
    assert result["attachments"][0]["reason_code"] == "epub_toc_refresh_candidate"


def _html_rows(count=3, chars=400):
    return [
        {"id": f"ATT:{i}", "text": "x" * chars,
         "metadata": {"attachmentKey": "ATT", "itemKey": "ITEM", "source_type": "html"}}
        for i in range(count)
    ]


def test_a_web_clipping_is_not_reported_as_a_missing_source():
    # inspect_source_structure had no html branch, so every clipping fell
    # through to source_available=False and 41 ordinary articles were ranked at
    # priority 45 as if their file could not be found.
    result = classify_flat_attachment("ATT", _html_rows())
    assert result["reason_code"] == "html_snapshot_is_inherently_flat"
    assert result["priority"] == 15
    assert result["gold_recommended"] is False


def test_a_long_web_clipping_is_still_inherently_flat():
    # Length is not evidence of structure for a web article; the 41 measured
    # here ran to 288,442 characters with no heading block at all.
    result = classify_flat_attachment("ATT", _html_rows(count=200, chars=2000))
    assert result["reason_code"] == "html_snapshot_is_inherently_flat"


def test_a_clipping_with_real_headings_is_still_a_recovery_candidate():
    rows = _html_rows(count=3)
    rows[0]["text"] = "1. Introduction"
    rows[1]["text"] = "2. Method"
    for row in rows[:2]:
        row["metadata"]["block_type"] = "heading"
    result = classify_flat_attachment("ATT", rows)
    assert result["reason_code"] == "html_body_heading_recovery_candidate"


def test_inspect_reports_html_as_present_rather_than_unavailable():
    from src.flat_structure_diagnostics import inspect_source_structure

    info = inspect_source_structure("html", _html_rows(), "ATT")
    assert info["source_available"] is True
    assert info["source_kind"] == "html_snapshot"
    assert "source_error" not in info
