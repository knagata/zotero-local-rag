from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

from src.pdf_toc_recovery import (
    HeadingAnchor, _effective_anchor_levels, _reference_section_pages, _splice_reference_chunks,
    align_headings, apply_anchors, try_ai_toc_fast_path,
)


def test_reference_section_pages_and_splice_replace_only_reference_chunks():
    structured = [
        ("a", "body one", {"page": 5, "reading_order": 1, "zone": "body"}),
        ("b", "flattened refs blob", {"page": 20, "reading_order": 1, "zone": "bibliography"}),
        ("c", "more refs blob", {"page": 21, "reading_order": 1, "zone": "bibliography"}),
    ]
    pages = _reference_section_pages(structured)
    assert pages == [20, 21]
    docling_refs = [
        ("ATT:docref:1", "Smith 2020 ...", {"page": 20, "reading_order": 1, "zone": "bibliography"}),
        ("ATT:docref:2", "Jones 2019 ...", {"page": 20, "reading_order": 2, "zone": "bibliography"}),
        ("ATT:docref:3", "Lee 2021 ...", {"page": 21, "reading_order": 1, "zone": "bibliography"}),
    ]
    spliced = _splice_reference_chunks(structured, docling_refs, pages)
    ids = [row[0] for row in spliced]
    # Body chunk kept; coarse bibliography blobs replaced by Docling entries.
    assert ids == ["a", "ATT:docref:1", "ATT:docref:2", "ATT:docref:3"]


def test_alignment_ignores_toc_dense_page_and_finds_body_in_order():
    headings = [
        {"title": "Chapter One", "level": 1, "kind": "chapter"},
        {"title": "First Topic", "level": 2, "kind": "section"},
        {"title": "Chapter Two", "level": 1, "kind": "chapter"},
        {"title": "Second Topic", "level": 2, "kind": "section"},
        {"title": "Chapter Three", "level": 1, "kind": "chapter"},
    ]
    records = {
        2: [
            {"text": "Chapter One", "reading_order": 0},
            {"text": "First Topic", "reading_order": 1},
            {"text": "Chapter Two", "reading_order": 2},
            {"text": "Second Topic", "reading_order": 3},
            {"text": "Chapter Three", "reading_order": 4},
        ],
        10: [{"text": "Chapter One", "reading_order": 0}],
        11: [{"text": "First Topic", "reading_order": 0}],
        30: [{"text": "Chapter Two", "reading_order": 0}],
        31: [{"text": "Second Topic", "reading_order": 0}],
        40: [{"text": "Chapter Three", "reading_order": 0}],
    }
    anchors, diagnostics = align_headings(headings, records)
    assert diagnostics["toc_pages"] == [2]
    assert diagnostics["body_coverage"] == 1.0
    assert [(row.title, row.page) for row in anchors] == [
        ("Chapter One", 10), ("First Topic", 11), ("Chapter Two", 30),
        ("Second Topic", 31), ("Chapter Three", 40),
    ]


def test_apply_anchors_switches_path_within_same_page():
    anchors = [
        HeadingAnchor("Chapter", 1, "chapter", 5, 0, 1.0),
        HeadingAnchor("First", 2, "section", 5, 2, 1.0),
        HeadingAnchor("Second", 2, "section", 5, 4, 1.0),
    ]
    chunks = [
        ("a", "intro", {"page": 5, "reading_order": 1}),
        ("b", "first body", {"page": 5, "reading_order": 3}),
        ("c", "second body", {"page": 5, "reading_order": 5}),
    ]
    result = apply_anchors(chunks, anchors)
    assert [row[2]["structure_path"] for row in result] == [
        ["Chapter"], ["Chapter", "First"], ["Chapter", "Second"],
    ]


def test_apply_anchors_tags_reference_and_note_zones():
    # Part E (dev-notes/current/77): AI-TOC chunks under a References/Notes
    # heading get zone=bibliography/endnote so reference extraction can find them.
    anchors = [
        HeadingAnchor("Chapter", 1, "chapter", 5, 0, 1.0),
        HeadingAnchor("References", 1, "references", 6, 0, 1.0),
        HeadingAnchor("Notes", 1, "notes", 7, 0, 1.0),
    ]
    chunks = [
        ("a", "body", {"page": 5, "reading_order": 1}),
        ("b", "Smith 2020 ...", {"page": 6, "reading_order": 1}),
        ("c", "1. see also ...", {"page": 7, "reading_order": 1}),
    ]
    result = apply_anchors(chunks, anchors)
    zones = [row[2].get("zone") for row in result]
    assert zones == [None, "bibliography", "endnote"]
    assert result[1][2]["structure_path"] == ["References"]


def test_apply_anchors_normalizes_missing_top_level_without_inventing_parent():
    anchors = [
        HeadingAnchor("Preface", 2, "front_matter", 1, 0, 1.0),
        HeadingAnchor("Chapter One", 2, "chapter", 2, 0, 1.0),
        HeadingAnchor("Section", 3, "section", 3, 0, 1.0),
        HeadingAnchor("Chapter Two", 2, "chapter", 4, 0, 1.0),
    ]
    chunks = [
        ("a", "preface", {"page": 1, "reading_order": 1}),
        ("b", "chapter", {"page": 2, "reading_order": 1}),
        ("c", "section", {"page": 3, "reading_order": 1}),
        ("d", "chapter", {"page": 4, "reading_order": 1}),
    ]

    result = apply_anchors(chunks, anchors)

    assert [row[2]["structure_path"] for row in result] == [
        ["Preface"], ["Chapter One"], ["Chapter One", "Section"], ["Chapter Two"],
    ]


def test_apply_anchors_closes_abstract_before_deeper_article_body():
    anchors = [
        HeadingAnchor("Abstract", 1, "abstract", 1, 0, 1.0),
        HeadingAnchor("Introduction", 2, "chapter", 1, 2, 1.0),
        HeadingAnchor("Method", 3, "section", 2, 0, 1.0),
        HeadingAnchor("Conclusion", 2, "chapter", 3, 0, 1.0),
        HeadingAnchor("References", 1, "references", 4, 0, 1.0),
    ]
    chunks = [
        ("a", "summary", {"page": 1, "reading_order": 1}),
        ("b", "intro", {"page": 1, "reading_order": 3}),
        ("c", "method", {"page": 2, "reading_order": 1}),
        ("d", "conclusion", {"page": 3, "reading_order": 1}),
        ("e", "citation", {"page": 4, "reading_order": 1}),
    ]

    result = apply_anchors(chunks, anchors)

    assert [row[2]["structure_path"] for row in result] == [
        ["Abstract"], ["Introduction"], ["Introduction", "Method"],
        ["Conclusion"], ["References"],
    ]


def test_anchor_level_normalization_keeps_existing_abstract_sibling_boundary():
    assert _effective_anchor_levels([]) == []
    anchors = [
        HeadingAnchor("Abstract", 1, "abstract", 1, 0, 1.0),
        HeadingAnchor("Introduction", 1, "chapter", 1, 2, 1.0),
    ]
    result = apply_anchors([
        ("a", "summary", {"page": 1, "reading_order": 1}),
        ("b", "intro", {"page": 1, "reading_order": 3}),
    ], anchors)
    assert [row[2]["structure_path"] for row in result] == [
        ["Abstract"], ["Introduction"],
    ]


def test_title_and_contents_do_not_advance_body_alignment_cursor():
    headings = [
        {"title": "A Book", "level": 1, "kind": "title"},
        {"title": "Contents", "level": 1, "kind": "contents"},
        {"title": "Chapter One", "level": 1, "kind": "chapter"},
    ]
    records = {
        1: [{"text": "A Book", "reading_order": 0}],
        2: [{"text": "Contents Chapter One", "reading_order": 0}],
        10: [{"text": "Chapter One", "reading_order": 0}],
        99: [{"text": "Contents of the archive", "reading_order": 0}],
    }
    anchors, diagnostics = align_headings(headings, records)
    assert [(row.title, row.page) for row in anchors] == [("Chapter One", 10)]
    assert diagnostics["body_coverage"] == 1.0


def test_alignment_keeps_body_heading_after_compact_toc_on_same_page():
    headings = [
        {"title": "Introduction", "level": 1, "kind": "chapter"},
        {"title": "Methods", "level": 1, "kind": "chapter"},
        {"title": "Results", "level": 1, "kind": "chapter"},
        {"title": "Discussion", "level": 1, "kind": "chapter"},
        {"title": "Conclusion", "level": 1, "kind": "chapter"},
    ]
    records = {
        1: [
            {"text": "Introduction 1 Methods 5 Results 10 Discussion 15 Conclusion 20",
             "reading_order": 0},
            {"text": "Introduction", "reading_order": 1},
            {"text": "This paper examines...", "reading_order": 2},
        ],
        5: [{"text": "Methods", "reading_order": 0}],
        10: [{"text": "Results", "reading_order": 0}],
        15: [{"text": "Discussion", "reading_order": 0}],
        20: [{"text": "Conclusion", "reading_order": 0}],
    }
    anchors, diagnostics = align_headings(headings, records)
    assert diagnostics["toc_pages"] == [1]
    assert diagnostics["body_coverage"] == 1.0
    assert [(row.title, row.page, row.reading_order) for row in anchors] == [
        ("Introduction", 1, 1), ("Methods", 5, 0), ("Results", 10, 0),
        ("Discussion", 15, 0), ("Conclusion", 20, 0),
    ]


def test_alignment_joins_split_blocks_and_removes_soft_hyphen():
    headings = [{
        "title": "Elements for a Cultural Studies of Design",
        "level": 1, "kind": "chapter",
    }]
    records = {20: [
        {"text": "Ele\u00adments for a Cultural", "reading_order": 0},
        {"text": "Studies of Design", "reading_order": 1},
    ]}
    anchors, diagnostics = align_headings(headings, records)
    assert diagnostics["body_coverage"] == 1.0
    assert [(row.page, row.reading_order) for row in anchors] == [(20, 0)]


def test_alignment_joins_sparse_reversed_title_and_subtitle_blocks():
    headings = [{
        "title": "Chapter Six The Position of Exhibitions - From Biennale to Triennale",
        "level": 1, "kind": "chapter",
    }]
    records = {40: [
        {"text": "From Biennale to Triennale", "reading_order": 0},
        {"text": "running header", "reading_order": 1},
        {"text": "author", "reading_order": 2},
        {"text": "Chapter Six The Position of Exhibitions", "reading_order": 3},
    ]}
    anchors, diagnostics = align_headings(headings, records)
    assert diagnostics["body_coverage"] == 1.0
    assert [(row.page, row.reading_order) for row in anchors] == [(40, 0)]


def test_alignment_uses_global_sequence_instead_of_greedy_late_match():
    headings = [
        {"title": "Alpha", "level": 1, "kind": "chapter"},
        {"title": "Beta", "level": 1, "kind": "chapter"},
    ]
    records = {
        10: [{"text": "Part Alpha", "reading_order": 0}],
        20: [{"text": "Beta", "reading_order": 0}],
        30: [{"text": "Alpha", "reading_order": 0}],
    }
    anchors, diagnostics = align_headings(headings, records, threshold=0.5)
    assert diagnostics["body_coverage"] == 1.0
    assert [(row.title, row.page) for row in anchors] == [
        ("Alpha", 10), ("Beta", 20),
    ]


def test_alignment_infers_pdf_offset_from_printed_pages():
    headings = [
        {"title": "One", "level": 1, "kind": "chapter", "printed_page": "1"},
        {"title": "Two", "level": 1, "kind": "chapter", "printed_page": "11"},
    ]
    records = {
        5: [{"text": "One", "reading_order": 0}],
        15: [{"text": "Two", "reading_order": 0}],
    }
    anchors, diagnostics = align_headings(headings, records)
    assert diagnostics["estimated_page_offset"] == 4
    assert [row.score for row in anchors] == [1.0, 1.0]


def test_fast_path_is_disabled_and_does_not_call_llm():
    with mock.patch.dict(os.environ, {"PDF_AI_TOC_FAST_PATH_ENABLE": "0"}, clear=False):
        result = try_ai_toc_fast_path(Path("missing.pdf"), "ITEM", [], {})
    assert not result.accepted
    assert result.reason == "feature_disabled"


def test_fast_path_skips_documents_below_minimum_page_count():
    env = {"PDF_AI_TOC_FAST_PATH_ENABLE": "1", "PDF_AI_TOC_MIN_PAGES": "30"}
    with mock.patch.dict(os.environ, env, clear=False), \
         mock.patch("src.pdf_toc_recovery.infer_toc") as infer:
        result = try_ai_toc_fast_path(
            Path("sample.pdf"), "ITEM", [], {"total_pages": 29},
        )
    assert not result.accepted
    assert result.reason == "below_minimum_pages"
    assert result.diagnostics == {"total_pages": 29, "minimum_pages": 30}
    infer.assert_not_called()


def test_fast_path_accepts_only_after_deterministic_coverage_gate():
    chunks = [("c1", "body", {"page": 10, "reading_order": 1})]
    inferred = {
        "document_type": "book",
        "headings": [
            {"title": "One", "level": 1, "kind": "chapter", "printed_page": "1"},
            {"title": "Two", "level": 1, "kind": "chapter", "printed_page": "20"},
        ],
    }
    records = {
        10: [{"text": "One", "reading_order": 0}],
        30: [{"text": "Two", "reading_order": 0}],
    }
    env = {
        "PDF_AI_TOC_FAST_PATH_ENABLE": "1", "PDF_AI_TOC_MIN_COVERAGE": "0.9",
        "PDF_AI_TOC_MIN_STRUCTURED_CHUNK_RATIO": "0.8",
        "PDF_AI_TOC_MIN_PAGES": "30",
    }
    with mock.patch.dict(os.environ, env, clear=False), \
         mock.patch("src.pdf_toc_recovery.infer_toc", return_value=inferred), \
         mock.patch("src.pdf_toc_recovery._page_records", return_value=records):
        result = try_ai_toc_fast_path(Path("sample.pdf"), "ITEM", chunks, {
            "is_scanned": False, "is_corrupted": False,
            "total_pages": 30,
            "scanned_ratio": 0.0, "corrupted_ratio": 0.0,
            "extraction_failure_ratio": 0.0,
        })
    assert result.accepted
    assert result.diagnostics["body_coverage"] == 1.0
    assert result.chunks[0][2]["structure_path"] == ["One"]


def test_fast_path_residual_ratio_gate_shares_fast_path_tolerance():
    # P4 (dev-notes/current/78, user approval 2026-07-26): the AI-TOC residual
    # ratio gate uses the same 2% tolerance as the PyMuPDF fast path
    # (PYMUPDF_NATIVE_OUTLINE_ANOMALY_RATIO_MAX) instead of rejecting on any
    # nonzero residual -- "text accepted but structure refused" at <=2% was an
    # unexplainable middle state.
    env = {"PDF_AI_TOC_FAST_PATH_ENABLE": "1", "PDF_AI_TOC_MIN_PAGES": "30"}
    base_quality = {
        "is_scanned": False, "is_corrupted": False, "total_pages": 100,
        "scanned_ratio": 0.0, "corrupted_ratio": 0.0, "extraction_failure_ratio": 0.0,
    }
    with mock.patch.dict(os.environ, env, clear=False), \
         mock.patch("src.pdf_toc_recovery.infer_toc", return_value={"headings": []}):
        # Within tolerance: the ratio gate passes, so the run reaches the model
        # call and stops at the (mocked) empty heading list instead -- proving
        # the ratios did not reject it. This used to lean on the per-item cloud
        # policy check as its stopping point; that gate was removed 2026-07-27.
        within = try_ai_toc_fast_path(
            Path("sample.pdf"), "ITEM", [], {**base_quality, "scanned_ratio": 0.01},
        )
        assert within.reason == "insufficient_inferred_headings"
        # Above tolerance: rejected with the tolerance-based reason string.
        above = try_ai_toc_fast_path(
            Path("sample.pdf"), "ITEM", [], {**base_quality, "corrupted_ratio": 0.03},
        )
        assert above.reason == "corrupted_ratio_above_tolerance"


def test_splice_is_abandoned_when_the_replacement_would_lose_the_section():
    """A shrunken re-parse must not authorise deleting what it replaces.

    Docling re-reads the reference pages to get one entry per chunk -- a
    boundary improvement, so the character count should survive. The only
    guard was that the replacement be non-empty, so a single surviving chunk
    licensed removing the whole section: four documents lost their entire
    endnotes, up to 40 pages each (2026-07-28).
    """
    structured = [
        ("a", "body text on an earlier page", {"page": 5, "reading_order": 1, "zone": "body"}),
        ("b", "x" * 4000, {"page": 220, "reading_order": 1, "zone": "endnote"}),
        ("c", "y" * 4000, {"page": 221, "reading_order": 1, "zone": "endnote"}),
    ]
    starved = [("d", "1. Adorno", {"page": 220, "reading_order": 1, "zone": "endnote"})]

    spliced = _splice_reference_chunks(structured, starved, [220, 221])

    assert spliced == list(structured), "the original chunks must be kept intact"
    assert sum(len(row[1]) for row in spliced) >= 8000


def test_splice_still_applies_when_the_replacement_carries_the_text():
    # The economy must not block the case the feature exists for: same text,
    # split into one chunk per entry.
    structured = [
        ("b", "1. Adorno 2. Benjamin 3. Cavell", {"page": 220, "reading_order": 1, "zone": "endnote"}),
    ]
    entries = [
        ("d1", "1. Adorno", {"page": 220, "reading_order": 1, "zone": "endnote"}),
        ("d2", "2. Benjamin", {"page": 220, "reading_order": 2, "zone": "endnote"}),
        ("d3", "3. Cavell", {"page": 220, "reading_order": 3, "zone": "endnote"}),
    ]

    spliced = _splice_reference_chunks(structured, entries, [220])

    assert [row[0] for row in spliced] == ["d1", "d2", "d3"]


def test_docling_reference_enrichment_stamps_the_attachment_key():
    """meta_base carried itemKey but not attachmentKey.

    Deletion and the source-truth check both key on attachmentKey, so every
    chunk this path produced was reachable by neither: not purged when the
    attachment was removed from Zotero, and invisible to the per-attachment
    orphan accounting even though it was exactly an orphan. 1,917 chunks
    across 9 items were in this state (2026-07-28, found auditing P1).
    """
    chunks = [("c1", "body", {"page": 10, "reading_order": 1})]
    inferred = {
        "document_type": "book",
        "headings": [
            {"title": "One", "level": 1, "kind": "chapter", "printed_page": "1"},
            {"title": "Two", "level": 1, "kind": "chapter", "printed_page": "20"},
        ],
    }
    records = {
        10: [{"text": "One", "reading_order": 0}],
        30: [{"text": "Two", "reading_order": 0}],
    }
    env = {
        "PDF_AI_TOC_FAST_PATH_ENABLE": "1", "PDF_AI_TOC_MIN_COVERAGE": "0.9",
        "PDF_AI_TOC_MIN_STRUCTURED_CHUNK_RATIO": "0.8", "PDF_AI_TOC_MIN_PAGES": "30",
        "PDF_AI_TOC_DOCLING_REFERENCES_ENABLE": "1",
    }
    captured_meta_base = {}

    def fake_extract(pdf_path, pages, *, attachment_key, meta_base, worker):
        captured_meta_base.update(meta_base)
        return []  # empty is fine; the splice guard leaves structured untouched

    with mock.patch.dict(os.environ, env, clear=False), \
         mock.patch("src.pdf_toc_recovery.infer_toc", return_value=inferred), \
         mock.patch("src.pdf_toc_recovery._page_records", return_value=records), \
         mock.patch("src.pdf_toc_recovery._reference_section_pages", return_value=[10]), \
         mock.patch("src.docling_extract.extract_reference_sections_with_docling",
                    side_effect=fake_extract):
        try_ai_toc_fast_path(Path("sample.pdf"), "ITEM", chunks, {
            "is_scanned": False, "is_corrupted": False, "total_pages": 30,
            "scanned_ratio": 0.0, "corrupted_ratio": 0.0, "extraction_failure_ratio": 0.0,
        }, docling_worker=object())

    assert captured_meta_base.get("itemKey") == "ITEM"
    assert captured_meta_base.get("attachmentKey"), "attachmentKey must be stamped, not omitted"


def test_ai_toc_reference_enrichment_skipped_without_a_docling_worker():
    # LX6XGB67 (2026-07-31): Docling segfaulted inside native OCR. Without a
    # crash-isolating worker the enrichment must not run in-process -- it is
    # optional, and the fail-closed path keeps the coarse AI-TOC chunks.
    chunks = [
        (f"ATT:p{page}:para0:part0", f"Body text on page {page}", {"page": page})
        for page in (10, 30)
    ]
    inferred = {
        "document_type": "monograph",
        "headings": [
            {"title": "One", "page": 10, "level": 1},
            {"title": "Two", "page": 30, "level": 1},
        ],
    }
    records = {
        10: [{"text": "One", "reading_order": 0}],
        30: [{"text": "Two", "reading_order": 0}],
    }
    env = {
        "PDF_AI_TOC_FAST_PATH_ENABLE": "1", "PDF_AI_TOC_MIN_COVERAGE": "0.9",
        "PDF_AI_TOC_MIN_STRUCTURED_CHUNK_RATIO": "0.8", "PDF_AI_TOC_MIN_PAGES": "30",
        "PDF_AI_TOC_DOCLING_REFERENCES_ENABLE": "1",
    }

    def exploding_extract(*_args, **_kwargs):
        raise AssertionError("Docling must not run without a worker")

    with mock.patch.dict(os.environ, env, clear=False), \
         mock.patch("src.pdf_toc_recovery.infer_toc", return_value=inferred), \
         mock.patch("src.pdf_toc_recovery._page_records", return_value=records), \
         mock.patch("src.pdf_toc_recovery._reference_section_pages", return_value=[10]), \
         mock.patch("src.docling_extract.extract_reference_sections_with_docling",
                    side_effect=exploding_extract):
        result = try_ai_toc_fast_path(Path("sample.pdf"), "ITEM", chunks, {
            "is_scanned": False, "is_corrupted": False, "total_pages": 30,
            "scanned_ratio": 0.0, "corrupted_ratio": 0.0, "extraction_failure_ratio": 0.0,
        })

    assert result.accepted
    assert "docling_reference_chunks" not in result.diagnostics
    # The skip must be visible: an enabled feature doing nothing is otherwise
    # indistinguishable from one that ran.
    assert result.diagnostics["docling_reference_enrichment_skipped"] == "no_docling_worker"
