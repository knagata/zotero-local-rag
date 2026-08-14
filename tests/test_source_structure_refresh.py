from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.source_structure_refresh import (
    _numbering_is_contiguous,
    _refresh_pdf_rows_from_mistral_cache,
    refresh_source_structure_metadata,
)


def _old_rows():
    return [{
        "id": "ATTACH01:epub:block9:part0",
        "text": "unchanged indexed text",
        "metadata": {
            "attachmentKey": "ATTACH01", "source_type": "epub",
            "locator": "epub:spine2:block7", "chapter": "Old flat section",
        },
    }]


def test_epub_refresh_maps_by_stable_spine_block_without_replacing_text():
    fresh = [(
        "different-new-chunk-id", "different freshly chunked text",
        {"locator": "epub:spine2:block6", "locator_end": "epub:spine2:block8",
         "structure_path": ["Chapter", "Section"],
         "structure_roles": ["chapter", "section"], "chapter": "Section"},
    )]
    with patch("src.source_structure_refresh._epub_source_path", return_value=Path("book.epub")), \
            patch("src.source_structure_refresh.extract_chunks_from_epub_snapshot",
                  return_value=(fresh, {})):
        rows, reports = refresh_source_structure_metadata(_old_rows())

    assert rows[0]["id"] == _old_rows()[0]["id"]
    assert rows[0]["text"] == "unchanged indexed text"
    assert rows[0]["metadata"]["structure_path"] == ["Chapter", "Section"]
    assert rows[0]["metadata"]["structure_roles"] == ["chapter", "section"]
    assert rows[0]["metadata"]["chapter"] == "Chapter"
    assert rows[0]["metadata"]["section"] == "Section"
    assert reports[0]["metadata_changed"] == 1


def test_epub_refresh_rejects_unmapped_existing_chunk():
    fresh = [(
        "new", "text", {"locator": "epub:spine3:block1", "structure_path": ["Other"]},
    )]
    with patch("src.source_structure_refresh._epub_source_path", return_value=Path("book.epub")), \
            patch("src.source_structure_refresh.extract_chunks_from_epub_snapshot",
                  return_value=(fresh, {})):
        with pytest.raises(RuntimeError, match="no fresh EPUB structure match"):
            refresh_source_structure_metadata(_old_rows())


def test_epub_refresh_rejects_ambiguous_structure_for_one_locator():
    fresh = [
        ("new:1", "a", {"locator": "epub:spine2:block7", "structure_path": ["One"]}),
        ("new:2", "b", {"locator": "epub:spine2:block7", "structure_path": ["Two"]}),
    ]
    with patch("src.source_structure_refresh._epub_source_path", return_value=Path("book.epub")), \
            patch("src.source_structure_refresh.extract_chunks_from_epub_snapshot",
                  return_value=(fresh, {})):
        with pytest.raises(RuntimeError, match="ambiguous fresh EPUB structure"):
            refresh_source_structure_metadata(_old_rows())


def test_epub_refresh_uses_start_heading_when_old_chunk_crosses_new_boundary():
    rows = _old_rows()
    rows[0]["metadata"]["locator_end"] = "epub:spine2:block8"
    fresh = [
        ("new:1", "chapter", {"locator": "epub:spine2:block7",
                              "structure_path": ["Chapter"]}),
        ("new:2", "section", {"locator": "epub:spine2:block8",
                              "structure_path": ["Chapter", "Section"]}),
    ]
    with patch("src.source_structure_refresh._epub_source_path", return_value=Path("book.epub")), \
            patch("src.source_structure_refresh.extract_chunks_from_epub_snapshot",
                  return_value=(fresh, {})):
        refreshed, _reports = refresh_source_structure_metadata(rows)
    assert refreshed[0]["metadata"]["structure_path"] == ["Chapter"]


def test_fixed_layout_epub_projects_toc_onto_existing_spine_locators():
    rows = [
        {"id": f"ATT:epub:spine{spine}:para0", "text": f"page {spine}",
         "metadata": {"attachmentKey": "ATT", "source_type": "epub",
                      "locator": f"epub:spine{spine}:para0"}}
        for spine in (0, 2, 4, 6)
    ]
    entries = {
        2: [{"path": ["Chapter One"], "roles": ["chapter"]}],
        6: [{"path": ["Chapter Two"], "roles": ["chapter"]}],
    }
    with patch("src.source_structure_refresh._epub_source_path", return_value=Path("book.epub")), \
            patch("src.source_structure_refresh.extract_chunks_from_epub_snapshot",
                  return_value=([], {"failure_reason": "fixed_layout_epub_requires_ocr"})), \
            patch("src.source_structure_refresh.get_epub_chapter_index_to_toc_entries",
                  return_value=entries):
        refreshed, reports = refresh_source_structure_metadata(rows)

    assert "structure_path" not in refreshed[0]["metadata"]
    assert refreshed[1]["metadata"]["structure_path"] == ["Chapter One"]
    assert refreshed[2]["metadata"]["structure_path"] == ["Chapter One"]
    assert refreshed[3]["metadata"]["structure_path"] == ["Chapter Two"]
    assert reports[0]["mapping_mode"] == "spine_toc"


def test_fixed_layout_epub_uses_only_common_parent_for_same_spine_fragments():
    rows = [{
        "id": "ATT:epub:spine2:para0", "text": "whole fixed-layout page",
        "metadata": {"attachmentKey": "ATT", "source_type": "epub",
                     "locator": "epub:spine2:para0"},
    }]
    entries = {2: [
        {"path": ["Chapter", "Section One"], "roles": ["chapter", "section"]},
        {"path": ["Chapter", "Section Two"], "roles": ["chapter", "section"]},
    ]}
    with patch("src.source_structure_refresh._epub_source_path", return_value=Path("book.epub")), \
            patch("src.source_structure_refresh.extract_chunks_from_epub_snapshot",
                  return_value=([], {"failure_reason": "fixed_layout_epub_requires_ocr"})), \
            patch("src.source_structure_refresh.get_epub_chapter_index_to_toc_entries",
                  return_value=entries):
        refreshed, _reports = refresh_source_structure_metadata(rows)

    assert refreshed[0]["metadata"]["structure_path"] == ["Chapter"]
    assert refreshed[0]["metadata"]["chapter"] == "Chapter"
    assert "section" not in refreshed[0]["metadata"]


def test_fixed_layout_epub_rejects_same_spine_fragments_without_common_parent():
    rows = [{
        "id": "ATT:epub:spine2:para0", "text": "ambiguous page",
        "metadata": {"attachmentKey": "ATT", "source_type": "epub",
                     "locator": "epub:spine2:para0"},
    }]
    entries = {2: [
        {"path": ["Chapter One"], "roles": ["chapter"]},
        {"path": ["Chapter Two"], "roles": ["chapter"]},
    ]}
    with patch("src.source_structure_refresh._epub_source_path", return_value=Path("book.epub")), \
            patch("src.source_structure_refresh.extract_chunks_from_epub_snapshot",
                  return_value=([], {"failure_reason": "fixed_layout_epub_requires_ocr"})), \
            patch("src.source_structure_refresh.get_epub_chapter_index_to_toc_entries",
                  return_value=entries):
        with pytest.raises(RuntimeError, match="no common parent"):
            refresh_source_structure_metadata(rows)


def test_pdf_refresh_reuses_existing_chunks_and_resolves_same_page_headings():
    rows = [
        {"id": f"PDFKEY01:p1:{index}", "text": text,
         "metadata": {"attachmentKey": "PDFKEY01", "source_type": "pdf", "page": 1,
                      "reading_order": index, "chapter": "Old flat heading"}}
        for index, text in enumerate(("Paper", "Introduction", "body argument"))
    ]
    with patch("src.source_structure_refresh._pdf_source_path", return_value=Path("paper.pdf")), \
            patch("src.source_structure_refresh.get_pdf_toc", return_value=[
                (1, "Paper", 1), (2, "Introduction", 1),
            ]):
        refreshed, reports = refresh_source_structure_metadata(rows)

    assert [row["text"] for row in refreshed] == [row["text"] for row in rows]
    assert refreshed[0]["metadata"]["structure_path"] == ["Paper"]
    assert refreshed[1]["metadata"]["structure_path"] == ["Paper", "Introduction"]
    assert refreshed[2]["metadata"]["structure_roles"] == ["chapter", "section"]
    assert refreshed[2]["metadata"]["chapter"] == "Paper"
    assert refreshed[2]["metadata"]["section"] == "Introduction"
    assert reports[0]["outline_entries"] == 2


def test_pdf_refresh_clears_old_outline_metadata_when_toc_disappears():
    rows = [{
        "id": "PDFKEY01:p1:0", "text": "body",
        "metadata": {
            "attachmentKey": "PDFKEY01", "source_type": "pdf", "page": 1,
            "structure_path": ["Old chapter"], "structure_roles": ["chapter"],
            "chapter": "Old chapter", "section": "Old section", "zone": "bibliography",
        },
    }]
    with patch("src.source_structure_refresh._pdf_source_path", return_value=Path("paper.pdf")), \
            patch("src.source_structure_refresh.get_pdf_toc", return_value=[]):
        refreshed, reports = refresh_source_structure_metadata(rows)

    for key in ("structure_path", "structure_roles", "chapter", "section", "zone"):
        assert key not in refreshed[0]["metadata"]
    assert reports[0]["metadata_changed"] == 1


def test_pdf_refresh_reapplies_accepted_persisted_ai_toc_anchors():
    rows = [
        {"id": "PDF:p1:0", "text": "chapter introduction",
         "metadata": {"attachmentKey": "PDF", "source_type": "pdf", "page": 1,
                      "reading_order": 1, "structure_path": ["stale"]}},
        {"id": "PDF:p1:1", "text": "chapter continuation",
         "metadata": {"attachmentKey": "PDF", "source_type": "pdf", "page": 1,
                      "reading_order": 2, "structure_path": ["stale"]}},
        {"id": "PDF:p2:0", "text": "section evidence",
         "metadata": {"attachmentKey": "PDF", "source_type": "pdf", "page": 2,
                      "reading_order": 1, "structure_path": ["stale"]}},
    ]
    payload = json.dumps([
        {"title": "Chapter One", "level": 1, "kind": "chapter",
         "page": 1, "reading_order": 0, "score": 0.99},
        {"title": "Section A", "level": 2, "kind": "section",
         "page": 2, "reading_order": 0, "score": 0.98},
    ])
    manifest = {"files": {"PDF": {"quality": {"ai_toc_diagnostics": {
        "accepted": True, "anchor_payload": payload,
    }}}}}
    with patch("src.source_structure_refresh._pdf_source_path", return_value=Path("paper.pdf")), \
            patch("src.source_structure_refresh.get_pdf_toc", return_value=[]), \
            patch("src.source_structure_refresh.load_manifest", return_value=manifest):
        refreshed, reports = refresh_source_structure_metadata(rows)

    assert [row["id"] for row in refreshed] == [row["id"] for row in rows]
    assert [row["text"] for row in refreshed] == [row["text"] for row in rows]
    assert [row["metadata"]["structure_path"] for row in refreshed] == [
        ["Chapter One"], ["Chapter One"], ["Chapter One", "Section A"],
    ]
    assert refreshed[2]["metadata"]["structure_roles"] == ["chapter", "section"]
    assert refreshed[2]["metadata"]["chapter"] == "Chapter One"
    assert refreshed[2]["metadata"]["section"] == "Section A"
    assert reports[0]["mapping_mode"] == "persisted_ai_toc_anchors"
    assert reports[0]["outline_entries"] == 2


def test_pdf_refresh_replays_mistral_cache_only_for_exact_chunk_ids():
    rows = [
        {"id": "PDF:p1:0", "text": "chapter", "metadata": {
            "attachmentKey": "PDF", "source_type": "pdf", "page": 1,
            "extraction_engine": "mistral_ocr",
        }},
        {"id": "PDF:p1:1", "text": "body", "metadata": {
            "attachmentKey": "PDF", "source_type": "pdf", "page": 1,
            "extraction_engine": "mistral_ocr",
        }},
    ]
    fresh = [
        ("PDF:p1:0", "different derived text", {"structure_path": ["Chapter"]}),
        ("PDF:p1:1", "different derived text", {"structure_path": ["Chapter"]}),
    ]
    with patch("src.source_structure_refresh._pdf_source_path", return_value=Path("paper.pdf")), \
            patch("src.source_structure_refresh.get_pdf_toc", return_value=[]), \
            patch("src.source_structure_refresh.source_digest", return_value="digest"), \
            patch("src.source_structure_refresh.load_result_any_model",
                  return_value=("model", {"pages": []})), \
            patch("src.source_structure_refresh.extract_chunks_from_mistral_ocr_result",
                  return_value=(fresh, {})):
        refreshed, reports = refresh_source_structure_metadata(rows)

    assert [row["text"] for row in refreshed] == ["chapter", "body"]
    assert all(row["metadata"]["structure_path"] == ["Chapter"] for row in refreshed)
    assert reports[0]["mapping_mode"] == "mistral_ocr_cache_exact_ids"
    assert reports[0]["mapped_chunks"] == 2


def test_pdf_refresh_rejects_mistral_cache_when_chunk_ids_differ():
    rows = [{"id": "PDF:p1:0", "text": "body", "metadata": {
        "attachmentKey": "PDF", "source_type": "pdf", "page": 1,
        "extraction_engine": "mistral_ocr",
    }}]
    fresh = [("PDF:p1:DIFFERENT", "body", {"structure_path": ["Chapter"]})]
    with patch("src.source_structure_refresh._pdf_source_path", return_value=Path("paper.pdf")), \
            patch("src.source_structure_refresh.get_pdf_toc", return_value=[]), \
            patch("src.source_structure_refresh.source_digest", return_value="digest"), \
            patch("src.source_structure_refresh.load_result_any_model",
                  return_value=("model", {"pages": []})), \
            patch("src.source_structure_refresh.extract_chunks_from_mistral_ocr_result",
                  return_value=(fresh, {})), \
            patch("src.source_structure_refresh._refresh_pdf_rows_from_numbered_body_headings",
                  return_value=None):
        refreshed, reports = refresh_source_structure_metadata(rows)

    assert "structure_path" not in refreshed[0]["metadata"]
    assert reports[0]["outline_entries"] == 0


def test_mistral_structure_replay_treats_unreadable_source_as_cache_miss():
    rows = [{"id": "PDF:p1:0", "text": "body", "metadata": {
        "extraction_engine": "mistral_ocr",
    }}]
    with patch("src.source_structure_refresh.source_digest", side_effect=OSError("gone")):
        assert _refresh_pdf_rows_from_mistral_cache(
            rows, "PDF", Path("paper.pdf"),
        ) is None


def test_mistral_structure_replay_treats_absent_cache_as_miss():
    rows = [{"id": "PDF:p1:0", "text": "body", "metadata": {
        "extraction_engine": "mistral_ocr",
    }}]
    with patch("src.source_structure_refresh.source_digest", return_value="digest"), \
            patch("src.source_structure_refresh.load_result_any_model", return_value=None):
        assert _refresh_pdf_rows_from_mistral_cache(
            rows, "PDF", Path("paper.pdf"),
        ) is None


def test_mistral_structure_replay_rejects_partial_structure_mapping():
    rows = [
        {"id": "PDF:p1:0", "text": "heading", "metadata": {
            "extraction_engine": "mistral_ocr",
        }},
        {"id": "PDF:p1:1", "text": "body", "metadata": {
            "extraction_engine": "mistral_ocr",
        }},
    ]
    fresh = [
        ("PDF:p1:0", "heading", {"structure_path": ["Chapter"]}),
        ("PDF:p1:1", "body", {}),
    ]
    with patch("src.source_structure_refresh.source_digest", return_value="digest"), \
            patch("src.source_structure_refresh.load_result_any_model",
                  return_value=("model", {"pages": []})), \
            patch("src.source_structure_refresh.extract_chunks_from_mistral_ocr_result",
                  return_value=(fresh, {})):
        assert _refresh_pdf_rows_from_mistral_cache(
            rows, "PDF", Path("paper.pdf"),
        ) is None


@pytest.mark.parametrize(
    ("accepted", "payload"),
    [
        (False, json.dumps([
            {"title": "Wrong One", "level": 1, "kind": "chapter",
             "page": 1, "reading_order": 0, "score": 1.0},
            {"title": "Wrong Two", "level": 1, "kind": "chapter",
             "page": 2, "reading_order": 0, "score": 1.0},
        ])),
        (True, "{not valid json"),
    ],
    ids=["rejected", "malformed"],
)
def test_pdf_refresh_does_not_reapply_unusable_persisted_ai_toc_payload(
    accepted, payload,
):
    rows = [{
        "id": "PDF:p1:0", "text": "ordinary body text",
        "metadata": {"attachmentKey": "PDF", "source_type": "pdf", "page": 1,
                     "reading_order": 1, "structure_path": ["stale"],
                     "chapter": "stale"},
    }]
    manifest = {"files": {"PDF": {"quality": {"ai_toc_diagnostics": {
        "accepted": accepted, "anchor_payload": payload,
    }}}}}
    with patch("src.source_structure_refresh._pdf_source_path", return_value=Path("paper.pdf")), \
            patch("src.source_structure_refresh.get_pdf_toc", return_value=[]), \
            patch("src.source_structure_refresh.load_manifest", return_value=manifest), \
            patch("src.source_structure_refresh.apply_anchors") as apply:
        refreshed, reports = refresh_source_structure_metadata(rows)

    apply.assert_not_called()
    assert "structure_path" not in refreshed[0]["metadata"]
    assert "chapter" not in refreshed[0]["metadata"]
    assert reports[0]["outline_entries"] == 0
    assert "mapping_mode" not in reports[0]


def test_pdf_refresh_recovers_conservative_japanese_numbered_hierarchy():
    titles = [
        "序言 プロトコルは実行のただなかで存在する", "第一部 脱中心化",
        "序章", "I 導入", "第一章 物理的メディア",
        "第一節 基礎", "II 詳論", "第二章 形式", "参考文献", "I 採用しない細目",
    ]
    rows = [
        {"id": f"PDF:{index}", "text": title,
         "metadata": {"attachmentKey": "PDF", "source_type": "pdf", "page": index + 1}}
        for index, title in enumerate(titles)
    ]
    for row in rows:
        row["metadata"]["block_type"] = "heading"
    with patch("src.source_structure_refresh._pdf_source_path", return_value=Path("paper.pdf")), \
            patch("src.source_structure_refresh.get_pdf_toc", return_value=[]):
        refreshed, reports = refresh_source_structure_metadata(rows)

    assert refreshed[0]["metadata"]["structure_path"] == [
        "序言 プロトコルは実行のただなかで存在する",
    ]
    assert refreshed[1]["metadata"]["structure_path"] == ["第一部 脱中心化"]
    assert refreshed[2]["metadata"]["structure_path"] == ["第一部 脱中心化", "序章"]
    assert refreshed[3]["metadata"]["structure_path"] == ["第一部 脱中心化", "序章", "I 導入"]
    assert refreshed[5]["metadata"]["structure_path"] == [
        "第一部 脱中心化", "第一章 物理的メディア", "第一節 基礎",
    ]
    assert refreshed[5]["metadata"]["chapter"] == "第一章 物理的メディア"
    assert refreshed[5]["metadata"]["section"] == "第一節 基礎"
    assert refreshed[8]["metadata"]["structure_path"] == ["参考文献"]
    assert refreshed[9]["metadata"]["structure_path"] == ["参考文献"]
    assert reports[0]["mapping_mode"] == "numbered_body_headings"


def test_pdf_heading_recovery_ignores_printed_toc_until_real_body_opener():
    # The contents entries share the contents page, which is what a printed
    # contents looks like: measured over the 25 candidates here that have one,
    # every entry sits on the contents page or the one after, and the first body
    # heading is 2 to 172 pages further on. A fixture that put one entry per
    # page described a layout no book has, and it was the only thing holding up
    # a contents region that advanced with each row it skipped -- which let a
    # real book's contents run to page 171 and swallow its four part openers.
    placed = [
        ("目次", 1), ("第一部 目次上の部", 1), ("第一章 目次上の章", 1),
        ("第一節 目次上の節", 1),
        ("序章", 2), ("I 本文の導入", 2), ("第一章 本文一", 3),
        ("第一節 本文一の節", 3), ("第二章 本文二", 4), ("第三章 本文三", 5),
    ]
    rows = [
        {"id": f"PDF:{index}", "text": title,
         "metadata": {"attachmentKey": "PDF", "source_type": "pdf", "page": page,
                      "block_type": "heading", "structure_path": ["stale"]}}
        for index, (title, page) in enumerate(placed)
    ]
    with patch("src.source_structure_refresh._pdf_source_path", return_value=Path("paper.pdf")), \
            patch("src.source_structure_refresh.get_pdf_toc", return_value=[]):
        refreshed, reports = refresh_source_structure_metadata(rows)

    for row in refreshed[:4]:
        assert "structure_path" not in row["metadata"]
    assert refreshed[4]["metadata"]["structure_path"] == ["序章"]
    assert refreshed[5]["metadata"]["structure_path"] == ["序章", "I 本文の導入"]
    assert refreshed[7]["metadata"]["structure_path"] == ["第一章 本文一", "第一節 本文一の節"]
    assert reports[0]["heading_counts"] == {
        "parts": 0, "chapters": 4, "sections": 1, "roman_subheadings": 1, "total": 6,
    }


# The body numbering has to continue from the opener rather than restart: a
# printed contents entry is now recognised by the level and number it carries,
# because a book that lists "PART II" and prints "PART TWO" shares no text
# between the two. A fixture with two chapter ones therefore describes a
# document no book has, and the second one reads as another contents entry.
@pytest.mark.parametrize("opener, body", [
    ("第一部 本論", ["第一章 導入", "第二章 展開", "第三章 展開", "第四章 結論"]),
    ("第一章 本論", ["第二章 展開", "第三章 展開", "第四章 結論", "第五章 総括"]),
])
def test_pdf_heading_recovery_exits_printed_toc_at_numbered_body_opener(opener, body):
    rows = [
        {"id": "PDF:toc", "text": "目次",
         "metadata": {"attachmentKey": "PDF", "source_type": "pdf", "page": 1,
                      "block_type": "heading"}},
        {"id": "PDF:toc-entry", "text": opener,
         "metadata": {"attachmentKey": "PDF", "source_type": "pdf", "page": 1,
                      "block_type": "text"}},
    ]
    for index, title in enumerate([opener, *body]):
        rows.append({
            "id": f"PDF:body:{index}", "text": title,
            "metadata": {"attachmentKey": "PDF", "source_type": "pdf", "page": index + 2,
                         "block_type": "heading"},
        })
    with patch("src.source_structure_refresh._pdf_source_path", return_value=Path("paper.pdf")), \
            patch("src.source_structure_refresh.get_pdf_toc", return_value=[]):
        refreshed, reports = refresh_source_structure_metadata(rows)

    assert "structure_path" not in refreshed[1]["metadata"]
    assert refreshed[2]["metadata"]["structure_path"] == [opener]
    assert reports[0]["mapping_mode"] == "numbered_body_headings"


def test_pdf_heading_recovery_requires_heading_evidence_except_for_part_openers():
    rows = [
        {"id": f"PDF:{index}", "text": title,
         "metadata": {"attachmentKey": "PDF", "source_type": "pdf", "page": index + 1,
                      "block_type": block_type}}
        for index, (title, block_type) in enumerate([
            ("第一部 本文ではtext扱いの部扉", "text"),
            ("第一章 本物の章", "heading"),
            ("第二章 本文中で言及しただけ", "text"),
            ("第一節 本物の節", "heading"),
            ("I 本物の小見出し", "heading"),
            ("第二章 次の本物の章", "heading"),
            ("第三章 さらに次の本物の章", "heading"),
        ])
    ]
    with patch("src.source_structure_refresh._pdf_source_path", return_value=Path("paper.pdf")), \
            patch("src.source_structure_refresh.get_pdf_toc", return_value=[]):
        refreshed, reports = refresh_source_structure_metadata(rows)

    assert refreshed[0]["metadata"]["structure_path"] == ["第一部 本文ではtext扱いの部扉"]
    assert refreshed[1]["metadata"]["structure_path"] == [
        "第一部 本文ではtext扱いの部扉", "第一章 本物の章",
    ]
    # The incidental prose-looking chapter must not become a boundary; it
    # remains content owned by the preceding real chapter.
    assert refreshed[2]["metadata"]["structure_path"] == refreshed[1]["metadata"]["structure_path"]
    assert reports[0]["heading_counts"]["chapters"] == 3


def test_pdf_heading_recovery_fails_closed_for_too_few_chapters():
    rows = [
        {"id": f"PDF:{index}", "text": title,
         "metadata": {"attachmentKey": "PDF", "source_type": "pdf", "page": index + 1,
                      "block_type": "heading", "structure_path": ["stale"],
                      "chapter": "stale"}}
        for index, title in enumerate([
            "第一章 一般的な語を含む見出し", "第一節 細目", "I 補足",
            "第二章 もう一つの見出し", "第二節 細目",
        ])
    ]
    with patch("src.source_structure_refresh._pdf_source_path", return_value=Path("paper.pdf")), \
            patch("src.source_structure_refresh.get_pdf_toc", return_value=[]):
        refreshed, reports = refresh_source_structure_metadata(rows)

    assert all("structure_path" not in row["metadata"] for row in refreshed)
    assert all("chapter" not in row["metadata"] for row in refreshed)
    assert reports[0]["outline_entries"] == 0
    assert "heading_counts" not in reports[0]


def test_pdf_heading_recovery_index_group_requires_kana_not_any_kanji_row():
    titles = [
        "第一章 一般の見出し", "第二章 もう一つの見出し", "五行",
        "第三章 三つ目の見出し", "第四章 四つ目の見出し", "第五章 五つ目の見出し",
    ]
    rows = [
        {"id": f"PDF:{index}", "text": title,
         "metadata": {"attachmentKey": "PDF", "source_type": "pdf", "page": index + 1,
                      "block_type": "heading"}}
        for index, title in enumerate(titles)
    ]
    with patch("src.source_structure_refresh._pdf_source_path", return_value=Path("paper.pdf")), \
            patch("src.source_structure_refresh.get_pdf_toc", return_value=[]):
        refreshed, _reports = refresh_source_structure_metadata(rows)

    # "五行" is an ordinary kanji heading, not a kana index-row marker, so it
    # must not be reclassified into the excluded "索引" zone.
    assert refreshed[2]["metadata"]["structure_path"] == ["第二章 もう一つの見出し"]
    assert "索引" not in refreshed[2]["metadata"]["structure_path"]


def test_pdf_heading_recovery_index_group_still_matches_real_kana_row():
    titles = [
        "第一章 一般の見出し", "第二章 もう一つの見出し", "第三章 三つ目の見出し",
        "第四章 四つ目の見出し", "第五章 五つ目の見出し", "ア行",
    ]
    rows = [
        {"id": f"PDF:{index}", "text": title,
         "metadata": {"attachmentKey": "PDF", "source_type": "pdf", "page": index + 1,
                      "block_type": "heading"}}
        for index, title in enumerate(titles)
    ]
    with patch("src.source_structure_refresh._pdf_source_path", return_value=Path("paper.pdf")), \
            patch("src.source_structure_refresh.get_pdf_toc", return_value=[]):
        refreshed, _reports = refresh_source_structure_metadata(rows)

    assert refreshed[5]["metadata"]["structure_path"] == ["索引"]


def test_pdf_heading_recovery_index_group_page_start_ignores_missing_page():
    titles = [
        "第一章 一般の見出し", "第二章 もう一つの見出し", "第三章 三つ目の見出し",
        "第四章 四つ目の見出し", "第五章 五つ目の見出し", "ア行",
    ]
    rows = [
        {"id": f"PDF:{index}", "text": title,
         "metadata": {"attachmentKey": "PDF", "source_type": "pdf",
                      "block_type": "heading"}}
        for index, title in enumerate(titles)
    ]
    with patch("src.source_structure_refresh._pdf_source_path", return_value=Path("paper.pdf")), \
            patch("src.source_structure_refresh.get_pdf_toc", return_value=[]):
        refreshed, _reports = refresh_source_structure_metadata(rows)

    # Without page metadata, the missing-page fallback must not anchor the
    # synthesized "索引" boundary back at the start of the document.
    assert refreshed[0]["metadata"]["structure_path"] == ["第一章 一般の見出し"]
    assert refreshed[5]["metadata"]["structure_path"] == ["索引"]


@pytest.mark.parametrize("intrinsic_zone", ["corrupted", "footnote"])
def test_pdf_refresh_preserves_intrinsic_zone_without_toc(intrinsic_zone):
    rows = [{
        "id": "PDFKEY01:p1:0", "text": "body",
        "metadata": {
            "attachmentKey": "PDFKEY01", "source_type": "pdf", "page": 1,
            "structure_path": ["Old chapter"], "zone": intrinsic_zone,
        },
    }]
    with patch("src.source_structure_refresh._pdf_source_path", return_value=Path("paper.pdf")), \
            patch("src.source_structure_refresh.get_pdf_toc", return_value=[]):
        refreshed, _reports = refresh_source_structure_metadata(rows)
    assert refreshed[0]["metadata"]["zone"] == intrinsic_zone


def test_pdf_refresh_clears_stale_zone_when_refreshed_path_is_empty():
    rows = [{
        "id": "PDFKEY01:p1:0", "text": "front matter before heading",
        "metadata": {
            "attachmentKey": "PDFKEY01", "source_type": "pdf", "page": 1,
            "structure_path": ["Bibliography"], "chapter": "Bibliography",
            "zone": "bibliography",
        },
    }]
    with patch("src.source_structure_refresh._pdf_source_path", return_value=Path("paper.pdf")), \
            patch("src.source_structure_refresh.get_pdf_toc", return_value=[
                (1, "Introduction", 2),
            ]), patch(
                "src.source_structure_refresh._resolve_record_structure_paths",
                return_value=[[]],
            ), patch(
                "src.source_structure_refresh.build_pdf_page_structure_path_lookup",
                return_value=lambda _page: [],
            ):
        refreshed, _reports = refresh_source_structure_metadata(rows)
    assert "zone" not in refreshed[0]["metadata"]


# A recovered run has to be the whole book, not a gapless tail of it.


def test_a_complete_numbered_run_from_one_is_accepted():
    assert _numbering_is_contiguous([1, 2, 3, 4])


def test_a_gap_rejects_the_run():
    assert not _numbering_is_contiguous([1, 3, 4])


def test_a_gapless_run_that_does_not_start_at_one_is_rejected():
    # 3, 4, 5 has no holes, but the extractor lost chapters 1 and 2. Storing it
    # would assert a five-chapter book begins at its third chapter.
    assert not _numbering_is_contiguous([3, 4, 5])
    assert not _numbering_is_contiguous([2, 3, 4])


def test_too_few_numbers_to_judge_are_left_alone():
    # Unnumbered openers report 0 and drop out; one lone number says nothing
    # about completeness, and rejecting it would block every short document.
    assert _numbering_is_contiguous([])
    assert _numbering_is_contiguous([0, 0])
    assert _numbering_is_contiguous([5])


def _pdf_rows(entries):
    """(text, block_type, page) triples as PDF chunks of one attachment."""
    return [
        {"id": f"PDF:{index}", "text": text,
         "metadata": {"attachmentKey": "PDF", "source_type": "pdf",
                      "page": page, "block_type": block_type}}
        for index, (text, block_type, page) in enumerate(entries)
    ]


def _recovered_paths(rows):
    with patch("src.source_structure_refresh._pdf_source_path", return_value=Path("b.pdf")), \
            patch("src.source_structure_refresh.get_pdf_toc", return_value=[]):
        refreshed, _reports = refresh_source_structure_metadata(rows)
    seen, last = [], None
    for row in refreshed:
        path = tuple((row.get("metadata") or {}).get("structure_path") or ())
        if path and path != last:
            seen.append(path)
        last = path
    return seen


def test_a_chapter_opener_labelled_page_furniture_is_still_a_boundary():
    # Layout extraction labels chapter openers "page_furniture" as readily as
    # "heading" -- in one book here five of fifteen landed in each -- and the
    # module only ever looked at headings, so the book came out flat.
    rows = _pdf_rows([
        ("CHAPTER ONE", "page_furniture", 1), ("body", "text", 1),
        ("CHAPTER TWO", "heading", 2), ("body", "text", 2),
        ("CHAPTER THREE", "page_furniture", 3), ("body", "text", 3),
        ("CHAPTER FOUR", "heading", 4), ("body", "text", 4),
        ("CHAPTER FIVE", "page_furniture", 5), ("body", "text", 5),
    ])
    assert _recovered_paths(rows) == [
        ("CHAPTER ONE",), ("CHAPTER TWO",), ("CHAPTER THREE",),
        ("CHAPTER FOUR",), ("CHAPTER FIVE",),
    ]


def test_a_running_header_repeating_the_chapter_name_is_not_a_boundary():
    # The header at the top of every page of chapter one carries the same words
    # as its opener. Admitting furniture without asking how often the document
    # repeats it would open a new chapter on every page.
    rows = _pdf_rows(
        [("CHAPTER ONE", "page_furniture", 1), ("body", "text", 1)]
        + [(text, kind, page) for page in range(2, 8)
           for text, kind in (("CHAPTER ONE", "page_furniture"), ("body", "text"))]
        + [("CHAPTER TWO", "heading", 8), ("body", "text", 8),
           ("CHAPTER THREE", "heading", 9), ("body", "text", 9),
           ("CHAPTER FOUR", "heading", 10), ("body", "text", 10),
           ("CHAPTER FIVE", "heading", 11), ("body", "text", 11)]
    )
    assert ("CHAPTER ONE",) not in _recovered_paths(rows)


def test_a_running_header_is_recognised_through_the_folio_it_prints():
    # "序章] 13", "序章] 15" -- the printed page number makes every occurrence
    # textually unique, so counting the raw text can never see the repetition.
    rows = _pdf_rows(
        [("第一章 誤解", "page_furniture", 1), ("body", "text", 1)]
        + [(f"第一章 誤解 | {page * 2}", "page_furniture", page)
           for page in range(2, 9)]
        + [("第二章 脱走", "heading", 9), ("body", "text", 9),
           ("第三章 結果", "heading", 10), ("body", "text", 10),
           ("第四章 和解", "heading", 11), ("body", "text", 11),
           ("第五章 現在", "heading", 12), ("body", "text", 12)]
    )
    paths = _recovered_paths(rows)
    assert not [path for path in paths if path[0].startswith("第一章 誤解 |")]


def test_a_printed_contents_is_recognised_when_it_numbers_differently():
    # The contents page says "PART II" where the body page says "PART TWO".
    # Matching on text alone found no repetition, so every contents line became
    # a part and the book got a spurious copy of its own hierarchy in front.
    rows = _pdf_rows([
        ("Contents", "heading", 1),
        ("PART I THE MYSTERY 11", "heading", 1),
        ("PART II FROM MYTH 87", "heading", 1),
        ("PART III THE JOURNEY 153", "heading", 1),
        ("PART IV LITTLE GIRLS 221", "heading", 2),
        ("PART V AN EVEN BALANCE 305", "heading", 2),
        ("PART ONE THE MYSTERY", "heading", 5), ("body", "text", 5),
        ("PART TWO FROM MYTH", "heading", 6), ("body", "text", 6),
        ("PART THREE THE JOURNEY", "heading", 7), ("body", "text", 7),
        ("PART FOUR LITTLE GIRLS", "heading", 8), ("body", "text", 8),
        ("PART FIVE AN EVEN BALANCE", "heading", 9), ("body", "text", 9),
    ])
    paths = _recovered_paths(rows)
    assert paths == [
        ("PART ONE THE MYSTERY",), ("PART TWO FROM MYTH",),
        ("PART THREE THE JOURNEY",), ("PART FOUR LITTLE GIRLS",),
        ("PART FIVE AN EVEN BALANCE",),
    ]


def test_the_contents_region_ends_with_the_contents_pages():
    # Without a bound the region can swallow the book: a chapter opener whose
    # number recurs in the back-of-book notes looks like a contents entry, and
    # then nothing ever ends the contents.
    rows = _pdf_rows([
        ("Contents", "heading", 1),
        ("CHAPTER ONE EPISTEMOLOGIES 17", "heading", 1),
        ("CHAPTER ONE", "page_furniture", 9), ("body", "text", 9),
        ("CHAPTER TWO", "page_furniture", 10), ("body", "text", 10),
        ("CHAPTER THREE", "page_furniture", 11), ("body", "text", 11),
        ("CHAPTER FOUR", "page_furniture", 12), ("body", "text", 12),
        ("CHAPTER FIVE", "page_furniture", 13), ("body", "text", 13),
    ])
    assert _recovered_paths(rows)[0] == ("CHAPTER ONE",)


def test_notes_grouped_by_chapter_do_not_open_those_chapters_again():
    # Notes collected at the back repeat every chapter heading to say which
    # chapter they serve. Read as openers they gave a five-chapter book ten.
    rows = _pdf_rows([
        ("CHAPTER ONE", "heading", 1), ("body", "text", 1),
        ("CHAPTER TWO", "heading", 2), ("body", "text", 2),
        ("CHAPTER THREE", "heading", 3), ("body", "text", 3),
        ("CHAPTER FOUR", "heading", 4), ("body", "text", 4),
        ("CHAPTER FIVE", "heading", 5), ("body", "text", 5),
        ("Notes", "heading", 6),
        ("CHAPTER ONE: EPISTEMOLOGIES", "heading", 6), ("note", "text", 6),
        ("CHAPTER TWO: TRUTH-TO-NATURE", "heading", 7), ("note", "text", 7),
    ])
    paths = _recovered_paths(rows)
    assert paths.count(("CHAPTER ONE",)) == 1
    assert ("CHAPTER FIVE", "Notes", "CHAPTER ONE: EPISTEMOLOGIES") in paths


def test_chapter_end_notes_do_not_stop_the_next_chapter_opening():
    # An edited collection puts notes at the end of every chapter. Folding all
    # later chapter headings under the notes region -- the first attempt at the
    # rule above -- left an eleven-chapter book with one chapter.
    rows = _pdf_rows([
        ("CHAPTER ONE", "heading", 1), ("body", "text", 1),
        ("Notes", "heading", 2), ("note", "text", 2),
        ("CHAPTER TWO", "heading", 3), ("body", "text", 3),
        ("Notes", "heading", 4), ("note", "text", 4),
        ("CHAPTER THREE", "heading", 5), ("body", "text", 5),
        ("CHAPTER FOUR", "heading", 6), ("body", "text", 6),
        ("CHAPTER FIVE", "heading", 7), ("body", "text", 7),
    ])
    paths = _recovered_paths(rows)
    assert ("CHAPTER TWO",) in paths
    assert ("CHAPTER ONE", "Notes") in paths
    assert ("CHAPTER TWO", "Notes") in paths


def test_one_prose_line_does_not_become_a_part_and_reparent_the_chapters():
    # PART_RE matches a bare Roman numeral followed by "THE", and a part opener
    # is admitted from any block type because layout extraction often files one
    # as text. "DID" is a valid Roman numeral, so this sentence scored 999 and
    # became a part with chapters three to five nested under it.
    rows = _pdf_rows([
        ("CHAPTER ONE", "heading", 1), ("body", "text", 1),
        ("CHAPTER TWO", "heading", 2), ("body", "text", 2),
        ("DID THE COMMITTEE ever meet again? The record is silent.", "text", 2),
        ("CHAPTER THREE", "heading", 3), ("body", "text", 3),
        ("CHAPTER FOUR", "heading", 4), ("body", "text", 4),
        ("CHAPTER FIVE", "heading", 5), ("body", "text", 5),
    ])
    assert _recovered_paths(rows) == [
        ("CHAPTER ONE",), ("CHAPTER TWO",), ("CHAPTER THREE",),
        ("CHAPTER FOUR",), ("CHAPTER FIVE",),
    ]


def test_chapter_numbers_may_restart_inside_each_part():
    # An edited collection or a multi-volume work numbers its chapters from one
    # again in every part. A single book-wide set of opened chapter numbers read
    # part two's chapters as repeats of part one's and dropped all of them,
    # leaving part two's whole body on the bare part node.
    rows = _pdf_rows(
        [("PART ONE", "heading", 1)]
        + [entry for chapter in range(1, 4) for entry in (
            (f"CHAPTER {chapter}", "heading", chapter + 1),
            (f"part one body {chapter}", "text", chapter + 1))]
        + [("PART TWO", "heading", 5)]
        + [entry for chapter in range(1, 4) for entry in (
            (f"CHAPTER {chapter}", "heading", chapter + 5),
            (f"part two body {chapter}", "text", chapter + 5))]
    )
    paths = _recovered_paths(rows)
    assert ("PART TWO", "CHAPTER 1") in paths
    assert ("PART TWO", "CHAPTER 3") in paths
    assert paths.count(("PART ONE", "CHAPTER 1")) == 1


def test_the_contents_region_does_not_advance_with_the_rows_it_skips():
    # The bound is pinned to the contents heading. When it moved with the region
    # instead, a book with a heading on every page never reached it, and the
    # contents ran until something else ended it -- in one book here nothing did
    # until page 171, so its real part openers were read as contents entries and
    # the recovered tree was the back-of-book notes index instead.
    rows = _pdf_rows(
        [("Contents", "heading", 1), ("CHAPTER ONE EPISTEMOLOGIES 17", "heading", 1)]
        # The body starts a few pages past the contents, as it does in every
        # measured book, and from there a heading sits on every page -- which is
        # what a moving bound could never get past.
        + [entry for index, name in enumerate(
            ["ONE", "TWO", "THREE", "FOUR", "FIVE"], start=1)
           for entry in (
               (f"CHAPTER {name}", "heading", index * 2 + 2),
               (f"body {index}", "text", index * 2 + 2),
               ("OBJECTIVITY", "page_furniture", index * 2 + 3),
               (f"more {index}", "text", index * 2 + 3))]
        + [("Notes", "heading", 14)]
        + [entry for name in ["ONE", "TWO", "THREE", "FOUR", "FIVE"]
           for entry in ((f"CHAPTER {name}: NOTES", "heading", 14), ("note", "text", 14))]
    )
    paths = _recovered_paths(rows)
    assert paths[0] == ("CHAPTER ONE",)
    assert ("CHAPTER FIVE",) in paths


def test_divisions_that_do_not_divide_the_document_are_refused():
    # A contents page whose own "Contents" heading was lost during extraction
    # leaves nothing to mark the region, so its lines are read as the divisions
    # they name. Japan-ness in Architecture came out with four parts, of which
    # three held two or three chunks -- their own contents lines -- while the
    # fourth held 1,711 of 1,718. The names look right; the weights show it is
    # one part wearing the name of the fourth.
    rows = _pdf_rows(
        [("PART ONE THE MYSTERY 11", "heading", 1),
         ("PART TWO FROM MYTH 87", "heading", 1),
         ("PART THREE THE JOURNEY 153", "heading", 1),
         ("PART FOUR LITTLE GIRLS 221", "heading", 1),
         ("PART FIVE AN EVEN BALANCE 305", "heading", 1)]
        + [(f"body {index}", "text", 2 + index // 20) for index in range(200)]
    )
    assert _recovered_paths(rows) == []


def test_a_book_whose_divisions_share_the_text_is_still_recovered():
    # The refusal is aimed at divisions that hold nothing, not at uneven ones:
    # a long first chapter followed by short ones is an ordinary book.
    rows = _pdf_rows(
        [("CHAPTER ONE", "heading", 1)]
        + [(f"long first chapter {index}", "text", 2 + index // 20) for index in range(100)]
        + [entry for chapter, name in enumerate(
            ["TWO", "THREE", "FOUR", "FIVE"], start=8)
           for entry in ([(f"CHAPTER {name}", "heading", chapter)]
                         + [(f"body {chapter}-{index}", "text", chapter) for index in range(10)])]
    )
    paths = _recovered_paths(rows)
    assert [path[0] for path in paths] == [
        "CHAPTER ONE", "CHAPTER TWO", "CHAPTER THREE", "CHAPTER FOUR", "CHAPTER FIVE",
    ]


# The pieces the recovery loop was split into, each asked its own question.

def _census(entries):
    from src.source_structure_refresh import _HeadingCensus
    return _HeadingCensus.of(_pdf_rows(entries))


def test_the_census_counts_headings_and_furniture_but_not_body_text():
    census = _census([
        ("CHAPTER ONE", "heading", 1), ("CHAPTER ONE", "page_furniture", 2),
        ("CHAPTER ONE", "text", 3),
    ])
    from src.source_structure_refresh import _repetition_key
    assert census.total[_repetition_key("CHAPTER ONE")] == 2


def test_a_folio_does_not_hide_a_repetition_from_the_census():
    # "序章] 13", "序章] 15" -- the printed page number makes every occurrence
    # textually unique, so counting the raw text sees no repetition at all.
    census = _census([(f"第一章 誤解 | {page * 2}", "page_furniture", page)
                      for page in range(1, 6)])
    assert census.repeats("第一章 誤解")


def test_the_census_says_when_a_heading_still_lies_ahead():
    census = _census([("PART II THE JOURNEY 87", "heading", 1),
                      ("PART TWO THE JOURNEY", "heading", 9)])
    # Matched on level and number: a contents page says "PART II" where the body
    # page says "PART TWO", and the two share no text.
    assert census.recurs_later("PART II THE JOURNEY 87")
    census.passing("PART II THE JOURNEY 87")
    census.passing("PART TWO THE JOURNEY")
    assert not census.recurs_later("PART TWO THE JOURNEY")


def test_the_contents_region_opens_on_either_language():
    from src.source_structure_refresh import _PrintedContents
    for heading in ("Contents", "CONTENTS", "目次", "Table of Contents"):
        contents = _PrintedContents()
        assert contents.opens_at(heading, 3), heading
        assert contents.heading_page == 3
    assert not _PrintedContents().opens_at("CHAPTER ONE", 3)


def test_the_contents_region_ends_a_page_after_its_heading():
    from src.source_structure_refresh import _PrintedContents
    contents = _PrintedContents()
    contents.opens_at("Contents", 1)
    census = _census([("CHAPTER ONE", "heading", 9)])
    assert contents.holds("CHAPTER ONE EPISTEMOLOGIES 17", 1, "heading", census)
    assert not contents.holds("CHAPTER ONE", 9, "heading", census)


def test_furniture_is_admitted_only_when_the_document_does_not_repeat_it():
    from src.source_structure_refresh import _admits_as_boundary
    once = _census([("CHAPTER ONE", "page_furniture", 1)])
    often = _census([("CHAPTER ONE", "page_furniture", page) for page in range(1, 9)])
    assert _admits_as_boundary("CHAPTER ONE", "page_furniture", once)
    assert not _admits_as_boundary("CHAPTER ONE", "page_furniture", often)
    # A heading block needs no such evidence, and a part opener is admitted from
    # any block type because layout extraction so often files one as text.
    assert _admits_as_boundary("CHAPTER ONE", "heading", often)
    assert _admits_as_boundary("PART ONE", "text", often)
    assert not _admits_as_boundary("CHAPTER ONE", "text", once)


def test_a_tree_reports_itself_unusable_before_it_is_stored():
    from src.source_structure_refresh import _RecoveredTree
    tree = _RecoveredTree()
    for index, title in enumerate(["CHAPTER ONE", "CHAPTER THREE", "CHAPTER FOUR",
                                   "CHAPTER FIVE", "CHAPTER SIX"]):
        tree.place(index * 10, title)
    # 1, 3, 4, 5, 6 -- the extractor lost chapter two, and storing this would
    # assert a six-chapter book has five.
    assert not tree.is_usable(100)
