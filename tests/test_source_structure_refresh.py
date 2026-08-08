from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.source_structure_refresh import (
    _numbering_is_contiguous,
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
    titles = [
        "目次", "第一部 目次上の部", "第一章 目次上の章", "第一節 目次上の節",
        "序章", "I 本文の導入", "第一章 本文一", "第一節 本文一の節",
        "第二章 本文二", "第三章 本文三",
    ]
    rows = [
        {"id": f"PDF:{index}", "text": title,
         "metadata": {"attachmentKey": "PDF", "source_type": "pdf", "page": index + 1,
                      "block_type": "heading", "structure_path": ["stale"]}}
        for index, title in enumerate(titles)
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


@pytest.mark.parametrize("opener", ["第一部 本論", "第一章 本論"])
def test_pdf_heading_recovery_exits_printed_toc_at_numbered_body_opener(opener):
    rows = [
        {"id": "PDF:toc", "text": "目次",
         "metadata": {"attachmentKey": "PDF", "source_type": "pdf", "page": 1,
                      "block_type": "heading"}},
        {"id": "PDF:toc-entry", "text": opener,
         "metadata": {"attachmentKey": "PDF", "source_type": "pdf", "page": 1,
                      "block_type": "text"}},
    ]
    for index, title in enumerate([
        opener, "第一章 導入", "第二章 展開", "第三章 展開", "第四章 結論",
    ]):
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
