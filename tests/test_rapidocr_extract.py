from __future__ import annotations

from src.rapidocr_extract import _merge_page_chunks
from src.text_utils import MAX_CHARS, TARGET_CHARS


def _line(page: int, ordinal: int, text: str):
    return (
        f"ATT:p{page}:para{ordinal}:part0",
        text,
        {"page": page, "locator": f"p{page}:para{ordinal}", "bbox": "{}"},
    )


def test_rapidocr_visual_lines_are_coalesced_to_retrieval_units_per_page():
    # A multi-column image can produce one RapidOCR record per visual line.
    # These must not become one vector per line, but page provenance remains a
    # hard boundary even when both pages contain short lines.
    page_one = [_line(1, index, f"page one visual line {index:03d}.") for index in range(80)]
    page_two = [_line(2, index, f"page two visual line {index:03d}.") for index in range(80)]

    merged = _merge_page_chunks(page_one) + _merge_page_chunks(page_two)

    assert len(merged) < 8
    assert {metadata["page"] for _chunk_id, _text, metadata in merged} == {1, 2}
    assert all(len(text) <= MAX_CHARS for _chunk_id, text, _metadata in merged)
    assert all(metadata.get("merge_count", 1) > 1 for _chunk_id, _text, metadata in merged)
    page_one_text = "\n".join(text for _id, text, md in merged if md["page"] == 1)
    page_two_text = "\n".join(text for _id, text, md in merged if md["page"] == 2)
    assert all(page_one_text.count(text) == 1 for _id, text, _md in page_one)
    assert all(page_two_text.count(text) == 1 for _id, text, _md in page_two)
    # Normal OCR pages should form chunks around the configured retrieval target
    # (the final remainder may be shorter), rather than 20--30 character lines.
    lengths = sorted(len(text) for _chunk_id, text, _metadata in merged)
    assert lengths[len(lengths) // 2] >= TARGET_CHARS
