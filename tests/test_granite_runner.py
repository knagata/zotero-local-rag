"""Fast tests for Granite doctag validation and bounded margin recovery."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "granite_runner_under_test", ROOT / "scripts" / "granite_runner.py",
)
assert SPEC and SPEC.loader
granite_runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(granite_runner)


def test_invalid_doctag_rejects_reversed_bbox_instead_of_sorting_it():
    text = "<doctag><picture><loc_31><loc_407><loc_75><loc_405></picture></doctag>"
    assert granite_runner.invalid_doctag_reason(text) == "reversed bbox"


def test_invalid_doctag_accepts_well_formed_distinct_regions():
    text = (
        "<doctag><text><loc_1><loc_2><loc_30><loc_40>one</text>"
        "<text><loc_2><loc_42><loc_31><loc_60>two</text></doctag>"
    )
    assert granite_runner.invalid_doctag_reason(text) is None


def test_repetitive_image_only_output_requires_retry():
    block = "<picture><loc_1><loc_2><loc_30><loc_40></picture>"
    assert granite_runner.truncate_repetitive_suffix(
        f"<doctag>{block * 8}</doctag>"
    ) is None


def test_repetitive_suffix_after_text_is_safely_truncated():
    text = "<text><loc_1><loc_2><loc_30><loc_40>body</text>"
    loop = "<picture><loc_4><loc_5><loc_20><loc_30></picture>" * 8
    sanitized = granite_runner.truncate_repetitive_suffix(
        f"<doctag>{text}{loop}</doctag>"
    )
    assert sanitized is not None
    assert "body" in sanitized
    assert sanitized.count("<picture>") == 0
    assert sanitized.endswith("</doctag>")


def test_one_reversed_bbox_on_textual_page_is_normalized():
    text = "<doctag><text><loc_30><loc_40><loc_1><loc_2>body</text></doctag>"
    normalized = granite_runner.normalize_isolated_reversed_bboxes(text)
    assert normalized is not None
    assert "<loc_1><loc_2><loc_30><loc_40>" in normalized


def test_many_reversed_boxes_without_text_are_not_hidden():
    block = "<picture><loc_30><loc_40><loc_1><loc_2></picture>"
    assert granite_runner.normalize_isolated_reversed_bboxes(
        f"<doctag>{block * 20}</doctag>"
    ) is None


def test_dark_margin_crop_is_conservative():
    ordinary = Image.new("RGB", (400, 300), "white")
    assert granite_runner.dark_margin_crop_box(ordinary) is None

    scan = Image.new("RGB", (800, 400), "black")
    draw = ImageDraw.Draw(scan)
    draw.rectangle((20, 20, 380, 380), fill="white")
    box = granite_runner.dark_margin_crop_box(scan)
    assert box is not None
    assert box[2] < 500
