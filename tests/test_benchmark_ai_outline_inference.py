from scripts.benchmark_ai_outline_inference import load_gold_csv, match_outlines, normalize_title


def test_normalize_title_tolerates_punctuation_and_case():
    assert normalize_title("Chapter 1: The Social") == normalize_title("CHAPTER 1 — The Social")


def test_match_outlines_reports_title_and_level_quality():
    truth = [
        {"title": "Part I", "level": 1},
        {"title": "First Chapter", "level": 2},
    ]
    prediction = [
        {"title": "PART I", "level": 1},
        {"title": "First Chapter", "level": 1},
        {"title": "Invented", "level": 2},
    ]
    score = match_outlines(truth, prediction)
    assert score["matched"] == 2
    assert score["title_precision"] == 0.6667
    assert score["title_recall"] == 1.0
    assert score["level_accuracy_on_matched"] == 0.5


def test_load_gold_csv_ignores_empty_tail_row(tmp_path):
    path = tmp_path / "gold.csv"
    path.write_text(
        "id,title,level,parent_id,pdf_page,kind\n"
        "n1,Introduction,1,,3,chapter\n"
        "n2,,,,,\n",
        encoding="utf-8",
    )
    assert load_gold_csv(path) == [{
        "id": "n1", "title": "Introduction", "level": 1,
        "page": 3, "parent_id": "", "kind": "chapter",
    }]
