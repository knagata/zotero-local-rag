from scripts.build_outline_gold_package import candidate_rows


def test_candidate_rows_keeps_disagreement_visible():
    rows = candidate_rows(
        [[1, "Chapter One", 10]],
        [{"title": "Chapter One", "level": 2, "page_hint": 8}],
    )
    assert rows[0]["bookmark_level"] == 1
    assert rows[0]["ai_level"] == 2
    assert rows[0]["level"] == ""
    assert rows[0]["candidate_sources"] == "bookmark+ai"


def test_candidate_rows_preserves_ai_only_candidate():
    rows = candidate_rows([], [{"title": "Introduction", "level": 1, "page_hint": 3}])
    assert rows[0]["title"] == "Introduction"
    assert rows[0]["candidate_sources"] == "ai"
