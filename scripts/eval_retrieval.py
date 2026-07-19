#!/usr/bin/env python3
"""Evaluate retrieval hit@k, MRR, and result-language mix from a JSONL gold set."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def score_ranked_items(ranked: list[str], expected: set[str], k: int) -> dict[str, float]:
    top = ranked[:k]
    first_rank = next((index for index, key in enumerate(top, start=1) if key in expected), None)
    return {"hit": float(first_rank is not None), "rr": 1.0 / first_rank if first_rank else 0.0}


def load_questions(path: Path) -> list[dict[str, Any]]:
    questions = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        row = json.loads(line)
        if not row.get("query") or not row.get("expected_item_keys"):
            raise ValueError(f"{path}:{line_number}: query and expected_item_keys are required")
        questions.append(row)
    return questions


def evaluate(
    questions: list[dict[str, Any]], search: Callable[..., dict[str, Any]],
    *, k: int, kwargs: dict[str, Any],
) -> dict[str, Any]:
    totals = {"hit": 0.0, "rr": 0.0}
    languages: dict[str, int] = {}
    details = []
    for row in questions:
        response = search(row["query"], k=k, **kwargs)
        results = response.get("results") or []
        ranked = [(result.get("meta") or {}).get("itemKey") for result in results]
        ranked = [key for key in ranked if key]
        score = score_ranked_items(ranked, set(row["expected_item_keys"]), k)
        totals["hit"] += score["hit"]
        totals["rr"] += score["rr"]
        for result in results:
            lang = str((result.get("meta") or {}).get("lang") or "unknown")
            languages[lang] = languages.get(lang, 0) + 1
        details.append({"id": row.get("id"), "ranked_item_keys": ranked, **score})
    denominator = max(len(questions), 1)
    return {
        "questions": len(questions), "hit_at_k": totals["hit"] / denominator,
        "mrr_at_k": totals["rr"] / denominator, "languages": languages, "details": details,
    }


@contextmanager
def _temporary_env(values: dict[str, str]):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qa", type=Path)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--include-hierarchical-v2", action="store_true",
        help="Also score legacy and canonical-node hierarchical routing. V2 needs prebuilt LLM node summaries.",
    )
    args = parser.parse_args()
    if args.k <= 0:
        parser.error("--k must be positive")
    import rag_mcp_server

    questions = load_questions(args.qa)
    conditions = {
        "semantic": {"auto_expand": False, "hybrid": False, "language_balance": False},
        "expanded": {"auto_expand": True, "hybrid": False, "language_balance": False},
        "hybrid": {"auto_expand": True, "hybrid": True, "language_balance": False},
        "balanced": {"auto_expand": True, "hybrid": True, "language_balance": True},
    }
    report = {
        name: evaluate(questions, rag_mcp_server.rag_search, k=args.k, kwargs=kwargs)
        for name, kwargs in conditions.items()
    }
    if args.include_hierarchical_v2:
        with _temporary_env({"HIERARCHICAL_SEARCH_V2_ENABLE": "0"}):
            report["hierarchical_legacy"] = evaluate(
                questions, rag_mcp_server.hierarchical_search, k=args.k,
                kwargs={"auto_expand": True},
            )
        with _temporary_env({"HIERARCHICAL_SEARCH_V2_ENABLE": "1"}):
            report["hierarchical_v2"] = evaluate(
                questions, rag_mcp_server.hierarchical_search, k=args.k,
                kwargs={"auto_expand": True},
            )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
