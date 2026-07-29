#!/usr/bin/env python3
"""Create a path-free, deterministic comparison of OCR bake-off reports."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "evaluations" / "ocr_bakeoff_v3" / "manifest.json"
DEFAULT_OUTPUT = ROOT / "tmp" / "ocr_bakeoff_v3" / "comparison"
METRIC_ORDER = (
    "heading_hierarchy", "reading_order", "zone_classification",
    "table_caption_retention", "locator_bbox_recovery", "tree_integrity",
    "text_accuracy",
)
STATUS_RANK = {
    "completed": 5, "ready": 4, "unavailable": 3, "blocked": 2, "failed": 1,
}
TIE_BREAK_RULE = [
    "higher mean total score",
    "higher mean metrics in METRIC_ORDER",
    "lower mean duration_seconds",
    "lower mean process_peak_rss_mb",
    "lexicographically smaller engine name",
]
DUPLICATE_RUN_RULE = (
    "prefer status completed > ready > unavailable > blocked > failed; then higher score "
    "and metrics; then lower duration and peak RSS"
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return round(result, 6) if math.isfinite(result) else None


def _identifier(value: Any, field: str) -> str:
    normalized = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", normalized):
        raise ValueError(f"invalid {field}: {normalized!r}")
    return normalized


def load_categories(manifest_path: Path) -> dict[str, str]:
    manifest = _load_object(manifest_path)
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError("manifest.samples must be a list")
    categories: dict[str, str] = {}
    for row in samples:
        if not isinstance(row, Mapping):
            continue
        sample_id = _identifier(row.get("id"), "sample id")
        category = _identifier(row.get("category"), "category")
        if sample_id in categories:
            raise ValueError(f"duplicate manifest sample id: {sample_id}")
        categories[sample_id] = category
    return categories


def _sanitize_run(run: Mapping[str, Any], categories: Mapping[str, str]) -> dict[str, Any]:
    sample_id = _identifier(run.get("sample_id"), "sample id")
    engine = _identifier(run.get("engine"), "engine")
    if sample_id not in categories:
        raise ValueError(f"sample is absent from manifest: {sample_id}")
    score_payload = run.get("score")
    score_payload = score_payload if isinstance(score_payload, Mapping) else {}
    metric_payload = score_payload.get("metrics")
    metric_payload = metric_payload if isinstance(metric_payload, Mapping) else {}
    metrics = {
        name: value for name in METRIC_ORDER
        if (value := _number(metric_payload.get(name))) is not None
    }
    return {
        "sample_id": sample_id,
        "category": categories[sample_id],
        "engine": engine,
        "status": str(run.get("status") or "unknown").strip().lower(),
        "score": _number(score_payload.get("total_score")),
        "metrics": metrics,
        "duration_seconds": _number(run.get("duration_seconds")),
        "process_peak_rss_mb": _number(run.get("process_peak_rss_mb")),
    }


def _duplicate_key(run: Mapping[str, Any]) -> tuple[Any, ...]:
    """Choose one duplicate run without depending on report or input order."""
    return (
        STATUS_RANK.get(str(run["status"]), 0),
        run["score"] if run["score"] is not None else -math.inf,
        tuple(run["metrics"].get(name, -math.inf) for name in METRIC_ORDER),
        -(run["duration_seconds"] if run["duration_seconds"] is not None else math.inf),
        -(run["process_peak_rss_mb"] if run["process_peak_rss_mb"] is not None else math.inf),
        json.dumps(run, ensure_ascii=False, sort_keys=True),
    )


def _mean(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return round(sum(present) / len(present), 6) if present else None


def _engine_summary(engine: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [run for run in runs if run["status"] == "completed" and run["score"] is not None]
    return {
        "engine": engine,
        "completed_runs": len(completed),
        "mean_score": _mean(run["score"] for run in completed),
        "mean_metrics": {
            name: value for name in METRIC_ORDER
            if (value := _mean(run["metrics"].get(name) for run in completed)) is not None
        },
        "mean_duration_seconds": _mean(run["duration_seconds"] for run in completed),
        "mean_process_peak_rss_mb": _mean(run["process_peak_rss_mb"] for run in completed),
    }


def _winner_key(summary: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -(summary["mean_score"] if summary["mean_score"] is not None else -math.inf),
        *(
            -(summary["mean_metrics"].get(name, -math.inf))
            for name in METRIC_ORDER
        ),
        summary["mean_duration_seconds"] if summary["mean_duration_seconds"] is not None else math.inf,
        summary["mean_process_peak_rss_mb"] if summary["mean_process_peak_rss_mb"] is not None else math.inf,
        summary["engine"],
    )


def aggregate_reports(
    reports: Iterable[Mapping[str, Any]], categories: Mapping[str, str],
) -> dict[str, Any]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for report in reports:
        runs = report.get("runs")
        if not isinstance(runs, list):
            raise ValueError("report.runs must be a list")
        for raw in runs:
            if not isinstance(raw, Mapping):
                raise ValueError("every report run must be an object")
            run = _sanitize_run(raw, categories)
            key = (run["sample_id"], run["engine"])
            if key not in selected or _duplicate_key(run) > _duplicate_key(selected[key]):
                selected[key] = run
    runs = sorted(selected.values(), key=lambda run: (run["sample_id"], run["engine"]))

    engines = sorted({run["engine"] for run in runs})
    engine_averages = [_engine_summary(
        engine, [run for run in runs if run["engine"] == engine],
    ) for engine in engines]
    category_winners = []
    for category in sorted(set(categories.values())):
        category_runs = [run for run in runs if run["category"] == category]
        summaries = [
            _engine_summary(engine, [run for run in category_runs if run["engine"] == engine])
            for engine in sorted({run["engine"] for run in category_runs})
        ]
        eligible = [summary for summary in summaries if summary["completed_runs"] > 0]
        winner = min(eligible, key=_winner_key) if eligible else None
        category_winners.append({
            "category": category,
            "winner": winner["engine"] if winner else None,
            "mean_score": winner["mean_score"] if winner else None,
            "completed_runs": winner["completed_runs"] if winner else 0,
        })
    return {
        "comparison_version": "ocr-bakeoff-comparison-v1",
        "metric_order": list(METRIC_ORDER),
        "tie_break_rule": TIE_BREAK_RULE,
        "duplicate_run_rule": DUPLICATE_RUN_RULE,
        "runs": runs,
        "category_winners": category_winners,
        "engine_averages": engine_averages,
    }


def _cell(value: Any) -> str:
    return str(value if value is not None else "—").replace("|", "\\|").replace("\n", " ")


def _metric_text(values: Mapping[str, Any]) -> str:
    return "; ".join(
        f"{name}={values[name]:.6f}" for name in METRIC_ORDER if name in values
    ) or "—"


def markdown_report(comparison: Mapping[str, Any]) -> str:
    lines = [
        "# OCRベイクオフ比較", "", "## Sample × engine", "",
        "| Sample | Category | Engine | Score | Metrics | Duration (s) | Peak RSS (MB) | Status |",
        "|---|---|---|---:|---|---:|---:|---|",
    ]
    for run in comparison.get("runs", []):
        score = f"{run['score']:.6f}" if run.get("score") is not None else "—"
        lines.append(
            f"| {_cell(run.get('sample_id'))} | {_cell(run.get('category'))} | "
            f"{_cell(run.get('engine'))} | {score} | {_metric_text(run.get('metrics', {}))} | "
            f"{_cell(run.get('duration_seconds'))} | "
            f"{_cell(run.get('process_peak_rss_mb'))} | {_cell(run.get('status'))} |"
        )
    lines.extend([
        "", "## カテゴリ別winner", "",
        "| Category | Winner | Mean score | Completed runs |", "|---|---|---:|---:|",
    ])
    for row in comparison.get("category_winners", []):
        lines.append(
            f"| {_cell(row.get('category'))} | {_cell(row.get('winner'))} | "
            f"{_cell(row.get('mean_score'))} | {_cell(row.get('completed_runs'))} |"
        )
    lines.extend([
        "", "## Engine平均", "",
        "| Engine | Completed runs | Mean score | Mean metrics | Mean duration (s) | Mean peak RSS (MB) |",
        "|---|---:|---:|---|---:|---:|",
    ])
    for row in comparison.get("engine_averages", []):
        lines.append(
            f"| {_cell(row.get('engine'))} | {_cell(row.get('completed_runs'))} | "
            f"{_cell(row.get('mean_score'))} | {_metric_text(row.get('mean_metrics', {}))} | "
            f"{_cell(row.get('mean_duration_seconds'))} | "
            f"{_cell(row.get('mean_process_peak_rss_mb'))} |"
        )
    lines.extend(["", "## 同点規則", ""])
    lines.extend(f"{index}. {rule}" for index, rule in enumerate(TIE_BREAK_RULE, start=1))
    lines.extend(["", f"重複run: {DUPLICATE_RUN_RULE}."])
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="+", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    comparison = aggregate_reports(
        [_load_object(path) for path in args.report], load_categories(args.manifest),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (args.output / "comparison.md").write_text(markdown_report(comparison), encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
