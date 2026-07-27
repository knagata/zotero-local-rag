#!/usr/bin/env python3
"""Run the local-only structured PDF extraction bake-off."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.extraction_engine import EngineRegistry, SourceRef, inspect_engine_availability
from src.ocr_bakeoff import (
    load_json, markdown_report, resolve_sample_path, score_result, validate_manifest,
    validate_truth,
)


DEFAULT_MANIFEST = ROOT / "evaluations" / "ocr_bakeoff_v3" / "manifest.json"


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    divisor = 1024**2 if sys.platform == "darwin" else 1024
    return round(value / divisor, 3)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local PDF V3 extraction bake-off")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--engine", action="append", help="Engine name; repeatable (default: all)")
    parser.add_argument("--sample", action="append", help="Sample id; repeatable (default: all)")
    parser.add_argument("--list", action="store_true", help="List samples, engines, and readiness")
    parser.add_argument("--dry-run", action="store_true", help="Validate selection without parsing PDFs")
    parser.add_argument("--output", type=Path, default=ROOT / "tmp" / "ocr_bakeoff_v3")
    return parser


def _selected(values: list[str] | None, available: list[str], kind: str) -> list[str]:
    chosen = available if not values or "all" in values else values
    unknown = sorted(set(chosen) - set(available))
    if unknown:
        raise ValueError(f"Unknown {kind}: {', '.join(unknown)}")
    return list(dict.fromkeys(chosen))


def _truth_path(manifest_path: Path, sample: dict[str, Any]) -> Path | None:
    value = str(sample.get("ground_truth") or "")
    return manifest_path.parent / value if value else None


def _evaluation_source(source_path: Path, sample: dict[str, Any], output: Path) -> Path:
    """Create a deterministic page subset so every engine sees the same evidence."""
    pages = sample.get("pages")
    if not isinstance(pages, list) or not pages:
        return source_path
    import fitz

    source = fitz.open(str(source_path))
    subset = fitz.open()
    try:
        for page in pages:
            index = int(page) - 1
            if index < 0 or index >= source.page_count:
                raise ValueError(f"sample page {page} is outside 1..{source.page_count}")
            subset.insert_pdf(source, from_page=index, to_page=index)
        target_dir = output / "sources"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{sample['id']}.pdf"
        subset.save(str(target), garbage=4, deflate=True)
        return target
    finally:
        subset.close()
        source.close()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = load_json(args.manifest)
    validate_manifest(manifest)
    samples = manifest.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("manifest.samples must be a list")
    registry = EngineRegistry()
    sample_by_id = {str(sample["id"]): sample for sample in samples}
    engine_names = _selected(args.engine, list(registry.names()), "engine")
    sample_ids = _selected(args.sample, list(sample_by_id), "sample")

    availability = {
        name: asdict(inspect_engine_availability(registry.get(name))) for name in engine_names
    }
    if args.list:
        print("Engines:")
        for name in engine_names:
            state = availability[name]
            print(f"  {name}: {'ready' if state['available'] else 'unavailable'} ({state['reason']})")
        print("Samples:")
        for sample_id in sample_ids:
            sample = sample_by_id[sample_id]
            source = resolve_sample_path(sample)
            truth = _truth_path(args.manifest, sample)
            print(
                f"  {sample_id}: source={'set' if source else 'missing:' + str(sample.get('path_env', ''))}; "
                f"truth={'ready' if truth and truth.is_file() else 'missing'}"
            )
        return 0

    runs: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        sample = sample_by_id[sample_id]
        source_path = resolve_sample_path(sample)
        truth_path = _truth_path(args.manifest, sample)
        for engine_name in engine_names:
            run: dict[str, Any] = {"sample_id": sample_id, "engine": engine_name}
            if source_path is None:
                run.update(status="blocked", reason=f"unset environment variable: {sample.get('path_env', '')}")
            elif not source_path.is_file():
                run.update(status="blocked", reason="configured source PDF does not exist")
            elif truth_path is None or not truth_path.is_file():
                run.update(status="blocked", reason="ground-truth annotation is missing")
            elif not availability[engine_name]["available"]:
                run.update(status="unavailable", reason=availability[engine_name]["reason"])
            elif args.dry_run:
                validate_truth(load_json(truth_path))
                run.update(status="ready")
            else:
                engine = registry.get(engine_name)
                started = time.monotonic()
                try:
                    evaluation_path = _evaluation_source(source_path, sample, args.output)
                    result = engine.parse(SourceRef(
                        evaluation_path, sample_id,
                        {"bakeoff_sample": sample_id, "source_pages": sample.get("pages") or []},
                    ))
                    raw = {
                        "engine": result.engine, "version": result.version,
                        "blocks": result.blocks, "quality": result.quality,
                        "diagnostics": result.diagnostics,
                    }
                    run["duration_seconds"] = round(time.monotonic() - started, 3)
                    run["process_peak_rss_mb"] = _peak_rss_mb()
                    run["score"] = score_result(load_json(truth_path), raw)
                    run["status"] = "completed"
                    raw_dir = args.output / "raw" / sample_id
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    (raw_dir / f"{engine_name}.json").write_text(
                        json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                    )
                except Exception as exc:
                    run.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            runs.append(run)

    report = {
        "bakeoff_version": manifest.get("version", "v3"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "availability": availability,
        "runs": runs,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (args.output / "report.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(run["status"] in {"ready", "completed", "unavailable", "blocked"} for run in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
