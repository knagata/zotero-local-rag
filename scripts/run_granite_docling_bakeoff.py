#!/usr/bin/env python3
"""Compare Granite-Docling MLX with cached standard-Docling English fixtures."""
from __future__ import annotations

import argparse
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

from src.docling_extract import _docling_items  # noqa: E402
from src.ocr_bakeoff import load_json, score_result  # noqa: E402


FIXTURE_ROOT = ROOT / "tmp" / "ocr_bakeoff_v3" / "docling_fixed"
EVALUATION_ROOT = ROOT / "evaluations" / "ocr_bakeoff_v3"
DEFAULT_OUTPUT = ROOT / "tmp" / "ocr_bakeoff_v3" / "granite_docling_mlx"
ENGLISH_SAMPLES = (
    "en_two_column", "tables_math", "scanned_pair", "notes_bibliography_book",
)


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return round(value / (1024**2 if sys.platform == "darwin" else 1024), 3)


def document_to_bakeoff_blocks(document: Any) -> list[dict[str, Any]]:
    """Adapt a VLM DoclingDocument to the existing engine-neutral scorer."""
    blocks: list[dict[str, Any]] = []
    page_items = _docling_items(document)
    for page in sorted(page_items):
        for item in page_items[page]:
            metadata = {
                "page": page,
                "locator": f"p{page}:order{item['reading_order']}",
                "block_type": item["block_type"],
                "reading_order": item["reading_order"],
                "zone": item["zone"],
                "structure_path": item.get("structure_path") or [],
                "provenance": item.get("provenance") or [],
            }
            provenance = metadata["provenance"]
            if provenance and isinstance(provenance[0], dict) and provenance[0].get("bbox"):
                metadata["bbox"] = provenance[0]["bbox"]
            blocks.append({
                "id": f"granite:p{page}:order{item['reading_order']}",
                "text": item["text"], "ordinal": len(blocks), "metadata": metadata,
            })
    return blocks


def _converter() -> Any:
    from docling.datamodel import vlm_model_specs
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import VlmPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline

    return DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=VlmPipelineOptions(
                vlm_options=vlm_model_specs.GRANITEDOCLING_MLX,
            ),
        ),
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="append", choices=ENGLISH_SAMPLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    samples = tuple(dict.fromkeys(args.sample or ENGLISH_SAMPLES))
    missing = [
        sample for sample in samples
        if not (FIXTURE_ROOT / "sources" / f"{sample}.pdf").is_file()
        or not (EVALUATION_ROOT / "annotations" / f"{sample}.json").is_file()
        or not (FIXTURE_ROOT / "raw" / sample / "docling.json").is_file()
    ]
    if missing:
        raise RuntimeError(f"Missing frozen fixture(s): {', '.join(missing)}")
    if args.dry_run:
        print(json.dumps({"ready": True, "samples": samples}, ensure_ascii=False))
        return 0

    converter = _converter()
    runs = []
    args.output.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        source = FIXTURE_ROOT / "sources" / f"{sample}.pdf"
        truth = load_json(EVALUATION_ROOT / "annotations" / f"{sample}.json")
        baseline = load_json(FIXTURE_ROOT / "raw" / sample / "docling.json")
        started = time.monotonic()
        result = converter.convert(str(source))
        raw = {
            "engine": "granite_docling_mlx",
            "version": "granite-docling-258M-mlx-bf16",
            "blocks": document_to_bakeoff_blocks(result.document),
            "quality": {"conversion_status": str(result.status)},
        }
        duration = round(time.monotonic() - started, 3)
        granite_score = score_result(truth, raw)
        baseline_score = score_result(truth, baseline)
        raw_dir = args.output / "raw" / sample
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "granite_docling_mlx.json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        runs.append({
            "sample_id": sample,
            "granite_score": granite_score,
            "standard_docling_score": baseline_score,
            "score_delta": round(
                granite_score["total_score"] - baseline_score["total_score"], 6,
            ),
            "granite_duration_seconds": duration,
            "process_peak_rss_mb": _peak_rss_mb(),
        })
        print(json.dumps(runs[-1], ensure_ascii=False), flush=True)
    report = {
        "schema_version": "granite-docling-bakeoff-1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "samples": list(samples), "runs": runs,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
