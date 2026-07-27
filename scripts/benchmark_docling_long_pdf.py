#!/usr/bin/env python3
"""Benchmark standard Docling or Granite-Docling MLX on one fixed PDF subset."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any


def document_metrics(document: Any) -> dict[str, Any]:
    labels: Counter[str] = Counter()
    characters = 0
    items = 0
    pages_with_content: set[int] = set()
    for item, _level in document.iterate_items():
        text = str(getattr(item, "text", "") or "")
        label = getattr(item, "label", "unknown")
        label_value = str(getattr(label, "value", label)).lower()
        labels[label_value] += 1
        characters += len(text)
        items += 1
        for prov in getattr(item, "prov", None) or []:
            page_no = getattr(prov, "page_no", None)
            if isinstance(page_no, int):
                pages_with_content.add(page_no)
    return {
        "items": items, "characters": characters,
        "pages_with_content": len(pages_with_content),
        "labels": dict(sorted(labels.items())),
    }


def converter_for(engine: str) -> Any:
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    if engine == "standard":
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        return DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=PdfPipelineOptions()),
        })
    from docling.datamodel import vlm_model_specs
    from docling.datamodel.pipeline_options import VlmPipelineOptions
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
    parser.add_argument("--engine", choices=("standard", "granite_mlx"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input does not exist: {args.input}")

    converter = converter_for(args.engine)
    started = time.monotonic()
    result = converter.convert(str(args.input))
    duration = time.monotonic() - started
    metrics = document_metrics(result.document)
    page_count = len(getattr(result.document, "pages", {}) or {})
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    report = {
        "schema_version": "docling-long-benchmark-1",
        "engine": args.engine,
        "input_name": args.input.name,
        "conversion_status": str(result.status),
        "pages": page_count,
        "duration_seconds": round(duration, 3),
        "seconds_per_page": round(duration / max(1, page_count), 3),
        "pages_per_minute": round(page_count * 60.0 / max(duration, 0.001), 3),
        "peak_rss_mb": round(rss / (1024**2 if sys.platform == "darwin" else 1024), 3),
        **metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
