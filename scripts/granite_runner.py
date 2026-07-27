#!/usr/bin/env python3
"""Extract one PDF with Granite-Docling, from inside the isolated Granite venv.

This file is never imported by the main application. It is executed by a
*different* interpreter -- the one in the Granite virtualenv -- because
``mlx-vlm`` requires ``transformers>=5.14`` while the Docling pipeline the rest
of the project uses pins ``<5.9`` on macOS. The two cannot share an
environment, so the boundary between them is a process boundary, and this is
the far side of it.

Protocol: one JSON request on stdin, one JSON response on stdout. Anything the
underlying libraries print goes to stderr, so stdout carries only the response
and the parent can parse it without stripping noise.

Chunking and normalisation are deliberately *not* reimplemented here: the
request is handed to ``extract_chunks_from_pdf_with_docling`` with a VlmPipeline
converter, so Granite output travels the same normalisation, zone assignment
and chunk-boundary rules as ordinary Docling output. Those modules import only
the standard library plus Docling, which the Granite venv also has.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_converter():
    """A DocumentConverter driving Granite-Docling through Docling's VlmPipeline."""
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
    try:
        request = json.loads(sys.stdin.read() or "{}")
    except ValueError as exc:
        json.dump({"status": "error", "message": f"invalid request: {exc}"}, sys.stdout)
        return 1

    pdf_path = str(request.get("pdf_path") or "")
    attachment_key = str(request.get("attachment_key") or "")
    meta_base = request.get("meta_base") or {}
    if not pdf_path or not attachment_key:
        json.dump(
            {"status": "error", "message": "pdf_path and attachment_key are required"},
            sys.stdout,
        )
        return 1

    try:
        from docling_extract import extract_chunks_from_pdf_with_docling

        chunks, quality_info = extract_chunks_from_pdf_with_docling(
            Path(pdf_path), attachment_key, dict(meta_base),
            converter=build_converter(),
        )
    except Exception as exc:  # noqa: BLE001 - reported to the parent, not raised
        json.dump({"status": "error", "message": f"{type(exc).__name__}: {exc}"}, sys.stdout)
        return 1

    quality_info = dict(quality_info)
    quality_info["parser"] = "granite_docling_mlx"
    quality_info["extraction_engine"] = "granite"
    json.dump(
        {
            "status": "ok",
            # Tuples do not survive JSON; the parent restores them.
            "chunks": [[cid, text, metadata] for cid, text, metadata in chunks],
            "quality_info": quality_info,
        },
        sys.stdout, ensure_ascii=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
