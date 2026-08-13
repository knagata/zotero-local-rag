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
import re
import sys
import tempfile
import traceback
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_LOCATIONS_RE = re.compile(
    r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"
)


class GraniteDoctagError(ValueError):
    """Granite returned structurally unsafe doctags for one or more pages."""


def invalid_doctag_reason(text: str) -> str | None:
    """Reject generated markup that Docling cannot safely assemble.

    Sorting a reversed box would hide a collapsed generation.  Fail closed so
    the caller can retry the source with dark scanner margins removed instead.
    """
    locations = [tuple(map(int, match)) for match in _LOCATIONS_RE.findall(text)]
    if any(right < left or bottom < top for left, top, right, bottom in locations):
        return "reversed bbox"
    return None


def truncate_repetitive_suffix(text: str) -> str | None:
    """Keep useful doctags before a long identical-location generation loop."""
    matches = list(_LOCATIONS_RE.finditer(text))
    run_start = 0
    run = 1
    for index, (previous, current) in enumerate(pairwise(matches), start=1):
        if current.group(0) == previous.group(0):
            run += 1
        else:
            run_start = index
            run = 1
        if run < 8:
            continue
        cut = matches[run_start].start()
        prefix = text[:cut]
        # A page containing only repeated image regions has not recovered its
        # textual content; it must take the margin-crop retry instead.
        if not re.search(
            r"<(?:text|title|section_header_level_\d+|list_item|caption|footnote)>",
            prefix,
        ):
            return None
        prefix = re.sub(r"<[a-zA-Z0-9_]+>\s*$", "", prefix)
        return prefix.rstrip() + "\n</doctag>"
    return text


def normalize_isolated_reversed_bboxes(text: str) -> str | None:
    """Repair a few bbox inversions only when the page otherwise has real text."""
    matches = list(_LOCATIONS_RE.finditer(text))
    reversed_count = sum(
        int(match.group(3)) < int(match.group(1))
        or int(match.group(4)) < int(match.group(2))
        for match in matches
    )
    if not reversed_count:
        return text
    has_text = re.search(
        r"<(?:text|title|section_header_level_\d+|list_item|caption|footnote)>", text,
    )
    if not has_text or reversed_count > max(4, int(len(matches) * 0.02)):
        return None

    def ordered(match):
        left, top, right, bottom = map(int, match.groups())
        left, right = sorted((left, right))
        top, bottom = sorted((top, bottom))
        return f"<loc_{left}><loc_{top}><loc_{right}><loc_{bottom}>"

    return _LOCATIONS_RE.sub(ordered, text)


def dark_margin_crop_box(image):
    """Return a conservative content box only for a large dark scanner matte."""
    from PIL import ImageOps

    width, height = image.size
    if width < 100 or height < 100:
        return None
    sample = ImageOps.grayscale(image)
    sample.thumbnail((1000, 1000))
    sw, sh = sample.size
    pixels = sample.load()

    def largest_bright_run(values):
        runs = []
        start = None
        for index, value in enumerate(values + [0.0]):
            if value >= 24 and start is None:
                start = index
            elif value < 24 and start is not None:
                runs.append((start, index))
                start = None
        return max(runs, key=lambda run: run[1] - run[0], default=None)

    col_run = largest_bright_run([
        sum(pixels[x, y] for y in range(sh)) / sh for x in range(sw)
    ])
    row_run = largest_bright_run([
        sum(pixels[x, y] for x in range(sw)) / sw for y in range(sh)
    ])
    if col_run is None or row_run is None:
        return None
    padding = max(2, int(min(sw, sh) * 0.01))
    left_s = max(0, col_run[0] - padding)
    top_s = max(0, row_run[0] - padding)
    right_s = min(sw, col_run[1] + padding)
    bottom_s = min(sh, row_run[1] + padding)
    retained = ((right_s - left_s) * (bottom_s - top_s)) / (sw * sh)
    # Small trims are more likely page shadows than a scanner canvas.  Preserve
    # the original page in that case rather than perturbing a successful input.
    if retained > 0.82:
        return None
    sx, sy = width / sw, height / sh
    return (
        int(left_s * sx), int(top_s * sy),
        max(int(right_s * sx), int(left_s * sx) + 1),
        max(int(bottom_s * sy), int(top_s * sy) + 1),
    )


def make_dark_margin_retry_pdf(source: Path, destination: Path) -> int:
    """Replace only matte pages; preserve every unaffected source PDF page."""
    import pypdfium2 as pdfium
    from pypdf import PdfReader, PdfWriter

    document = pdfium.PdfDocument(str(source))
    replacements = {}
    changed = 0
    try:
        for page_number, page in enumerate(document):
            image = page.render(scale=2.0).to_pil().convert("RGB")
            crop_box = dark_margin_crop_box(image)
            if crop_box is not None:
                image = image.crop(crop_box)
                replacement = destination.with_name(f"cropped-{page_number}.pdf")
                image.save(replacement, "PDF", resolution=144.0)
                replacements[page_number] = replacement
                changed += 1
            image.close()
        if not changed:
            return 0
        source_reader = PdfReader(source)
        writer = PdfWriter()
        for page_number, source_page in enumerate(source_reader.pages):
            replacement = replacements.get(page_number)
            if replacement is None:
                writer.add_page(source_page)
            else:
                writer.add_page(PdfReader(replacement).pages[0])
        with destination.open("wb") as handle:
            writer.write(handle)
        return changed
    finally:
        document.close()
        for replacement in replacements.values():
            replacement.unlink(missing_ok=True)


def build_converter():
    """A DocumentConverter driving Granite-Docling through Docling's VlmPipeline."""
    from docling.datamodel import vlm_model_specs
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import VlmPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline

    class GuardedVlmPipeline(VlmPipeline):
        def _turn_dt_into_doc(self, conv_res):
            invalid = []
            for page in conv_res.pages:
                response = page.predictions.vlm_response
                text = "" if response is None else response.text
                truncated = truncate_repetitive_suffix(text)
                if truncated is None:
                    invalid.append(f"p{page.page_no}: repeated bbox generation")
                    continue
                normalized = normalize_isolated_reversed_bboxes(truncated)
                if normalized is None:
                    invalid.append(f"p{page.page_no}: collapsed reversed bboxes")
                    continue
                if response is not None and normalized != text:
                    response.text = normalized
                reason = invalid_doctag_reason(normalized)
                if reason:
                    invalid.append(f"p{page.page_no}: {reason}")
            if invalid:
                raise GraniteDoctagError("; ".join(invalid))
            return super()._turn_dt_into_doc(conv_res)

    return DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=GuardedVlmPipeline,
            pipeline_options=VlmPipelineOptions(
                vlm_options=vlm_model_specs.GRANITEDOCLING_MLX,
            ),
        ),
    })


def _caused_by_invalid_doctags(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, GraniteDoctagError):
            return True
        current = current.__cause__ or current.__context__
    return False


def extract_with_recovery(pdf_path: Path, attachment_key: str, meta_base: dict):
    from docling_extract import extract_chunks_from_pdf_with_docling

    try:
        return extract_chunks_from_pdf_with_docling(
            pdf_path, attachment_key, meta_base, converter=build_converter(),
        )
    except Exception as exc:
        if not _caused_by_invalid_doctags(exc):
            raise
        with tempfile.TemporaryDirectory(prefix="granite-margin-retry-") as directory:
            retry_pdf = Path(directory) / "cropped.pdf"
            cropped_pages = make_dark_margin_retry_pdf(pdf_path, retry_pdf)
            if not cropped_pages:
                raise
            chunks, quality_info = extract_chunks_from_pdf_with_docling(
                retry_pdf, attachment_key, meta_base, converter=build_converter(),
            )
            quality_info = dict(quality_info)
            quality_info["granite_dark_margin_retry"] = True
            quality_info["granite_dark_margin_cropped_pages"] = cropped_pages
            return chunks, quality_info


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
        chunks, quality_info = extract_with_recovery(
            Path(pdf_path), attachment_key, dict(meta_base),
        )
    except Exception as exc:  # noqa: BLE001 - reported to the parent, not raised
        # Docling wraps the useful cause (for example an invalid Granite bbox)
        # in the generic ``Pipeline VlmPipeline failed`` exception.  Preserve
        # the bounded exception chain across the JSON process boundary.
        detail = "".join(traceback.format_exception(exc)).strip()
        json.dump({
            "status": "error",
            "message": f"{type(exc).__name__}: {exc}",
            "traceback": detail[-8000:],
        }, sys.stdout)
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
