"""Bounded, in-memory rendering of a Zotero PDF page for visual source checks."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import fitz


def render_source_page(path: Path, pdf_page: int, *, crop: list[float] | None = None,
                       rotation: int = 0) -> tuple[dict[str, Any], bytes]:
    if isinstance(pdf_page, bool) or not isinstance(pdf_page, int) or pdf_page < 1:
        raise ValueError("pdf_page must be a 1-based PDF page number (not the printed page number)")
    if rotation not in (0, 90, 180, 270):
        raise ValueError("rotation must be 0, 90, 180 or 270 degrees clockwise")
    region = crop if crop is not None else [0., 0., 1., 1.]
    if (len(region) != 4 or any(not isinstance(v, (int, float)) or isinstance(v, bool)
                              or not math.isfinite(v) for v in region)
            or not (0 <= region[0] < region[2] <= 1 and 0 <= region[1] < region[3] <= 1)):
        raise ValueError("crop must be normalized [left, top, right, bottom] within 0..1")
    with fitz.open(path) as document:
        if not document.is_pdf:
            raise ValueError("source is not a PDF; this tool currently supports PDF attachments only")
        if document.needs_pass:
            raise ValueError("PDF is encrypted; unlock it in Zotero before checking the page")
        if pdf_page > len(document):
            raise ValueError(f"pdf_page exceeds the PDF's {len(document)} pages")
        page = document[pdf_page - 1]
        bounds = page.rect
        clip = fitz.Rect(bounds.x0 + region[0]*bounds.width, bounds.y0 + region[1]*bounds.height,
                         bounds.x0 + region[2]*bounds.width, bounds.y0 + region[3]*bounds.height)
        if clip.width < 1 or clip.height < 1:
            raise ValueError("crop is too small; select a larger region")
        scale = 1800 / max(clip.width, clip.height)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip,
                                colorspace=fitz.csRGB, alpha=False)
        if rotation:
            from PIL import Image
            import io
            image = Image.frombytes('RGB', (pixmap.width, pixmap.height), pixmap.samples)
            output = io.BytesIO()
            image.rotate(-rotation, expand=True).save(output, format='JPEG', quality=90)
            encoded = output.getvalue()
            width, height = (pixmap.height, pixmap.width) if rotation in (90, 270) else (pixmap.width, pixmap.height)
        else:
            encoded = pixmap.tobytes('jpeg', jpg_quality=90)
            width, height = pixmap.width, pixmap.height
        if len(encoded) > 6 * 1024**2:
            raise ValueError("rendered image is too large; request a smaller crop")
        return {"pdf_page": pdf_page, "page_label": page.get_label(), "total_pages": len(document),
                "crop": region, "rotation": rotation, "width": width, "height": height,
                "image_sha256": hashlib.sha256(encoded).hexdigest(), "mime_type": "image/jpeg",
                "writes_performed": False}, encoded
