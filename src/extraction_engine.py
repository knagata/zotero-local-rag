"""Engine-neutral source extraction contracts for structured ingestion V3.

Adapters return source-order blocks and their provenance; the ingestion pipeline
then applies one shared zone, chunking, structure, and indexing implementation.
No adapter output is itself a database canonical form.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
from typing import Any, Dict, Protocol, Sequence

try:
    from .env_utils import load_dotenv_native
    from .pdf_provenance import SCAN_DERIVED_CLASSES
except ImportError:  # direct execution with src/ on sys.path
    from env_utils import load_dotenv_native
    from pdf_provenance import SCAN_DERIVED_CLASSES

# Imported by value, not module, to keep this module free of the LLM client
# import chain that ocr_layer_audit pulls in.
OCR_LAYER_DEGRADED = "degraded"


@dataclass(frozen=True)
class SourceRef:
    path: Path
    attachment_key: str
    metadata: Dict[str, Any]
    source_type: str = "pdf"


@dataclass(frozen=True)
class EngineCapabilities:
    requires: str = "none"  # none | mps | cuda
    min_memory_gb: int = 0
    approx_pages_per_min: float | None = None
    outline: bool = False
    headings: bool = False
    zones: bool = False
    tables: bool = False
    bbox: bool = False
    reading_order: bool = False
    ocr: bool = False
    cloud: bool = False


@dataclass(frozen=True)
class EngineAvailability:
    available: bool
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    engine: str
    version: str
    blocks: list[Dict[str, Any]]
    quality: Dict[str, Any]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class ExtractionEngine(Protocol):
    name: str
    version: str

    def capabilities(self) -> EngineCapabilities: ...
    def parse(self, source: SourceRef) -> ExtractionResult: ...


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def mps_available() -> bool:
    """Return whether this process can use Apple Metal without importing torch eagerly."""
    try:
        import torch  # type: ignore
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def local_memory_gb() -> int:
    """Best-effort physical-memory estimate used only for safe engine selection."""
    try:
        return max(0, int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3))
    except (AttributeError, OSError, ValueError):
        return 0


class ChunkAdapter:
    """Bridge existing extractors while preserving their output provenance."""
    name = "base"
    version = "v3-adapter-1"
    _capabilities = EngineCapabilities()

    def capabilities(self) -> EngineCapabilities:
        return self._capabilities

    def extract(self, source: SourceRef) -> tuple[Sequence[tuple[str, str, Dict[str, Any]]], Dict[str, Any]]:
        raise NotImplementedError

    def parse(self, source: SourceRef) -> ExtractionResult:
        chunks, quality = self.extract(source)
        blocks = []
        for ordinal, (chunk_id, text, metadata) in enumerate(chunks):
            md = dict(metadata)
            md.update({"extraction_engine": self.name, "extraction_version": self.version})
            blocks.append({"id": chunk_id, "text": text, "ordinal": ordinal, "metadata": md})
        return ExtractionResult(self.name, self.version, blocks, dict(quality))


class PyMuPDFEngine(ChunkAdapter):
    name = "pymupdf"
    _capabilities = EngineCapabilities(
        approx_pages_per_min=120.0, outline=True, bbox=True, reading_order=True,
    )

    def extract(self, source: SourceRef):  # type: ignore[no-untyped-def]
        try:
            from .pdf_extract import extract_chunks_from_pdf
        except ImportError:  # direct execution with src/ on sys.path
            from pdf_extract import extract_chunks_from_pdf
        return extract_chunks_from_pdf(source.path, source.attachment_key, source.metadata)


class DoclingEngine(ChunkAdapter):
    name = "docling"
    _capabilities = EngineCapabilities(
        min_memory_gb=8, headings=True, zones=True, tables=True, bbox=True,
        reading_order=True, ocr=True,
    )

    def extract(self, source: SourceRef):  # type: ignore[no-untyped-def]
        try:
            from .docling_extract import extract_chunks_from_pdf_with_docling
        except ImportError:  # direct execution with src/ on sys.path
            from docling_extract import extract_chunks_from_pdf_with_docling
        return extract_chunks_from_pdf_with_docling(source.path, source.attachment_key, source.metadata)


class NDLocrLiteEngine(ChunkAdapter):
    name = "ndlocr_lite"
    _capabilities = EngineCapabilities(
        min_memory_gb=8, headings=True, zones=True, tables=True, bbox=True,
        reading_order=True, ocr=True,
    )

    def extract(self, source: SourceRef):  # type: ignore[no-untyped-def]
        try:
            from .ndlocr_extract import extract_chunks_from_pdf_with_ndlocr
        except ImportError:  # direct execution with src/ on sys.path
            from ndlocr_extract import extract_chunks_from_pdf_with_ndlocr
        return extract_chunks_from_pdf_with_ndlocr(source.path, source.attachment_key, source.metadata)


class YomitokuEngine(ChunkAdapter):
    """CC BY-NC-SA 4.0 licensed; non-commercial personal research use only."""

    name = "yomitoku"
    _capabilities = EngineCapabilities(
        min_memory_gb=4, headings=True, zones=True, tables=True, bbox=True,
        reading_order=True, ocr=True,
    )

    def extract(self, source: SourceRef):  # type: ignore[no-untyped-def]
        try:
            from .yomitoku_extract import extract_chunks_from_pdf_with_yomitoku
        except ImportError:  # direct execution with src/ on sys.path
            from yomitoku_extract import extract_chunks_from_pdf_with_yomitoku
        return extract_chunks_from_pdf_with_yomitoku(source.path, source.attachment_key, source.metadata)


class MistralOCREngine(ChunkAdapter):
    """Cloud fallback for when local structuring fails. Opt-in only; see
    ``inspect_engine_availability`` for the API-key + explicit-flag gate."""

    name = "mistral_ocr"
    _capabilities = EngineCapabilities(
        headings=True, zones=True, tables=True, bbox=True, reading_order=True, ocr=True, cloud=True,
    )

    def extract(self, source: SourceRef):  # type: ignore[no-untyped-def]
        try:
            from .mistral_ocr_extract import extract_chunks_from_pdf_with_mistral_ocr
        except ImportError:  # direct execution with src/ on sys.path
            from mistral_ocr_extract import extract_chunks_from_pdf_with_mistral_ocr
        return extract_chunks_from_pdf_with_mistral_ocr(source.path, source.attachment_key, source.metadata)


def inspect_engine_availability(engine: ExtractionEngine) -> EngineAvailability:
    """Check local prerequisites without importing models or processing files."""
    details: Dict[str, Any] = {
        "requires": engine.capabilities().requires,
        "memory_gb": local_memory_gb(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "min_memory_gb": engine.capabilities().min_memory_gb,
        "approx_pages_per_min": engine.capabilities().approx_pages_per_min,
    }
    enough_memory = (
        not engine.capabilities().min_memory_gb
        or not details["memory_gb"]
        or details["memory_gb"] >= engine.capabilities().min_memory_gb
    )
    if engine.name == "pymupdf":
        ok = _module_available("fitz")
        return EngineAvailability(ok, "ready" if ok else "PyMuPDF (fitz) is not installed", details)
    if engine.name == "docling":
        installed = _module_available("docling")
        ok = installed and enough_memory
        reason = "ready" if ok else ("Docling is not installed" if not installed else "insufficient memory")
        return EngineAvailability(ok, reason, details)
    if engine.name == "ndlocr_lite":
        configured = os.environ.get("NDLOCR_BIN", "").strip()
        binary = configured if configured and Path(configured).expanduser().is_file() else shutil.which("ndlocr-lite")
        details["binary"] = str(binary or "")
        ok = bool(binary) and enough_memory
        reason = "ready" if ok else ("NDLOCR-Lite executable was not found" if not binary else "insufficient memory")
        return EngineAvailability(ok, reason, details)
    if engine.name == "yomitoku":
        installed = _module_available("yomitoku")
        ok = installed and enough_memory
        reason = "ready" if ok else ("yomitoku is not installed" if not installed else "insufficient memory")
        return EngineAvailability(ok, reason, details)
    if engine.name == "mistral_ocr":
        # A configured key alone does not enable this in the bake-off: cloud
        # sends of library content require a second, explicit opt-in so a
        # routine local run can never silently upload documents.
        load_dotenv_native(Path(__file__).resolve().parents[1])
        configured = bool(os.environ.get("MISTRAL_OCR_API_KEY", "").strip())
        allowed = os.environ.get("OCR_BAKEOFF_ALLOW_CLOUD", "0").strip() == "1"
        details.update({"api_key_configured": configured, "cloud_bakeoff_allowed": allowed})
        ok = configured and allowed
        if not configured:
            reason = "MISTRAL_OCR_API_KEY is not configured"
        elif not allowed:
            reason = "cloud bake-off requires OCR_BAKEOFF_ALLOW_CLOUD=1 (explicit opt-in)"
        else:
            reason = "ready"
        return EngineAvailability(ok, reason, details)
    caps = engine.capabilities()
    if caps.cloud:
        return EngineAvailability(False, "cloud engines are excluded from the local bake-off", details)
    return EngineAvailability(enough_memory, "ready" if enough_memory else "insufficient memory", details)


class EngineRegistry:
    """Resolve explicit configuration before capability and local fallback rules."""
    def __init__(self, engines: Sequence[ExtractionEngine] | None = None):
        supplied = engines or (
            PyMuPDFEngine(), DoclingEngine(), NDLocrLiteEngine(), YomitokuEngine(), MistralOCREngine(),
        )
        self._engines = {engine.name: engine for engine in supplied}

    def get(self, name: str) -> ExtractionEngine:
        try:
            return self._engines[name]
        except KeyError as exc:
            raise ValueError(f"Unknown extraction engine: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(self._engines)

    def select(self, *, requested: str | None = None, no_cloud: bool = False) -> ExtractionEngine:
        if requested:
            engine = self.get(requested)
            if no_cloud and engine.capabilities().cloud:
                raise ValueError("no-cloud attachment cannot use a cloud extraction engine")
            return engine
        # Docling is the current local layout baseline; a missing optional
        # dependency is handled by the caller through this deterministic fallback.
        return self.get("docling") if "docling" in self._engines else self.get("pymupdf")


def pymupdf_fast_path_rejection_reason(quality_info: Dict[str, Any]) -> str | None:
    """Return a stable rejection reason, or ``None`` when the fast path passes.

    Native PDF outlines are more trustworthy than an inferred outline and can
    tolerate a very small number of anomalous pages. Covers, maps, and indexes
    routinely trigger the page-level scan/content heuristics even when the
    embedded body text is healthy. AI-recovered outlines remain fail-closed.

    A missing outline is *not* on its own a rejection reason: a document that
    genuinely has no native TOC (``has_outline=False``) and gets no AI-TOC
    match has no structure to recover in the first place, so escalating it to
    Mistral OCR would gain nothing -- Mistral OCR cannot invent a table of
    contents any better than the source PDF has. It only makes sense to
    escalate on structure grounds when a native outline *does* exist but its
    presence doesn't line up with clean text (the tolerance checks below);
    plain unstructured-but-readable text should just be indexed as-is (user
    decision 2026-07-26, following the M5TQ4HLZ case where a 127-page
    catalog with no native TOC was needlessly deferred to Mistral OCR purely
    for lacking an outline, despite PyMuPDF text being fine).

    The scanned/corrupted tolerance (``PYMUPDF_NATIVE_OUTLINE_ANOMALY_RATIO_MAX``,
    default 0.02) applies the same regardless of ``has_native_outline``: this
    used to be a 2%-vs-0% asymmetry (any residual page failed the fast path
    outright when there was no native outline to "earn" tolerance), but now
    that a missing outline is no longer treated as inherently suspicious
    (previous paragraph), there's no remaining reason to also demand a
    stricter page-anomaly bar for it -- the same repair-then-tolerate
    reasoning applies whether or not the document happens to have a TOC
    (user decision 2026-07-26). ``has_native_outline`` still selects which
    reason string is returned, for diagnostics.

    IMPORTANT: ``quality_info["scanned_ratio"]`` and ``["corrupted_ratio"]``
    are consumed here *after* ``index_from_zotero.py`` has already tried to
    repair what it can -- when ``PDF_SCANNED_PAGE_PATCH_ENABLE=1`` /
    ``PDF_CORRUPTED_PAGE_PATCH_ENABLE=1``, every page the Docling sub-PDF
    patch attempted (E2c/E2d, note 77) is removed from these ratios by
    ``pdf_extract.recompute_scanned_quality_after_patch`` /
    ``recompute_corrupted_quality_after_patch`` before this function ever
    sees them, regardless of whether the patch recovered clean text or only
    confirmed there was nothing recoverable. So in the common case (patch
    enabled, no exception) these ratios are **not** the raw per-document scan/
    corruption footprint PyMuPDF measured -- they are what's left *unresolved*
    after repair, which for a successful patch run is 0.0 for the patched
    pages regardless of the native-outline tolerance below. The tolerance
    checks here are therefore a fallback that only bites when the patch is
    disabled, raised an exception, or (rarely) left a partial residual it
    couldn't attempt -- repair is tried first; tolerating/rejecting a ratio is
    the last resort for whatever repair didn't resolve (user decision
    2026-07-26). Callers that need the pre-repair raw counts should read the
    ``scanned_pages``/``corrupted_pages`` list lengths captured by
    ``extract_chunks_from_pdf`` before any patch step, not this function.
    """
    has_native_outline = bool(quality_info.get("has_outline"))

    # Deterministic extraction defects (pdf_provenance.detect_text_defects).
    # These are artefacts of *extraction*, not of OCR: letter-spaced text
    # ("p r o b l e m s") and dropped ft/fi/fl ligatures ("sofware") come from
    # the text layer being read wrongly, so re-extracting with a layout engine
    # fixes them while re-OCRing a born-digital page would not. They are
    # checked before the scan/corruption ratios because they can occur in a
    # document those ratios call perfectly healthy -- the two worst-affected
    # documents in the library measured 0 scanned and 0 corrupted pages.
    for defect in quality_info.get("text_defects") or []:
        return f"text_defect_{defect}"

    source_class = str(quality_info.get("source_class") or "")
    if source_class:
        # Scan-derived text is a derivative of the page image, so it is only
        # canonical once measured (note 79 stage 2, ocr_layer_audit). Being a
        # scan is not itself disqualifying: the measured rates run from 0.00%
        # to well over the threshold, and an ABBYY-OCRed 567-page book came in
        # at 0.25% -- re-OCRing that would spend cloud budget to replace good
        # text. Only a measured-degraded layer is rejected; an unmeasured one
        # passes, because failing to measure is not evidence of a problem.
        #
        # This replaces the old is_scanned check, which counted pages *lacking*
        # text and so missed every OCRed scan in the library -- 67 documents,
        # all is_scanned=False because OCR had put text on every page.
        if (
            source_class in SCAN_DERIVED_CLASSES
            and str(quality_info.get("ocr_layer_quality") or "") == OCR_LAYER_DEGRADED
        ):
            return "ocr_layer_degraded"
    elif quality_info.get("is_scanned"):
        # No classification available (classify_pdf_source raised): fall back
        # to the legacy document-level flag rather than silently passing.
        return "document_scanned"
    if quality_info.get("is_corrupted"):
        return "document_corrupted"

    extraction_failure_ratio = float(
        quality_info.get("extraction_failure_ratio") or 0.0
    )
    if extraction_failure_ratio > 0.0:
        return "extraction_failure_ratio_nonzero"

    # Residual (post-patch) ratios, not raw detection counts -- see the
    # docstring above. When the E2c/E2d patches are enabled and succeed,
    # these are 0.0 for every page the patch attempted; a nonzero value here
    # means either the patch is disabled/failed, or a page was left
    # unattempted (e.g. patch gated off by is_scanned/is_corrupted already
    # being true at the document level).
    residual_scanned_ratio = float(quality_info.get("scanned_ratio") or 0.0)
    residual_corrupted_ratio = float(quality_info.get("corrupted_ratio") or 0.0)
    # Same tolerance regardless of has_native_outline (user decision
    # 2026-07-26) -- see the docstring above. has_native_outline only picks
    # which reason string comes back, not how strict the check is.
    tolerance = float(
        os.environ.get("PYMUPDF_NATIVE_OUTLINE_ANOMALY_RATIO_MAX", "0.02")
    )
    if has_native_outline:
        if residual_scanned_ratio > tolerance:
            return "scanned_ratio_above_native_outline_tolerance"
        if residual_corrupted_ratio > tolerance:
            return "corrupted_ratio_above_native_outline_tolerance"
        return None

    if residual_scanned_ratio > tolerance:
        return "scanned_ratio_nonzero"
    if residual_corrupted_ratio > tolerance:
        return "corrupted_ratio_nonzero"
    return None


def pymupdf_fast_path_passes(quality_info: Dict[str, Any]) -> bool:
    """Deterministic gate for using PyMuPDF's own PDF result as canonical
    instead of escalating to Docling, per the approved routing
    (``evaluations/ocr_bakeoff_v3/results/routing_proposal.md``: "PDF embedded
    text・単純layout" candidacy requires coverage, no scanned/corrupted
    pages, and a usable embedded outline — all four gates below must pass).

    Only PyMuPDF's own ``quality_info`` (as returned by
    ``pdf_extract.extract_chunks_from_pdf``, and possibly amended in-place by
    the E2c/E2d Docling sub-PDF patches -- see
    ``pymupdf_fast_path_rejection_reason``'s docstring for what "amended"
    means for the scanned/corrupted ratios) is consulted; this never invokes
    another engine. The caller is expected to fall through to Docling when
    this returns False (multi-column/table layout, missing outline, or any
    page needing OCR all fail the gate on purpose — Docling scored far
    higher than PyMuPDF on every bake-off category except the outline+
    embedded-text ones this gate targets).
    """
    return pymupdf_fast_path_rejection_reason(quality_info) is None


_ENGINE_NAME_NORMALIZATION = {
    "ndlocr-lite": "ndlocr_lite",
}

_EXTRACTION_QUALITY_ALLOW_LIST = (
    "parser", "is_scanned", "is_corrupted", "scanned_ratio", "corrupted_ratio",
    "extraction_failure_ratio", "poppler_fallback_ratio", "ocr_fallback_ratio",
    "has_outline",
    "avg_corruption_score", "max_corruption_score", "total_pages", "fallback_engine",
    "model", "lite", "device", "dpi",
    "ai_toc_recovery_status", "ai_toc_body_coverage", "ai_toc_matched_count",
    # Partial-coverage adoption: the chunk is real text from a document that
    # was only partly recovered.  Carrying it here is what makes "reprocess
    # everything that came out incomplete" a query rather than a re-scan.
    "quality_uncertain", "quality_uncertain_reason",
    "source_coverage_adopted", "source_coverage_covered_ratio",
)


def resolve_extraction_engine(quality_info: Dict[str, Any], existing: str | None) -> str:
    """Resolve the provenance-tracking engine name for a V3 chunk.

    Extractors that already stamp ``extraction_engine`` on their own (EPUB/HTML
    DOM paths, adapter-based engines) take precedence and are returned as-is.
    Otherwise this is derived from the document's ``quality_info``: a Mistral
    OCR cloud fallback (``fallback_engine``) wins over the primary parser, the
    primary ``parser`` name is used next (normalizing ``ndlocr-lite`` to the
    canonical ``ndlocr_lite`` engine name), and plain PyMuPDF extraction with
    no quality-info parser at all falls back to ``"pymupdf"``.
    """
    if existing:
        return existing
    quality_info = quality_info or {}
    fallback_engine = quality_info.get("fallback_engine")
    if fallback_engine:
        return str(fallback_engine)
    parser = quality_info.get("parser")
    if parser:
        parser = str(parser)
        return _ENGINE_NAME_NORMALIZATION.get(parser, parser)
    return "pymupdf"


def summarize_extraction_quality(quality_info: Dict[str, Any]) -> str:
    """Compact JSON summary of the reprocessing-relevant signals in ``quality_info``.

    Only allow-listed keys that are actually present in ``quality_info`` are
    included, so the resulting string stays small and every field in it is a
    genuine signal for selecting reprocessing candidates later. Chroma metadata
    values must be scalars, so this returns a ``json.dumps`` string rather than
    a dict.
    """
    quality_info = quality_info or {}
    summary = {
        key: quality_info[key]
        for key in _EXTRACTION_QUALITY_ALLOW_LIST
        if key in quality_info
    }
    return json.dumps(summary, ensure_ascii=False, separators=(",", ":"))


__all__ = [
    "ChunkAdapter", "DoclingEngine", "EngineAvailability",
    "EngineCapabilities", "EngineRegistry", "ExtractionEngine", "ExtractionResult",
    "MistralOCREngine", "NDLocrLiteEngine", "PyMuPDFEngine", "SourceRef", "YomitokuEngine",
    "inspect_engine_availability", "local_memory_gb", "mps_available",
    "pymupdf_fast_path_passes", "pymupdf_fast_path_rejection_reason",
    "resolve_extraction_engine", "summarize_extraction_quality",
]
