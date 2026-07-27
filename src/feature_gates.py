# src/feature_gates.py
"""Decide which optional features are available from what is configured.

Optional features here each depend on an external resource: a paid LLM, a paid
OCR API, a (free but rate-limited) Semantic Scholar key. Historically each also
had its own on/off flag, which meant a flag and its resource could disagree --
a key present but the flag off, or the flag on with no key, both producing
confusing half-states and a setup wizard that had to write a long list of
variables to keep them consistent.

Two rules replace that (user decision 2026-07-27):

**Everything defaults to off.** The setup wizard writes the values it needs, so
a default that tries to be clever only makes the resulting configuration harder
to read. An unconfigured install does nothing until asked.

**A feature that is on without its resource is an error, not a downgrade.**
``verify_enabled_features()`` reports every such mismatch so an entry point can
stop with a message naming the feature and the missing key. Silently skipping
the work would leave the operator wondering why nothing happened -- which is
exactly what a missing key used to cause.

The design and the per-feature mapping live in
``dev-notes/current/80_setup_wizard_profiles.md``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

try:
    from .env_utils import load_dotenv_native
except ImportError:  # direct execution with src/ on sys.path
    from env_utils import load_dotenv_native

#: Provider prefixes that mean "a real, reachable model is configured".
#: ``codex_cli`` is deliberately included: it is a local CLI agent rather than
#: a metered API, but it is still an LLM the features can call.
_LLM_PROVIDERS = ("deepseek", "anthropic", "gemini", "openai_compat", "codex_cli")

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _env(name: str) -> str:
    return str(os.environ.get(name) or "").strip()


def _ensure_env_loaded() -> None:
    load_dotenv_native(Path(__file__).resolve().parents[1])


def llm_configured() -> bool:
    """Whether any LLM role points at a usable provider.

    Gates the AI-TOC fast path, the OCR-layer audit, query expansion, LLM
    hierarchical summaries and LLM reference extraction -- the features that
    make a metered call per document.
    """
    _ensure_env_loaded()
    for role in ("LLM_CHEAP", "LLM_STANDARD", "LLM_REVIEW"):
        spec = _env(role).casefold()
        if any(spec.startswith(f"{provider}:") for provider in _LLM_PROVIDERS):
            return True
    return False


def cloud_ocr_configured() -> bool:
    """Whether Mistral OCR can be called at all."""
    _ensure_env_loaded()
    return bool(_env("MISTRAL_OCR_API_KEY"))


def citation_network_configured() -> bool:
    """Whether Semantic Scholar is usable.

    The key is free, but without one the public rate limit is too low for the
    citation pipeline to make progress, so an unkeyed install is treated as not
    having the feature rather than as having a very slow one.
    """
    _ensure_env_loaded()
    return bool(_env("S2_API_KEY"))


def flag_enabled(name: str, *, default: bool = False) -> bool:
    """Resolve a feature flag. Unset means off unless a caller says otherwise."""
    _ensure_env_loaded()
    raw = _env(name).casefold()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    return default


# --- per-feature resolution ------------------------------------------------
# Each of these is the single place its feature asks "am I on?", so the answer
# cannot drift between call sites.


#: feature flag -> the resource it cannot work without. Used both to report a
#: misconfiguration and to describe features in status output.
FEATURE_REQUIREMENTS = {
    "PDF_AI_TOC_FAST_PATH_ENABLE": ("AI TOC fast path", "an LLM_* role", llm_configured),
    "OCR_LAYER_AUDIT_ENABLE": ("OCR layer audit", "an LLM_* role", llm_configured),
    "QUERY_EXPANSION_ENABLE": ("query expansion", "an LLM_* role", llm_configured),
    "LLM_SUMMARIES_ENABLE": ("LLM hierarchical summaries", "an LLM_* role", llm_configured),
    "LLM_REFERENCE_EXTRACTION_ENABLE": (
        "LLM reference extraction", "an LLM_* role", llm_configured,
    ),
    "PDF_MISTRAL_TOC_QUEUE_ENABLE": (
        "Mistral OCR batch queue", "MISTRAL_OCR_API_KEY", cloud_ocr_configured,
    ),
    "CITATION_NETWORK_ENABLE": (
        "Citation Network", "S2_API_KEY", citation_network_configured,
    ),
}


def ai_toc_enabled() -> bool:
    return flag_enabled("PDF_AI_TOC_FAST_PATH_ENABLE")


def ocr_layer_audit_enabled() -> bool:
    return flag_enabled("OCR_LAYER_AUDIT_ENABLE")


def query_expansion_enabled() -> bool:
    return flag_enabled("QUERY_EXPANSION_ENABLE")


def llm_summaries_enabled() -> bool:
    return flag_enabled("LLM_SUMMARIES_ENABLE")


def llm_reference_extraction_enabled() -> bool:
    return flag_enabled("LLM_REFERENCE_EXTRACTION_ENABLE")


def mistral_batch_queue_enabled() -> bool:
    return flag_enabled("PDF_MISTRAL_TOC_QUEUE_ENABLE")


def citation_network_enabled() -> bool:
    return flag_enabled("CITATION_NETWORK_ENABLE")


def pdf_structure_recovery_enabled() -> bool:
    """Choice (A): recover document structure for PDFs, or index them flat.

    Needs no external resource, so it is not in FEATURE_REQUIREMENTS. With this
    off, PDFs become PyMuPDF paragraph chunks (and local OCR output where there
    is no text layer), while EPUB and HTML keep their DOM structuring -- that
    costs nothing and needs no service.
    """
    return flag_enabled("PDF_STRUCTURE_RECOVERY_ENABLE")


#: Choice (C): which engine structures a PDF, chosen per size bucket.
STRUCTURE_ENGINES = ("docling", "granite", "mistral")
DEFAULT_ENGINE_SHORT = "docling"
DEFAULT_ENGINE_LONG = "docling"
DEFAULT_ENGINE_PAGE_BOUNDARY = 30


#: Granite needs its own virtualenv: mlx-vlm wants transformers>=5.14 while the
#: Docling pipeline pins <5.9 on macOS, so the two cannot share one env.
DEFAULT_GRANITE_PYTHON = "tmp/granite_docling_venv/bin/python"


def granite_configured() -> bool:
    """Whether Granite can actually be used as a structuring engine.

    Both halves are required. The isolated environment left over from the
    bake-off satisfies the first on its own, which would let an operator select
    an engine the pipeline has no adapter to call -- the route would match
    neither the Docling nor the Mistral branch and the document would come out
    unstructured with nothing reported.
    """
    _ensure_env_loaded()
    configured = _env("GRANITE_VENV_PYTHON") or DEFAULT_GRANITE_PYTHON
    path = Path(configured)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    if not path.exists():
        return False
    return (Path(__file__).resolve().parent / "granite_worker.py").exists()


def structure_engine_page_boundary() -> int:
    """Page count separating the "short" and "long" engine choices.

    Deliberately its own setting rather than a reuse of
    ``PDF_AI_TOC_MIN_PAGES``: that one asks "is this long enough to have a
    table of contents", while this asks "which engine should carry the cost".
    The defaults coincide at 30; the questions do not.
    """
    _ensure_env_loaded()
    try:
        return max(1, int(_env("PDF_STRUCTURE_ENGINE_PAGE_BOUNDARY") or DEFAULT_ENGINE_PAGE_BOUNDARY))
    except ValueError:
        return DEFAULT_ENGINE_PAGE_BOUNDARY


def _engine(name: str, default: str) -> str:
    _ensure_env_loaded()
    value = _env(name).casefold()
    return value if value in STRUCTURE_ENGINES else default


def structure_engine_for(total_pages: int) -> str:
    """The engine choice (C) assigns to a document of this length.

    Each option trades differently, which is why the choice is exposed rather
    than inferred: Docling is free and fast, Granite is free and more accurate
    but ~2.3x slower and needs its own venv, Mistral is fast and accurate but
    charges per page.
    """
    if total_pages < structure_engine_page_boundary():
        return _engine("PDF_STRUCTURE_ENGINE_SHORT", DEFAULT_ENGINE_SHORT)
    return _engine("PDF_STRUCTURE_ENGINE_LONG", DEFAULT_ENGINE_LONG)


def verify_enabled_features() -> list[str]:
    """Return one message per feature switched on without its resource.

    An empty list means the configuration is coherent. Entry points should stop
    on a non-empty result rather than proceeding with the feature quietly
    inert: "I enabled it and nothing happened" is the failure this prevents.
    """
    problems: list[str] = []
    # Choosing an engine that cannot run is the same class of mistake as
    # enabling a feature without its key, so it is reported the same way.
    for setting, default in (
        ("PDF_STRUCTURE_ENGINE_SHORT", DEFAULT_ENGINE_SHORT),
        ("PDF_STRUCTURE_ENGINE_LONG", DEFAULT_ENGINE_LONG),
    ):
        raw = _env(setting).casefold()
        if raw and raw not in STRUCTURE_ENGINES:
            problems.append(
                f"{setting}={raw!r} is not one of {', '.join(STRUCTURE_ENGINES)}."
            )
        elif raw == "mistral" and not cloud_ocr_configured():
            problems.append(
                f"{setting}=mistral but MISTRAL_OCR_API_KEY is not configured."
            )
        elif raw == "granite" and not granite_configured():
            problems.append(
                f"{setting}=granite but the isolated Granite environment was not "
                f"found, or the adapter is not installed (GRANITE_VENV_PYTHON, "
                f"default {DEFAULT_GRANITE_PYTHON}; adapter src/granite_worker.py). "
                "Granite needs its own virtualenv because mlx-vlm and Docling "
                "require incompatible transformers versions."
            )
    for flag, (label, requirement, available) in FEATURE_REQUIREMENTS.items():
        if flag_enabled(flag) and not available():
            problems.append(
                f"{label} is enabled ({flag}=1) but {requirement} is not configured."
            )
    return problems


def enabled_features() -> dict[str, bool]:
    """Snapshot for status output and diagnostics."""
    return {
        "llm_configured": llm_configured(),
        "cloud_ocr_configured": cloud_ocr_configured(),
        "s2_key_configured": citation_network_configured(),
        "pdf_structure_recovery": pdf_structure_recovery_enabled(),
        "ai_toc": ai_toc_enabled(),
        "ocr_layer_audit": ocr_layer_audit_enabled(),
        "query_expansion": query_expansion_enabled(),
        "llm_summaries": llm_summaries_enabled(),
        "llm_reference_extraction": llm_reference_extraction_enabled(),
        "mistral_batch_queue": mistral_batch_queue_enabled(),
        "citation_network": citation_network_enabled(),
    }


__all__ = [
    "DEFAULT_ENGINE_LONG", "DEFAULT_ENGINE_SHORT", "FEATURE_REQUIREMENTS",
    "STRUCTURE_ENGINES", "granite_configured", "structure_engine_for",
    "structure_engine_page_boundary", "ai_toc_enabled", "citation_network_configured",
    "citation_network_enabled", "cloud_ocr_configured", "enabled_features",
    "flag_enabled", "llm_configured", "llm_reference_extraction_enabled",
    "llm_summaries_enabled", "mistral_batch_queue_enabled",
    "ocr_layer_audit_enabled", "pdf_structure_recovery_enabled",
    "query_expansion_enabled", "verify_enabled_features",
]
