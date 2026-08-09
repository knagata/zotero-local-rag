"""Explicit state boundaries for the Zotero indexing workflow."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DiscoveryResult:
    attachments: list[Any]
    preflight_notes: list[dict[str, Any]] | None
    excluded_attachments: list[Any] = field(default_factory=list)
    inventory_attachments: list[Any] = field(default_factory=list)
    preferred_pdf_attachments: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class NoteIndexOutcome:
    manifest: dict[str, dict[str, Any]]
    updated: int
    skipped: int
    deleted_stale: int


@dataclass(frozen=True)
class QualityWarning:
    attachment_key: str
    title: str
    reasons: tuple[str, ...]
    quality: dict[str, Any]


@dataclass(frozen=True)
class ResolvedAttachmentSource:
    path: Path
    source_type: str
    mtime: float
    size: int


@dataclass(frozen=True)
class ReparseDecision:
    force_docling: bool = False
    force_ndlocr: bool = False
    force_mistral: bool = False


@dataclass(frozen=True)
class PdfExtraction:
    """What came of trying to read one PDF, and whether anything came at all.

    ``deferred`` means the file was handed to the Mistral OCR batch rather than
    extracted: it is queued, the manifest already records that, and nothing the
    caller does with chunks applies to it. It is the one way out of this route
    that is not a result, and it was a bare ``continue`` reaching out of 665
    lines into the loop around them.
    """

    chunks: list
    quality: dict
    deferred: bool = False


@dataclass(frozen=True)
class SourceVerdict:
    """Whether an attachment still holds what was indexed from it, and so what
    the run owes it: extraction, a quality reading only, or nothing.

    ``signature`` is the content hash if one was worth computing -- the file
    matched a prior row on modification time and size, and that row carried a
    signature, so only a same-size replacement could still be hiding. It is
    carried out of the decision because the manifest row written later reuses
    it rather than hashing the same unchanged bytes twice.
    """

    #: "index" -- extract and re-index; "quality_only" -- the bytes are
    #: unchanged but their quality has not been read, or was asked for again;
    #: "skip" -- unchanged and already understood.
    action: str
    signature: str | None = None
