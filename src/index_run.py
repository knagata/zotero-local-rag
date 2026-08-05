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
