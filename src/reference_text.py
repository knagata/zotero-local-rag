"""Conservative classification helpers for extracted bibliography text."""
from __future__ import annotations

import re


SHORT_FORM_REFERENCE_RE = re.compile(
    r"\b(?:ibid|op\.?\s*cit|loc\.?\s*cit)\b|同書|同前|前掲",
    re.IGNORECASE,
)


def is_short_form_reference(value: str | None) -> bool:
    """Return True when a reference depends on a preceding bibliographic entry."""
    return bool(SHORT_FORM_REFERENCE_RE.search(value or ""))
