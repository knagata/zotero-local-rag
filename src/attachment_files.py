"""Resolve a local attachment file consistently for ingestion and freshness."""
from pathlib import Path


def resolve_attachment_file(path: str | Path) -> Path:
    source = Path(path).expanduser()
    if source.is_dir():
        for name in ('index.html', 'index.htm'):
            candidate = source / name
            if candidate.is_file():
                return candidate
        candidates = sorted(candidate for candidate in source.iterdir()
                            if candidate.is_file() and candidate.suffix.casefold() in {'.html', '.htm'})
        if candidates:
            return candidates[0]
    return source
