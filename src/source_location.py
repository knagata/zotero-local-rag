"""Resolve renamed stored attachments without guessing across documents."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .manifest import content_signature


def resolve_source_path(attachment_key: str, entry: dict[str, Any], *,
                        storage_root: Path | None = None) -> Path:
    saved = Path(str(entry.get("pdf_path") or ""))
    if saved.is_file():
        return saved
    if not re.fullmatch(r"[A-Z0-9]{8}", attachment_key):
        raise FileNotFoundError("添付IDが不正です。保存場所を確認してください。")
    root = storage_root or Path(os.environ.get("ZOTERO_DATA_DIR", str(Path.home() / "Zotero"))) / "storage"
    folder = root / attachment_key
    signature = entry.get("content_signature")
    if not signature:
        raise FileNotFoundError("原本が移動し、内容の照合情報がありません。Zoteroから再取込してください。")
    candidates = []
    if folder.is_dir() and not folder.is_symlink():
        for candidate in folder.iterdir():
            if candidate.is_symlink() or not candidate.is_file() or candidate.suffix.lower() != saved.suffix.lower():
                continue
            size = candidate.stat().st_size
            if size == entry.get("size") and content_signature(candidate, size) == signature:
                candidates.append(candidate)
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"原本の移動先を一意に照合できません（一致候補{len(candidates)}件）。"
            "保存場所・同期・内容変更を確認し、Zoteroから再取込してください。"
        )
    return candidates[0]
