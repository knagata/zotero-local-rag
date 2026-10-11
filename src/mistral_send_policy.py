"""Explicit per-item Mistral send exclusion, independent of OCR quality."""
from __future__ import annotations

import re
from typing import Any

import httpx

try:
    from .zotero_source_localapi import local_api_url, zotero_api_headers
except ImportError:  # direct src entrypoint
    from zotero_source_localapi import local_api_url, zotero_api_headers

NO_MISTRAL_OCR_TAG = 'no-mistral-ocr'


def has_send_exclusion(*tag_groups: Any) -> bool:
    for group in tag_groups:
        for value in group or ():
            tag = value.get('tag', '') if isinstance(value, dict) else value
            if isinstance(tag, str) and tag.strip().casefold() == NO_MISTRAL_OCR_TAG:
                return True
    return False


def assert_mistral_send_allowed(attachment_key: str) -> None:
    """Read fresh tags immediately before paid send; errors fail closed.

    Cached responses and staged-result adoption do not call this gate because
    they send nothing. A missing/unreachable Zotero must never imply permission.
    """
    if not re.fullmatch(r'[A-Z0-9]{8}', attachment_key):
        raise ValueError('Mistral送信前の確認には有効なZotero添付IDが必要です。')
    with httpx.Client(timeout=10, headers=zotero_api_headers()) as client:
        response = client.get(local_api_url(f'items/{attachment_key}'))
        response.raise_for_status()
        raw = response.json()
        data = raw.get('data', raw)
        if data.get('itemType') != 'attachment':
            raise ValueError('Mistral送信対象がZotero添付ではありません。')
        groups = [data.get('tags', [])]
        parent = data.get('parentItem')
        if parent:
            if not re.fullmatch(r'[A-Z0-9]{8}', str(parent)):
                raise ValueError('Zotero親アイテムIDが不正です。送信を中止しました。')
            response = client.get(local_api_url(f'items/{parent}'))
            response.raise_for_status()
            raw = response.json()
            groups.append(raw.get('data', raw).get('tags', []))
    if has_send_exclusion(*groups):
        raise RuntimeError(
            f'{attachment_key}: no-mistral-ocrタグによりMistral OCRへの送信を禁止しています。'
            '品質不足の記録は保持します。再送が必要な場合だけZoteroでタグを外してください。')
