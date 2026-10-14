"""Resolve human page references and validate manual chunk classifications."""
from __future__ import annotations

from src.document_structure import ZONE_POLICIES


def resolve_toc(rows: list[dict], toc: list[dict], *, allow_unresolved: bool = False,
                total_pages: int | None = None) -> list[dict]:
    indexed = {row['id']: row for row in rows}
    pages: dict[int, list[str]] = {}
    for row in rows:
        page = row['metadata'].get('page')
        if str(page).isdigit() and int(page) > 0:
            pages.setdefault(int(page), []).append(row['id'])
    result = []
    for original in toc:
        entry = dict(original)
        anchor, page = entry.get('chunk_id'), entry.get('page')
        if page is not None and (isinstance(page, bool) or not isinstance(page, int)
                                 or page < 1 or total_pages is not None and page > total_pages):
            raise ValueError('開始PDFページが範囲外です。紙面番号ではなく物理ページを指定してください。')
        if anchor:
            if anchor not in indexed:
                raise ValueError('目次の開始位置が現在のチャンクと一致しません。')
            actual = indexed[anchor]['metadata'].get('page')
            if page is not None and str(actual) != str(page):
                raise ValueError('開始ページとページ内の開始位置が一致しません。')
        elif page in pages:
            entry['chunk_id'] = pages[page][0]
        elif not allow_unresolved or page is None:
            raise ValueError('目次の開始ページにチャンクがありません。下書きは保存できますが、反映前に開始位置を修正してください。')
        result.append(entry)
    return result


def validate_zones(rows: list[dict], zones: dict[str, str]) -> None:
    indexed = {row['id'] for row in rows}
    if len(zones) > 500 or any(key not in indexed or zone not in ZONE_POLICIES for key, zone in zones.items()):
        raise ValueError('分類の修正は現在のチャンクと既知の分類を指定し、500件以内にしてください。')
