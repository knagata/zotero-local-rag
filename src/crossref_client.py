"""Crossref API クライアント — アブストラクトと書誌情報の取得。

S2 を引用関係（被引用・参照）の取得に専念させ、以下を Crossref で補う:
  1. 外部論文ノードのアブストラクト表示（概要パネル）
  2. Citation Network更新で S2 に見つからなかった資料の書誌情報フォールバック
     （年・被引用数）

Crossref（https://api.crossref.org）は認証不要・登録不要。User-Agent に
mailto を付けると "polite pool" に振り分けられ安定したレスポンスが得られる。
CROSSREF_MAILTO 環境変数で指定する（未設定でも動作する）。
"""
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
import urllib.parse
from html import unescape
from typing import Any, Dict, Optional

_CROSSREF_BASE = "https://api.crossref.org/works/"
_MIN_INTERVAL = 1.0   # 連続呼び出し間の最小間隔（秒）。礼儀的なスペーシング。
_last_call = 0.0


class CrossrefError(Exception):
    """一時的な取得失敗（ネットワーク/5xx 等）。404 など「見つからない」とは区別する。
    呼び出し側はこれを受けたらキャッシュせず再試行可能として扱うこと。"""


def _user_agent() -> str:
    mailto = os.environ.get("CROSSREF_MAILTO", "").strip()
    base = "ZoteroLocalRAG/1.0 (https://github.com/zotero-local-rag)"
    return f"{base} mailto:{mailto}" if mailto else base


def _normalize_doi(doi: str) -> str:
    """各種 DOI 表記から裸の DOI を取り出す。"""
    doi = (doi or "").strip()
    for prefix in ("https://doi.org/", "http://doi.org/",
                   "https://dx.doi.org/", "http://dx.doi.org/", "doi:"):
        if doi.lower().startswith(prefix):
            doi = doi[len(prefix):]
            break
    return doi.strip()


def _clean_abstract(raw: str) -> Optional[str]:
    """Crossref の abstract は JATS XML。タグ除去・エンティティ復号・空白圧縮する。"""
    if not raw:
        return None
    # <jats:title>Abstract</jats:title> 等の見出しタグは中身ごと落とす
    text = re.sub(r"<jats:title>.*?</jats:title>", " ", raw,
                  flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)        # 残りのタグを除去
    text = unescape(text)                        # &amp; などを復号
    text = re.sub(r"\s+", " ", text).strip()     # 空白を1個に圧縮
    text = re.sub(r"^Abstract[:\s]+", "", text, flags=re.IGNORECASE)  # 先頭の "Abstract" ラベル
    return text or None


def _extract_year(msg: Dict[str, Any]) -> Optional[int]:
    for key in ("published", "published-print", "published-online", "issued"):
        parts = (msg.get(key) or {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            try:
                return int(parts[0][0])
            except (ValueError, TypeError):
                pass
    return None


def _fmt_authors(authors: Optional[list], max_authors: int = 5) -> str:
    names = []
    for a in authors or []:
        family = a.get("family", "")
        given = a.get("given", "")
        name = (f"{given} {family}").strip() if (family or given) else a.get("name", "")
        if name:
            names.append(name)
    if not names:
        return ""
    if len(names) > max_authors:
        return ", ".join(names[:max_authors]) + " et al."
    return ", ".join(names)


def _normalize_message(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    titles = msg.get("title") or []
    title = titles[0].strip() if titles and isinstance(titles[0], str) else ""
    title = re.sub(r"<[^>]+>", " ", unescape(title))
    title = re.sub(r"\s+", " ", title).strip()
    if not title:
        return None
    containers = msg.get("container-title") or []
    publishers = msg.get("publisher") or ""
    return {
        "doi": msg.get("DOI") or None,
        "title": title,
        "authors": _fmt_authors(msg.get("author")),
        "year": _extract_year(msg),
        "container": containers[0] if containers else None,
        "publisher": publishers or None,
        "type": msg.get("type") or None,
        "source": "crossref",
    }


def search_crossref(
    bibliographic: str, *, rows: int = 5, timeout: float = 15.0,
) -> list[Dict[str, Any]]:
    """Search Crossref using a citation/title string and normalize candidates."""
    global _last_call
    query = (bibliographic or "").strip()
    if not query or rows <= 0:
        return []
    elapsed = time.time() - _last_call
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    params = {"query.bibliographic": query, "rows": min(rows, 20)}
    mailto = os.environ.get("CROSSREF_MAILTO", "").strip()
    if mailto:
        params["mailto"] = mailto
    request = urllib.request.Request(
        "https://api.crossref.org/works?" + urllib.parse.urlencode(params),
        headers={"User-Agent": _user_agent(), "Accept": "application/json"},
    )
    _last_call = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise CrossrefError(str(exc)) from exc
    items = (payload.get("message") or {}).get("items") or []
    return [candidate for item in items if isinstance(item, dict) and (candidate := _normalize_message(item))]


def fetch_crossref_by_doi(doi: str, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
    """DOI から Crossref メタデータを取得する。

    Returns:
        - dict: 取得成功（abstract は無い場合 None）
        - None: Crossref に該当 DOI が無い（404 = 確定的に「見つからない」）
    Raises:
        CrossrefError: 一時的な取得失敗（ネットワーク/タイムアウト/5xx）。
            呼び出し側はキャッシュせず再試行可能として扱う。
    """
    global _last_call
    doi = _normalize_doi(doi)
    if not doi:
        return None

    # 礼儀的なスペーシング（単発オンデマンド呼び出しでは通常スリープしない）
    elapsed = time.time() - _last_call
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_call = time.time()

    url = _CROSSREF_BASE + urllib.parse.quote(doi, safe="")
    req = urllib.request.Request(
        url, headers={"User-Agent": _user_agent(), "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None  # Crossref に該当なし（確定）
        print(f"[crossref] HTTP {e.code} for {doi}", file=sys.stderr)
        raise CrossrefError(f"HTTP {e.code}") from e
    except Exception as e:
        print(f"[crossref] request error for {doi}: {e}", file=sys.stderr)
        raise CrossrefError(str(e)) from e

    msg = data.get("message") or {}
    titles = msg.get("title") or []
    return {
        "doi":            msg.get("DOI") or doi,
        "title":          titles[0] if titles else None,
        "authors":        _fmt_authors(msg.get("author")),
        "year":           _extract_year(msg),
        "abstract":       _clean_abstract(msg.get("abstract") or ""),
        "citation_count": msg.get("is-referenced-by-count"),
    }
