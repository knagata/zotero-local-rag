#!/usr/bin/env python3
"""
S2 API 診断スクリプト
失敗しているクエリを単発で投げてレスポンスヘッダーを確認する。

使い方:
  uv run scripts/s2_diagnose.py
  uv run scripts/s2_diagnose.py --isbn 978-3-319-90539-6
  uv run scripts/s2_diagnose.py --query "The Voice in the Cinema"
"""
import argparse
import os
import sys
import json
import time
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# .env 読み込み
env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

S2_API_KEY = os.environ.get("S2_API_KEY", "")


def s2_get(url: str, label: str) -> None:
    headers = {"User-Agent": "ZoteroLocalRAG/1.0-diagnose"}
    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"  URL: {url}")
    print(f"  API key: {'SET (' + S2_API_KEY[:8] + '...)' if S2_API_KEY else 'NOT SET'}")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode())
            print("  Status: 200 OK")
            # 成功レスポンスのヘッダーも表示
            interesting = ["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset",
                           "Retry-After", "Content-Type"]
            for h in interesting:
                v = resp.headers.get(h)
                if v:
                    print(f"  {h}: {v}")
            print()
            # 結果サマリー
            if "paperId" in body:
                print(f"  Found: {body.get('title', '(no title)')} ({body.get('year', '?')})")
                print(f"  paperId: {body['paperId']}")
                print(f"  citations: {body.get('citationCount', '?')}")
            elif "data" in body:
                data = body["data"]
                print(f"  Search results: {len(data)} hit(s)")
                for i, p in enumerate(data[:3]):
                    print(f"    [{i+1}] {p.get('title', '?')} ({p.get('year', '?')}) — {p.get('paperId', '?')}")
            else:
                print(f"  Response: {json.dumps(body)[:200]}")
    except urllib.error.HTTPError as e:
        print(f"  Status: {e.code} {e.reason}")
        interesting = ["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset",
                       "Retry-After", "Content-Type", "x-amzn-RequestId", "x-amzn-ErrorType"]
        for h in interesting:
            v = e.headers.get(h) if e.headers else None
            if v:
                print(f"  {h}: {v}")
        # ヘッダーが一つも無かった場合
        if e.headers:
            header_names = [h for h in interesting if e.headers.get(h)]
            if not header_names:
                print("  (rate-limit headers: none returned)")
        try:
            body_text = e.read().decode()[:300]
            print(f"  Body: {body_text}")
        except Exception:
            pass
    except Exception as e:
        print(f"  Error: {e}")


def run_error_items() -> None:
    """DB の error アイテムを最大5件取得して単発リクエスト"""
    try:
        import sqlite3
        conn = sqlite3.connect(str(ROOT / "data" / "relations.db"))
        rows = conn.execute(
            "SELECT item_key, doi, isbn FROM item_citation_status WHERE s2_status='error' LIMIT 5"
        ).fetchall()
        conn.close()
    except Exception as e:
        print(f"DB read error: {e}")
        return

    if not rows:
        print("error ステータスのアイテムはありません")
        return

    for item_key, doi, isbn in rows:
        if doi:
            url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{urllib.parse.quote(doi)}?fields=paperId,title,year,citationCount"
            s2_get(url, f"item {item_key}  DOI:{doi}")
        elif isbn:
            first_isbn = isbn.split()[0]
            url = f"https://api.semanticscholar.org/graph/v1/paper/ISBN:{first_isbn}?fields=paperId,title,year,citationCount"
            s2_get(url, f"item {item_key}  ISBN:{first_isbn}")
        else:
            print(f"\nitem {item_key}: DOI/ISBN なし（タイトル検索が必要）")
        time.sleep(2.6)  # 最小インターバル


def main() -> None:
    parser = argparse.ArgumentParser(description="S2 API 診断スクリプト")
    parser.add_argument("--isbn", help="ISBN で検索")
    parser.add_argument("--doi",  help="DOI で検索")
    parser.add_argument("--query", help="タイトル全文で検索")
    parser.add_argument("--db-errors", action="store_true",
                        help="DB の error アイテムを一括診断（最大5件）")
    args = parser.parse_args()

    if args.isbn:
        url = f"https://api.semanticscholar.org/graph/v1/paper/ISBN:{args.isbn}?fields=paperId,title,year,citationCount"
        s2_get(url, f"ISBN:{args.isbn}")
    elif args.doi:
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{urllib.parse.quote(args.doi)}?fields=paperId,title,year,citationCount"
        s2_get(url, f"DOI:{args.doi}")
    elif args.query:
        encoded = urllib.parse.quote(args.query)
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded}&limit=5&fields=paperId,title,year,citationCount"
        s2_get(url, f"search: {args.query}")
    elif args.db_errors:
        run_error_items()
    else:
        # 引数なし: DB の error アイテムを診断
        run_error_items()

    print(f"\n{'─'*60}")


if __name__ == "__main__":
    main()
