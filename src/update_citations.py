import argparse
import difflib
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["RAYON_RS_NUM_CPUS"] = "1"
import sys
import sqlite3
import time
import json
import urllib.request
import urllib.parse
from pathlib import Path

# Adjust sys.path to ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from citation_mapper import map_item_global_citations, map_item_local_references

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from env_utils import load_dotenv_native
from zotero_source_localapi import local_api_url, zotero_api_headers

load_dotenv_native(PROJECT_ROOT)

API_KEY = os.environ.get("ZOTERO_API_KEY", "")

#: 両ステップを走り切った状態。通常更新ではスキップし、--force でのみ再処理する。
#: "not_found" もここに含める: S2 に無い資料を毎回問い合わせ直すのは無駄であり、
#: 検索ロジックを改善したときは --force で再走査する運用にする。
#: "limited" もここに含める: max_citations に達して意図的に打ち切った状態で、
#: 再試行しても同じ上限に当たる。上限を上げたときは --force で再走査する。
TERMINAL_STATUSES = frozenset({"mapped", "not_found", "limited"})

#: S2 offers no way to ask for the most-cited citing papers first, so a cut at
#: N keeps an arbitrary N rather than the top N -- which is why the reduction
#: belongs at rendering, where a criterion exists, and the fetch takes the lot.
#: The most-cited work in this library has 7,755 citations and none reaches
#: 10,000; the whole library's citations come to 151,906 rows. This is left as a
#: safety valve rather than removed: if identification goes wrong and lands on a
#: canonical paper with hundreds of thousands of citations, the run should stop
#: rather than page through all of them.
MAX_CITATIONS = 25_000


def _zotero_request(endpoint: str, params: dict = None, method: str = "GET", data: dict = None, headers: dict = None):
    url = local_api_url(endpoint)
    if params:
        query = urllib.parse.urlencode(params)
        url = f"{url}?{query}"

    # Zotero picks its own response schema when the version header is absent,
    # and this path had been sending only the key -- the least pinned of the
    # three callers, and the one that also writes back to the library.
    req_headers = zotero_api_headers(**(headers or {}))

    req_data = None
    if data:
        req_data = json.dumps(data).encode('utf-8')
        req_headers["Content-Type"] = "application/json"
        
    req = urllib.request.Request(url, data=req_data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            res_body = response.read().decode('utf-8')
            return json.loads(res_body) if res_body else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8') if hasattr(e, 'read') else ""
        print(f"[ERROR] Zotero API HTTP Error {e.code} on {url}: {body}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[ERROR] Zotero API Request Error on {url}: {e}", file=sys.stderr)
        return None

def _unwrap_item(raw: dict):
    # Depending on local vs web API
    if "data" in raw:
        return raw.get("key"), raw["data"]
    return raw.get("key"), raw

def resolve_epub_path(att_data: dict, zotero_data_dir: str):
    key = att_data.get("key")
    if not key:
        return None
        
    link_mode = att_data.get("linkMode")
    path = att_data.get("path", "")
    filename = att_data.get("filename", "")
    
    # For storage-based attachments (Zotero internal storage)
    if link_mode == "imported_file" and filename:
        return os.path.join(zotero_data_dir, "storage", key, filename)
    elif path.startswith("storage:"):
        filename = path.replace("storage:", "")
        return os.path.join(zotero_data_dir, "storage", key, filename)
        
    return None

def get_all_items():
    print("[PROGRESS] Fetching all items from Zotero Local API...", file=sys.stderr)
    start = 0
    limit = 100
    all_items = []
    while True:
        batch = _zotero_request("items", params={"format": "json", "limit": limit, "start": start})
        if batch is None:
            # _zotero_request already printed the underlying error. Distinguish
            # this from "the library genuinely has zero items" (an empty list),
            # which would otherwise silently look identical -- both used to
            # fall straight through to "Found 0 potential items" (2026-08-01,
            # traced from a run where Zotero simply wasn't running).
            if start == 0:
                # Covers both a connection failure and an HTTP error (401,
                # 500, ...) -- _zotero_request already logged which one, so
                # this doesn't guess a specific cause.
                raise RuntimeError(
                    f"Could not fetch items from the Zotero Local API at "
                    f"{local_api_url('')} -- see the error above."
                )
            break
        if not batch or not isinstance(batch, list):
            break

        for raw in batch:
            _, ad = _unwrap_item(raw)
            if ad.get("itemType") not in ("attachment", "note", "annotation"):
                all_items.append(raw)
        
        start += len(batch)
        if len(batch) < limit:
            break
            
    print(f"[PROGRESS] Found {len(all_items)} potential items.", file=sys.stderr)
    return all_items

def _openalex_author_tokens(work: dict) -> set:
    """Surnames OpenAlex lists for a work, comparable to Zotero's creator tokens."""
    from citation_mapper import _external_surnames
    return _external_surnames(
        ((authorship.get("author") or {}).get("display_name") or "")
        for authorship in work.get("authorships") or []
    )


def query_openalex(title: str, author: str, creators: str = ""):
    """Resolve a DOI for a work, or None when no candidate can be verified.

    A title-similarity match alone is not enough to accept a DOI here: OpenAlex
    indexes *book reviews* under titles nearly identical to the book's, and the
    resolved DOI is both stored locally and PATCHed into the user's Zotero
    library. Adopting a reviewer's DOI mislabels the user's own bibliographic
    record and then makes find_s2_paper_id import the review's citations as the
    book's (observed for Pratt "Imperial Eyes" -> Lorimer's review
    10.1086/600773, and Preziosi "Grasping the World" -> Hooper-Greenhill's
    10.1093/jdh/epi034).
    """
    from citation_mapper import _creator_name_tokens

    # Skip if title is mostly non-ASCII (Japanese etc.) — OpenAlex search unreliable
    non_ascii_ratio = sum(1 for c in title if ord(c) > 0x7F) / max(len(title), 1)
    if non_ascii_ratio > 0.3:
        return None

    print(f"    -> [OpenAlex] Resolving DOI for: {title} ({author})", file=sys.stderr)
    query = f"{title} {author}".strip()
    encoded_query = urllib.parse.quote(query)
    url = f"https://api.openalex.org/works?search={encoded_query}&mailto=zotero-local-rag@example.com"

    req = urllib.request.Request(url, headers={"User-Agent": "ZoteroLocalRAG/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
        results = data.get("results", [])
        if not results:
            return None

        # Require title similarity >= 0.6 before accepting a match
        title_lower = title.lower()
        def _sim(work: dict) -> float:
            t = (work.get("title") or "").lower()
            return difflib.SequenceMatcher(None, title_lower[:120], t[:120]).ratio()

        wanted = _creator_name_tokens(creators or author)
        candidates = [w for w in results if _sim(w) >= 0.6]
        # A review is never the work itself, whatever its title says.
        candidates = [w for w in candidates if str(w.get("type") or "").casefold() != "review"]
        if wanted:
            # Records listing no author at all cannot be checked either way, so
            # they stay eligible; a record naming *different* people is rejected.
            candidates = [
                w for w in candidates
                if not _openalex_author_tokens(w) or (wanted & _openalex_author_tokens(w))
            ]
        if not candidates:
            print("    -> [OpenAlex] No candidate passed title/author verification.", file=sys.stderr)
            return None

        best_match = max(candidates, key=_sim)

        doi = best_match.get("doi")
        if doi:
            if doi.startswith("https://doi.org/"):
                doi = doi.replace("https://doi.org/", "")
            return doi
    except Exception as e:
        print(f"    -> [OpenAlex] Error resolving: {e}", file=sys.stderr)
    return None

def _zotero_web_patch_doi(item_key: str, doi: str, version: object) -> None:
    """Zotero Web API (api.zotero.org) 経由でアイテムに DOI を書き戻す。"""
    api_key = os.environ.get("ZOTERO_API_KEY", "")
    user_id = os.environ.get("ZOTERO_USER_ID", "")
    if not api_key or not user_id:
        return

    url = f"https://api.zotero.org/users/{user_id}/items/{item_key}"
    headers = zotero_api_headers(api_key, **{"Content-Type": "application/json"})
    if version is not None:
        headers["If-Unmodified-Since-Version"] = str(version)

    data = json.dumps({"DOI": doi}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="PATCH")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status == 204:
                print("    -> DOI saved to Zotero.", file=sys.stderr)
            else:
                print(f"    -> Zotero Web API returned {resp.status}.", file=sys.stderr)
    except urllib.error.HTTPError as e:
        print(f"    -> Zotero Web API error {e.code}: {e.reason}", file=sys.stderr)
    except Exception as e:
        print(f"    -> Zotero Web API error: {e}", file=sys.stderr)


def _run_epub_step(item_key: str, item_data: dict, zotero_data_dir: str, epub_budget: int = 50) -> None:
    """EPUB参照抽出ステップ（step 2）を実行する。"""
    print("  [EPUB] Checking for EPUB attachment to extract outgoing references...", file=sys.stderr)
    try:
        children = _zotero_request(f"items/{item_key}/children")
        if not isinstance(children, list):
            children = []

        epub_key = None
        epub_data = None
        for child in children:
            _, att_data = _unwrap_item(child)
            filename = att_data.get("filename", "")
            if filename.lower().endswith(".epub"):
                epub_key = att_data.get("key")
                epub_data = att_data
                break

        if not epub_key:
            print("        -> Skip: No EPUB attachment found.", file=sys.stderr)
        else:
            epub_path = resolve_epub_path(epub_data, zotero_data_dir)
            if not epub_path or not os.path.exists(epub_path):
                print(f"        -> Error: EPUB file not found on disk ({epub_key}).", file=sys.stderr)
            else:
                res2 = map_item_local_references(item_key, epub_path, epub_budget=epub_budget)
                msg2 = res2.get('message', '')
                print(f"        -> Result: {msg2}", file=sys.stderr)
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(f"        -> Exception: {e}", file=sys.stderr)


def process_item(item_key: str, item_data: dict, zotero_data_dir: str, skip_s2: bool = False, epub_budget: int = 50) -> tuple:
    """
    1アイテムの引用ネットワーク構築を実行する。
    skip_s2=True の場合は S2 API ステップをスキップし EPUB 抽出のみ実行。
    完了ステータスの書き込みは呼び出し元が行う。

    Returns:
        ``(s2_ok, s2_resolved)``。

        - ``s2_ok``: S2 ステップが正常終了したら True。429 リトライ枯渇などで
          失敗したら False（呼び出し元は "error" を記録し、次回再処理させる）。
        - ``s2_resolved``: S2 が実際にこの資料を同定できたら True。False の場合
          "mapped" を書いてはいけない（S2上の身元が無いのに mapped と記録すると、
          以後 --all から永久にスキップされ、再検索の機会が失われる）。
        - ``s2_retryable``: 失敗が再試行に値するなら True。False の場合、マッパー
          自身が理由（"limited" 等）を記録済みなので、呼び出し元は上書きしない。
    """
    title = item_data.get("title", "")
    year = item_data.get("date", "")[:4] if item_data.get("date") else ""
    doi = item_data.get("DOI", "")
    isbn = item_data.get("ISBN", "")

    creators_list = item_data.get("creators", [])
    creators = ", ".join([
        (c.get("lastName", "") + " " + c.get("firstName", "")).strip()
        if "lastName" in c else c.get("name", "")
        for c in creators_list
    ])

    # ── Step 1: S2 API（被引用取得） ──────────────────────────
    s2_ok = True
    # skip_s2 は "s2_done"（S2が同定済みで関係も保存済み）からの再開なので解決済み扱い。
    s2_resolved = True
    # 失敗が再試行に値するか。値しないなら、マッパーが書いた理由（"limited" 等）を
    # 呼び出し元が "error" で塗り潰してはいけない。
    s2_retryable = True
    if skip_s2:
        print("  [1/2] S2 step: skipped (already done).", file=sys.stderr)
    else:
        # DOI Lookup & Write-back via Zotero Web API
        if not doi and not isbn and title:
            author = creators_list[0].get("lastName", "") if creators_list else ""
            resolved_doi = query_openalex(title, author, creators)
            if resolved_doi:
                doi = resolved_doi
                print(f"    -> Resolved DOI: {resolved_doi}.", file=sys.stderr)
                _zotero_web_patch_doi(item_key, resolved_doi, item_data.get("version"))

        # DOI/ISBN を DB に保存（S2処理前なので "pending" として記録。
        # ここで "mapped" を書くと途中クラッシュ時に次回スキップされてしまう）
        if doi or isbn:
            from db_relations import update_item_citation_status
            update_item_citation_status(item_key, "pending", doi=doi or None, isbn=isbn or None)

        print("  [1/2] Fetching citations from Semantic Scholar...", file=sys.stderr)
        try:
            res1 = map_item_global_citations(
                item_key, title=title, year=year, creators=creators,
                doi=doi, isbn=isbn, max_citations=MAX_CITATIONS,
            )
            s1_status = res1.get('status')
            msg = res1.get('message', '')
            if s1_status == "success":
                s2_resolved = bool(res1.get("s2_resolved", True))
                print(f"        -> Success: {msg}", file=sys.stderr)
            else:
                print(f"        -> {s1_status}: {msg}", file=sys.stderr)
                s2_ok = False
                s2_retryable = bool(res1.get("retryable", True))
        except Exception as e:
            print(f"        -> Exception: {e}", file=sys.stderr)
            s2_ok = False

    # ── Step 2: EPUB参照抽出 ─────────────────────────────────
    print("  [2/2] Extracting references from EPUB...", file=sys.stderr)
    _run_epub_step(item_key, item_data, zotero_data_dir, epub_budget=epub_budget)
    return s2_ok, s2_resolved, s2_retryable

def _fmt_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    else:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        return f"{h}h {m}m"


def _get_chunk_count(item_key: str) -> int:
    """ChromaDB SQLite から指定アイテムのチャンク数を返す。取得失敗時は -1。"""
    try:
        from citation_mapper import CHROMA_DIR, _get_segment_meta
        db_path = os.path.join(CHROMA_DIR, "chroma.sqlite3")
        if not os.path.exists(db_path):
            return -1
        seg = _get_segment_meta()
        conn = sqlite3.connect(db_path, timeout=5)
        try:
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM embeddings e
                JOIN embedding_metadata m ON m.id = e.id
                    AND m.key = 'itemKey' AND m.string_value = ?
                WHERE e.segment_id = ?
                """,
                (item_key, seg["metadata_segment_id"]),
            ).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()
    except Exception:
        return -1


def main():
    parser = argparse.ArgumentParser(description="Update citation relations for Zotero items.")
    parser.add_argument("--item", type=str, help="Item key to process")
    parser.add_argument("--all", action="store_true", help="Process all items in the database")
    parser.add_argument("--force", action="store_true", help="Force update even if item is already mapped")
    parser.add_argument("--resume-skipped", action="store_true",
                        help="Resolve s2_status='skipped' EPUB references with larger budget")
    parser.add_argument("--epub-budget", type=int, default=None,
                        help="Max S2 lookups per EPUB item (default: 50 for --all/--item, 200 for --resume-skipped)")
    parser.add_argument(
        "--unresolved-only", action="store_true",
        help="With --all: only items S2 never identified (no s2_paper_id). "
             "Implies --force, since those items carry a terminal status and "
             "would otherwise be skipped. Use after improving the lookup to "
             "re-search just them, instead of re-fetching the whole library.",
    )
    args = parser.parse_args()
    if args.unresolved_only:
        args.force = True

    if not args.item and not args.all and not args.resume_skipped:
        parser.print_help()
        sys.exit(0)

    api_key = os.environ.get("S2_API_KEY", "")
    if not api_key:
        print("\n" + "="*60, file=sys.stderr)
        print("[ERROR] S2_API_KEY is required for Citation Network updates.", file=sys.stderr)
        print("Run Setup.command and choose Citation Network, or set the key in .env.", file=sys.stderr)
        print("="*60 + "\n", file=sys.stderr)
        sys.exit(2)

    zotero_data_dir = os.environ.get("ZOTERO_DATA_DIR", os.path.expanduser("~/Zotero"))

    # Pre-initialize embedding model and validate ChromaDB data
    try:
        from citation_mapper import _get_emb_fn, _get_segment_meta
        print("[PROGRESS] Initializing embedding model and checking ChromaDB...", file=sys.stderr)
        _get_segment_meta()  # Validates that data exists
        _get_emb_fn()        # Loads model and warms up
        print("[PROGRESS] Ready.", file=sys.stderr)
    except Exception as e:
        print(f"[WARNING] Initialization issue: {e}", file=sys.stderr)

    from db_relations import get_item_citation_status, update_item_citation_status

    if args.resume_skipped:
        from db_relations import get_items_with_skipped_epub_refs
        from citation_mapper import resolve_skipped_epub_refs
        budget = args.epub_budget or 200
        # 'error'（429 リトライ枯渇等で失敗した行）も再解決対象に含める
        statuses = ('skipped', 'error')
        item_keys = get_items_with_skipped_epub_refs(statuses=statuses)
        if not item_keys:
            print("[DONE] No items with skipped/error EPUB references found.", file=sys.stderr)
        else:
            print(f"[PROGRESS] Found {len(item_keys)} items with skipped/error EPUB refs. Budget: {budget}/item.", file=sys.stderr)
            for idx, key in enumerate(item_keys, 1):
                print(f"\n[{idx}/{len(item_keys)}] {key}", file=sys.stderr)
                try:
                    result = resolve_skipped_epub_refs(key, budget=budget, statuses=statuses)
                    print(f"        -> {result['message']}", file=sys.stderr)
                except Exception as e:
                    print(f"  [ERROR] {e}", file=sys.stderr)
        return

    if args.item:
        raw = _zotero_request(f"items/{args.item}")
        if raw:
            _, item_data = _unwrap_item(raw)
            status = get_item_citation_status(args.item)
            if status in TERMINAL_STATUSES and not args.force:
                print(f"[SKIP] Item {args.item} already processed (status={status}). Use --force to re-process.", file=sys.stderr)
            else:
                skip_s2 = (status == "s2_done" and not args.force)
                if skip_s2:
                    print("[RESUME] S2 step already done. Running EPUB step only.", file=sys.stderr)
                s2_ok, s2_resolved, s2_retryable = process_item(
                    args.item, item_data, zotero_data_dir, skip_s2=skip_s2,
                    epub_budget=args.epub_budget or 50)
                if s2_ok:
                    final = "mapped" if s2_resolved else "not_found"
                    update_item_citation_status(args.item, final)
                    print(f"[DONE] Item {args.item} marked as '{final}'.", file=sys.stderr)
                elif s2_retryable:
                    update_item_citation_status(args.item, "error")
                    print(f"[WARN] S2 step failed for {args.item}; status set to 'error' (will retry next run).", file=sys.stderr)
                else:
                    print(f"[WARN] S2 step stopped for {args.item} for a reason that will not "
                          f"change on retry; keeping the status the mapper recorded.", file=sys.stderr)
        else:
            print(f"Error: Could not fetch item {args.item} from Zotero API.")

    elif args.all:
        try:
            items = get_all_items()
        except RuntimeError as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            sys.exit(1)
        if args.unresolved_only:
            from db_relations import get_item_s2_paper_id
            before = len(items)
            items = [
                raw for raw in items
                if not get_item_s2_paper_id(_unwrap_item(raw)[0])
            ]
            print(
                f"[PROGRESS] --unresolved-only: {len(items)} of {before} items have "
                f"no s2_paper_id.", file=sys.stderr,
            )
        total = len(items)
        if total == 0:
            print("[PROGRESS] No items found.", file=sys.stderr)
            return

        stats = {"processed": 0, "skipped": 0, "resumed": 0, "error": 0, "limited": 0, "meta_updated": 0}
        run_start = time.time()
        processed_times: list[float] = []  # wall-clock seconds per non-skipped item

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"  全 {total} 件を処理します（force={args.force}）", file=sys.stderr)
        print("  ステータス凡例: スキップ=両ステップ完了済み / 再開=S2完了・EPUB未実行", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)

        for idx, raw in enumerate(items, start=1):
            key, item_data = _unwrap_item(raw)
            title = item_data.get("title", "(no title)")
            year  = (item_data.get("date") or "")[:4]
            label = f"{title[:50]}…" if len(title) > 50 else title
            label = f"{label} ({year})" if year else label

            # ── ステータス確認 ───────────────────────────────────
            status = get_item_citation_status(key)
            if status in TERMINAL_STATUSES and not args.force:
                # 両ステップ完了済み → S2 はスキップするが isbn/doi は Zotero から同期。
                # 既存ステータスは保持する（not_found を mapped に昇格させない）。
                new_doi  = item_data.get("DOI",  "") or None
                new_isbn = item_data.get("ISBN", "") or None
                from db_relations import sync_item_citation_identifiers
                if sync_item_citation_identifiers(key, doi=new_doi, isbn=new_isbn):
                    stats["meta_updated"] += 1
                stats["skipped"] += 1
                if stats["skipped"] <= 3 or stats["skipped"] % 20 == 0:
                    meta_note = f"  doi/isbn synced={stats['meta_updated']}" if stats["meta_updated"] else ""
                    print(f"  [{idx}/{total}] SKIP {key}  ({stats['skipped']} skipped{meta_note})", file=sys.stderr)
                continue

            skip_s2 = (status == "s2_done" and not args.force)

            # ── progress header ──────────────────────────────────
            elapsed_total = time.time() - run_start
            remaining_items = total - idx
            if processed_times:
                avg_sec = sum(processed_times) / len(processed_times)
                eta_sec = avg_sec * (remaining_items + 1)
                eta_str = f"ETA {_fmt_duration(eta_sec)}"
            else:
                eta_str = "ETA --"

            chunk_count = _get_chunk_count(key)
            chunk_str = f"{chunk_count} chunks" if chunk_count >= 0 else "chunks: ?"
            mode_str = "RESUME (S2済み・EPUBのみ)" if skip_s2 else "PROCESS"

            print(f"\n[{idx}/{total}]  残り {remaining_items} 件  |  {key}  {chunk_str}  [{mode_str}]", file=sys.stderr)
            print(f"  {label}", file=sys.stderr)
            print(
                f"  経過 {_fmt_duration(elapsed_total)}"
                f"  ·  {eta_str}"
                f"  ·  完了 {stats['processed']} / 再開 {stats['resumed']} / スキップ {stats['skipped']} / エラー {stats['error']}",
                file=sys.stderr,
            )
            print(f"{'─'*60}", file=sys.stderr)

            # ── 処理・計測 ───────────────────────────────────────
            item_start = time.time()
            try:
                s2_ok, s2_resolved, s2_retryable = process_item(
                    key, item_data, zotero_data_dir, skip_s2=skip_s2,
                    epub_budget=args.epub_budget or 50)
                if s2_ok:
                    update_item_citation_status(key, "mapped" if s2_resolved else "not_found")
                    if skip_s2:
                        stats["resumed"] += 1
                    else:
                        stats["processed"] += 1
                elif s2_retryable:
                    # S2 ステップ失敗（429 リトライ枯渇など）→ 次回の --all で再処理されるよう "error" を記録
                    update_item_citation_status(key, "error")
                    stats["error"] += 1
                    print(f"  [WARN] S2 step failed for {key}; status set to 'error' (will retry next run).", file=sys.stderr)
                else:
                    # 再試行しても同じ結果になる打ち切り（max_citations 到達など）。
                    # マッパーが理由を書いているので上書きしない。"error" にすると
                    # 毎回の --all が同じ上限に当たり続ける。
                    stats["limited"] += 1
                    print(f"  [WARN] S2 step stopped for {key} for a reason that will not change "
                          f"on retry; keeping the status the mapper recorded.", file=sys.stderr)
            except Exception as e:
                print(f"  [ERROR] {e}", file=sys.stderr)
                update_item_citation_status(key, "error")
                stats["error"] += 1
            finally:
                item_elapsed = time.time() - item_start
                processed_times.append(item_elapsed)
                print(f"  → 処理時間: {_fmt_duration(item_elapsed)}", file=sys.stderr)

        # ── EPUB 参照のエラー行を再試行（429 リトライ枯渇等で失敗したもの） ──
        if not args.force:
            from db_relations import get_items_with_skipped_epub_refs
            from citation_mapper import resolve_skipped_epub_refs
            error_items = get_items_with_skipped_epub_refs(statuses=('error',))
            if error_items:
                print(f"\n[PROGRESS] Retrying EPUB refs that previously failed (s2_status='error') "
                      f"on {len(error_items)} items...", file=sys.stderr)
                for r_idx, r_key in enumerate(error_items, 1):
                    print(f"  [{r_idx}/{len(error_items)}] {r_key}", file=sys.stderr)
                    try:
                        result = resolve_skipped_epub_refs(r_key, budget=args.epub_budget or 200,
                                                           statuses=('error',))
                        print(f"        -> {result['message']}", file=sys.stderr)
                    except Exception as e:
                        print(f"  [ERROR] {e}", file=sys.stderr)

        # ── final summary ────────────────────────────────────────
        total_time = time.time() - run_start
        print(f"\n{'='*60}", file=sys.stderr)
        print("  完了サマリー", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(f"  合計件数  : {total}", file=sys.stderr)
        print(f"  処理済み  : {stats['processed']}  (両ステップ新規実行)", file=sys.stderr)
        print(f"  再開      : {stats['resumed']}  (S2済み・EPUBのみ実行)", file=sys.stderr)
        print(f"  スキップ  : {stats['skipped']}  (両ステップ完了済み・メタデータのみ同期)", file=sys.stderr)
        print(f"  メタ同期  : {stats['meta_updated']}  (スキップ中に DOI/ISBN を Zotero から更新)", file=sys.stderr)
        print(f"  エラー    : {stats['error']}", file=sys.stderr)
        print(f"  総処理時間: {_fmt_duration(total_time)}", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
