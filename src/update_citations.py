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

load_dotenv_native(PROJECT_ROOT)

API_BASE = os.environ.get("ZOTERO_LOCAL_API_BASE", "http://127.0.0.1:8080").rstrip("/")
API_PREFIX = os.environ.get("ZOTERO_LOCAL_API_PREFIX", "").strip("/")
API_KEY = os.environ.get("ZOTERO_API_KEY", "")

def _zotero_request(endpoint: str, params: dict = None, method: str = "GET", data: dict = None, headers: dict = None):
    url = f"{API_BASE}/{API_PREFIX}/{endpoint}"
    if params:
        query = urllib.parse.urlencode(params)
        url = f"{url}?{query}"
        
    req_headers = {}
    if API_KEY:
        req_headers["Zotero-API-Key"] = API_KEY
    if headers:
        req_headers.update(headers)
        
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

def query_openalex(title: str, author: str):
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

        best_match = max(results, key=_sim)
        if _sim(best_match) < 0.6:
            return None

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
    headers = {
        "Zotero-API-Key": api_key,
        "Zotero-API-Version": "3",
        "Content-Type": "application/json",
    }
    if version is not None:
        headers["If-Unmodified-Since-Version"] = str(version)

    data = json.dumps({"DOI": doi}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="PATCH")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status == 204:
                print(f"    -> DOI saved to Zotero.", file=sys.stderr)
            else:
                print(f"    -> Zotero Web API returned {resp.status}.", file=sys.stderr)
    except urllib.error.HTTPError as e:
        print(f"    -> Zotero Web API error {e.code}: {e.reason}", file=sys.stderr)
    except Exception as e:
        print(f"    -> Zotero Web API error: {e}", file=sys.stderr)


def _run_epub_step(item_key: str, item_data: dict, zotero_data_dir: str, epub_budget: int = 50) -> None:
    """EPUB参照抽出ステップ（step 2）を実行する。"""
    print(f"  [EPUB] Checking for EPUB attachment to extract outgoing references...", file=sys.stderr)
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
            print(f"        -> Skip: No EPUB attachment found.", file=sys.stderr)
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


def process_item(item_key: str, item_data: dict, zotero_data_dir: str, skip_s2: bool = False, epub_budget: int = 50) -> bool:
    """
    1アイテムの引用ネットワーク構築を実行する。
    skip_s2=True の場合は S2 API ステップをスキップし EPUB 抽出のみ実行。
    完了ステータス ("mapped") の書き込みは呼び出し元が行う。

    Returns:
        S2 ステップが正常終了したら True。429 リトライ枯渇などで失敗したら False
        （呼び出し元は "mapped" ではなく "error" を記録し、次回再処理させること）。
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
    if skip_s2:
        print(f"  [1/2] S2 step: skipped (already done).", file=sys.stderr)
    else:
        # DOI Lookup & Write-back via Zotero Web API
        if not doi and not isbn and title:
            author = creators_list[0].get("lastName", "") if creators_list else ""
            resolved_doi = query_openalex(title, author)
            if resolved_doi:
                doi = resolved_doi
                print(f"    -> Resolved DOI: {resolved_doi}.", file=sys.stderr)
                _zotero_web_patch_doi(item_key, resolved_doi, item_data.get("version"))

        # DOI/ISBN を DB に保存（S2処理前なので "pending" として記録。
        # ここで "mapped" を書くと途中クラッシュ時に次回スキップされてしまう）
        if doi or isbn:
            from db_relations import update_item_citation_status
            update_item_citation_status(item_key, "pending", doi=doi or None, isbn=isbn or None)

        print(f"  [1/2] Fetching citations from Semantic Scholar...", file=sys.stderr)
        try:
            res1 = map_item_global_citations(
                item_key, title=title, year=year, creators=creators,
                doi=doi, isbn=isbn, max_citations=5000,
            )
            s1_status = res1.get('status')
            msg = res1.get('message', '')
            if s1_status == "success":
                print(f"        -> Success: {msg}", file=sys.stderr)
            else:
                print(f"        -> {s1_status}: {msg}", file=sys.stderr)
                s2_ok = False
        except Exception as e:
            print(f"        -> Exception: {e}", file=sys.stderr)
            s2_ok = False

    # ── Step 2: EPUB参照抽出 ─────────────────────────────────
    print(f"  [2/2] Extracting references from EPUB...", file=sys.stderr)
    _run_epub_step(item_key, item_data, zotero_data_dir, epub_budget=epub_budget)
    return s2_ok

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
    args = parser.parse_args()

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
        print(f"[PROGRESS] Initializing embedding model and checking ChromaDB...", file=sys.stderr)
        _get_segment_meta()  # Validates that data exists
        _get_emb_fn()        # Loads model and warms up
        print(f"[PROGRESS] Ready.", file=sys.stderr)
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
            if status == "mapped" and not args.force:
                print(f"[SKIP] Item {args.item} is already fully mapped. Use --force to re-process.", file=sys.stderr)
            else:
                skip_s2 = (status == "s2_done" and not args.force)
                if skip_s2:
                    print(f"[RESUME] S2 step already done. Running EPUB step only.", file=sys.stderr)
                s2_ok = process_item(args.item, item_data, zotero_data_dir, skip_s2=skip_s2,
                                     epub_budget=args.epub_budget or 50)
                if s2_ok:
                    update_item_citation_status(args.item, "mapped")
                    print(f"[DONE] Item {args.item} marked as fully mapped.", file=sys.stderr)
                else:
                    update_item_citation_status(args.item, "error")
                    print(f"[WARN] S2 step failed for {args.item}; status set to 'error' (will retry next run).", file=sys.stderr)
        else:
            print(f"Error: Could not fetch item {args.item} from Zotero API.")

    elif args.all:
        items = get_all_items()
        total = len(items)
        if total == 0:
            print("[PROGRESS] No items found.", file=sys.stderr)
            return

        stats = {"processed": 0, "skipped": 0, "resumed": 0, "error": 0, "meta_updated": 0}
        run_start = time.time()
        processed_times: list[float] = []  # wall-clock seconds per non-skipped item

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"  全 {total} 件を処理します（force={args.force}）", file=sys.stderr)
        print(f"  ステータス凡例: スキップ=両ステップ完了済み / 再開=S2完了・EPUB未実行", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)

        for idx, raw in enumerate(items, start=1):
            key, item_data = _unwrap_item(raw)
            title = item_data.get("title", "(no title)")
            year  = (item_data.get("date") or "")[:4]
            label = f"{title[:50]}…" if len(title) > 50 else title
            label = f"{label} ({year})" if year else label

            # ── ステータス確認 ───────────────────────────────────
            status = get_item_citation_status(key)
            if status == "mapped" and not args.force:
                # 両ステップ完了済み → S2 はスキップするが isbn/doi は Zotero から同期
                new_doi  = item_data.get("DOI",  "") or None
                new_isbn = item_data.get("ISBN", "") or None
                from db_relations import update_item_citation_status as _uics
                _uics(key, "mapped", doi=new_doi, isbn=new_isbn)
                if new_doi or new_isbn:
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
                s2_ok = process_item(key, item_data, zotero_data_dir, skip_s2=skip_s2,
                                     epub_budget=args.epub_budget or 50)
                if s2_ok:
                    update_item_citation_status(key, "mapped")
                    if skip_s2:
                        stats["resumed"] += 1
                    else:
                        stats["processed"] += 1
                else:
                    # S2 ステップ失敗（429 リトライ枯渇など）→ 次回の --all で再処理されるよう "error" を記録
                    update_item_citation_status(key, "error")
                    stats["error"] += 1
                    print(f"  [WARN] S2 step failed for {key}; status set to 'error' (will retry next run).", file=sys.stderr)
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
        print(f"  完了サマリー", file=sys.stderr)
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
