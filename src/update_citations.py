import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["RAYON_RS_NUM_CPUS"] = "1"
import sys
import time
import json
import urllib.request
import urllib.parse
from pathlib import Path

# Adjust sys.path to ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from citation_mapper import map_item_global_citations, map_item_local_references

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_dotenv_native() -> None:
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip()
                        if len(v) >= 2 and (
                            (v.startswith('"') and v.endswith('"'))
                            or (v.startswith("'") and v.endswith("'"))
                        ):
                            v = v[1:-1]
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

load_dotenv_native()

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
            
        # Pick the work with the highest citation count to avoid matching book reviews
        best_match = max(results, key=lambda x: x.get("cited_by_count", 0))
        doi = best_match.get("doi")
        
        if doi:
            if doi.startswith("https://doi.org/"):
                doi = doi.replace("https://doi.org/", "")
            return doi
    except Exception as e:
        print(f"    -> [OpenAlex] Error resolving: {e}", file=sys.stderr)
    return None

def process_item(item_key: str, item_data: dict, zotero_data_dir: str, force: bool = False):
    from db_relations import get_item_citation_status
    
    status = get_item_citation_status(item_key)
    if status == "mapped" and not force:
        print(f"[SKIP] Item {item_key} is already mapped.", file=sys.stderr)
        return

    print(f"========================================", file=sys.stderr)
    print(f"[PROGRESS] Processing item: {item_key}", file=sys.stderr)
    
    title = item_data.get("title", "")
    year = item_data.get("date", "")[:4] if item_data.get("date") else ""
    doi = item_data.get("DOI", "")
    isbn = item_data.get("ISBN", "")
    
    # Extract creators
    creators_list = item_data.get("creators", [])
    creators = ", ".join([
        (c.get("lastName", "") + " " + c.get("firstName", "")).strip() 
        if "lastName" in c else c.get("name", "") 
        for c in creators_list
    ])
    
    # DOI Lookup & Write-back
    if not doi and not isbn and title:
        author = creators_list[0].get("lastName", "") if creators_list else ""
        resolved_doi = query_openalex(title, author)
        if resolved_doi:
            print(f"    -> Resolved DOI: {resolved_doi}. Updating Zotero...", file=sys.stderr)
            patch_payload = {"DOI": resolved_doi}
            version = item_data.get("version")
            headers = {"If-Unmodified-Since-Version": str(version)} if version else {}
            
            patch_res = _zotero_request(f"items/{item_key}", method="PATCH", data=patch_payload, headers=headers)
            if patch_res is not None:
                print(f"    -> Successfully saved DOI to Zotero.", file=sys.stderr)
                doi = resolved_doi
            else:
                print(f"    -> Failed to save DOI to Zotero.", file=sys.stderr)
    
    # 1. Global citations (Semantic Scholar)
    print(f"  [1/2] Fetching incoming citations from Semantic Scholar...", file=sys.stderr)
    try:
        res1 = map_item_global_citations(item_key, title=title, year=year, creators=creators, doi=doi, isbn=isbn, max_citations=5000)
        status = res1.get('status')
        msg = res1.get('message', '')
        if status == "success":
            print(f"        -> Success: {msg}", file=sys.stderr)
        elif status == "api_error":
            print(f"        -> API Error: {msg}", file=sys.stderr)
        elif status == "no_data_found":
            print(f"        -> No Data: {msg}", file=sys.stderr)
        else:
            print(f"        -> Error: {msg}", file=sys.stderr)
    except Exception as e:
        print(f"        -> Exception: {e}", file=sys.stderr)

    # 2. Local references (EPUB)
    print(f"  [2/2] Checking for EPUB attachment to extract outgoing references...", file=sys.stderr)
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
                res2 = map_item_local_references(item_key, epub_path)
                msg2 = res2.get('message', '')
                print(f"        -> Result: {msg2}", file=sys.stderr)
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(f"        -> Exception: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Update citation relations for Zotero items.")
    parser.add_argument("--item", type=str, help="Item key to process")
    parser.add_argument("--all", action="store_true", help="Process all items in the database")
    parser.add_argument("--force", action="store_true", help="Force update even if item is already mapped")
    args = parser.parse_args()
    
    if not args.item and not args.all:
        parser.print_help()
        sys.exit(0)
            
    api_key = os.environ.get("S2_API_KEY", "")
    if not api_key:
        print("\n" + "="*60, file=sys.stderr)
        print("⚠️  [WARNING] Semantic Scholar API Key (S2_API_KEY) is NOT set!", file=sys.stderr)
        print("Without an API key, the rate limit is strictly enforced (1 request / 3 seconds).", file=sys.stderr)
        print("="*60 + "\n", file=sys.stderr)
        time.sleep(3)
        
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
    
    if args.item:
        raw = _zotero_request(f"items/{args.item}")
        if raw:
            _, item_data = _unwrap_item(raw)
            process_item(args.item, item_data, zotero_data_dir, force=args.force)
        else:
            print(f"Error: Could not fetch item {args.item} from Zotero API.")
    elif args.all:
        items = get_all_items()
        count = 0
        total = len(items)
        for raw in items:
            key, item_data = _unwrap_item(raw)
            count += 1
            print(f"\n[PROGRESS] Processing {count}/{total}...", file=sys.stderr)
            process_item(key, item_data, zotero_data_dir, force=args.force)

if __name__ == "__main__":
    main()
