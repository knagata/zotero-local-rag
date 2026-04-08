import os
import sys
from pathlib import Path

# Setup Path
ROOT = Path("/Users/knag/Documents/GitHub/zotero-local-rag")
sys.path.insert(0, str(ROOT / "src"))

# Environment Variables
os.environ["CHROMA_DIR"] = str(ROOT / "data" / "chroma")
os.environ["EMB_PROFILE"] = "fast"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["EMB_MODEL"] = str(ROOT / "data" / "models" / "paraphrase-multilingual-MiniLM-L12-v2")

from rag_mcp_server import rag_search, search_items

def test():
    print("=== Test 1: include_item_keys ===")
    # First, find some item keys
    items = search_items(query="intelligence", k=2)
    keys = [it['itemKey'] for it in items['items']]
    print(f"Targeting keys: {keys}")
    
    res1 = rag_search(query="intelligence", k=10, include_item_keys=keys)
    res_keys = set(r['meta']['itemKey'] for r in res1['results'])
    print(f"Result itemKeys: {res_keys}")
    
    if res_keys.issubset(set(keys)):
        print("✅ Success: All results belong to the specified item keys.")
    else:
        print("❌ Failure: Results included item keys outside the filter.")

    print("\n=== Test 2: exclude_chunk_ids ===")
    # First search
    res2a = rag_search(query="intelligence", k=3)
    ids_to_exclude = [r['id'] for r in res2a['results']]
    print(f"Excluding IDs: {ids_to_exclude}")
    
    # Second search with exclusion
    res2b = rag_search(query="intelligence", k=5, exclude_chunk_ids=ids_to_exclude)
    new_ids = [r['id'] for r in res2b['results']]
    print(f"New IDs: {new_ids}")
    
    overlap = set(ids_to_exclude).intersection(set(new_ids))
    if not overlap:
        print("✅ Success: No excluded IDs found in the new results.")
    else:
        print(f"❌ Failure: Overlapping IDs found: {overlap}")

if __name__ == "__main__":
    test()
