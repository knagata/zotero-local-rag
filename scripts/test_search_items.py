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

from rag_mcp_server import search_items

def test():
    print("=== Test 1: Single query item search ===")
    res1 = search_items(query="artificial intelligence", k=5)
    print(f"Unique Items: {len(res1['items'])}")
    for i, it in enumerate(res1['items']):
        print(f"  [{i}] {it['itemKey']} | RRF: {it['rrf_score']:.6f} | Dist: {it['distance']:.4f} | {it['title']}")

    print("\n=== Test 2: Multi-query item search ===")
    queries = ["AI", "artificial intelligence", "neural networks"]
    res2 = search_items(query=queries, k=10)
    print(f"Queries: {queries}")
    print(f"Unique Items: {len(res2['items'])}")
    for i, it in enumerate(res2['items']):
        # If an item hits multiple times, RRF should be notably high
        print(f"  [{i}] {it['itemKey']} | RRF: {it['rrf_score']:.6f} | Dist: {it['distance']:.4f} | {it['title']}")

if __name__ == "__main__":
    test()
