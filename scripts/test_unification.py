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
os.environ["MIN_RETURN_CHARS"] = "50"

from rag_mcp_server import rag_search

def test():
    print("=== Test 1: Single query (string) ===")
    res1 = rag_search(query="artificial intelligence", k=5)
    print(f"Results: {len(res1['results'])}")
    for r in res1['results'][:2]:
         print(f"  ID: {r['id']}, Distance: {r['distance']:.4f}, RRF: {r['rrf_score']:.6f}")
    
    print("\n=== Test 2: Multi-query (list) ranking and RRF ===")
    queries = ["artificial intelligence", "machine learning", "neural networks"]
    res2 = rag_search(query=queries, k=10)
    
    print(f"Queries: {queries}")
    print(f"Top 5 unique results (Sorted by RRF):")
    for i, r in enumerate(res2['results'][:5]):
         # Calculate expected min RRF for a single hit at rank 1: 1/(60+1) = 0.01639
         # Multiple hits will result in higher scores.
         print(f"  [{i}] ID: {r['id']}, RRF: {r['rrf_score']:.6f}, Dist: {r['distance']:.4f}")

    # Verify sorting
    rrf_scores = [r['rrf_score'] for r in res2['results']]
    if rrf_scores == sorted(rrf_scores, reverse=True):
        print("\n✅ Success: Results are correctly sorted by RRF score descending.")
    else:
        print("\n❌ Failure: Results are NOT sorted by RRF score correctly.")

if __name__ == "__main__":
    test()
