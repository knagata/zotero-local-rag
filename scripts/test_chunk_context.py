import os
import sys
from pprint import pprint
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

from rag_mcp_server import rag_search, get_chunk_context

def test():
    print("=== Step 1: Find a chunk ===")
    res1 = rag_search(query="artificial intelligence", k=1)
    if not res1["results"]:
        print("No results found.")
        return
    
    first_hit = res1["results"][0]
    chunk_id = first_hit["id"]
    print(f"Target Chunk ID: {chunk_id}")
    print(f"Original Text snippet: {first_hit['text'][:100]}...\n")

    print(f"=== Step 2: Fetch Context for {chunk_id} ===")
    ctx = get_chunk_context(chunk_id=chunk_id, window=2)
    
    if "error" in ctx:
        print(f"Error: {ctx['error']}")
        return
        
    print(f"Returned {len(ctx['chunk_ids_included'])} chunks.")
    print(f"Included IDs: {ctx['chunk_ids_included']}")
    print("\n--- Context Text ---")
    print(ctx["context_text"])
    print("--------------------")

if __name__ == "__main__":
    test()
