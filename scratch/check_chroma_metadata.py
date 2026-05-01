
import os
import chromadb
from chromadb.utils import embedding_functions

ROOT = os.path.dirname(os.path.abspath(__name__))
CHROMA_DIR = os.path.join(ROOT, "data", "chroma")

def check_metadata():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    # Try to find the collection
    collections = client.list_collections()
    print(f"Collections: {[c.name for c in collections]}")
    
    for coll_name in [c.name for c in collections]:
        coll = client.get_collection(name=coll_name)
        count = coll.count()
        print(f"Collection {coll_name} has {count} items.")
        
        if count > 0:
            # peek 10 and check metadata
            peek = coll.peek(10)
            metadatas = peek.get("metadatas", [])
            has_chapter = any("chapter" in m for m in metadatas if m)
            print(f"  Sample has chapter metadata: {has_chapter}")
            
            # search for any item with chapter metadata
            # Chroma doesn't have a direct "exists" filter for keys in all versions easily without knowing a value
            # but we can try to query with a where filter if we know what values might look like, or just get some and check.
            
            all_metas = coll.get(limit=100, include=["metadatas"])["metadatas"]
            chapter_count = sum(1 for m in all_metas if m and "chapter" in m and m["chapter"])
            print(f"  Found {chapter_count}/100 items with chapter metadata in first 100 items.")

if __name__ == "__main__":
    check_metadata()
