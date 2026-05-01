
import os
import sys
from pathlib import Path

# Add src to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from rag_mcp_server import get_document_outline

def test_outline():
    # EPUB test
    print("Testing EPUB outline (IL65X3KK)...")
    epub_out = get_document_outline("IL65X3KK")
    print(epub_out)
    
    # PDF test (SW9Q8VWQ)
    print("\nTesting PDF outline (SW9Q8VWQ)...")
    pdf_out = get_document_outline("SW9Q8VWQ")
    print(pdf_out)

    # Missing key test
    print("\nTesting missing key...")
    missing_out = get_document_outline("MISSING")
    print(missing_out)

if __name__ == "__main__":
    test_outline()
