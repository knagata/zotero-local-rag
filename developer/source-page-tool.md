# Visual PDF verification over MCP

`get_source_page` returns MCP TextContent provenance plus ImageContent, using
FastMCP's Image helper. A client can visually compare paper with retrieved OCR
without opening a server file path. The public input is a Zotero attachment key
and physical 1-based PDF page, or a search chunk ID whose metadata supplies them.
Explicit pages allow neighboring-page inspection; contradictory attachment IDs
are rejected. Notes/EPUB chunks without PDF page metadata require a PDF source.

Source paths come only from the identified attachment's trusted Local API data
and the existing Zotero local-file resolver. The tool cannot accept arbitrary
paths or fetch URLs. Missing local files produce a sync/retry message. Reading a
PDF uses no native Chroma writer and does not invoke OCR or hosted LLM services.
Page rendering runs in a worker thread to avoid blocking MCP's event loop.

`src/source_page.py` renders with PyMuPDF to an in-memory RGB JPEG. Normalized crop
coordinates refer to the displayed page before the optional clockwise rotation.
The longest image edge is 1800 pixels; cropping makes small text larger without
increasing the image bound. Inputs reject nonfinite/out-of-range crops, tiny
regions, invalid pages and rotations. Encrypted/non-PDF sources are rejected.
The response includes physical page, stored PDF page label, total pages, crop,
rotation, pixel dimensions and image SHA256. A PDF page label can be empty or
incorrect; it is not an inferred printed folio. Images larger than 6 MiB fail
with a crop suggestion. No rendered-image cache or permanent file is written.

The image supplies evidence; it does not automatically overwrite OCR. The MCP
instructions connect visual inspection to report_chunk_quality for substantial
damage. Existing correction/reingestion controls remain authoritative. Tests in
tests/test_source_page.py cover actual PDF rendering and MCP image serialization,
bounds, sync errors, and read-only source preservation.
