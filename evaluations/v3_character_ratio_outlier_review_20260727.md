# V3 character-ratio outlier review

Reviewed: 2026-07-27  
Source: `evaluations/v3_cutover_audit_20260723_live_sources.json`  
Scope: all 21 items outside the 0.8–1.5 legacy/V3 character-ratio band.

The ratio is a review signal, not an invariant: V3 adds zone/structure headings,
recovers material hidden by legacy extraction, and intentionally stops indexing a
Zotero attachment that has since been deleted.  Every source was checked against
its active attachment, extraction route, first/last source evidence, and (where
relevant) the legacy attachment set.

## Findings and disposition

| Item | Initial ratio | Disposition | Evidence / action |
|---|---:|---|---|
| `287Z7U6X` | 1.552 | explained increase | HTML DOM retains the full slide/article body; legacy snapshot segmentation had only 11 chunks. |
| `2A3GN9AK` | 1.548 | explained increase | Docling reconstructs the 20-page cemetery guide’s positioned text more completely than legacy extraction. |
| `2TDL9NLG` | 1.969 | explained increase | EPUB DOM preserves front matter, endnotes, index and TOC as distinct V3 zones. |
| `4CY8EIIB` | 0.706 | explained decrease | Same active PDF and complete first/last body evidence; the legacy-only surplus is garbled TOC/leader OCR-like text. V3 PyMuPDF normalization removes that noise. |
| `6VPN3EEQ` | 0.703 | **repaired** | A Japanese chronology’s short event rows were discarded before short-record merging. Merge now precedes the isolated-short filter; local re-extraction is 156,752 characters. |
| `BF8ZPCJP` | 2.948 | explained increase | EPUB DOM recovered book body and publisher paratext that legacy extraction under-counted. |
| `D3RVUXHY` | 0.662 | **repaired** | Publisher conversion used long `<h3>` elements for ordinary prose; the old DOM path treated them only as headings. Long malformed headings now become body blocks; local re-extraction is 515,831 characters. |
| `FP5UXECM` | 1.880 | explained increase | HTML DOM captures the article rather than legacy page-shell fragments. |
| `GXUKYRGH` | 1.792 | explained increase | Docling reconstructs the 9-page PDF’s spaced/positioned text. |
| `J33Z2LHV` | 0.007 | **repaired** | A single semantic blockquote suppressed the book-wide div fallback. Per-spine fallback recovers the body; local re-extraction is 59,323 characters. |
| `JQYWTQP8` | 0.014 | **repaired** | A table/figcaption made the semantic route non-empty while the div article body was skipped. Per-spine fallback recovers it; local re-extraction is 140,277 characters. |
| `QLC8CB3V` | 1.934 | explained increase | Docling recovers the Japanese article layout and bibliography more fully. |
| `R9C3UQTI` | 0.007 | **repaired** | Caption-only semantic blocks suppressed the div article body. Per-spine fallback recovers it; local re-extraction is 59,344 characters. |
| `RFHEXBIS` | 1.608 | explained increase | Docling recovers the short PDF’s positioned text; source remains quality-tagged for its noisy glyphs. |
| `RTVZVXT8` | 0.655 | explained decrease | Legacy contained two attachments; active Zotero/V3 contains only `C66HF59V`. The absent `Z3C43DNL` had 582,746 legacy characters and is marked `parent_item_deleted`, so it must not be reintroduced. |
| `S3CBHQMI` | 1.564 | explained increase | HTML DOM retains the full Comic Natalie interview body rather than the legacy partial snapshot. |
| `UE97SG9P` | 9.187 | explained increase | HTML DOM indexes the complete article (85 chunks) versus the legacy 8-chunk teaser. |
| `WVLWDTYL` | 54.947 | explained increase | Legacy had only 246 characters; Docling recovered the 19-page Japanese thesis/article. |
| `XBG2FYC7` | 6.310 | explained increase | HTML DOM captures the full annotated article instead of the legacy shell. |
| `YIRUY4AP` | 1.666 | explained increase | PyMuPDF V3 reconstruction includes more of the Japanese dissertation body. |
| `ZQQZTBVK` | 1.656 | explained increase | Docling recovers the article bibliography and a footnote as separate V3 zones. |

## Repair contract

The five repairs are deliberately general, not item-specific:

1. Choose a leaf-`div` EPUB fallback per spine document when semantic blocks
   contain less than 2,000 characters and the fallback is decisively richer.
2. Retain an `h1`–`h6` element longer than 240 characters as
   `malformed_heading_body`, rather than treating publisher-mislabelled prose
   as a heading.
3. Merge adjacent short records before excluding isolated sub-40-character
   labels, preserving chronology rows without admitting standalone UI noise.

The five named items must be force-reparsed into the V3 collection, then the
cutover audit rerun. This review closes only when the rerun has no unexplained
decrease and manifest/Chroma/FTS integrity remains clean.
