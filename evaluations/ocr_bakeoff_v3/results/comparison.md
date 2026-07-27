# OCRベイクオフ比較

## Sample × engine

| Sample | Category | Engine | Score | Metrics | Duration (s) | Peak RSS (MB) | Status |
|---|---|---|---:|---|---:|---:|---|
| embedded_text_pair | embedded_text | docling | 0.972940 | heading_hierarchy=1.000000; reading_order=1.000000; zone_classification=1.000000; table_caption_retention=1.000000; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.549008 | 0.569 | 1614.547 | completed |
| embedded_text_pair | embedded_text | mistral_ocr | 0.972940 | heading_hierarchy=1.000000; reading_order=1.000000; zone_classification=1.000000; table_caption_retention=1.000000; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.549008 | 3.457 | 78.547 | completed |
| embedded_text_pair | embedded_text | ndlocr_lite | 0.739762 | heading_hierarchy=0.000000; reading_order=1.000000; zone_classification=1.000000; table_caption_retention=1.000000; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.329366 | 3.729 | 276.578 | completed |
| embedded_text_pair | embedded_text | pymupdf | 0.753466 | heading_hierarchy=0.000000; reading_order=1.000000; zone_classification=1.000000; table_caption_retention=1.000000; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.557772 | 0.015 | 57.625 | completed |
| embedded_text_pair | embedded_text | yomitoku | 0.972934 | heading_hierarchy=1.000000; reading_order=1.000000; zone_classification=1.000000; table_caption_retention=1.000000; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.548894 | 9.984 | 2712.203 | completed |
| en_two_column | english_two_column | docling | 0.755102 | heading_hierarchy=1.000000; reading_order=0.000000; zone_classification=1.000000; table_caption_retention=1.000000; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.585028 | 3.22 | 1049.453 | completed |
| en_two_column | english_two_column | mistral_ocr | 0.975102 | heading_hierarchy=1.000000; reading_order=1.000000; zone_classification=1.000000; table_caption_retention=1.000000; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.585028 | 2.001 | 72.547 | completed |
| en_two_column | english_two_column | ndlocr_lite | 0.731948 | heading_hierarchy=0.000000; reading_order=1.000000; zone_classification=1.000000; table_caption_retention=1.000000; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.199134 | 3.41 | 76.391 | completed |
| en_two_column | english_two_column | pymupdf | 0.365102 | heading_hierarchy=0.000000; reading_order=0.000000; zone_classification=0.500000; table_caption_retention=1.000000; locator_bbox_recovery=0.500000; tree_integrity=1.000000; text_accuracy=0.085028 | 0.062 | 50.797 | completed |
| en_two_column | english_two_column | yomitoku | 0.975102 | heading_hierarchy=1.000000; reading_order=1.000000; zone_classification=1.000000; table_caption_retention=1.000000; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.585028 | 13.716 | 2457.453 | completed |
| ja_horizontal | japanese_horizontal | docling | 0.973221 | heading_hierarchy=1.000000; reading_order=1.000000; zone_classification=1.000000; table_caption_retention=1.000000; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.553691 | 0.586 | 1142.719 | completed |
| ja_horizontal | japanese_horizontal | mistral_ocr | 0.973310 | heading_hierarchy=1.000000; reading_order=1.000000; zone_classification=1.000000; table_caption_retention=1.000000; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.555172 | 4.044 | 76.281 | completed |
| ja_horizontal | japanese_horizontal | ndlocr_lite | 0.360968 | heading_hierarchy=0.000000; reading_order=0.000000; zone_classification=0.500000; table_caption_retention=1.000000; locator_bbox_recovery=0.500000; tree_integrity=1.000000; text_accuracy=0.016129 | 2.648 | 86.844 | completed |
| ja_horizontal | japanese_horizontal | pymupdf | 0.370435 | heading_hierarchy=0.000000; reading_order=0.000000; zone_classification=0.500000; table_caption_retention=1.000000; locator_bbox_recovery=0.500000; tree_integrity=1.000000; text_accuracy=0.173913 | 0.031 | 56.094 | completed |
| ja_horizontal | japanese_horizontal | yomitoku | 0.973221 | heading_hierarchy=1.000000; reading_order=1.000000; zone_classification=1.000000; table_caption_retention=1.000000; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.553691 | 6.883 | 2712.203 | completed |
| ja_vertical | japanese_vertical | docling | 0.370270 | heading_hierarchy=0.000000; reading_order=0.000000; zone_classification=0.500000; table_caption_retention=1.000000; locator_bbox_recovery=0.500000; tree_integrity=1.000000; text_accuracy=0.171171 | 2.507 | 1290.609 | completed |
| ja_vertical | japanese_vertical | mistral_ocr | 0.665969 | heading_hierarchy=0.000000; reading_order=1.000000; zone_classification=0.500000; table_caption_retention=1.000000; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.599476 | 0.809 | 76.984 | completed |
| ja_vertical | japanese_vertical | ndlocr_lite | 0.363904 | heading_hierarchy=0.000000; reading_order=0.000000; zone_classification=0.500000; table_caption_retention=1.000000; locator_bbox_recovery=0.500000; tree_integrity=1.000000; text_accuracy=0.065068 | 2.185 | 86.844 | completed |
| ja_vertical | japanese_vertical | pymupdf | 0.363986 | heading_hierarchy=0.000000; reading_order=0.000000; zone_classification=0.500000; table_caption_retention=1.000000; locator_bbox_recovery=0.500000; tree_integrity=1.000000; text_accuracy=0.066434 | 0.004 | 56.094 | completed |
| ja_vertical | japanese_vertical | yomitoku | 0.665969 | heading_hierarchy=0.000000; reading_order=1.000000; zone_classification=0.500000; table_caption_retention=1.000000; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.599476 | 5.076 | 2712.203 | completed |
| no_outline | no_outline | docling | 0.728631 | heading_hierarchy=0.000000; reading_order=1.000000; zone_classification=1.000000; table_caption_retention=1.000000; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.143852 | 0.446 | 1614.547 | completed |
| no_outline | no_outline | mistral_ocr | 0.724767 | heading_hierarchy=0.000000; reading_order=1.000000; zone_classification=1.000000; table_caption_retention=1.000000; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.079450 | 2.08 | 78.703 | completed |
| no_outline | no_outline | ndlocr_lite | 0.734881 | heading_hierarchy=0.000000; reading_order=1.000000; zone_classification=1.000000; table_caption_retention=1.000000; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.248016 | 2.472 | 276.578 | completed |
| no_outline | no_outline | pymupdf | 0.363659 | heading_hierarchy=0.000000; reading_order=0.000000; zone_classification=0.500000; table_caption_retention=1.000000; locator_bbox_recovery=0.500000; tree_integrity=1.000000; text_accuracy=0.060976 | 0.009 | 57.797 | completed |
| no_outline | no_outline | yomitoku | 0.726349 | heading_hierarchy=0.000000; reading_order=1.000000; zone_classification=1.000000; table_caption_retention=1.000000; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.105819 | 5.804 | 2712.203 | completed |
| notes_bibliography_book | notes_and_bibliography | docling | 0.490964 | heading_hierarchy=0.500000; reading_order=0.000000; zone_classification=0.500000; table_caption_retention=1.000000; locator_bbox_recovery=0.500000; tree_integrity=1.000000; text_accuracy=0.349398 | 4.239 | 1614.547 | completed |
| notes_bibliography_book | notes_and_bibliography | mistral_ocr | 0.589561 | heading_hierarchy=0.000000; reading_order=1.000000; zone_classification=0.000000; table_caption_retention=1.000000; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.826025 | 1.418 | 77.922 | completed |
| notes_bibliography_book | notes_and_bibliography | ndlocr_lite | 0.570537 | heading_hierarchy=0.000000; reading_order=1.000000; zone_classification=0.000000; table_caption_retention=1.000000; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.508951 | 3.286 | 276.578 | completed |
| notes_bibliography_book | notes_and_bibliography | pymupdf | 0.571074 | heading_hierarchy=0.000000; reading_order=1.000000; zone_classification=0.000000; table_caption_retention=1.000000; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.517905 | 0.009 | 57.297 | completed |
| notes_bibliography_book | notes_and_bibliography | yomitoku | 0.589621 | heading_hierarchy=0.000000; reading_order=1.000000; zone_classification=0.000000; table_caption_retention=1.000000; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.827010 | 5.282 | 2712.203 | completed |
| scanned_pair | scanned | docling | 0.753176 | heading_hierarchy=0.000000; reading_order=1.000000; zone_classification=1.000000; table_caption_retention=1.000000; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.552941 | 8.106 | 1614.547 | completed |
| scanned_pair | scanned | mistral_ocr | 0.973165 | heading_hierarchy=1.000000; reading_order=1.000000; zone_classification=1.000000; table_caption_retention=1.000000; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.552758 | 3.78 | 78.656 | completed |
| scanned_pair | scanned | ndlocr_lite | 0.736192 | heading_hierarchy=0.000000; reading_order=1.000000; zone_classification=1.000000; table_caption_retention=1.000000; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.269859 | 3.413 | 276.578 | completed |
| scanned_pair | scanned | pymupdf | 0.220000 | heading_hierarchy=0.000000; reading_order=0.000000; zone_classification=0.000000; table_caption_retention=1.000000; locator_bbox_recovery=0.000000; tree_integrity=1.000000; text_accuracy=0.000000 | 0.002 | 57.625 | completed |
| scanned_pair | scanned | yomitoku | 0.751770 | heading_hierarchy=0.000000; reading_order=1.000000; zone_classification=1.000000; table_caption_retention=1.000000; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.529502 | 8.995 | 2712.203 | completed |
| tables_math | tables_and_math | docling | 0.970144 | heading_hierarchy=1.000000; reading_order=1.000000; zone_classification=1.000000; table_caption_retention=1.000000; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.502405 | 1.802 | 1142.719 | completed |
| tables_math | tables_and_math | mistral_ocr | 0.911918 | heading_hierarchy=1.000000; reading_order=1.000000; zone_classification=1.000000; table_caption_retention=0.500000; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.531973 | 1.39 | 73.875 | completed |
| tables_math | tables_and_math | ndlocr_lite | 0.830199 | heading_hierarchy=1.000000; reading_order=1.000000; zone_classification=1.000000; table_caption_retention=0.000000; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.169976 | 3.226 | 82.781 | completed |
| tables_math | tables_and_math | pymupdf | 0.850144 | heading_hierarchy=1.000000; reading_order=1.000000; zone_classification=1.000000; table_caption_retention=0.000000; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.502405 | 0.041 | 53.531 | completed |
| tables_math | tables_and_math | yomitoku | 0.487879 | heading_hierarchy=1.000000; reading_order=0.000000; zone_classification=0.500000; table_caption_retention=0.000000; locator_bbox_recovery=0.500000; tree_integrity=1.000000; text_accuracy=0.464646 | 7.526 | 2712.203 | completed |

## カテゴリ別winner

| Category | Winner | Mean score | Completed runs |
|---|---|---:|---:|
| embedded_text | docling | 0.97294 | 1 |
| english_two_column | yomitoku | 0.975102 | 1 |
| japanese_horizontal | mistral_ocr | 0.97331 | 1 |
| japanese_vertical | yomitoku | 0.665969 | 1 |
| no_outline | ndlocr_lite | 0.734881 | 1 |
| notes_and_bibliography | yomitoku | 0.589621 | 1 |
| scanned | mistral_ocr | 0.973165 | 1 |
| tables_and_math | docling | 0.970144 | 1 |

## Engine平均

| Engine | Completed runs | Mean score | Mean metrics | Mean duration (s) | Mean peak RSS (MB) |
|---|---:|---:|---|---:|---:|
| mistral_ocr | 8 | 0.848342 | heading_hierarchy=0.625000; reading_order=1.000000; zone_classification=0.812500; table_caption_retention=0.937500; locator_bbox_recovery=1.000000; tree_integrity=1.000000; text_accuracy=0.534861 | 2.372375 | 76.6894 |
| yomitoku | 8 | 0.767856 | heading_hierarchy=0.500000; reading_order=0.875000; zone_classification=0.750000; table_caption_retention=0.875000; locator_bbox_recovery=0.937500; tree_integrity=1.000000; text_accuracy=0.526758 | 7.90825 | 2680.3592 |
| docling | 8 | 0.751806 | heading_hierarchy=0.562500; reading_order=0.625000; zone_classification=0.875000; table_caption_retention=1.000000; locator_bbox_recovery=0.875000; tree_integrity=1.000000; text_accuracy=0.425937 | 2.684375 | 1385.461 |
| ndlocr_lite | 8 | 0.633549 | heading_hierarchy=0.125000; reading_order=0.750000; zone_classification=0.750000; table_caption_retention=0.875000; locator_bbox_recovery=0.875000; tree_integrity=1.000000; text_accuracy=0.225812 | 3.046125 | 179.8965 |
| pymupdf | 8 | 0.482233 | heading_hierarchy=0.125000; reading_order=0.375000; zone_classification=0.500000; table_caption_retention=0.875000; locator_bbox_recovery=0.625000; tree_integrity=1.000000; text_accuracy=0.245554 | 0.021625 | 55.8575 |

## 同点規則

1. higher mean total score
2. higher mean metrics in METRIC_ORDER
3. lower mean duration_seconds
4. lower mean process_peak_rss_mb
5. lexicographically smaller engine name

## v2改訂に関する注記（スコアラー修正・再計算）

v2 (2026-07-21) fixes a region-matching bug in src/ocr_bakeoff.py: _match_regions used to compare a short ground-truth anchor against a candidate block's FULL text via SequenceMatcher.ratio(), which is diluted whenever the block is a much longer multi-sentence paragraph than the anchor, and did not strip markdown emphasis (*_`#) that OCR engines emit around otherwise-verbatim text, breaking exact substring containment. Both are fixed (partial-match ratio measured against the anchor's own length; markdown decoration stripped before matching only, not for text_accuracy scoring). Re-scoring all engines' already-cached raw output (no re-extraction) shows docling and pymupdf unchanged (no bug hit their samples) while ndlocr_lite's mean rose from 0.542092 to 0.633549 (embedded_text_pair, scanned_pair) and would have raised mistral_ocr's first pass from 0.803 to 0.848342 (no_outline).
