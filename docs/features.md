# 機能と必要な設定

[READMEへ戻る](../README.md)

## 一覧

| 機能 | Core | Citation | LLM |
|---|:---:|:---:|:---:|
| Zotero書誌検索 | ✓ | ✓ | ✓ |
| 本文の意味検索・語彙検索 | ✓ | ✓ | ✓ |
| 前後文脈・目次 | ✓ | ✓ | ✓ |
| ローカル抽出型要約 | ✓ | ✓ | ✓ |
| 関連資料検索 | ✓ | ✓ | ✓ |
| 被引用・参照文献 | — | ✓ | ✓ |
| 引用グラフ・未所蔵文献候補 | — | ✓ | ✓ |
| LLMクエリ拡張 | — | — | ✓ |
| LLM要約（文書構造からのbottom-up） | — | — | ✓ |
| PDF/HTMLの高品質参考文献抽出・AI目次 | — | — | ✓ |

## Core local RAG

必要なのはローカル埋め込みモデルだけです。索引作成後は、Zotero書誌、PDF・HTML・EPUB本文、Zoteroノートを検索できます。

主なMCPツール:

- `search_zotero_items`: タイトル・著者・年で検索
- `search_items`: 関連資料を本文なしで絞り込み
- `rag_search`: 関連段落を検索（ベクトル＋FTS語彙のhybrid）
- `get_chunk_context`: 前後段落を取得
- `get_document_outline`: 目次を取得
- `hierarchical_search`: 資料・章・段落を横断検索

本文抽出はソース種別ごとにルーティングされます。EPUB/HTMLはDOM構造化抽出でzone（本文・参考文献・脚注等）を付与し、参照や注は1エントリ=1チャンクで境界保持します。PDFは埋め込みテキストとoutlineがあればPyMuPDF、無構造の長編はAI目次推定（LLM設定時）、それ以外はDoclingです。スキャンPDFはOCR層なし、またはOCR層品質gate不合格なら、30頁未満はDocling、30頁以上はMistral OCR Batchへ保留します。RapidOCRは通常PDFでは使わず、固定レイアウトEPUBと明示re-OCRに限定します。

## Citation Network

Semantic Scholar、OpenAlex等へのインターネット接続と `S2_API_KEY` が必要です。

できること:

- 外部論文からの被引用取得
- 参考文献・脚注・巻末注の抽出と照合（取り込み時に境界保持したV3チャンクから復元）
- 参照・被引用グラフ
- 複数資料から参照される未所蔵文献の提案
- 英語学術論文のGROBID enrichment（任意・review-only）

詳細: [Citation Network](citation-network.md)

## LLM-assisted

DeepSeek API、OpenAI互換サーバー、Anthropic/Gemini等を利用します。

LLMなしでも構造的抽出とローカル抽出型要約は可能です。LLMは、無構造な長編PDFのAI目次推定、文書構造からのbottom-up要約、参考文献抽出、クエリ展開の品質を高めます。

本文をクラウドへ送る処理は、送信ポリシーがない場合は停止します。詳細: [LLMとプライバシー](llm-and-privacy.md)
