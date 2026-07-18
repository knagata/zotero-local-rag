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
| LLM要約・事例抽出 | — | — | ✓ |
| PDF/HTMLの高品質参考文献抽出 | — | — | ✓ |

## Core local RAG

必要なのはローカル埋め込みモデルだけです。索引作成後は、Zotero書誌、PDF・HTML・EPUB本文、Zoteroノートを検索できます。

主なMCPツール:

- `search_zotero_items`: タイトル・著者・年で検索
- `search_items`: 関連資料を本文なしで絞り込み
- `rag_search`: 関連段落を検索
- `get_chunk_context`: 前後段落を取得
- `get_document_outline`: 目次を取得
- `hierarchical_search`: 資料・章・段落を横断検索

## Citation Network

Semantic Scholar、OpenAlex等へのインターネット接続と `S2_API_KEY` が必要です。

できること:

- 外部論文からの被引用取得
- EPUBの脚注・巻末注・参考文献の抽出と照合
- 参照・被引用グラフ
- 複数資料から参照される未所蔵文献の提案

詳細: [Citation Network](citation-network.md)

## LLM-assisted

DeepSeek API、Codex CLI、Claude CLI、OpenAI互換サーバー等を利用します。

LLMなしでもEPUBの構造的抽出と簡易参考文献抽出は可能です。LLMは、複雑なPDF/HTML、注記、要約、クエリ展開の品質を高めます。

本文をクラウドへ送る処理は、送信ポリシーがない場合は停止します。詳細: [LLMとプライバシー](llm-and-privacy.md)
