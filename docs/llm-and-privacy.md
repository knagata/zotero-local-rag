# LLMとプライバシー

[READMEへ戻る](../README.md)

## LLMを使う機能

- 日英クエリ拡張
- 文書構造からのbottom-up要約（資料・章）
- 無構造な長編PDFのAI目次推定
- PDF/HTMLの参考文献抽出
- 曖昧な参考文献候補の分類・整理

## プロバイダ

| プロバイダ | 準備 |
|---|---|
| DeepSeek | `DEEPSEEK_API_KEY` |
| Codex CLI | ローカルでログイン済みのCodex CLI |
| Claude CLI | ローカルでログイン済みのClaude CLI |
| OpenAI互換 | `LLM_OPENAI_BASE_URL`と必要に応じてAPIキー |
| Anthropic API | `ANTHROPIC_API_KEY`と追加依存 |

モデルは用途別の3段階で指定します。

```dotenv
LLM_CHEAP=deepseek:deepseek-v4-flash
LLM_STANDARD=deepseek:deepseek-v4-pro
LLM_REVIEW=deepseek:deepseek-v4-pro
```

`LLM_CHEAP` は要約やクエリ拡張などの大量処理、`LLM_STANDARD` は通常作業と一次フォールバック、`LLM_REVIEW` は品質確認と最終フォールバックに使います。

## クラウド送信ポリシー

資料単位のクラウド除外タグは**2026-07-27に撤去しました**。

索引に入れた資料は `rag_search` がチャンクとして返し、それはアシスタントへ渡ります。
つまり「索引には入れるがクラウドへは出さない」は成立しません。意味のある判断は
**その資料をライブラリに入れるかどうか**であり、資料単位の拒否ではありませんでした。

実害もありました。タグ照会はZoteroへのHTTP問い合わせで、失敗時はfail-closedで
「送信不可」を返すため、**Zoteroが一瞬落ちただけで全スキャンPDFが不要な再OCRへ回る**
という挙動になっていました。

クラウド利用の可否は機能フラグで制御します。**既定はすべて0**で、`Setup.command` の
Custom設定で機能ごとに明示的に有効化されます。課金が発生するのはDeepSeek（LLM）と
Mistral OCRだけで、Semantic Scholarの鍵は無料です（ただし鍵なしではrate limitで
実用にならないため、鍵が無ければCitation Networkは有効化しません）。

| フラグ | 対象 |
|---|---|
| `LLM_CHEAP` / `LLM_STANDARD` / `LLM_REVIEW` | 要約・参照抽出・品質判定 |
| `PDF_AI_TOC_FAST_PATH_ENABLE` | AI目次（冒頭20頁を送信） |
| `PDF_MISTRAL_TOC_QUEUE_ENABLE` | Mistral OCR Batch（ファイル全体を送信） |

Customで `PDF_AI_TOC_FAST_PATH_ENABLE=1` を選ぶAI目次推定も有料です。これは階層要約とは
別の処理で、DB構築中にも実行され得ます。`Setup.command`でDB構築を始める前に、この機能を
有効にするか決めてください。DB整合性の確認後にのみ階層AI要約へ進む場合は、構築時は`0`にします。

## LLM要約

文書構造V3の葉からbottom-upにAI要約を生成し、`__sum_node` 検索索引を更新します。

```bash
uv run python scripts/build_structure_summaries.py --all --mode llm --embed \
  --database-gate data/quality/server_database_gate.json
```

既定のローカル抽出型要約は外部送信しません。

```bash
uv run python scripts/build_structure_summaries.py --all --mode extractive --embed
```

全件AI要約の生成はユーザーが明示承認した場合だけ実行します（[環境設定](configuration.md#メンテナンス時のai要約) 参照）。
サーバーではさらに、フェーズ2のZotero・原本・DB監査が現在のDB世代に対して合格していなければ、
フェーズ3のAI要約CLIはAPIを呼び出さず停止します。

## 参考文献抽出

まず書き込みなしで確認します。

```bash
uv run python -m src.extract_references --item ITEMKEY
```

レビューキューへ保存:

```bash
uv run python -m src.extract_references --item ITEMKEY --stage
uv run python scripts/review_references.py list --status pending --limit 20
```

モデル出力だけでWorkを統合せず、原文、著者、書名、刊年、DOI/ISBN等を決定的に再検証します。不確実な候補は別著作として保留します。

英語学術論文向けのGROBID enrichmentは埋め込みとは独立して実行します。詳細は [開発・保守](development.md) を参照してください。
