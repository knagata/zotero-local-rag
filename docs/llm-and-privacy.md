# LLMとプライバシー

[READMEへ戻る](../README.md)

## LLMを使う機能

- 日英クエリ拡張
- 高品質な資料・章要約
- 逐語根拠付きの事例抽出
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

タスク別にモデルを変えられます。

```dotenv
LLM_DEFAULT=deepseek:deepseek-v4-pro
LLM_EXPAND=deepseek:deepseek-v4-pro
LLM_SUMMARY=codex_cli:gpt-5.6-luna
LLM_EXTRACT=deepseek:deepseek-v4-pro
```

## クラウド送信ポリシー

推奨はZoteroタグによるブラックリストです。

```dotenv
SUMMARY_EXCLUDE_TAGS=private,confidential,no-cloud
EXTRACT_EXCLUDE_TAGS=private,confidential,no-cloud
```

タグ確認ができない場合も送信しません。全資料を送信してよい場合だけ、明示的に設定します。

```dotenv
SUMMARY_ALLOW_CLOUD_ALL=1
EXTRACT_ALLOW_CLOUD_ALL=1
```

除外タグはGit管理外の `.env.policy` に分離できます。

## LLM要約

```bash
uv run python -m src.build_summaries --mode llm
```

既定のローカル抽出型要約は外部送信しません。

```bash
uv run python -m src.build_summaries
```

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

## 夜間要約

夜間実行は初期状態で無効です。利用する場合は [環境設定](configuration.md#夜間実行) を参照してください。
