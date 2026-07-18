# トラブルシューティング

[READMEへ戻る](../README.md)

## 最初に確認する

MCPから `server_status` を呼び出します。ターミナルでは設定状態を確認できます。

```bash
uv run scripts/setup_wizard.py --status
```

ログ:

- MCPツール: `get_debug_logs`
- ファイル: `data/zotero-rag.log`
- 夜間処理: `data/nightly_summaries.log`

## よくある問題

| 症状 | 対応 |
|---|---|
| ChromaDBが見つからない | `CHROMA_DIR`を確認し、ライブラリ更新を実行 |
| コレクションが空 | 初回インデックスを実行 |
| 埋め込みモデルエラー | `EMB_PROFILE`、`EMB_MODEL`、オフライン設定を確認 |
| 更新後の検索結果が古い | MCPの `force_reload_index` を実行 |
| S2の429 | 時間を置いてCitation更新を再実行。必要ならAPIキーを設定 |
| LLM処理が停止する | APIキー/CLIログインと送信ポリシーを確認 |

## 日本語PDFの文字が崩れる

Tesseractを追加できます。

```bash
# macOS
brew install tesseract tesseract-lang

```

索引品質の確認:

```bash
uv run src/index_from_zotero.py --check-quality --progress
```

高品質パーサーで再処理:

```bash
uv run src/index_from_zotero.py --reparse-corrupted --progress
```

## 更新を途中で止めた

ライブラリ更新はロックとmanifest、Citation更新はDBステータスを使って再開できます。通常の統合commandをもう一度実行してください。

## それでも解決しない

`server_status`、最新ログ、実行したコマンド、エラーメッセージを添えてIssueを作成してください。秘密値や本文データは含めないでください。
