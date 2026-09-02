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
- 要約バッチ報告: `data/quality/maintenance-summary-report.json`
- Citation Graph: `data/citation-graph.stdout.log`、`data/citation-graph.stderr.log`
- Citation Graph OAuth: `data/citation-graph-remote.stdout.log`、`data/citation-graph-remote.stderr.log`

## よくある問題

| 症状 | 対応 |
|---|---|
| ChromaDBが見つからない | `CHROMA_DIR`を確認し、ライブラリ更新を実行 |
| コレクションが空 | 初回インデックスを実行 |
| 埋め込みモデルエラー | ウィザードをオンラインで再実行してモデルを取得。`EMB_MODEL`が別マシンの絶対パスなら削除して再実行 |
| 更新後の検索結果が古い | MCPの `force_reload_index` を実行 |
| S2の429 | 時間を置いてCitation更新を再実行。必要ならAPIキーを設定 |
| LLM処理が停止する | APIキー/CLIログインと送信ポリシーを確認 |
| ローカルCitation Graphを開けない | `manage_citation_graph_service.py status`で`local_launch_agent`と`local_graph`を確認 |
| 外部Citation Graphを開けない | `local_oauth_proxy`、`public_oauth_proxy`、`tailscale funnel status`を確認 |
| Googleログイン後に403 | ログインした確認済みメールが`REMOTE_MCP_ALLOWED_GOOGLE_EMAILS`に含まれるか確認 |
| Googleからredirect URIエラー | `CITATION_GRAPH_PUBLIC_URL/auth/callback`をGoogle Cloud Consoleへ完全一致で登録 |
| 管理画面が「別のjobを実行中」と表示 | `/admin/`の実行中stepを確認し、完了を待つ。必要時だけ`STOP`で停止 |
| 更新状況が「再確認待ち」 | 書込みjob後の読み取り専用確認が待機または実行中。実行履歴を確認し、他jobがなければ完了を待つ |
| 更新状況が「期限切れ」 | 定期確認Agentと直近jobを確認する。`manage_citation_graph_service.py status`で`update_check_launch_agent=loaded`を確認し、必要なら画面から即時確認 |
| 管理jobが`interrupted` | OS再起動等でrunnerが失われた。ログ末尾とDB状態を確認してから同じ処理を再実行 |

Citation Graphの常駐状態と再起動:

```bash
uv run python scripts/manage_citation_graph_service.py status
uv run python scripts/manage_citation_graph_service.py restart
tailscale funnel status
```

LaunchAgentが未登録なら、7234・7244番を使う手動プロセスを終了してから`install`します。plistへ秘密値は
保存されないため、OAuth設定はGit管理外の`.env`を確認します。

ブラウザ管理jobの記録とログは`data/admin_jobs/`です。画面に出ない詳細を確認するときも、実行中PIDを
手作業でkillする前に管理画面の停止を使ってください。停止APIはrunner本人と確認できないPIDを拒否します。

## 日本語PDFの文字が崩れる

フォント復号に失敗したページだけは、Tesseractを追加すると改善する場合があります。
画像だけのPDF全体を処理する主OCRではありません。

Customで`Setup.command`を実行すると、Tesseract本体と日本語言語データを検出し、
不足していればHomebrewからインストールするか確認します。手動で導入する場合:

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
