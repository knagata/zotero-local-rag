# 初回セットアップ

[READMEへ戻る](../README.md)

## 1. 必要なもの

- Zotero Desktop
- Python 3.10
- [uv](https://docs.astral.sh/uv/)
- このリポジトリのローカルコピー

macOSでuvを導入する例:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. ウィザードを起動する

macOSでは `Setup.command` をダブルクリックします。ターミナルからは次を実行します。

```bash
uv run scripts/setup_wizard.py
```

### Zoteroデータフォルダ

通常は `~/Zotero` です。フォルダ内に `zotero.sqlite` と `storage/` があることを確認してください。

### 埋め込みモデル

| 選択 | 向いている用途 |
|---|---|
| `fast` | 通常利用。軽量で、日英を含む多言語検索に対応 |
| `bge` | より高品質な多言語検索。ダウンロードと実行負荷が大きい |

最初は `fast` を推奨します。モデルを変更すると別のベクトル索引が必要です。

### 機能段階

- Core: ローカル検索だけを設定
- Citation Network: Semantic Scholar連携を追加（APIキー必須）
- LLM-assisted: 要約や参考文献抽出にLLMを追加

後からウィザードを再実行して拡張できます。

Citation Networkを使う場合は、事前に[Semantic Scholar APIページ](https://www.semanticscholar.org/product/api)の「Request an API Key」からキーを申請してください。キーなしの共有枠は実用上の制限が厳しいため、このプロジェクトでは必須として扱います。

## 3. Claude Desktopへ接続する

ウィザードでClaude Desktopへの登録を選ぶのが簡単です。手動登録する場合のサーバーコマンド:

```bash
uv --directory /absolute/path/to/zotero-local-rag run python -u src/rag_mcp_server.py
```

Claude Desktopの設定例:

```json
{
  "mcpServers": {
    "zotero-rag": {
      "command": "/absolute/path/to/uv",
      "args": [
        "--directory", "/absolute/path/to/zotero-local-rag",
        "run", "python", "-u", "src/rag_mcp_server.py"
      ]
    }
  }
}
```

このプロジェクトはPyPI未公開なので、`uvx zotero-local-rag`ではなくローカルディレクトリから実行します。

## 4. サーバー用の設定だけを行う

サーバーへ配置する場合は、Claude Desktop登録を行わない `--server` を使います。

```bash
uv run scripts/setup_wizard.py --server
```

Setupは`.env`と接続設定を作るだけで、DB構築・埋め込み・AI目次・OCR・階層AI要約を実行しません。
これらは料金や長時間処理を伴い得るため、サーバー上で次のworkflowを順番に実行します。

```bash
bash Server-Database-Workflow.command
```

1. V3 DBをゼロから構築する（階層要約は生成しない）
2. Zoteroの対象・原本coverage・DB整合性を監査する
3. 合格したDB世代に対してのみ、有料の階層AI要約を生成・索引化する
4. 階層要約と要約索引を監査する

フェーズ2が成功しない限りフェーズ3は開始できません。`PDF_AI_TOC_FAST_PATH_ENABLE=1` の
Full構成では、フェーズ1中にAI目次推定がAPIを使用し得る点にも注意してください。

## 5. 動作を確認する

フェーズ2まで完了した後にClaude Desktopを再起動し、`server_status`を呼び出します。
`status: ok`とコレクション件数が確認できれば完了です。

設定状態だけを秘密値なしで確認する場合:

```bash
uv run scripts/setup_wizard.py --status
```

問題がある場合は [トラブルシューティング](troubleshooting.md) を参照してください。
