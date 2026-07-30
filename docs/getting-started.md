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
ウィザードは選択したモデルを `data/models/` に初回ダウンロードし、`.env`には実際のローカルパスを保存します。
ネットワークに接続できない場合は、モデルを手動配置してから再実行してください。

PDF構造化で `Granite` を選んだ場合も、Graniteが失敗した際の安全なフォールバックとして通常の
`Docling`を使うため、ウィザードはDocling本体もプロジェクト環境へ導入します。

### セットアップ方式

- Minimal: APIや追加エンジンを使わず、PDFを平文として索引化
- Custom: Citation Network、PDF構造化、短いPDFと長いPDFの構造化エンジン、
  AI目次・OCR監査・クエリ拡張・階層要約・参考文献抽出を一項目ずつ選択

Customではページ境界の前後それぞれにDocling、Granite、Mistralを独立して指定できます。
例えば、短いPDFと長いPDFの両方をGraniteにする構成も選べます。Graniteの専用環境が
未導入の場合は、Graniteを選んだ後の確認に同意するとウィザードが作成します
（Apple Silicon搭載Macのみ）。

CustomではNDLOCR-Liteも検出します。未導入の場合は、約450MBの無料ローカルツールとして
公式GitHubリポジトリの検証済みタグからインストールするか確認します。NDLOCR-Liteは
日本語の明示的な再OCRと固定レイアウトEPUBに使われ、通常の画像PDFは構造化設定に応じて
Docling／Granite／Mistralで処理されます。

Tesseractと日本語言語データも検出します。Customで不足している場合は、Homebrewから
インストールするか確認します。Tesseractは画像PDF全体の主OCRではなく、フォント復号に
失敗したページだけを補助的に再読取します。

APIキーは入力中に画面へ表示されません。後からウィザードを再実行して構成を変更できます。

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

`.env`と接続設定を保存した後、SetupはDB構築が必要かどうかを判定して案内します
（真の初回は確認なしで進められ、既存DBがある状態でのプロファイル変更は`REBUILD`の
入力確認が必要です）。構築後は続けてDB監査（Zotero対象・原本coverage・DB整合性、
非破壊）の実行も案内されます。ここをスキップした場合は、後で
`Maintenance-Widget.command`から監査だけ実行できます。

有料の階層AI要約は、DB監査に合格した世代に対してのみ`Maintenance-Widget.command`から
生成・索引化できます（少量バッチの差分実行、または`SUMMARIZE`入力確認を伴う全件一括生成）。
DB監査に合格しない限り開始できません。Customで`PDF_AI_TOC_FAST_PATH_ENABLE=1` を選んだ
場合は、DB構築中にAI目次推定がAPIを使用し得る点にも注意してください。

## 5. 動作を確認する

DB構築（と、できればDB監査）まで完了した後にClaude Desktopを再起動し、`server_status`を
呼び出します。`status: ok`とコレクション件数が確認できれば完了です。

設定状態だけを秘密値なしで確認する場合:

```bash
uv run scripts/setup_wizard.py --status
```

問題がある場合は [トラブルシューティング](troubleshooting.md) を参照してください。
