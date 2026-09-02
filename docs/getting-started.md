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

### 外部のClaudeから接続する（任意）

`rag_mcp_server.py`のstdio接続は変更せず、認証付きHTTP専用の
`src/rag_mcp_http_server.py`を併設できます。検索実装とデータベースは両者で共通です。
Remote MCPは認証なしでは起動せず、HTTP待受も`127.0.0.1`に限定されます。

1. Tailscale Funnelで使うMacの固定`https://...ts.net`名を確認します。
2. Google Cloud ConsoleでOAuth同意画面と「ウェブアプリケーション」のOAuthクライアントを作り、
   承認済みのリダイレクトURIを
   `https://<Macの名前>.<tailnet>.ts.net/auth/callback`にします。
3. `.env`へ次を保存します。秘密値をGitへ追加しないでください。

```dotenv
REMOTE_MCP_PUBLIC_URL=https://<Macの名前>.<tailnet>.ts.net
REMOTE_MCP_GOOGLE_CLIENT_ID=<Google OAuthクライアントのClient ID>
REMOTE_MCP_GOOGLE_CLIENT_SECRET=<Google OAuthクライアントのClient secret>
REMOTE_MCP_ALLOWED_GOOGLE_EMAILS=<許可する自分のGoogleメールアドレス>
REMOTE_MCP_HOST=127.0.0.1
REMOTE_MCP_PORT=8000
```

4. HTTP MCPを起動します。

```bash
uv run python -u src/rag_mcp_http_server.py
```

5. 別のターミナルでFunnelを常駐設定します。

```bash
tailscale funnel --bg 8000
```

ClaudeのRemote Connectorには
`https://<Macの名前>.<tailnet>.ts.net/mcp`を登録します。Funnel URLは公開URLなので、
Google OAuthと`REMOTE_MCP_ALLOWED_GOOGLE_EMAILS`の両方を外さないでください。OAuth同意画面が
テスト運用中なら、このメールアドレスをGoogle側のテストユーザーにも追加します。MacのTailscale
クライアントがFunnelに対応する導入形態であることも事前に確認してください。

#### Remote MCPを自動起動する

接続確認後はmacOS LaunchAgentへ登録します。Googleの秘密値はplistへ複製せず、起動時に
Git追跡外の`.env`から読みます。

```bash
uv run python scripts/manage_remote_mcp_service.py install
uv run python scripts/manage_remote_mcp_service.py status
```

LaunchAgentはログイン時に起動し、異常終了時は10秒以上空けて再起動します。運用コマンド:

```bash
uv run python scripts/manage_remote_mcp_service.py restart
uv run python scripts/manage_remote_mcp_service.py uninstall
```

ログは`data/remote-mcp.stdout.log`と`data/remote-mcp.stderr.log`です。Tailscale Funnelの
`--bg`設定はTailscaleが保持するため、MCPプロセスとは別にHerdrで常駐させる必要はありません。
リポジトリがmacOSの`Documents`配下にありLaunchAgentのPythonがファイルアクセスを許可されて
いない場合、installはhealth確認に失敗してAgentを自動で外します。システム設定の
「プライバシーとセキュリティ」→「フルディスクアクセス」で表示されたPython実行ファイルを
許可するか、リポジトリを`Documents`外へ移してからinstallを再実行してください。

### Citation Graphを外部ブラウザへ公開する（任意）

Citation Graphも、従来のlocalhost表示を残したまま、別ポートのTailscale FunnelとGoogle OAuthで
限定公開できます。Remote MCPのGoogle Client ID／Secretと許可メールを共用しますが、公開originと
callbackは8443番を含む別URIです。設定、自動起動、動作確認は
[Show Citation Networkガイド](show-citation-network.md#外部コンピュータからgoogleログインで開く)を
参照してください。

## 4. サーバー用の設定だけを行う

サーバーへ配置する場合は、Claude Desktop登録を行わない `--server` を使います。

```bash
uv run scripts/setup_wizard.py --server
```

`.env`と接続設定を保存した後、SetupはDB構築が必要かどうかを判定して案内します
（真の初回は確認なしで進められ、既存DBがある場合は設定変更の有無にかかわらず常に
再構築の案内が出ます。プロファイル変更時は`REBUILD`の入力が必須、それ以外の設定
変更時は任意でEnterによりスキップできます）。構築後は続けてDB監査（Zotero対象・原本coverage・DB整合性、
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
