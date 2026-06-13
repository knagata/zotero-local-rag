# Zotero Local RAG (MCP サーバー)

ローカルのZoteroライブラリとそれに紐づくPDF/HTML/EPUBドキュメントを、LLM（Claude、Cursor、Zed、Windsurfなど）に直接接続する Model Context Protocol (MCP) サーバーです。

決定論的なフィルタリング（確実な絞り込み）を活用することで、コンテキストの文脈を最大化しつつ、トークン消費量を節約する設計となっています。

## ✨ 主な機能

- **ローカル環境で実行**: ローカルZoteroストレージから直接段落を抽出し、インデックス化します。リモートへの依存は埋め込みモデル（HuggingFace等で完全オフライン・キャッシュ可能）のみです。
- **段階的な検索**: 用途に合わせて検索レイヤーを最適化しています。
  1. `search_zotero_items`: ベクトルインデックスを介さず、Zotero Local APIからタイトル・著者名・出版年などを超高速で直接検索（部分一致など）できる書誌検索。
  2. `search_items`: ベクトル検索を利用しつつ本文テキストを返さず、メタデータとRRF（密度ベース）スコアのみで関連資料をスクリーニング。
  3. `rag_search`: 意味検索（セマンティック検索）による、ピンポイントな段落レベルのテキスト抽出。
  4. `get_chunk_context`: 指定した段落前後の文脈をデータベースから直接取得。
- **引用・被引用ネットワーク分析 (Semantic Scholar連携)**:
  - `build_citation_network`: 指定した資料について、EPUBからの引用先抽出（References）とSemantic Scholarからの被引用取得（Citations）を両方とも一括で実行してデータベースを構築。
  - `get_references_for_item` / `get_chunk_references`: 特定の段落チャンクが「どの文献を引用しているか」の参照先ネットワークを取得・分析。
  - `get_cited_chunks_for_item` / `get_citations_for_chunk`: 特定の段落チャンクが「外部の論文からどのような文脈で引用されているか」の被引用ネットワークを取得・分析。
- **高度な最適化**:
  - **Reciprocal Rank Fusion (RRF)**: 複数クエリからの検索結果をシームレスに統合し、「キーワードの密度が高い」資料や段落を上位に引き上げます。
  - **既知IDの除外 (`exclude_chunk_ids`)**: LLMが直前のやり取りで既に読んだテキストのチャンクIDを自動的にブラックリスト化し、毎回の検索で100%新しい情報だけを取得してトークンを節約します。
- **サーバー状態の確認 (`server_status`)**: ChromaDBの接続状態・コレクション数・埋め込みモデルの設定をClaudeから直接確認できます。
- **インデックスの強制リロード (`force_reload_index`)**: インデクサー実行後に検索結果が反映されない場合、ChromaDBのインデックスを強制再読み込みします。
- **デバッグログの取得 (`get_debug_logs`)**: サーバーのログファイルを参照し、エラーやイベントのトレースバックをClaudeから直接確認できます。

## 🚀 インストールとセットアップ

このパッケージは、高速な依存関係解決のためにパッケージマネージャー `uv` に依存しています。

### 1. 必須要件

システムに [uv](https://github.com/astral-sh/uv) がインストールされていることを確認してください。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. ワンクリック・セットアップ (推奨)

手動で環境変数を設定しなくても、ターミナルを開かずにダブルクリックだけでセットアップ可能なウィザードを用意しています。

- **Mac ユーザー**: `Setup.command` をダブルクリック
- **Windows ユーザー**: `Setup.bat` をダブルクリック

ウィザードでは以下のステップを順に案内します。Enterを押すだけで進めます（2回目以降の起動では設定変更をスキップできます）。

1. **Zoteroデータフォルダの指定** — デフォルトは `~/Zotero`
2. **埋め込みモデルの選択** — `fast`（デフォルト・軽量多言語）または `bge`（BGE-M3・高精度多言語）
3. **Claude DesktopへのMCP設定の自動登録** — `claude_desktop_config.json` に正しいコマンドを自動で書き込みます（デフォルトはスキップ）
4. **埋め込みインデクサーの実行** — ZoteroのPDF/HTML/EPUBをベクトル化してChroma DBに保存します（デフォルトは実行）

> **埋め込みの更新**: Zoteroに新しい文献を追加した後は、ウィザードを再起動してEnterを連打するだけで差分インデックスが更新されます。

### 3. アプリケーションの更新（新バージョンへの移行）

GitHubから最新バージョンを自動でダウンロードして上書きする更新スクリプトを用意しています。`.env` とインデックスデータ（`data/`）は保持されます。

- **Mac ユーザー**: `Software-Update.command` をダブルクリック
- **Windows ユーザー**: `Software-Update.bat` をダブルクリック

更新後はClaude Desktopを再起動してください。

### 4. 引用ネットワークの更新（Semantic Scholar連携）

Zotero文献の被引用情報（どの論文に引用されているか）をSemantic Scholar APIから取得してデータベースに保存します。`build_citation_network` ツールをMCP経由でClaudeに依頼するか、以下のスクリプトから手動で実行できます。

- **Mac ユーザー**: `Citation-Update.command` をダブルクリック
- **Windows ユーザー**: `Citation-Update.bat` をダブルクリック

実行時は対象（特定アイテム指定 / 全アイテム一括 / スキップ済みEPUB参照の再解決）を選択するメニューが表示されます。Semantic Scholar APIのレート制限（APIキーあり: 2.5秒間隔、なし: 3.5秒間隔）により、大規模ライブラリの全件更新は時間がかかります。

> **S2 APIキー**: `S2_API_KEY` を `.env` に設定するとレート制限が緩和されます。また `ZOTERO_USER_ID` + `ZOTERO_API_KEY` を設定すると、解決した DOI が Zotero ライブラリに自動書き戻しされます（詳細は下記の環境変数表を参照）。

> **詳細ヘルプ**: メニューの選び方、ステータスの意味、429エラーやエラー回復の仕組みについては [`CITATION_UPDATE_GUIDE.md`](./CITATION_UPDATE_GUIDE.md) を参照してください。エラーが出た場合も Force rebuild は不要で、メニュー3の再実行だけで該当分が自動回収されます。

### 5. 環境変数（手動設定の場合）

手動でMCP設定を記述する場合に必要な環境変数です。

| 変数名 | 説明 | デフォルト |
|---|---|---|
| `CHROMA_DIR` | ChromaDBの保存先（絶対パス） | `<プロジェクトルート>/data/chroma` |
| `EMB_PROFILE` | 埋め込みモデルプロファイル（`fast` または `bge`） | `fast` |
| `EMB_MODEL` | 埋め込みモデルの明示的な指定（ローカルパスまたはHugging Face ID） | プロファイルから自動選択 |
| `EMB_DEVICE` | 推論デバイス（`cpu`、`mps`、`cuda`） | `cpu`（bge: macは`mps`） |
| `HF_HUB_OFFLINE` | `1` に設定するとHugging Faceへのアクセスを無効化 | — |
| `ZOTERO_LOCAL_API_BASE` | Zotero Local HTTP APIのベースURL | `http://127.0.0.1:23119/api` |
| `ZOTERO_LOCAL_API_PREFIX` | ZoteroローカルAPIのパスプレフィックス | `users/0` |
| `ZOTERO_API_KEY` | Zotero Web APIキー（`ZOTERO_USER_ID` と合わせて設定すると DOI 書き戻しが有効になる） | — |
| `ZOTERO_USER_ID` | Zotero ユーザーID（数値。zotero.org/settings/keys で確認可能） | — |
| `S2_API_KEY` | Semantic Scholar APIキー（設定すると間隔が 3.5s → 2.5s に短縮） | — |

---

## 🛠 各クライアントでの使用方法

> **注意**: このMCPサーバーはPyPIには公開されていません。`uvx zotero-local-rag` は動作しません。ローカルのプロジェクトディレクトリを指定した `uv run` を使用してください。

### Claude Desktop

`claude_desktop_config.json` に以下を追記してください。セットアップウィザードを使った場合は自動で登録されます。

```json
{
  "mcpServers": {
    "zotero-rag": {
      "command": "/Users/<username>/.local/bin/uv",
      "args": [
        "--directory",
        "/absolute/path/to/zotero-local-rag",
        "run",
        "python",
        "-u",
        "src/rag_mcp_server.py"
      ],
      "env": {
        "CHROMA_DIR": "/absolute/path/to/zotero-local-rag/data/chroma",
        "EMB_PROFILE": "fast"
      }
    }
  }
}
```

`command` のuvのパスは `which uv` で確認できます。

### Cursor

Cursorの `Settings` → `Features` → `MCP` にて：

1. **+ Add new MCP server** をクリック
2. Name: `zotero-rag`
3. Type: `command`
4. Command:
   ```bash
   uv --directory /absolute/path/to/zotero-local-rag run python -u src/rag_mcp_server.py
   ```
   環境変数 `CHROMA_DIR` は別途 `env` フィールドで指定してください。

### Zed

Zedの設定画面 (`settings.json`) にて：

```json
{
  "context_servers": {
    "zotero-rag": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/zotero-local-rag",
        "run",
        "python",
        "-u",
        "src/rag_mcp_server.py"
      ],
      "env": {
        "CHROMA_DIR": "/absolute/path/to/zotero-local-rag/data/chroma",
        "EMB_PROFILE": "fast"
      }
    }
  }
}
```

---

## 🔍 サーバーが反応しない場合

Claudeからサーバーが応答しないと感じた場合は、`server_status` ツールを呼び出してください。以下の情報が返ります：

- ChromaDBのパスが存在するか
- コレクション名とドキュメント数
- 埋め込みモデルの設定
- エラーがあれば具体的な原因と対処法

一般的な原因と対処法：

| 症状 | 原因 | 対処 |
|---|---|---|
| `chroma_dir_exists: false` | インデクサーが未実行 | ウィザードを起動してインデクサーを実行 |
| `No collections found` | インデクサーが途中で失敗 | ウィザードを再実行 |
| `EMB resolve error` | モデルがオフラインキャッシュにない | 一度オンラインで実行してモデルをキャッシュ |

---

## 📖 LLM向けのベストプラクティス

このパッケージには、提供されたツールをLLMが最適に活用するための指示書 `ZOTERO_RAG_GUIDE.md` が同梱されています。自律的な反復リサーチタスクを行わせるために、この手順書をシステムプロンプトに組み込むか、事前に読み込ませることを推奨します。

---

## 🗺 アーキテクチャ

システムの全体図・処理パイプライン・モジュール一覧・データストア構成は [`ARCHITECTURE.md`](./ARCHITECTURE.md) を参照してください。
