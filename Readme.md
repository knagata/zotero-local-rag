# Zotero Local RAG (MCP サーバー)

ローカルのZoteroライブラリとそれに紐づくPDF/HTMLドキュメントを、LLM（Claude、Cursor、Zed、Windsurfなど）に直接接続する Model Context Protocol (MCP) サーバーです。

決定論的なフィルタリング（確実な絞り込み）を活用することで、コンテキストの文脈を最大化しつつ、トークン消費量を節約する設計となっています。

## ✨ 主な機能

- **ローカル環境で実行**: ローカルZoteroストレージから直接段落を抽出し、インデックス化します。リモートへの依存は埋め込みモデル（HuggingFace等で完全オフライン・キャッシュ可能）のみです。
- **段階的な検索**: RAGを3つのレイヤーに分割し、用途別に最適化しています。
  1. `search_items`: 本文テキストを返さず、メタデータとRRF（密度ベース）スコアのみによる書誌スクリーニング。
  2. `rag_search`: 意味検索による、ピンポイントな段落レベルのテキスト抽出。
  3. `get_chunk_context`: 指定した段落前後の文脈をデータベースから直接取得。
- **高度な最適化**:
  - **Reciprocal Rank Fusion (RRF)**: 複数クエリからの検索結果をシームレスに統合し、「キーワードの密度が高い」資料や段落を上位に引き上げます。
  - **既知IDの除外 (`exclude_chunk_ids`)**: LLMが直前のやり取りで既に読んだテキストのチャンクIDを自動的にブラックリスト化し、毎回の検索で100%新しい情報だけを取得してトークンを節約します。

## 🚀 インストールとセットアップ

このパッケージは、高速な依存関係解決のためにパッケージマネージャー `uv` に依存しています。

### 1. 必須要件
システムに [uv](https://github.com/astral-sh/uv) がインストールされていることを確認してください。

### 2. ワンクリック・セットアップ (推奨)
手動で環境変数を設定しなくても、ターミナルを開かずにダブルクリックだけでセットアップ可能なウィザードを用意しています。
*   **Mac ユーザー**: `Zotero_Local_RAG_Setup.command` をダブルクリック
*   **Windows ユーザー**: `Zotero_Local_RAG_Setup.bat` をダブルクリック

画面の指示に従って Enter を押すだけで、Zoteroフォルダの指定、埋め込みモデルのダウンロード、そしてChroma DBへのインデックス作成までが全自動で行われます。（設定内容は自動的に `.env` ファイルに保存されます。）

### 3. 環境変数 (手動設定の場合)
手動で設定する場合や、クライアントのJSON設定内で指定する場合は、以下の設定を必要とします。
- `CHROMA_DIR`: ローカルのベクトルデータベースが保存されるディレクトリの絶対パス（例: `/Users/username/data/zotero_chroma`）。
- `ZOTERO_LOCAL_API_BASE`: （オプション）プロキシなどを使用する場合のZotero DB APIのURL。

*注意: このMCPサーバーでファイルを検索させる前に、必ず上記のセットアップスクリプトまたは内部のインデクサースクリプトを使ってZoteroのファイルをChroma DBにインデックス化しておく必要があります。*

---

## 🛠 各クライアントでの使用方法

`zotero-local-rag` はユニバーサルなMCP仕様に準拠しているため、サポートされているどんなIDEやAIエージェントでも設定に追加するだけで実行可能です。

### Claude Desktop
`claude_desktop_config.json` に以下を追記してください：

```json
{
  "mcpServers": {
    "zotero-rag": {
      "command": "uvx",
      "args": [
        "zotero-local-rag"
      ],
      "env": {
        "CHROMA_DIR": "/absolute/path/to/your/chroma_dir",
        "EMB_PROFILE": "fast"
      }
    }
  }
}
```

### Cursor
Cursorの `Settings` -> `Features` -> `MCP` にて：
1. **+ Add new MCP server** をクリック
2. Name: `zotero-rag`
3. Type: `command`
4. Command:
   ```bash
   CHROMA_DIR=/absolute/path/to/your/chroma_dir uvx zotero-local-rag
   ```

### Zed
Zedの設定画面 (`settings.json`) にて：
```json
{
  "context_servers": {
    "zotero-rag": {
      "command": "uvx",
      "args": ["zotero-local-rag"],
      "env": {
        "CHROMA_DIR": "/absolute/path/to/your/chroma_dir"
      }
    }
  }
}
```

## 📖 LLM向けのベストプラクティス
このパッケージには、提供されたツールをLLMが最適に活用するための指示書である `ZOTERO_RAG_GUIDE.md` が同梱されています。自律的な反復リサーチタスクを行わせるために、この手順書をシステムプロンプトに組み込むか、事前に読み込ませることを推奨します。
