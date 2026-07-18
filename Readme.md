# Zotero Local RAG

ZoteroのローカルライブラリにあるPDF・HTML・EPUBを索引化し、Claude Desktop、Cursor、ZedなどからMCPで検索・参照するためのローカルRAGサーバーです。基本検索はローカルで完結し、外部APIやLLMは必要な機能だけ後から追加できます。

## 機能は三段階です

| 段階 | 追加要件 | 主にできること |
|---|---|---|
| 1. Core local RAG | ローカル埋め込みモデル | Zotero書誌検索、本文の意味検索・語彙検索、前後文脈、目次、資料・章の抽出型要約、関連資料検索 |
| 2. Citation Network | インターネット接続。`S2_API_KEY`は任意 | Semantic Scholarの被引用取得、EPUBの参照文献照合、引用グラフ、未所蔵文献候補、Crossref・CiNii・NDLによる書誌照合 |
| 3. LLM-assisted | DeepSeek API、Codex CLI、Claude CLI、またはローカル互換API | クエリ拡張、LLM要約・事例抽出、PDF/HTMLからの高品質な参考文献抽出、曖昧な書誌候補の整理 |

この区分には二つの注意点があります。

- Semantic ScholarはAPIキーなしでも共有枠で動作します。キーは速度と安定性のために推奨されますが、必須ではありません。
- EPUBの構造的な参照抽出と簡易ヒューリスティック抽出はLLMなしでも可能です。LLMはPDF/HTMLや複雑な注記の精度、自動要約・整理を高める追加機能です。

外部機能はCoreの上に追加されます。最初はCoreだけで開始し、必要になった時点でセットアップウィザードを再実行して拡張できます。

## クイックスタート

### 必須要件

- Python 3.10
- [uv](https://docs.astral.sh/uv/)
- Zoteroデスクトップとローカルライブラリ

macOSでuvを導入する例:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### セットアップウィザード（推奨）

macOSでは `Setup.command` をダブルクリックします。ターミナルからは次を実行できます。

```bash
uv run scripts/setup_wizard.py
```

ウィザードでは次の項目を選べます。

1. Zoteroデータフォルダ
2. 埋め込みモデル: `fast`（軽量）または `bge`（BGE-M3、高品質・重量級）
3. 機能段階: Core、Citation Network、LLM-assisted
4. Citationを選んだ場合の任意のSemantic Scholar APIキー
5. LLMを選んだ場合のプロバイダとクラウド送信ポリシー
6. Claude DesktopへのMCP登録
7. 初回インデックスの作成

設定はGit管理外の `.env` に機能別のまとまりで保存されます。再実行すると現在の段階と設定済み機能を確認でき、既存の未知の設定も保持されます。APIキーは入力中も画面に表示されません。

秘密値を表示せず現在の設定状態だけを確認するには、次を使います。

```bash
uv run scripts/setup_wizard.py --status
```

### 初回動作確認

インデックス完了後にClaude Desktopなどを再起動し、MCPから `server_status` を呼び出します。次に、例えば以下を依頼します。

```text
Zoteroから「修復的司法」に関係する資料を検索して
```

Coreだけでも `search_zotero_items`、`search_items`、`rag_search`、`get_chunk_context`、`get_document_outline` を利用できます。

## 日常の使い方

### Zoteroライブラリを更新する

Zoteroへ資料を追加・変更した後は、macOSで `Library-Update.command` をダブルクリックします。通常はメニュー1で差分だけ索引化されます。

```bash
uv run src/index_from_zotero.py --progress
```

### 抽出型要約を更新する（Core）

`Summary-Update.command` をダブルクリックするか、次を実行します。本文は外部へ送信されません。

```bash
uv run python -m src.build_summaries
```

### Citation Networkを更新する（Level 2）

`Citation-Update.command` をダブルクリックします。通常はメニュー3を選ぶと未処理・エラー分だけ再開します。

```bash
uv run src/update_citations.py --all
```

処理にはSemantic Scholar/OpenAlexへの接続が必要です。`S2_API_KEY`がなくても動作しますが、共有枠のため低速です。429エラーや再開方法は [CITATION_UPDATE_GUIDE.md](CITATION_UPDATE_GUIDE.md) を参照してください。

引用グラフは `Show-Citation-Graph.command` で開けます。

### LLM機能を使う（Level 3）

LLM要約:

```bash
uv run python -m src.build_summaries --mode llm
```

参考文献抽出のプレビュー:

```bash
uv run python -m src.extract_references --item ITEMKEY
```

参考文献抽出の通常操作は `Citation-Update.command` のメニュー6（プレビュー）と7（保存）からも実行できます。保存前後に原文根拠と安定識別子を検証し、不確実な候補は審査キューへ保留します。

利用可能なMCPツールと推奨検索手順は [ZOTERO_RAG_GUIDE.md](ZOTERO_RAG_GUIDE.md) を参照してください。

## クラウド送信の安全設定

LLM要約とLLM参考文献抽出は、送信ポリシーがない場合にfail-closedで停止します。推奨設定はZoteroタグによるブラックリストです。

```dotenv
SUMMARY_EXCLUDE_TAGS=private,confidential,no-cloud
EXTRACT_EXCLUDE_TAGS=private,confidential,no-cloud
```

すべての資料を送信してよい場合だけ、明示的に次を設定します。

```dotenv
SUMMARY_ALLOW_CLOUD_ALL=1
EXTRACT_ALLOW_CLOUD_ALL=1
```

秘密値と分けたい場合は、Git管理外の `.env.policy` に除外タグを保存できます。タグ確認ができない場合も、安全のため送信は停止します。

## `.env` の管理

通常はウィザードで管理してください。手動設定のひな形は [.env.example](.env.example) にあります。

### Level 1: Core

| 変数 | 用途 | 既定値 |
|---|---|---|
| `FEATURE_LEVEL` | ウィザードで選んだ設定段階。機能制限ではなく管理用マーカー | `core` |
| `ZOTERO_DATA_DIR` | `zotero.sqlite` と `storage/` がある場所 | `~/Zotero` |
| `CHROMA_DIR` | ベクトル索引の保存先 | `data/chroma` |
| `EMB_PROFILE` | `fast` または `bge` | `fast` |
| `EMB_MODEL` | 埋め込みモデルのローカルパスまたはHugging Face ID | プロファイルから選択 |
| `EMB_DEVICE` | `cpu`、`mps`、`cuda` | 環境から選択 |
| `HF_HUB_OFFLINE` | `1`でHugging Faceへのアクセスを禁止 | 任意 |

`FEATURE_LEVEL`は説明・管理用であり、セキュリティ境界ではありません。外部送信の可否はAPIキーと送信ポリシーで決まります。

### Level 2: Citation Network

| 変数 | 用途 | 必須性 |
|---|---|---|
| `S2_API_KEY` | Semantic Scholarの専用枠 | 任意。未設定時は共有枠 |
| `ZOTERO_USER_ID` + `ZOTERO_API_KEY` | OpenAlexで解決したDOIをZotero Web APIへ書き戻す | 任意 |
| `CINII_APP_ID` | CiNii Research v2による書誌照合 | 任意 |
| `CROSSREF_MAILTO` | Crossref polite pool用連絡先 | 任意 |

### Level 3: LLM-assisted

タスク設定は `provider:model` 形式です。カンマ区切りでフォールバック順を指定できます。

```dotenv
LLM_DEFAULT=deepseek:deepseek-v4-pro
LLM_EXPAND=deepseek:deepseek-v4-pro
LLM_SUMMARY=deepseek:deepseek-v4-pro
LLM_EXTRACT=deepseek:deepseek-v4-pro
DEEPSEEK_API_KEY=...
```

対応プロバイダ:

- `deepseek`: `DEEPSEEK_API_KEY`
- `codex_cli`: ローカルのCodexログインと利用枠
- `claude_cli`: ローカルのClaudeログインと利用枠
- `openai_compat`: `LLM_OPENAI_BASE_URL` と必要に応じて `LLM_OPENAI_API_KEY`
- `anthropic`: `ANTHROPIC_API_KEY` と `uv sync --extra llm-anthropic`
- `gemini`: 後方互換の生成LLMプロバイダ。埋め込みには使用不可

## MCPクライアントへの接続

Claude Desktopはウィザードから自動登録できます。手動設定例:

```json
{
  "mcpServers": {
    "zotero-rag": {
      "command": "/absolute/path/to/uv",
      "args": [
        "--directory", "/absolute/path/to/zotero-local-rag",
        "run", "python", "-u", "src/rag_mcp_server.py"
      ],
      "env": {
        "CHROMA_DIR": "/absolute/path/to/zotero-local-rag/data/chroma",
        "EMB_PROFILE": "fast"
      }
    }
  }
}
```

CursorやZedでも、作業ディレクトリをこのリポジトリに指定して次のコマンドをMCPサーバーとして登録します。

```bash
uv run python -u src/rag_mcp_server.py
```

このプロジェクトはPyPI未公開のため、`uvx zotero-local-rag` ではなくローカルディレクトリから実行してください。

## 任意機能

### 日本語PDFのOCR

一部のPDFでテキストが崩れる場合にTesseractを利用できます。

```bash
# macOS
brew install tesseract tesseract-lang

# Debian/Ubuntu
sudo apt install tesseract-ocr tesseract-ocr-jpn
```

高度な再OCR候補キューはNDLOCR-Liteにも対応します。元のZotero添付ファイルは変更しません。

### 夜間要約

夜間処理は初期状態では無効です。`.env` で時刻・件数・Codex週次残量の下限を管理できます。

```dotenv
NIGHTLY_ENABLE=1
NIGHTLY_START_TIME=03:30
NIGHTLY_LAUNCH_MODE=terminal
NIGHTLY_MAX_HOURS=5
NIGHTLY_MAX_ITEMS=20
NIGHTLY_MIN_WEEKLY_REMAINING_PERCENT=20
```

macOSへ登録または状態確認:

```bash
scripts/install_nightly_launchd.sh
scripts/install_nightly_launchd.sh --check
```

Documents配下などmacOSの保護対象にリポジトリがある場合は、`NIGHTLY_LAUNCH_MODE=terminal`を使用します。ログはTerminalと `data/nightly_summaries.log` に表示・保存されます。

### アプリケーション更新

macOSでは `Software-Update.command` をダブルクリックできます。`.env`、`data/`、`.venv/`、`.claude/` は保持されます。更新後はClaude DesktopなどのMCPクライアントを再起動してください。

## データとバックアップ

索引、DB、ログ、品質評価、バックアップはすべて `data/` 以下に置かれ、Gitには追加されません。通常運用で重要なのは現在の `relations.db`、`chroma/`、`lexical.sqlite3`、`manifest.json`です。

大規模な修復前だけ `data/backups/` にスナップショットを作成し、修復・整合性検査・回帰テストが完了したら古い中間バックアップを削除してください。バックアップは自動同期されないため、必要なら利用者側の暗号化バックアップへコピーしてください。

## トラブルシューティング

まずMCPから `server_status` を呼び出してください。

| 症状 | 確認事項 |
|---|---|
| `chroma_dir_exists: false` | `CHROMA_DIR`と初回インデックスを確認 |
| `No collections found` | `Library-Update.command`またはインデクサーを実行 |
| `EMB resolve error` | `EMB_PROFILE`とモデルのローカルパスを確認 |
| S2の429 | 待ってメニュー3を再実行。必要なら`S2_API_KEY`を設定 |
| LLM処理が即停止 | APIキー、CLIログイン、除外タグ/全許可ポリシーを確認 |

ログはMCPの `get_debug_logs` または `data/zotero-rag.log` で確認できます。

## 開発・高度な運用

```bash
# 全回帰テスト
uv run python -m unittest discover -s tests -q

# 参考文献候補の状態確認
uv run python scripts/review_references.py list --status pending --limit 20

# 検索品質評価
uv run python scripts/eval_retrieval.py data/quality/gold_qa.jsonl --k 10
```

詳細資料:

- [ZOTERO_RAG_GUIDE.md](ZOTERO_RAG_GUIDE.md): MCPツールと検索ワークフロー
- [CITATION_UPDATE_GUIDE.md](CITATION_UPDATE_GUIDE.md): Citation Networkの更新・再開・レート制限
- [ARCHITECTURE.md](ARCHITECTURE.md): 処理フロー、モジュール、データストア
- [.env.example](.env.example): 全設定のひな形

## ライセンス

[MIT License](LICENSE)
