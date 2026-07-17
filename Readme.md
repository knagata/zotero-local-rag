# Zotero Local RAG (MCP サーバー)

ローカルのZoteroライブラリとそれに紐づくPDF/HTML/EPUBドキュメントを、LLM（Claude、Cursor、Zed、Windsurfなど）に直接接続する Model Context Protocol (MCP) サーバーです。

決定論的なフィルタリング（確実な絞り込み）を活用することで、コンテキストの文脈を最大化しつつ、トークン消費量を節約する設計となっています。

## ✨ 主な機能

- **ローカル環境で実行**: ローカルZoteroストレージから直接段落を抽出し、インデックス化します。リモートへの依存は埋め込みモデル（HuggingFace等で完全オフライン・キャッシュ可能）のみです。
- **段階的な検索**: 用途に合わせて検索レイヤーを最適化しています。
  1. `search_zotero_items`: ベクトルインデックスを介さず、Zotero Local APIからタイトル・著者名・出版年などを超高速で直接検索（部分一致など）できる書誌検索。
  2. `search_items`: ベクトル検索を利用しつつ本文テキストを返さず、メタデータとRRF（密度ベース）スコアのみで関連資料をスクリーニング。
  3. `rag_search`: 意味検索（セマンティック検索）による、ピンポイントな段落レベルのテキスト抽出。
  4. `hierarchical_search`: 資料要約・章要約で候補を絞り、直接段落検索も融合する俯瞰検索。
  5. `get_chunk_context`: 指定した段落前後の文脈をデータベースから直接取得。
- **多言語ハイブリッド検索**: 日英クエリ拡張、FTS5 trigram語彙検索、任意の日英結果クオータを利用できます。
- **要約・事例レイヤー**: ローカル抽出型要約を無償で構築でき、明示的に選んだ場合のみ共通LLMクライアントで高品質要約と事例抽出を行います。
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

### LLMプロバイダ設定（拡張機能用）

クエリ拡張・要約・参考文献抽出などの段階的に追加される機能は、共通のLLM設定を使います。
指定形式は `provider:model` で、カンマ区切りにすると左から順にフォールバックします。

```bash
# 全タスクの既定（未指定時もこの値）
export LLM_DEFAULT="gemini:gemini-3.1-flash-lite"

# タスク別の上書き例
export LLM_EXPAND="gemini:gemini-3.1-flash-lite"
export LLM_SUMMARY="codex_cli:gpt-5.6-luna"
export LLM_EXTRACT="gemini:gemini-3.1-flash-lite"
```

対応プロバイダは `gemini`、`anthropic`、`deepseek`、`openai_compat`、`codex_cli`、
`claude_cli` です。Geminiには `GEMINI_API_KEY`、Anthropicには `ANTHROPIC_API_KEY`、
DeepSeekには `DEEPSEEK_API_KEY` が必要です（比較済みの予備プロバイダで、要約の既定には
使いません）。Anthropic SDKは
`uv sync --extra llm-anthropic` で追加できます。Ollama・LM Studio・vLLMなどは
`LLM_OPENAI_BASE_URL`（例: `http://localhost:11434/v1`）を設定して `openai_compat` を使います。
CLIプロバイダは各CLIで事前にサインインされている必要があります。
永続化する場合は [`.env.example`](.env.example) から必要な項目だけを `.env` にコピーしてください。

参考文献抽出は `Citation-Update` のメニュー6で書き込みなしのプレビュー、メニュー7で保存できます。
CiNii Research v2を解決先に使う場合は、公式登録で得た `CINII_APP_ID` を設定してください。
NDL Search SRUはアプリケーションIDなしで利用します。

全量保存前のレビューには、引用グラフを変更しないステージングキューを使用できます。

```bash
# ローカルヒューリスティックで候補をレビューキューへ保存
uv run python -m src.extract_references --item ITEMKEY --stage --heuristic

# pending候補の確認・判定
uv run python scripts/review_references.py list --status pending --limit 20
uv run python scripts/review_references.py set 123 rejected --note "本文の誤検出"

# approved候補のうち、原文中にDOI/ISBNが実在するものだけcommit
uv run python scripts/review_references.py commit-approved --limit 20

# completenessの自動レポート（precision/recallは目視評価が必要）
uv run python -m src.reference_quality_report --status pending
```

要約索引は次のコマンドで差分構築できます。既定の `extractive` は本文を外部へ送信しません。

```bash
# ローカル抽出型要約 + 要約コレクション
uv run python -m src.build_summaries

# LLM要約（本文を設定済みプロバイダへ送信するため、除外タグを設定して明示実行）
SUMMARY_EXCLUDE_TAGS=private,confidential uv run python -m src.build_summaries --mode llm

# 既存ChromaからFTS5を初回構築
uv run python -m src.lexical_index --rebuild

# 既存チャンクにlangを後付け（埋め込み再計算なし、FTSも同期再構築）
uv run python -m src.backfill_languages --batch-size 5000
```

`SUMMARY_EXCLUDE_TAGS` / `EXTRACT_EXCLUDE_TAGS` が未設定の場合、本文を外部LLMへ送る処理は
fail-closedで停止します。全資料の送信を許可する場合だけ、対応する
`SUMMARY_ALLOW_CLOUD_ALL=1` / `EXTRACT_ALLOW_CLOUD_ALL=1` を明示設定してください。
秘密値と分けたい場合は、Git除外される `.env.policy` に除外タグだけを記録できます。

### 夜間Codex要約（品質ゲート後に有効化）

> **現在は有効化しないでください。** ゲート2は条件付き合格しましたが、監査で発見した
> 非本文メタ応答の修正とゲート3が完了するまで、夜間ランナーは無効のままです。

まずDBを書き換えない比較を行います。

```bash
uv run python scripts/compare_summary_models.py \
  --item ITEMKEY --llm codex_cli:gpt-5.6-luna --max-sections 3

# APIプロバイダのDB非書込み並列比較（小さい並列数から開始）
uv run python scripts/compare_summary_models.py \
  --item ITEMKEY --llm deepseek:deepseek-v4-pro --max-sections 4 --workers 4
```

再OCR候補は、DBを変更せずJSONとMarkdownへ出力できます。

```bash
uv run python scripts/list_reocr_candidates.py \
  --comparison data/quality/summary-comparison.json \
  --json-output data/quality/reocr-candidates.json \
  --markdown-output data/quality/reocr-candidates.md
```

候補JSONを明示して再OCRする場合だけ、言語ルーティングが有効になります。`ja` は
NDLOCR-Lite、`en` / `other` はDoclingへ送られます。元のZotero PDFは変更しません。
再抽出に成功すると古い要約・事例は無効化されるため、続けてLuna要約を再生成してください。

```bash
uv run src/index_from_zotero.py --progress \
  --reocr-candidates data/quality/reocr-candidates.json --reocr-limit 2

uv run python -m src.build_summaries --mode llm --force --item ITEMKEY \
  --llm codex_cli:gpt-5.6-luna
```

`scripts/run_grounding_gate.py` は指定資料の要約DBを置換する手動品質ゲート専用です。
`--write-database` の明示指定が必要で、通常運用や夜間runnerからは呼び出しません。

夜間ランナーは既定で最大5時間・20資料、Codex単独で実行し、レート制限時は正常停止します。
Codexの週次利用枠も実行前、各資料の開始前、各LLMリクエストの直前に確認し、既定では残量が20%以下になると処理を見送ります。
残量を取得できない場合も安全のため見送ります。週次枠のリセット後は、次回の夜間実行から
自動的に再開します。閾値は `NIGHTLY_MIN_WEEKLY_REMAINING_PERCENT` で変更できます。
再OCR工程は別ガード `NIGHTLY_REOCR_ENABLE=1` がない限り実行されません。有効時も候補上位2件が既定上限で、
索引・要約のバックアップ、文字数・異常反復・grounding破棄率の前後レポートを
`data/nightly_reocr_report.json` に保存します。品質ゲート失敗時は、その夜の後続処理を停止します。

```bash
scripts/nightly_summaries.sh --check
NIGHTLY_ENABLE=1 NIGHTLY_MAX_HOURS=5 NIGHTLY_MAX_ITEMS=20 scripts/nightly_summaries.sh

# ゲート3完了後、再OCR候補も明示的に処理する場合のみ
NIGHTLY_ENABLE=1 NIGHTLY_REOCR_ENABLE=1 \
  NIGHTLY_REOCR_CANDIDATES=data/quality/reocr-candidates.json \
  scripts/nightly_summaries.sh
```

夜間実行時刻は `.env` の `NIGHTLY_START_TIME=03:30` のように24時間表記で指定します。
macOSのシステムタイムゾーンが使われるため、JST設定のMacではJST 03:30です。設定後に
`scripts/install_nightly_launchd.sh` を実行すると、launchd設定を生成・登録します。時刻を変更した場合も
同じコマンドを再実行してください。`scripts/install_nightly_launchd.sh --check` は設定を変更せず状態を表示します。

### 3. アプリケーションの更新（新バージョンへの移行）

GitHubから最新バージョンを自動でダウンロードして上書きする更新スクリプトを用意しています。`.env` とインデックスデータ（`data/`）は保持されます。

- **Mac ユーザー**: `Software-Update.command` をダブルクリック
- **Windows ユーザー**: `Software-Update.bat` をダブルクリック

更新後はClaude Desktopを再起動してください。

### 4. 日本語PDFのテキスト抽出品質を上げる（任意）

一部のPDF（独自フォントを使用しているもの）では、PyMuPDF がテキストを正しく抽出できない場合があります。そのような場合、Tesseract OCR をフォールバックとして使用することで、ページを画像化してOCRし、正しいテキストを取得できます。

```bash
# macOS
brew install tesseract tesseract-lang

# Linux
sudo apt install tesseract-ocr tesseract-ocr-jpn
```

インストール後は自動的に日本語＋英語のOCRが有効になります。Tesseract が無い場合もエラーにはならず、通常のPyMuPDF抽出のみで動作します。

### 5. 引用ネットワークの更新（Semantic Scholar連携）

Zotero文献の被引用情報（どの論文に引用されているか）をSemantic Scholar APIから取得してデータベースに保存します。`build_citation_network` ツールをMCP経由でClaudeに依頼するか、以下のスクリプトから手動で実行できます。

- **Mac ユーザー**: `Citation-Update.command` をダブルクリック
- **Windows ユーザー**: `Citation-Update.bat` をダブルクリック

実行時は対象（特定アイテム指定 / 全アイテム一括 / スキップ済みEPUB参照の再解決）を選択するメニューが表示されます。Semantic Scholar APIのレート制限（APIキーあり: 2.5秒間隔、なし: 3.5秒間隔）により、大規模ライブラリの全件更新は時間がかかります。

> **S2 APIキー**: `S2_API_KEY` を `.env` に設定するとレート制限が緩和されます。また `ZOTERO_USER_ID` + `ZOTERO_API_KEY` を設定すると、解決した DOI が Zotero ライブラリに自動書き戻しされます（詳細は下記の環境変数表を参照）。

> **詳細ヘルプ**: メニューの選び方、ステータスの意味、429エラーやエラー回復の仕組みについては [`CITATION_UPDATE_GUIDE.md`](./CITATION_UPDATE_GUIDE.md) を参照してください。エラーが出た場合も Force rebuild は不要で、メニュー3の再実行だけで該当分が自動回収されます。

### 6. 環境変数（手動設定の場合）

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
