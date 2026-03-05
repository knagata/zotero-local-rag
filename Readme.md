# zotero-local-rag

Zotero の文献データベースをベクトル化し、AIエージェントなどを用いて高速に横断検索を行うためのツールです。

Zotero のローカル同期済み添付（PDF / Web Snapshot(HTML) / EPUB）と Notes から本文を抽出し、段落（必要に応じて分割）単位で埋め込みして Chroma に格納します。
ローカル環境でAIエージェントから MCP ツールとして呼び出すことを念頭に設計されており、 **段落単位のセマンティック検索** と **前後文脈（context）** を返します。

README は、ProプランのClaude Desktopを念頭に記述されていますが、例えば Cursor Agent などでも利用可能です。Cursor は無料プランでのテストが確認できています。Ollama や LM Studio が選択肢になると思います。

---

## できること

- Zotero 添付（Attachment）の本文を抽出して Chroma にメタ情報とともにインデックス
  - PDF：Zotero key｜ローカルパス｜ページ番号と段落番号　など
  - Web Snapshot（HTML）：Zotero key｜ローカルパス｜段落番号　など
  - EPUB：Zotero key｜ローカルパス｜章インデックス／章をまたいだグローバル段落番号　など
- Zotero Notes をインデックス（ただし `rag_search` のデフォルト検索対象に含めない）
- `rag_search` による段落レベル検索（上位 `k` 件）
- `where` によるメタデータフィルタ
- `context_window` による前後段落の付与
- `manifest.json` による差分更新（mtime/size ベース）

---

## 前提

- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** がインストール済みであること
  - Python バージョン管理・venv 作成・パッケージインストールを一括で行う
  - macOS / Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- Zotero 7
- 添付ファイルはローカルに同期済み（Zotero の `storage/` 配下）
- Claude Desktop（最低でもProプランがローカルMCPの利用に必要）

本プロジェクトでは、**依存関係をインストールした Python 実体と、 Claude Desktop が起動する MCP サーバの Python 実体が一致していること**が重要です。
uv を使うことで `.venv/` に依存が固定され、MCP 設定では `.venv/bin/python`（Windows: `.venv\Scripts\python.exe`）を指定するだけで一致が保証されます。

---

## 依存関係

- chromadb
- sentence-transformers
- huggingface-hub>=0.34.0,<1.0
- pymupdf
- fastmcp
- httpx
- typing_extensions
- trafilatura
- EbookLib
- pydantic>=2.12,<3

補足：`sentence-transformers` は内部で torch 等に依存します。`uv sync` が適切なバイナリを自動で選択します。

---

## 事前準備

### Zotero の設定

Zotero の環境設定から"詳細"→"各種設定"

`□ Allow other applications on this computer to communicate with Zotero`
に☑をいれる。

変更後は Zotero を再起動。再起動後に設定が反映されます。

### Claude Desktop 設定ファイルの確認

"設定"→"開発者"内の"ローカルMCPサーバー"から"設定を編集"をクリックし、 `claude_desktop_config.json` の場所を確認しておく。

なお設定は設定ファイルの変更を保存後に Claude の再起動することで有効になります。Windows の場合は、 Claude の画面を閉じてもバックグランドで Claudeが生きているので、かならずタスクマネージャーで確認し、確実に終了してから再度起動するよう注意してください（ OS ごと再起動してもよい）。

---

## クイックスタート

以下のパスやコマンドは macOS を前提に記述されています。 **Windows 環境を用いる場合は「 Windows 環境について」を参照してください。**

#### 1. 依存関係を一括インストール（初回のみ）

```bash
make setup
```

`uv sync` が実行され、Python 3.10 の自動取得・`.venv/` 作成・パッケージインストールまで完了します。

#### 2. 環境変数の確認（Zotero の場所・Chroma の保存先など）

Zotero のデータディレクトリが標準位置（`$HOME/Zotero`）ではない場合は、**先に手動で指定**してください：

```bash
export ZOTERO_DATA_DIR="/ABS/PATH/TO/Zotero"
```

※ `ZOTERO_DATA_DIR` は `storage/` と `zotero.sqlite` を含むディレクトリです。

#### 3. 埋め込みモデルを事前にキャッシュ（初回のみ・オンライン）

既定の高速モデル（多言語）を `data/models/` に保存します。

```bash
make cache-model
```

高精度モデル（bge-m3）を使う場合：

```bash
EMB_PROFILE=bge make cache-model
```

#### 4. Zotero 添付をインデックス（Zotero → Chroma）

添付（PDF / HTML / EPUB）から本文を抽出し、段落単位で Chroma に保存します。この際、Zoteroが起動していないとライブラリにアクセスできません。"Allow other application on this computer to communicate with Zotero"が有効な状態でZoteroを起動しておいてください。

初回は時間がかかることがあります。

```bash
make sync
```

うまくいかない場合は、まず以下で前提を確認できます：
```bash
make check
```

#### 5. Claude Desktop の MCP 設定（起動時に自動でサーバ起動）

MCP サーバは Terminal から手動起動せず、`claude_desktop_config.json` に登録して **Claude 起動時に自動起動** させます。

注意：`command` には **`.venv/bin/python` の絶対パス** を必ず指定してください。
`python` や `python3` のような相対指定は行わないでください。

例：

```json
{
  "mcpServers": {
    "zotero-rag": {
      "command": "/Users/user/Documents/zotero-local-rag/.venv/bin/python",
      "args": ["-u", "/Users/user/Documents/zotero-local-rag/src/rag_mcp_server.py"],
      "env": {
        "CHROMA_DIR": "/Users/user/Documents/zotero-local-rag/data/chroma",
        "EMB_PROFILE": "fast",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1"
      }
    }
  }
}
```

`.venv/bin/python` の絶対パスを確認するには：

```bash
echo "$(pwd)/.venv/bin/python"
```

設定後に Claude Desktop を再起動し、コネクタに `zotero-rag` が出るか確認してください。

---

## Windows 環境について

本プロジェクトは macOS / Linux を主対象としていますが、Windows でも **uv** を使えば運用できます。

- **PowerShell または cmd.exe** で作業します（Git Bash は不要）
- **Windows では make を使いません**
  - `make sync` 等の代わりに、下記のように Python スクリプトを直接実行してください。

### クイックスタート（Windows / PowerShell）

#### 1. **uv のインストール**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

インストール後、PowerShell を再起動します。

#### 2. 依存関係をインストール

プロジェクトディレクトリで：

```powershell
uv sync
```

#### 3. 環境変数を設定

```powershell
# Zotero のデータディレクトリ（storage/ と zotero.sqlite を含む場所）
$env:ZOTERO_DATA_DIR = "C:\Users\user\Zotero"
$env:CHROMA_DIR = "C:\Users\user\Documents\zotero-local-rag\data\chroma"
$env:EMB_PROFILE = "fast"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
```

#### 4. 埋め込みモデルを事前にキャッシュ（初回のみ・オンライン）

```powershell
$env:HF_HUB_OFFLINE = "0"
$env:TRANSFORMERS_OFFLINE = "0"
uv run python -c @"
from huggingface_hub import snapshot_download
snapshot_download(
  repo_id='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
  local_dir='data/models/paraphrase-multilingual-MiniLM-L12-v2',
  local_dir_use_symlinks=False,
)
print('cached')
"@
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
```

#### 5. インデックス作成

```powershell
uv run python src/index_from_zotero.py --progress
```

### GPU 利用（任意）

- NVIDIA GPU があり CUDA が使える場合：`$env:EMB_DEVICE = "cuda"`
- それ以外：`$env:EMB_DEVICE = "cpu"`

### Claude Desktop の MCP 設定（Windows）

- `command` には `.venv\Scripts\python.exe` の **絶対パス** を指定します。

例：

```json
{
  "mcpServers": {
    "zotero-rag": {
      "command": "C:\\Users\\user\\Documents\\zotero-local-rag\\.venv\\Scripts\\python.exe",
      "args": ["-u", "C:\\Users\\user\\Documents\\zotero-local-rag\\src\\rag_mcp_server.py"],
      "env": {
        "CHROMA_DIR": "C:\\Users\\user\\Documents\\zotero-local-rag\\data\\chroma",
        "EMB_PROFILE": "fast",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1"
      }
    }
  }
}
```

---

## インデックス作成（Zotero → Chroma）

### Makefile を使う（macOS、推奨）

初回構築・更新：

```bash
make sync
```

強制再構築（Chroma と manifest を削除して全件再作成）：

```bash
make rebuild
```

添付解決状況のダンプ（デバッグ用）：

```bash
make dump
```

初回利用時・Zoteroライブラリが更新時にsyncを行ってください。初回のsyncには時間がかかる場合があります。
埋め込みモデルを変更する際には、rebuildが必要です。

### 直接実行する（Windows）

初回構築・更新

```powershell
uv run python src/index_from_zotero.py --progress
```

強制再構築（Chroma と manifest を削除して全件再作成）：

```powershell
uv run python src/index_from_zotero.py --rebuild --progress
```

添付解決状況のダンプ（デバッグ用）：

```powershell
uv run python src/index_from_zotero.py --dump-attachments --progress
```

よく使うオプション：

- `--collection <COLLECTION_KEY>`：特定コレクションだけ対象
- `--require-data-dir`：`ZOTERO_DATA_DIR` が不正なら即エラー（安全）
- `--rebuild`：強制再構築

---

## 埋め込みモデルのキャッシュ

このプロジェクトは **オフライン運用を基本** にしています（`env.sh` は `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` をデフォルトで有効化）。
そのため、埋め込みモデルは **事前に `./data/models/` 配下へ配置** しておきます。

モデルは 1 台（1 環境）につき **ひとつだけ** 用意すれば十分です（`EMB_PROFILE` で選択）。

### fast（既定）: paraphrase-multilingual-MiniLM-L12-v2

```bash
make cache-model
```

### bge（高精度）: bge-m3

```bash
EMB_PROFILE=bge make cache-model
```

キャッシュ後は通常どおり `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` のまま実行してください。

---

## MCP サーバ起動

```bash
make serve
```

Claudeから利用する場合には使用しません。

---

## ツール

Claudeからzotero-local-ragの使い方はアクセスできる仕様なので、ユーザーはClaudeに以下の情報を伝える必要はありません。
プロンプトにzotero-local-ragを利用して文献検索を行うように記載するだけで大丈夫です。
Claudeがアクセスできる内容を知りたい場合や、細かい指示を指定したい場合は以下を参照してください。

### `rag_search`

段落レベルのセマンティック検索。

- `query: str`（必須）
  - 検索クエリ
- `k: int = 10`
  - 返す件数
- `where: dict | None = None`
  - メタデータフィルタ（必要なときだけ）
- `context_window: int = 1`
  - 前後に付ける段落数（0で無効）

---

### `where`（メタデータフィルタ）の詳細

`where` は Chroma のメタデータフィルタです。検索対象を絞りたいときだけ使います。

想定している主なキー（例）：

- `itemKey`（Zotero item key）
- `attachmentKey`（添付の key）
- `noteKey`（Note の key）
- `year`
- `title`
- `path` / `pdf_path`
- `page`（PDF のみ）
- `source_type`（`pdf` / `html` / `epub` / `note`）

例：特定文献（itemKey）に限定

```json
{"itemKey": "BGZ9UFUJ"}
```

例：Notes のみ（Note を探すときに明示）

```json
{"source_type": "note"}
```

例：EPUB のみに限定

```json
{"source_type": "epub"}
```

例：複数候補のどれか（IN）

```json
{"itemKey": {"$in": ["BGZ9UFUJ", "UWABANQQ"]}}
```

例：年の範囲（AND）

```json
{"$and": [{"year": {"$gte": 2010}}, {"year": {"$lte": 2019}}]}
```

---

## トラブルシューティング

### 1 インデックスの作成時にメモリ不足で止まってしまう

インデックス作成時のバッチサイズが大きいとメモリ不足に陥る場合があります。
バッチサイズはデフォルトでは 128 ですが 64 や 32 など、小さいバッチサイズで再度実行してみてください。

```bash
BATCH_SIZE=64 make sync
```

### 2 Claude で `MCP zotero-rag: Server disconnected`

多くの場合、MCP サーバが例外で終了しています。Claude のログで原因を確認します。

```bash
tail -n 200 ~/Library/Logs/Claude/mcp*.log
```

### 3 Hugging Face に接続できない / オフラインでモデルが見つからない

`make cache-model` でモデルのキャッシュが必要です。特にオフラインモードでは、`EMB_MODEL` をローカルディレクトリにするか、同じ Python 環境の Hugging Face キャッシュにモデルが存在している必要があります。

### 4 `MuPDF error: ...` が出る

破損気味の PDF で PyMuPDF が警告を出すことがあります。処理は継続しますが、抽出品質が低い場合は PDF の差し替え（再取得）を検討してください。

---

## 環境変数

`env.sh` がデフォルト値を持ちます。

### 必須 / 推奨

- `ZOTERO_DATA_DIR`（推奨）
  - Zotero データディレクトリ（`storage/` と `zotero.sqlite` を含む）
  - デフォルト: `$HOME/Zotero`

- `CHROMA_DIR`
  - Chroma DBの保存先（デフォルト: `./data/chroma`）

### 埋め込みモデル（導入時にどちらか選ぶ）

- `EMB_PROFILE`：埋め込みモデルのプリセット（`fast` / `bge`）
  - `fast`（既定）：`./data/models/paraphrase-multilingual-MiniLM-L12-v2`（高速）
  - `bge`：`./data/models/bge-m3`（高精度）

- `EMB_MODEL`（任意）：埋め込みモデルの指定（Hugging Face の repo id またはローカルディレクトリ）。
  - `EMB_MODEL` を明示した場合は `EMB_PROFILE` より優先されます。
  - **オフライン運用（下記）ではローカルディレクトリ指定が最も確実** です。

- `EMB_DEVICE`（任意）：推論デバイス（例: `cpu`, macOS でGPUを用いる場合 `mps`）。
  - `env.sh` は `EMB_PROFILE` に応じて妥当なデフォルトを選びます。

### オフライン運用

- `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`
  - デフォルト有効（ネットワークアクセスを防止）
  - 初回キャッシュ時だけ一時的に `0` にして実行します（`make cache-model` が自動で処理）。

### 任意（必要なときだけ）

- `CHROMA_COLLECTION`：コレクション名（未設定なら次元で自動サフィックス）
- `PDF_CACHE_DIR`：Local API フォールバック時のダウンロード先（デフォルト: `./data/pdf_cache`）
- `MANIFEST_PATH`：差分更新用 manifest（デフォルト: `./data/manifest.json`）

- `BATCH_SIZE`：バッチサイズ（デフォルト: 256）
  - これ以上のチャンクが溜まると、削除→upsert してフラッシュします。
  - `col.upsert(...)` に渡すサブバッチサイズも同じ値を使います（メモリスパイク対策）。
  - （目安）Apple Silicon の統合メモリ環境で重い場合は小さめ（例: 32〜128）にすると安定しやすいです。

- `PROGRESS=1`：進捗ログを有効化
- `MAX_CHARS`：段落分割後の最大文字数（デフォルト: 1200）
- `MIN_CHUNK_CHARS`：最小チャンク長（デフォルト: 200）
- `MAX_HTML_BYTES`：巨大 Web Snapshot の読み込み上限（デフォルト: 10000000）

---

## 運用メモ

- 埋め込みモデルを切り替える場合（例：`EMB_PROFILE=fast` ↔ `EMB_PROFILE=bge` / `EMB_MODEL` の変更）は **埋め込み次元が変わり得る** ため、`CHROMA_COLLECTION` を分けるか `--rebuild` が必要です。`CHROMA_COLLECTION` を未設定にしておけば自動で `..._384` / `..._1024` のように分かれます。
- 文献追加後は `make sync` を再実行すると差分更新されます。
- 別PCに移す場合は、uv をインストール → `make setup` → モデルの事前キャッシュ → MCP 設定、の順が確実です。
