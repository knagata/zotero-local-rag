# zotero-local-rag

Zotero のローカル同期済み添付（PDF / Web Snapshot(HTML) / EPUB）と Notes から本文を抽出し、 **段落（必要に応じて分割）**単位で埋め込みして Chroma に格納します。  
Claude Desktop から MCP ツールとして呼び出すことを念頭に設計されており、 **段落レベルのセマンティック検索** と **前後文脈（context）**を返します。


---

## できること

- Zotero 添付（Attachment）の本文を抽出して Chroma にインデックス
  - PDF：ページ→段落
  - Web Snapshot（HTML）：本文抽出（trafilatura 優先 / フォールバックあり）
  - EPUB：章（XHTML/HTML）→本文抽出→段落
- Zotero Notes をインデックス（ただし `rag_search` のデフォルト検索対象に含めない）
- `rag_search` による段落レベル検索（上位 `k` 件）
- `where` によるメタデータフィルタ
- `context_window` による前後段落の付与
- `manifest.json` による差分更新（mtime/size ベース）

---

## 前提

- **pyenv を用いて Python のバージョンを固定して利用することを前提とします**
  - 動作確認済み：Python 3.10.x（例: 3.10.17）
- Zotero 7
- 添付ファイルはローカルに同期済み（Zotero の `storage/` 配下）
- Claude Desktop（MCP 経由でツール利用する場合）

本プロジェクトでは、**依存関係をインストールした Python 実体と、 Claude Desktop が起動する MCP サーバの Python 実体が一致していること**が重要です。  
そのため、 Python の実体を明示的に固定できる **pyenv の利用を強く推奨**します。 README は pyenv が事前に導入されている前提で書かれています。

また、 README はユーザー名が `user` 、プロジェクトディレクトリが `/Users/user/Documents/zotero-local-rag` であることを前提に記述しています。環境が異なる場合は、例示しているパスを適宜読み替えてください。

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
- EbookLib==0.18
- pydantic>=2.12,<3

補足：`sentence-transformers` は内部で torch 等に依存します。環境（CPU/GPU）により最適な入れ方が変わるため、まずは通常の `pip install -r requirements.txt` を推奨します。

---

## インストール

```bash
cd zotero-local-rag
python -m pip install -U pip
python -m pip install -r requirements.txt
```

---

## クイックスタート（最短手順）

0. Python バージョンを pyenv で固定

   ```bash
   pyenv install 3.10.17
   pyenv local 3.10.17
   ```

   正しく切り替わっていることを確認します：
   ```bash
   python -c "import sys; print(sys.executable)"
   ```

   出力が `.pyenv/versions/3.10.17/bin/python` を指していればOKです。

1. 依存関係をインストール
   ```bash
   cd /Users/user/Documents/zotero-local-rag
   python -m pip install -U pip
   python -m pip install -r requirements.txt
   ```

2. 環境変数を読み込む（Zotero の場所・Chroma の保存先・オフライン設定など）
   ```bash
   . ./env.sh
   ```

   Zotero のデータディレクトリが標準位置（`$HOME/Zotero`）ではない場合は、**先に手動で指定**してください：
   ```bash
   export ZOTERO_DATA_DIR="/ABS/PATH/TO/Zotero"
   ```
   ※ `ZOTERO_DATA_DIR` は `storage/` と `zotero.sqlite` を含むディレクトリです。 `/ABS/PATH/TO/Zotero` を自身の環境におけるZoteroのディレクトリの絶対パスに置き換えてください。

3. 埋め込みモデルを事前にキャッシュ（初回のみ・オンライン）

   既定の高速モデル（多言語）を `data/models/` に保存します。
   ```bash
   HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
   python - <<'PY'
   from huggingface_hub import snapshot_download
   snapshot_download(
     repo_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
     local_dir="data/models/paraphrase-multilingual-MiniLM-L12-v2",
     local_dir_use_symlinks=False,
   )
   print("cached: data/models/paraphrase-multilingual-MiniLM-L12-v2")
   PY
   ```

4. Zotero 添付をインデックス（Zotero → Chroma）

   添付（PDF / HTML / EPUB）から本文を抽出し、段落単位で Chroma に保存します。
   初回は時間がかかることがあります。
   ```bash
   make sync
   ```

   うまくいかない場合は、まず以下で前提を確認できます：
   ```bash
   make check
   ```

5. Claude Desktop の MCP 設定（起動時に自動でサーバ起動）

   MCP サーバは Terminal から手動起動せず、`claude_desktop_config.json` に登録して **Claude 起動時に自動起動**させます。

   注意：`command` には **pyenv で固定した Python の実体（`sys.executable` で確認できる絶対パス）**を必ず指定してください。  
   `python` や `python3` のような相対指定は行わないでください。

   - `command` は **依存をインストールした Python** を絶対パスで指定
   - `args` は `-u` 付きで `src/rag_mcp_server.py` を指定
   - `env` には最低限 `CHROMA_DIR` と（必要なら）`EMB_PROFILE` / オフライン設定を指定

   例：
   ```json
   {
     "mcpServers": {
       "zotero-rag": {
         "command": "/Users/user/.pyenv/versions/3.10.17/bin/python",
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

   設定後に Claude Desktop を再起動し、コネクタに `zotero-rag` が出るか確認してください。

---

## Windows 環境について

本プロジェクトは macOS / Linux を主対象としていますが、Windows でも **Git Bash** または **WSL**（Ubuntu 等）を使えば運用できます。

- **PowerShell / cmd.exe は対象外**（この README は Bash 前提です）
- **Windows では make を使いません**  
  - `make sync` 等の代わりに、下記のように Python スクリプトを直接実行してください。
- `env.sh` は Bash 用なので、Git Bash / WSL で `source`（または `. ./env.sh`）して使います。

### 1 事前条件

- Python 3.10.x（例：3.10.17）
- Zotero のデータがローカルに存在すること（`storage/` と `zotero.sqlite` を含むディレクトリ）

### 2 実行例（Git Bash / WSL 共通）

```bash
cd /Users/user/Documents/zotero-local-rag

# 環境変数（必要に応じて編集/上書き）
. ./env.sh

# Zotero データの場所は Windows では特にズレやすいので、基本は明示指定を推奨
export ZOTERO_DATA_DIR="/ABS/PATH/TO/Zotero"

# インデックス作成（Zotero → Chroma）
python src/index_from_zotero.py --progress

# MCP サーバ（手動起動する場合。通常は Claude の設定で自動起動させる）
python -u src/rag_mcp_server.py
```

### 3 ZOTERO_DATA_DIR の例

- **WSL から Windows 側の Zotero データを参照する場合**（例：C ドライブ上の `C:\\Users\\user\\Zotero`）
  ```bash
  export ZOTERO_DATA_DIR="/mnt/c/Users/user/Zotero"
  ```

- **Git Bash から参照する場合**（同じく `C:\\Users\\user\\Zotero`）
  ```bash
  export ZOTERO_DATA_DIR="/c/Users/user/Zotero"
  ```

※ `ZOTERO_DATA_DIR` は **必ず `storage/` と `zotero.sqlite` を含むディレクトリ**を指す必要があります。

### 4 GPU 利用（任意）

- NVIDIA GPU があり CUDA が使える場合：`EMB_DEVICE=cuda`
- それ以外：`EMB_DEVICE=cpu`

例：
```bash
export EMB_DEVICE=cuda
```

### 5 注意点

- Windows はパス表記や権限周りで問題が出やすいので、まずは少量の添付で `index_from_zotero.py` が正常に完走することを確認してから本格運用してください。
- Claude Desktop 連携（MCP）の設定ファイル位置や書式は OS / Claude のバージョンで変わり得るため、基本は Claude 側のドキュメントに従ってください（本 README の設定例は macOS 前提です）。

---

## 事前キャッシュ（オフライン前提）

このプロジェクトは **オフライン運用を基本**にしています（`env.sh` は `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` をデフォルトで有効化）。
そのため、埋め込みモデルは **事前に `./data/models/` 配下へ配置**しておきます。

モデルは 1 台（1 環境）につき **ひとつだけ**用意すれば十分です（`EMB_PROFILE` で選択）。

### fast（既定）: paraphrase-multilingual-MiniLM-L12-v2

`./data/models/paraphrase-multilingual-MiniLM-L12-v2/` にスナップショットを保存します。

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
  repo_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  local_dir="data/models/paraphrase-multilingual-MiniLM-L12-v2",
  local_dir_use_symlinks=False,
)
print("cached: data/models/paraphrase-multilingual-MiniLM-L12-v2")
PY
```

### bge（高精度）: bge-m3

`./data/models/bge-m3/` にスナップショットを保存します。

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
  repo_id="BAAI/bge-m3",
  local_dir="data/models/bge-m3",
  local_dir_use_symlinks=False,
)
print("cached: data/models/bge-m3")
PY
```

キャッシュ後は通常どおり `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` のまま実行してください。

補足：すでに Hugging Face のグローバルキャッシュにダウンロード済みでも動く場合がありますが、環境差（Python 実行環境・キャッシュ場所・権限）で詰まりやすいので、**プロジェクト内 `data/models/` に固定**する運用を推奨します。

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
  - **オフライン運用（下記）ではローカルディレクトリ指定が最も確実**です。

- `EMB_DEVICE`（任意）：推論デバイス（例: `cpu`, macOS でGPUを用いる場合 `mps`）。
  - `env.sh` は `EMB_PROFILE` に応じて妥当なデフォルトを選びます。

### オフライン運用

- `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`
  - デフォルト有効（ネットワークアクセスを防止）
  - 初回キャッシュ時だけ一時的に `0` にして実行します（上記「事前キャッシュ」参照）。

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

## インデックス作成（Zotero → Chroma）

### Makefile を使う（推奨）

`env.sh` を読み込んでから実行します。

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

### 直接実行する

```bash
. ./env.sh
python src/index_from_zotero.py --progress
```

よく使うオプション：

- `--collection <COLLECTION_KEY>`：特定コレクションだけ対象
- `--require-data-dir`：`ZOTERO_DATA_DIR` が不正なら即エラー（安全）
- `--rebuild`：強制再構築

---

## MCP サーバ起動（ローカル）

```bash
make serve
```

Claudeから利用する場合には使用しません。

---

## Claude Desktop（MCP）設定

重要：**依存をインストールした Python を `command` に指定**してください（例：pyenv の python の絶対パス）。

できるだけシンプルにするなら、`data/models/` に事前キャッシュした前提で、Claude 側の `env` は最低限これだけで動きます：

- `CHROMA_DIR`
- `EMB_PROFILE`（`fast` か `bge`）
- `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`

※ 通常は `EMB_MODEL` を指定しなくても、プロジェクト内の既定パス（`data/models/...`）を自動で使います。既定以外の場所に置く場合だけ `EMB_MODEL` にローカルディレクトリを指定してください。

```json
{
  "mcpServers": {
    "zotero-rag": {
      "command": "/Users/user/.pyenv/versions/3.10.17/bin/python",
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

例：`EMB_PROFILE=bge` を使う場合は `data/models/bge-m3/` が必要です（別場所なら `EMB_MODEL=/path/to/bge-m3`）。

`-u` は標準入出力のバッファリングを抑え、ログ追跡がしやすくなります。

---

## ツール（Claude から呼ぶ）

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

### 1 Claude で `MCP zotero-rag: Server disconnected`

多くの場合、MCP サーバが例外で終了しています。Claude のログで原因を確認します。

```bash
tail -n 200 ~/Library/Logs/Claude/mcp*.log
```

### 2 Hugging Face に接続できない / オフラインでモデルが見つからない

事前キャッシュが必要です（上記「事前キャッシュ」参照）。特にオフラインモードでは、`EMB_MODEL` をローカルディレクトリにするか、同じ Python 環境の Hugging Face キャッシュにモデルが存在している必要があります。

### 3 `MuPDF error: ...` が出る

破損気味の PDF で PyMuPDF が警告を出すことがあります。処理は継続しますが、抽出品質が低い場合は PDF の差し替え（再取得）を検討してください。

---


## 運用メモ

- 埋め込みモデルを切り替える場合（例：`EMB_PROFILE=fast` ↔ `EMB_PROFILE=bge` / `EMB_MODEL` の変更）は **埋め込み次元が変わり得る** ため、`CHROMA_COLLECTION` を分けるか `--rebuild` が必要です。`CHROMA_COLLECTION` を未設定にしておけば自動で `..._384` / `..._1024` のように分かれます。
- 文献追加後は `make sync` を再実行すると差分更新されます。
- 別PCに移す場合は、同じ Python で `pip install -r requirements.txt` → モデルの事前キャッシュ → MCP 設定、の順が確実です。