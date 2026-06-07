# Zotero Local RAG — アーキテクチャ図

---

## 1. システム全体図

```mermaid
flowchart TB
  subgraph CLIENT["👤 クライアント"]
    CD["Claude Desktop\n(LLM / MCPクライアント)"]
    LUC["Library-Update\n.command / .bat"]
    CUC["Citation-Update\n.command / .bat"]
    SC["Setup\n.command / .bat"]
  end

  subgraph CORE["⚙️ コアプログラム (src/)"]
    MCP["rag_mcp_server.py\nMCPサーバー"]
    IDX["index_from_zotero.py\nインデクサー"]
    UCT["update_citations.py\n引用情報CLIツール"]
  end

  subgraph DATA["💾 データストア (data/)"]
    CHROMA[("ChromaDB\nchroma/\n段落ベクトル＋メタデータ")]
    RELS[("relations.sqlite3\n引用・参照ネットワーク")]
    MANI["manifest.json\nインデックス更新状態"]
    LOG["zotero-rag.log\nサーバーログ"]
    S2L["s2_rate.lock\nS2 APIレート管理"]
  end

  subgraph EXT["🌐 外部リソース"]
    ZAPP["Zotero Desktop App\n(ローカル API :23119)"]
    S2API["Semantic Scholar API\n(被引用データ)"]
    HF["HuggingFace モデル\n(埋め込みベクトル)"]
    FILES["ローカルファイル\nPDF / EPUB / HTML / Note"]
  end

  CD -- "MCP プロトコル" --> MCP
  LUC -- "uv run" --> IDX
  CUC -- "uv run" --> UCT
  SC -- "setup_wizard.py" --> IDX

  MCP -- "ベクトル検索" --> CHROMA
  MCP -- "引用DB参照" --> RELS
  MCP -- "書誌検索" --> ZAPP
  MCP -- "ログ書き込み" --> LOG

  IDX -- "チャンク保存" --> CHROMA
  IDX -- "状態保存" --> MANI
  IDX -- "添付ファイル一覧" --> ZAPP
  IDX -- "ファイル読み込み" --> FILES
  IDX -- "埋め込み" --> HF

  UCT -- "引用DB書き込み" --> RELS
  UCT -- "被引用取得" --> S2API
  UCT -- "チャンク参照" --> CHROMA
  UCT -- "レート管理" --> S2L

  ZAPP -. "ローカルストレージ" .-> FILES
```

---

## 2. インデックス構築パイプライン

`Library-Update.command` → `src/index_from_zotero.py` の処理フロー。

```mermaid
flowchart LR
  subgraph INPUT["入力"]
    ZA["Zotero Local API\n(:23119)"]
    PF["PDF ファイル"]
    EF["EPUB ファイル"]
    HF2["HTML スナップショット"]
    NF["Zotero ノート"]
  end

  subgraph EXTRACT["テキスト抽出 (src/)"]
    PE["pdf_extract.py\nページ単位で段落抽出\n(docling fallback対応)"]
    EE["html_extract.py\nEPUB章ごとに段落抽出"]
    HE["html_extract.py\nHTML本文抽出"]
    NE["note_extract.py\nノートHTML→テキスト"]
  end

  subgraph PROC["後処理 (src/)"]
    TU["text_utils.py\n長文分割・短文結合\n繰り返し行除去\nCJK/欧文判定"]
    CD2["chapter_detect.py\nPDF目次・EPUB章タイトル付与"]
  end

  subgraph STORE["保存 (src/)"]
    EMB["embedder.py\nモデルロード・コレクション初期化"]
    MAN["manifest.py\n処理済みキー管理"]
    CHROMA2[("ChromaDB\ndata/chroma/\nembedding_id, ベクトル\nitemKey, page, chapter等")]
    MANI2["manifest.json\nattachmentKey→mtime"]
  end

  ZA -- "添付ファイル一覧取得" --> INPUT
  PF --> PE
  EF --> EE
  HF2 --> HE
  NF --> NE

  PE --> TU
  EE --> TU
  HE --> TU
  NE --> TU

  TU --> CD2
  CD2 --> EMB
  MAN --> EMB
  EMB -- "upsert" --> CHROMA2
  EMB -- "完了記録" --> MANI2
```

### 主要な chunk ID 命名規則

| ソースタイプ | chunk_id の例 |
|---|---|
| PDF | `ABCDEFGH:p12:para3:part0` (p=ページ番号) |
| EPUB | `ABCDEFGH:c5:para2:part0` (c=章インデックス) |
| HTML | `ABCDEFGH:h0:para7:part0` |
| Note | `ABCDEFGH:n0:para1:part0` |

---

## 3. RAG クエリフロー（実行時）

Claude が `rag_search` / `search_items` を呼び出したときの処理。

```mermaid
flowchart LR
  CLD["Claude Desktop"] -- "ツール呼び出し\n(MCP)" --> MCP2

  subgraph MCP2["rag_mcp_server.py"]
    direction TB
    QE["クエリをベクトル化\n(embedder経由)"]
    RRF["RRF スコアリング\n複数クエリ結果を統合"]
    FILT["フィルタリング\nexclude_chunk_ids\ninclude_item_keys\nwhere条件"]
    CTX["文脈拡張\nget_chunk_context\n前後チャンク取得"]
  end

  subgraph CHROMA3["🗄 ChromaDB"]
    VEC["ベクトル近傍探索\n(HNSW)"]
    META["メタデータ取得\n(SQLite)"]
  end

  subgraph ZAPI2["Zotero Local API"]
    BMETA["書誌メタデータ\n(タイトル・著者・年等)"]
  end

  QE --> VEC
  VEC --> META
  META --> RRF
  RRF --> FILT
  FILT --> CTX
  CTX -- "結果返却" --> CLD

  BMETA -- "search_zotero_items\nget_item_details" --> CLD
```

---

## 4. 引用ネットワーク構築フロー

`build_citation_network` ツール（または `Citation-Update.command`）の処理フロー。

```mermaid
flowchart TD
  START["build_citation_network\n(item_key)"] --> ZITEM

  subgraph PREP["準備"]
    ZITEM["Zotero API\nタイトル・DOI・ISBN取得"]
    EPUBP["EPUB ファイルパス解決"]
  end

  subgraph LOCAL["ローカル参照抽出\n(epub_reference_extractor.py)"]
    EFOOTNOTE["EPUB 脚注・章末注テキスト抽出"]
    CMATCH["citation_mapper.py\nmap_item_local_references()\nチャンクとのコサイン類似度照合"]
    RINSERT["db_relations.py\ninsert_reference()\n参照レコード保存"]
  end

  subgraph GLOBAL["グローバル被引用取得\n(citation_mapper.py)"]
    FIND["find_s2_paper_id()\nS2 API で論文ID特定\n(DOI/タイトル/著者で検索)"]
    S2REQ["s2_request()\nS2 Citations エンドポイント\n(1 req/sec レート管理)"]
    RETRY["429 時 指数バックオフ\n5s → 10s → 20s"]
    CEMBED["_load_chunks_for_item()\nアイテムチャンクを SQLite から取得\n(オンザフライ埋め込み + キャッシュ)"]
    CINSERT["db_relations.py\ninsert_citation()\n被引用レコード保存"]
  end

  subgraph RATELOCK["レート管理\n(fcntl.flock)"]
    S2LOCK2["s2_rate.lock\n_s2_wait_and_claim()\nプロセス間排他制御"]
  end

  ZITEM --> EPUBP
  ZITEM --> FIND
  EPUBP --> EFOOTNOTE
  EFOOTNOTE --> CMATCH
  CMATCH --> RINSERT
  RINSERT --> RELS2

  FIND --> S2REQ
  S2REQ <-- "レート管理" --> S2LOCK2
  S2REQ --> RETRY
  S2REQ --> CEMBED
  CEMBED --> CINSERT
  CINSERT --> RELS2

  RELS2[("relations.sqlite3\n引用・参照テーブル")]
```

---

## 5. ソースモジュール一覧

| ファイル | 役割 | 主な関数・クラス |
|---|---|---|
| `rag_mcp_server.py` | MCPサーバー本体。ツール定義・クエリ処理・Chroma接続管理 | 全 MCP ツール関数、`_col()`, `_z_api()` |
| `index_from_zotero.py` | インデックス構築 CLI。Zotero 添付ファイルを全件処理 | `main_async()`, `_upsert_in_subbatches()` |
| `update_citations.py` | 引用ネットワーク更新 CLI（Citation-Update 用） | `main()`, `process_item()`, `get_all_items()` |
| `citation_mapper.py` | S2 API 連携・引用チャンク照合・レート管理 | `map_item_global_citations()`, `map_item_local_references()`, `s2_request()`, `_s2_wait_and_claim()`, `_load_chunks_for_item()` |
| `db_relations.py` | relations.sqlite3 の CRUD 操作 | `insert_citation()`, `insert_reference()`, `get_citations_for_chunk()`, `get_references_for_item()` |
| `embedder.py` | 埋め込みモデルのロード・ChromaDB コレクション初期化 | `get_collection()`, `_resolve_embedder_settings()` |
| `zotero_source_localapi.py` | Zotero ローカル HTTP API (:23119) クライアント | `ZoteroLocalAPI`, `get_item()`, `get_attachments()` |
| `pdf_extract.py` | PDF ページからの段落抽出（pdfminer ベース） | `extract_chunks_from_pdf()`, `extract_paragraphs_from_pdf_page()` |
| `html_extract.py` | HTML/EPUB スナップショットからの段落抽出 | `extract_chunks_from_html_snapshot()`, `extract_chunks_from_epub_snapshot()` |
| `note_extract.py` | Zotero ノート（HTML）のテキスト抽出 | `index_notes()` |
| `docling_extract.py` | Docling ライブラリを使った PDF 抽出（fallback） | `extract_chunks_from_pdf_with_docling()` |
| `epub_reference_extractor.py` | EPUB の脚注・章末注から参照文献テキスト抽出 | `extract_epub_references()` |
| `text_utils.py` | テキスト分割・正規化・品質判定ユーティリティ | `split_long_paragraph()`, `merge_short_chunk_records()`, `looks_like_gibberish()`, `joiner_for_text()` |
| `chapter_detect.py` | PDF 目次・EPUB 章タイトルの取得 | `get_pdf_toc()`, `get_epub_chapter_index_to_title()`, `build_pdf_page_chapter_lookup()` |
| `manifest.py` | インデックス処理済み状態の読み書き | `load_manifest()`, `save_manifest()` |

---

## 6. データストア一覧

| ファイル / DB | 場所 | 種別 | 内容 |
|---|---|---|---|
| `chroma/` | `data/chroma/` | ChromaDB (SQLite + HNSW) | 段落テキストの埋め込みベクトル、および `itemKey`, `attachmentKey`, `page`, `chapter`, `source_type` 等のメタデータ |
| `manifest.json` | `data/` | JSON | 処理済み添付ファイルの `attachmentKey → mtime` マップ。差分インデックス更新に使用 |
| `relations.sqlite3` | `data/` | SQLite | 引用ネットワーク。`citations` テーブル（外部論文からの被引用）と `references` テーブル（EPUB からの参照先）を格納 |
| `zotero.sqlite3` | Zotero App フォルダ | SQLite | Zotero が管理する書誌データ。本システムは直接アクセスせず、ローカル API 経由で取得 |
| `zotero-rag.log` | `data/` | テキスト | MCP サーバーのデバッグログ。`get_debug_logs` ツールで参照可能 |
| `s2_rate.lock` | `data/` | テキスト | Semantic Scholar API のレート管理用タイムスタンプ。`fcntl.flock` によるプロセス間排他制御 |
| `models/` | `data/models/` | バイナリ | HuggingFace からキャッシュされた埋め込みモデルファイル |

---

## 7. 設定ファイル一覧

| ファイル | 内容 |
|---|---|
| `.env` | 環境変数設定（`ZOTERO_LOCAL_API_BASE`, `S2_API_KEY`, `EMB_PROFILE` 等） |
| `pyproject.toml` | Python 依存パッケージ定義（uv 用） |
| `env.sh` | シェル環境変数のサンプル（手動設定用） |
| `.claude/settings.json` | Claude Code の権限設定（MCP サーバー停止コマンドの拒否ルール等） |
