# Zotero Local RAG MCP リファレンス

このMCPサーバーは、ユーザーのZoteroライブラリに対する探索・抽出を行うツール群を提供します。最大の特徴は「トークン消費の最適化」と「高度なスコアリング（RRF）」です。状況に応じて自律的に最適なツールとパラメータを選択してください。

---

## 🛠 利用可能なツール

用途と解像度（コスト）に応じて使い分けてください。

### 1. `search_zotero_items` (Zotero直接クイック検索・超高速)

ベクトルインデックスを介さず、Zotero Local HTTP APIを直接用いてタイトル、著者、出版年などから超高速で部分一致検索を行います。「あのタイトルの論文はあるか？」「あの著者の論文はあるか？」といった明確な目的がある場合に最適です。

- **`query`**: 検索ワード（タイトル、著者名、出版年など）。
- **`limit`**: 最大取得件数 (デフォルト `20`)。
- **`qmode`**: 検索モード。`"titleCreatorYear"`（タイトル・著者・出版年のみを検索 - 推奨・超高速）または `"everything"`（添付ファイル本文を含むすべてを検索）。

### 2. `search_items` (資料レベル・低コスト)

本文を返さず、ベクトル検索に合致した資料のメタデータのみを返します。特定のテーマについて、広い範囲から関連文献の当たりをつけるのに最適です。

- **`query`**: 文字列、または検索語のリスト。
- **`k`**: 取得件数 (デフォルト `10`)。コストが低いため、多めに取得して俯瞰することが可能です。
- **`where`**: Chromaメタデータフィルター（`source_type`、`itemKey` などで絞り込み可能）。
- **`include_notes`**: Zoteroノートを検索対象に含めるか (デフォルト `false`)。

### 3. `rag_search` (段落レベル・高コスト)

検索に合致した具体的な段落（チャンク）の本文テキストを抽出します。

- **`query`**: 文字列、または検索語のリスト。
- **`k`**: 取得件数 (デフォルト `5`)。
- **`where`**: Chromaメタデータフィルター。
- **`include_item_keys`**: 特定の資料 (`itemKey`) に検索範囲を絞り込むリスト。
- **`exclude_chunk_ids`**: 既に取得したチャンク (`id`) を除外し、新たな情報を取得するためのリスト。
- **`context_window`**: ヒットした段落の前後何段落を含めるか (デフォルト `0`)。必要がない限り0に保ちます。
- **`include_notes`**: Zoteroノートを検索対象に含めるか (デフォルト `false`)。

### 4. `get_chunk_context` (文脈拡張・低コスト)

`rag_search` で取得した特定のチャンクについて、その周辺の段落を再検索（ベクトル計算）なしで取得します。

- **`chunk_id`**: 起点となるチャンクID（例: `ABCDEFGH:p12:para3:part0`）。
- **`window`**: 前後に展開する段落数 (デフォルト `2`)。

### 5. `get_item_details` (詳細情報取得)

特定の資料の抄録（Abstract）、タグ、全著者リストなどのZoteroメタデータを取得します。

- **`item_key`**: 対象の資料の Zotero `itemKey`。

### 6. `list_recent_items` (ブラウジング)

ライブラリに直近で追加・更新された文献一覧を取得します。

- **`limit`**: 取得件数 (デフォルト `20`)。

### 7. `get_document_outline` (目次取得)

PDFまたはEPUBドキュメントの目次（章構造）を取得します。章単位で検索範囲を絞り込む際の事前調査に使います。

- **`attachment_key`**: ZoteroのattachmentKey（例: `ABCDEFGH`）。

### 8. `server_status` (サーバー状態確認)

サーバーの稼働状態と設定を確認します。応答が得られない場合や動作が不安定な場合に最初に呼び出してください。

返り値に含まれる情報：
- `status`: `"ok"` または `"error"`
- `chroma_dir` / `chroma_dir_exists`: ChromaDBのパスと存在確認
- `collections`: コレクション名とドキュメント数
- `emb_model` / `emb_profile` / `emb_device`: 埋め込みモデルの設定
- `emb_model_loaded`: 埋め込みモデルが既にメモリにロード済みかどうか
- `errors`: エラーがある場合の原因と対処法

### 9. `force_reload_index` (インデックスの強制リロード)

ChromaDBのインデックスとメタデータを強制的にリロードします。インデクサーを新たに実行した直後に「Error finding id」エラーや検索結果が空になる場合に使用してください。引数はありません。

### 10. `get_debug_logs` (デバッグログの取得)

サーバーのログファイルから最新N行を取得します。ChromaDBエラー、コレクションのリロードイベント、クエリ失敗時のトレースバックを診断するために使用してください。

- **`lines`**: 取得するログ行数（デフォルト `100`、最新行から遡って返します）

### 11. `build_citation_network` (引用・被引用ネットワークの一括構築)

特定のZotero文献について、以下の両方を一括で実行し、引用ネットワークのデータベースを構築します。
1. **引用先の抽出**: EPUBから注釈（脚注・章末注）を自動抽出し、参照頻度（cite_count）の高い順に最大50件をSemantic Scholarで解決して保存。
2. **被引用の抽出**: Semantic Scholar APIを利用して「外部のどの論文から引用されているか」を取得して保存。

- **`item_key`**: 対象のZotero `itemKey`。

> **EPUB参照バジェットについて**: 脚注が大量にある書籍（編著など）では、重要度の低い参照は `s2_status='skipped'` として保留されます。`Citation-Update` のメニュー項目5「Resume skipped EPUB refs」（`--resume-skipped`）でバジェットを増やして後から再解決できます。

### 12. `get_references_for_item` / `get_chunk_references` (参照文献の取得)

構築済みのデータベースから、特定の資料（または段落チャンク）が「どの文献を引用しているか」を取得します。
- `get_references_for_item`: 資料全体に対する参照文献の一覧を取得
- `get_chunk_references`: 特定の段落チャンクIDに対する参照文献を取得

### 13. `get_cited_chunks_for_item` / `get_chunk_citations` (被引用の取得)

マッピング済みのデータベースから、特定の資料（または段落チャンク）が「外部の論文からどのような文脈で引用されているか」を取得します。これにより、「この文献の中で最も外部から議論されている（引用されている）重要な段落はどこか？」を特定できます。
- `get_cited_chunks_for_item`: 資料全体で被引用数が多い順にチャンク一覧を取得
- `get_chunk_citations`: 特定の段落チャンクIDに対する被引用コンテキストの一覧を取得

---

## ⚙️ 最適化仕様と返り値の解釈

### マルチクエリ・バッチ検索による最適化

`search_items` および `rag_search` の `query` パラメータには、可能な限り **文字列のリスト** (例: `["AI", "人工知能", "LLM"]`) を渡してください。
1回の呼び出しで全キーワードの検索と重複排除が実行されるため、トークンと計算コストの節約になります。

### スコア指標 (`distance` と `rrf_score`)

各種検索ツールの返り値には以下の2つのスコアが含まれます。

- **`distance` (距離)**: 値が小さい（0に近い）ほど、テキストの意味的な一致度が高い（特定の箇所がクエリと強く合致している）ことを示します。
- **`rrf_score` (RRFスコア)**: 値が大きいほど重要度が高い。複数のクエリキーワードでヒットしたり、一つの資料内で何度もヒットしたりした場合に加算されるため、**「資料/段落としてのコンセンサスや密度の高さ」**を表します。

### `where` フィルターで使用可能なメタデータキー

| キー | 型 | 説明 |
|---|---|---|
| `title` | str | 資料タイトル |
| `year` | int | 出版年 |
| `creators` | str | 著者（`;` 区切り） |
| `itemKey` | str | ZoteroのアイテムID |
| `attachmentKey` | str | 添付ファイルID |
| `source_type` | str | `"pdf"` / `"html"` / `"epub"` / `"note"` |
| `page` | int | PDFのページ番号（1始まり） |
| `page_label` | str | PDFのページラベル（例: `"xii"`, `"15"`） |
| `chapter` | str | 章タイトル |
| `section` | str | セクションタイトル（PDF） |

---

## 📋 推奨ワークフロー

### 目的別の調査フロー

**ケースA：特定のタイトルや著者、出版年が既に分かっている場合 (超高速・ピンポイント)**
```
1. search_zotero_items(query="著者名やタイトルの一部", limit=5)
   → Zotero Local APIから直接itemKey（およびメタデータ）を即座に取得

2. rag_search(query="具体的な質問や調べたい内容", include_item_keys=["KEY1"], k=5)
   → その資料に絞ってセマンティック検索を行い具体的な段落を抽出

3. get_chunk_context(chunk_id="...", window=2)
   → 必要に応じて前後の文脈を取得（再検索なし）
```

**ケースB：テーマについて広い範囲から関連文献を探したい場合 (セマンティック検索)**
```
1. search_items(query=["キーワード1", "キーワード2"], k=10)
   → ベクトルインデックスによる関連文献の一覧とitemKeyを取得

2. rag_search(query="具体的な質問や調べたい内容", include_item_keys=["KEY1", "KEY2"], k=5)
   → 絞り込んだ資料から具体的な段落を抽出

3. get_chunk_context(chunk_id="...", window=2)
   → 必要に応じて前後の文脈を取得（再検索なし）
```

**ケースC：文献の引用・被引用ネットワークを分析したい場合**
```
1. build_citation_network(item_key="KEY1") 
   → Zotero内のEPUBから引用先を抽出し、外部APIから被引用データを取得して一括でデータベースを構築

2. get_cited_chunks_for_item(item_key="KEY1")
   → 外部から最も多く引用されている重要な段落のランキングを取得

3. get_chunk_citations(chunk_id="...")
   → その段落が「どんな論文で、どのように言及されているか」の文脈テキストを取得し、学術的な評価や影響を分析
```

### 新しい視点が欲しい場合（既読スキップ）

```python
# rag_search の exclude_chunk_ids に既に読んだIDを渡す
rag_search(
    query=["..."],
    exclude_chunk_ids=["KEY:p3:para2:part0", "KEY:p5:para1:part0"],
    k=5
)
```

### 章単位での精密検索

```
1. get_document_outline(attachment_key="ABCDEFGH")
   → 目次を取得して章タイトルを把握

2. rag_search(query=[...], where={"chapter": "第2章"}, k=5)
   → 特定の章のみを対象に検索
```

### サーバーが反応しない場合

```
server_status()
→ status / errors を確認して原因を特定
```
