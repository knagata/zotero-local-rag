# Claude利用ガイド・MCPツールリファレンス

[READMEへ戻る](../README.md)

この文書は、ClaudeがZoteroライブラリを効率よく探索するためのツール選択と検索手順をまとめたものです。現行の取り込み・検索はV3データプレーン（`zotero_paragraphs_v3`）です。

---

## 🛠 利用可能なツール

用途と解像度（コスト）に応じて使い分けてください。

### `search_zotero_items`（Zotero直接検索）

ベクトルインデックスを介さず、Zotero Local HTTP APIを直接用いてタイトル、著者、出版年などから超高速で部分一致検索を行います。「あのタイトルの論文はあるか？」「あの著者の論文はあるか？」といった明確な目的がある場合に最適です。

- **`query`**: 検索ワード（タイトル、著者名、出版年など）。
- **`limit`**: 最大取得件数 (デフォルト `20`)。
- **`qmode`**: 検索モード。`"titleCreatorYear"`（タイトル・著者・出版年のみを検索 - 推奨・超高速）または `"everything"`（添付ファイル本文を含むすべてを検索）。

### `search_items`（資料レベル）

本文を返さず、ベクトル検索に合致した資料のメタデータのみを返します。特定のテーマについて、広い範囲から関連文献の当たりをつけるのに最適です。

- **`query`**: 文字列、または検索語のリスト。原則として日英ペア、同義語、固有名詞の原綴りとカタカナ表記を併記します。
- **`k`**: 取得件数 (デフォルト `10`)。コストが低いため、多めに取得して俯瞰することが可能です。
- **`where`**: Chromaメタデータフィルター（`source_type`、`itemKey` などで絞り込み可能）。
- **`include_notes`**: Zoteroノートを検索対象に含めるか (デフォルト `false`)。

### `rag_search`（段落レベル）

検索に合致した具体的な段落（チャンク）の本文テキストを抽出します。

- **`query`**: 文字列、または検索語のリスト。原則として日英ペア、同義語、固有名詞の原綴りとカタカナ表記を併記します。
- **`k`**: 取得件数 (デフォルト `5`)。
- **`where`**: Chromaメタデータフィルター。
- **`include_item_keys`**: 特定の資料 (`itemKey`) に検索範囲を絞り込むリスト。
- **`exclude_chunk_ids`**: 既に取得したチャンク (`id`) を除外し、新たな情報を取得するためのリスト。
- **`context_window`**: ヒットした段落の前後何段落を含めるか (デフォルト `0`)。必要がない限り0に保ちます。
- **`include_notes`**: Zoteroノートを検索対象に含めるか (デフォルト `false`)。
- **`auto_expand`**: 日英対訳・同義語を自動生成するか (デフォルト `true`)。既に日英両方を渡した通常検索では自動的に省略されます。
- **`hybrid`**: ベクトル検索とローカル FTS5 の語彙検索を RRF で統合するか (デフォルト `true`)。固有名詞や専門用語の一致に有効です。
- **`language_balance`**: 候補に日英両方がある場合、最終結果に各言語を最大2件ずつ確保するか (デフォルト `false`)。利用には `lang` メタデータを含む再インデックスが必要です。
- **`search_mode`**: 通常は `"default"`、民族誌的・経験的事例を探す場合は `"case"`。事例モードでは仮想事例文（HyDE）と抽象度展開を併用し、前後1段落を既定で返します。

### `hierarchical_search`（資料・章→段落）

俯瞰的な質問、関連文献の発見、複数資料の比較では最初に使います。資料要約と章要約から候補を選び、その候補内の段落検索と全体への直接検索をRRFで統合します。

検索可能なLLMノード要約は`__sum_node`索引から候補選定に使われます。要約の欠落や索引障害が
あっても資料を見失いにくいよう、既定では全体への直接段落検索（direct fallback）も併用します。

- **`k`**: 最終的に返す根拠段落数 (デフォルト `8`)。
- **`k_items`**: 要約層で残す候補資料数 (デフォルト `12`)。
- **`include_direct`**: 要約で取りこぼした資料を拾う全体段落検索を併用するか (デフォルト `true`)。
- **`return_summaries`**: 候補資料と段落に要約スニペットを付けるか (デフォルト `true`)。

### `get_item_summary` / `rag_search(search_mode="case")`

- `get_item_summary(item_key=...)`: 保存済みの資料要約全文をローカルDBから取得します。
  検索対象として採択されたLLM資料要約だけを返し、未生成の場合は空になります。
- 要約は検索候補を絞るための索引であり、引用可能な事実源ではありません。研究上の主張や
  書誌情報は、検索結果の原文チャンクまたはZotero原資料で必ず確認してください。
- 事例を探すときは `rag_search(query=..., search_mode="case")` を使います。構造化事例DBは
  廃止済みで、これは索引を介さず原文段落をHyDE・抽象度展開つきで直接検索するモードです。

### `get_chunk_context`（前後文脈）

`rag_search` で取得した特定のチャンクについて、その周辺の段落を再検索（ベクトル計算）なしで取得します。

- **`chunk_id`**: 起点となるチャンクID（例: `ABCDEFGH:p12:para3:part0`）。
- **`window`**: 前後に展開する段落数 (デフォルト `2`)。

### `get_item_details`（書誌詳細）

特定の資料の抄録（Abstract）、タグ、全著者リストなどのZoteroメタデータを取得します。

- **`item_key`**: 対象の資料の Zotero `itemKey`。

### `list_recent_items`（最近の資料）

ライブラリに直近で追加・更新された文献一覧を取得します。

- **`limit`**: 取得件数 (デフォルト `20`)。

### `get_document_outline`（目次）

PDFまたはEPUBドキュメントの目次（章構造）を取得します。章単位で検索範囲を絞り込む際の事前調査に使います。

- **`attachment_key`**: ZoteroのattachmentKey（例: `ABCDEFGH`）。

### `server_status`（状態確認）

サーバーの稼働状態と設定を確認します。応答が得られない場合や動作が不安定な場合に最初に呼び出してください。

返り値に含まれる情報：
- `status`: `"ok"` または `"error"`
- `chroma_dir` / `chroma_dir_exists`: ChromaDBのパスと存在確認
- `collections`: コレクション名とドキュメント数
- `emb_model` / `emb_profile` / `emb_device`: 埋め込みモデルの設定
- `emb_model_loaded`: 埋め込みモデルが既にメモリにロード済みかどうか
- `errors`: エラーがある場合の原因と対処法

### `force_reload_index`（索引再読込）

ChromaDBのインデックスとメタデータを強制的にリロードします。インデクサーを新たに実行した直後に「Error finding id」エラーや検索結果が空になる場合に使用してください。引数はありません。

### `get_debug_logs`（ログ）

サーバーのログファイルから最新N行を取得します。ChromaDBエラー、コレクションのリロードイベント、クエリ失敗時のトレースバックを診断するために使用してください。

- **`lines`**: 取得するログ行数（デフォルト `100`、最新行から遡って返します）

### `build_citation_network`（引用ネットワーク構築）

特定のZotero文献について、以下の両方を一括で実行し、引用ネットワークのデータベースを構築します。
1. **引用先の抽出**: 取り込み時に境界保持されたV3チャンク（bibliography/endnote/footnote zone）から参照候補を復元し、参照頻度（cite_count）の高い順に最大50件をSemantic Scholarで解決して保存（旧EPUB二重パースは廃止）。
2. **被引用の抽出**: Semantic Scholar APIを利用して「外部のどの論文から引用されているか」を取得して保存。

- **`item_key`**: 対象のZotero `itemKey`。

> **参照バジェットについて**: 参照が大量にある書籍（編著など）では、重要度の低い参照は `s2_status='skipped'` として保留されます。`uv run src/update_citations.py --resume-skipped` でバジェットを増やして後から再解決できます。

### `get_references_for_item` / `get_chunk_references`

構築済みのデータベースから、特定の資料（または段落チャンク）が「どの文献を引用しているか」を取得します。
- `get_references_for_item`: 資料全体に対する参照文献の一覧を取得
- `get_chunk_references`: 特定の段落チャンクIDに対する参照文献を取得

結果の `relation_key` は誤関係の報告に使う安定IDです。`context_count` と `raw_reference_count` は原文根拠の有無を判断する手がかりです。

### `get_cited_chunks_for_item` / `get_chunk_citations`

マッピング済みのデータベースから、特定の資料（または段落チャンク）が「外部の論文からどのような文脈で引用されているか」を取得します。これにより、「この文献の中で最も外部から議論されている（引用されている）重要な段落はどこか？」を特定できます。
- `get_cited_chunks_for_item`: 資料全体で被引用数が多い順にチャンク一覧を取得
- `get_chunk_citations`: 特定の段落チャンクIDに対する被引用コンテキストの一覧を取得

### `report_citation_relation` / `list_citation_relation_reports`

- `report_citation_relation(relation_key, reason, details)`: 具体的な根拠がある誤関係を人間の確認待ちとして報告
- `list_citation_relation_reports(status=...)`: `pending`、`disabled`、`kept`、`all` の報告状態を参照

Semantic Scholarの関係は原則として信頼します。分野の違いや意外性だけでは報告してはいけません。原資料の参考文献一覧に存在しない、識別子が別著作を指す、引用方向が逆などの具体的根拠がある場合だけ報告します。

報告は即時Disableではありません。最終判断は `Maintenance-Widget.command` で人間が行います。ClaudeはDisableやKeepを実行できません。

### `report_summary_quality` / `list_summary_quality_reports`

- `report_summary_quality(...)`: 資料要約・章要約に、原文と矛盾する記述や裏付けのない主張など具体的根拠がある問題を発見したとき、人間の確認待ちとして報告します。
- `list_summary_quality_reports(status=...)`: `pending` などの報告状態を参照します。
- 報告は助言的・可逆で、実行時点でレコードが削除・確定されるわけではありません。単に意外・話題が離れているだけでは報告しません。

### `suggest_unowned_works`（未所蔵文献）

引用ネットワークを集計し、複数の所蔵文献から参照されているのにライブラリにない文献を
ランキングします。LLMや外部APIは呼ばず、構築済みのローカルDBだけを使います。

- **`scope_item_keys`**: 省略時はライブラリ全体。指定時はそのアイテム群に隣接する文献だけを集計。
- **`direction`**: `"references"` は所蔵文献が引用する重要文献、`"citations"` は複数の所蔵文献を引用する隣接研究を発見。
- **`k`**: 最大取得件数（デフォルト `20`）。
- **`min_citing_items`**: 何件以上の異なる所蔵文献に隣接することを求めるか（デフォルト `2`）。

### `related_items`（関連資料）

指定した所蔵文献に近い別の所蔵文献を、根拠つきで返します。

- **`item_key`**: 起点となるZoteroアイテム。
- **`method`**: `"coupling"`（共有参考文献）、`"cocitation"`（同じ外部論文からの共引用）、
  `"semantic"`（本文チャンク平均ベクトル）、
  `"hybrid"`（引用2方式+意味類似の等重みRRF、デフォルト）。
- **`k`**: 最大取得件数（デフォルト `10`）。

結果の `evidence` には共有参照数、共引用元数、内容類似度が含まれます。

### `extract_references_for_item` / `confirm_reference_match`

PDF・HTML・EPUBの参考文献候補を構造化し、正準worksグラフへ解決します。

- `extract_references_for_item` は既定で `dry_run=true` のためDBを書き換えません。
- `use_llm=true` では候補テキストを `LLM_STANDARD` へ送信します。資料単位の除外タグは2026-07-27に撤去され、`LLM_STANDARD` の設定有無が唯一のゲートです。
- 保存時は DOI/ISBN、CiNii Research (`CINII_APP_ID` 設定時)、NDL Searchの順に候補を照合し、低信頼結果も根拠とともに保持します。
- `confirm_reference_match(edge_id, work_id)` で低信頼エッジを正しいworkへ付け替え、`work_id` を省略すると棄却します。

### `promote_chapters` / `detect_translation`

- `promote_chapters` は章著者が複数章で明確に異なる場合だけ章を子workへ昇格します。既定はdry-runです。
- `detect_translation` はZotero Extraの明示原題、次にNDLの原タイトル候補を使います。未検証のLLM推測だけではリンクを作りません。既定はdry-runです。

---

## ⚙️ 最適化仕様と返り値の解釈

### マルチクエリ・バッチ検索による最適化

`search_items` および `rag_search` の `query` パラメータには、可能な限り **文字列のリスト**を渡してください。日本語資料と英語資料の片方だけに検索結果が偏らないよう、次の組み合わせを基本とします。

- 同じ概念を表す日本語クエリと英語クエリ
- 分野で使われる同義語、上位概念、具体語
- 人名・地名・専門用語の原綴りとカタカナ表記

例: `rag_search(query=["贈与論 互酬性", "gift exchange reciprocity Mauss", "モース 贈与"])`

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
| `lang` | str | `"ja"` / `"zh"` / `"en"` / `"other"` |
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
   → V3チャンク（参考文献・注のzone）から引用先を抽出し、外部APIから被引用データを取得して一括でデータベースを構築

2. get_cited_chunks_for_item(item_key="KEY1")
   → 外部から最も多く引用されている重要な段落のランキングを取得

3. get_chunk_citations(chunk_id="...")
   → その段落が「どんな論文で、どのように言及されているか」の文脈テキストを取得し、学術的な評価や影響を分析
```

**ケースD：テーマと無関係な文献も含めて事例を探したい場合**
```
1. rag_search(
       query=["威信財の再分配の事例", "prestige goods redistribution ethnographic case"],
       search_mode="case",
       k=10
   )
   → 文献テーマで事前に絞らず、全チャンクをHyDE・抽象度展開つきで直接検索

2. get_chunk_context(chunk_id="...", window=3)
   → 有望な事例の前後を広げ、記述の主体・地域・時期を確認

3. rag_search(
       query=["具体的な実践名", "原綴り・地域名"],
       where={"itemKey": "KEY1", "chapter": "該当章"},
       search_mode="case",
       k=10
   )
   → ヒットした資料・章の内部を具体語で掘り下げる
```

事例検索では、資料レベルのテーマが検索対象と一致しないことがあるため、最初から
`search_items` で候補資料を限定しないでください。LLM設定やネットワークに問題がある場合も、
自動拡張だけを省略して元クエリによる検索を継続します。

**ケースE：引用関係をたどって新しい文献を探したい場合**
```
1. related_items(item_key="KEY1", method="hybrid", k=10)
   → 所蔵文献の中から、引用構造または本文内容が近い資料を根拠つきで発見

2. suggest_unowned_works(direction="references", min_citing_items=2, k=20)
   → ライブラリ内の複数文献が参照する未所蔵の基礎文献・重要文献を発見

3. suggest_unowned_works(
       scope_item_keys=["KEY1", "KEY2"],
       direction="citations",
       min_citing_items=2,
       k=20
   )
   → 関心のある所蔵文献群をまとめて引用している隣接研究を発見

4. DOI・タイトルを確認してZoteroへ追加し、ライブラリ更新後にrag_searchで本文を探索
```

結果にはランキング根拠となる所蔵 `itemKey` が含まれます。現段階の同一文献判定は
S2 ID、DOI、正規化タイトルによる暫定方式で、将来 `works` 正準IDへ移行予定です。

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
