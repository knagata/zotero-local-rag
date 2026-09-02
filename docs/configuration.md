# 環境設定

[READMEへ戻る](../README.md)

通常はセットアップウィザードで `.env` を管理します。

```bash
uv run scripts/setup_wizard.py
uv run scripts/setup_wizard.py --status
```

手動設定のひな形は [`.env.example`](../.env.example) です。`.env`と`.env.policy`はGitに追加されません。
`Setup.command` と `setup_wizard.py --server` は設定と埋め込みモデルの初回ダウンロードを行います。
DB構築・埋め込み処理・クラウドAPIは実行しません。

## セットアップ方式

初期設定は `Setup.command` が行います。**機能フラグの既定はすべて0**で、ウィザードが
選択に応じて明示的に書き込みます。有効なのに必要な鍵が無い場合は、機能名と不足鍵名を
挙げて起動時に停止します（黙って縮退しません）。

| | 1 Minimal | 2 Custom |
|---|---|---|
| PDF構造化 | しない（平文チャンク） | する／しないを選択 |
| 構造化エンジン | — | ページ境界の前後ごとにDocling／Granite／Mistral |
| LLM機能 | なし | 5機能を個別に選択 |
| Citation Network | なし | S2キーを入力して有効化可能 |
| **課金** | **なし** | 選択内容による |

CustomではCitation Network、PDF構造化、AI目次、OCR監査、クエリ拡張、階層要約、
参考文献抽出を一項目ずつ設定します。PDF構造化を有効にすると、ページ境界未満と以上の
エンジンを独立して選べるため、`Granite / Granite`や`Docling / Mistral`など任意の
組み合わせが可能です。Graniteは常に選択肢へ表示され、専用環境が無い状態で選ぶと、
Apple Silicon搭載Macではウィザードが導入確認後に専用venvを作成します。
APIキー入力には非表示入力を使い、入力直前にもその旨を表示します。

### 構造化エンジンの性格

| エンジン | 費用 | 速度 | 精度 | 備考 |
|---|---|---|---|---|
| Docling | 無料 | 早い | 標準 | 表・数式は最良 |
| Granite | 無料 | 約2.3倍遅い | 総合で最良 | 専用venv必須・Apple Silicon限定 |
| Mistral OCR | **有料**（頁単位） | 早い | 長編スキャンで最良 | Batch queue経由。既存チャンクは採用時まで不変 |

通常の画像PDFは、構造化ONなら選択したDocling／Granite／Mistral、OFFならDoclingの
平文OCR経路で文字抽出します。NDLOCR-LiteとRapidOCRは固定レイアウトEPUBと明示的な
再OCRで使います。**NDLOCR-Liteは日本語資料に強く推奨**し、Customでは未導入時に
ウィザードから無料のローカルツールとして導入できます。

## Core

| 変数 | 用途 | 既定 |
|---|---|---|
| `FEATURE_LEVEL` | セットアップ方式の記録（`core`／`custom`）。表示用で、機能の可否は個々のフラグが決める | `core` |
| `ZOTERO_DATA_DIR` | Zoteroデータフォルダ | `~/Zotero` |
| `CHROMA_DIR` | ベクトル索引のディレクトリ | `data/chroma` |
| `EMB_PROFILE` | `fast`または`bge` | `fast` |
| `EMB_MODEL` | ダウンロード済み埋め込みモデルのローカルパス（手動設定時はIDも可） | `data/models/` 配下 |
| `EMB_DEVICE` | `cpu`、`mps`、`cuda` | 環境から選択 |
| `HF_HUB_OFFLINE` | `1`でHugging Faceへの接続を禁止 | 任意 |
| `CHROMA_HNSW_SYNC_THRESHOLD` | HNSW索引を永続化する頻度。大きくすると再取り込みが高速化する | `100` |

`FEATURE_LEVEL`は管理用で、セキュリティ境界ではありません。

## Remote MCP（任意）

Remote MCPは`src/rag_mcp_http_server.py`だけが読みます。従来のstdio版
`src/rag_mcp_server.py`には影響しません。

| 変数 | 用途 | 既定 |
|---|---|---|
| `REMOTE_MCP_PUBLIC_URL` | OAuth discoveryとcallbackに使う公開HTTPS origin。`/mcp`は付けない | 必須 |
| `REMOTE_MCP_GOOGLE_CLIENT_ID` | Google OAuthクライアントのClient ID | 必須 |
| `REMOTE_MCP_GOOGLE_CLIENT_SECRET` | Google OAuthクライアントのClient secret | 必須 |
| `REMOTE_MCP_ALLOWED_GOOGLE_EMAILS` | 接続を許可する確認済みGoogleメールアドレス。カンマ区切り | 必須 |
| `REMOTE_MCP_HOST` | HTTP待受。Funnel経由を強制するためloopback以外は拒否 | `127.0.0.1` |
| `REMOTE_MCP_PORT` | Funnelが転送するローカルHTTPポート | `8000` |

公開MCP endpointは`REMOTE_MCP_PUBLIC_URL`に`/mcp`を加えたURL、Google OAuthクライアントの
callbackは`/auth/callback`を加えたURLです。設定不足、HTTPの公開URL、loopback以外の待受、
許可ユーザーなしはいずれも起動前に拒否します。

## Citation GraphのGoogle OAuth公開（任意）

従来の`citation_graph/server.py`は引き続き`127.0.0.1`だけで待ち受け、ローカルでは
`http://localhost:7234`を認証なしで利用できます。`citation_graph/remote_proxy.py`は別プロセスの
公開専用入口で、Google OAuthを通過した許可メールのブラウザだけをローカルサーバーへ転送します。

| 変数 | 用途 | 既定 |
|---|---|---|
| `CITATION_GRAPH_PUBLIC_URL` | Funnelで公開するHTTPS origin。pathは付けない | 必須 |
| `CITATION_GRAPH_SESSION_SECRET` | ブラウザセッション署名鍵。32文字以上のランダム値 | 必須 |
| `CITATION_GRAPH_LOCAL_URL` | 従来のローカルCitation Graph | `http://127.0.0.1:7234` |
| `CITATION_GRAPH_REMOTE_HOST` | OAuth proxyの待受。loopback以外は拒否 | `127.0.0.1` |
| `CITATION_GRAPH_REMOTE_PORT` | Funnelが転送するOAuth proxyのポート | `7244` |

Google Client ID、Client secret、許可メールは`REMOTE_MCP_GOOGLE_CLIENT_ID`、
`REMOTE_MCP_GOOGLE_CLIENT_SECRET`、`REMOTE_MCP_ALLOWED_GOOGLE_EMAILS`を共用します。Google側には
`CITATION_GRAPH_PUBLIC_URL/auth/callback`を承認済みリダイレクトURIとして追加します。

## Citation Network

| 変数 | 用途 |
|---|---|
| `S2_API_KEY` | Semantic Scholar APIキー。Citation Networkでは必須 |
| `ZOTERO_USER_ID` + `ZOTERO_API_KEY` | 解決したDOIのZotero書き戻し。任意 |
| `CINII_APP_ID` | CiNii Research v2。任意 |
| `CROSSREF_MAILTO` | Crossref polite pool。任意 |

## LLM

LLMロールはプロバイダ接頭辞つきで指定します（例: `deepseek:deepseek-v4-pro`）。

| 変数 | 用途 |
|---|---|
| `LLM_CHEAP` | 大量処理（leaf要約、クエリ拡張） |
| `LLM_STANDARD` | 親要約の還元・AI目次推定・参考文献抽出と一次フォールバック |
| `LLM_REVIEW` | 品質確認と最終フォールバック |
| `DEEPSEEK_API_KEY` | いずれかのロールが `deepseek:*` を使う場合に必須 |
| `DEEPSEEK_THINKING` / `DEEPSEEK_REASONING_EFFORT` | DeepSeekの思考モード・推論強度 |
| `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` | ロールが `anthropic:` / `gemini:` の場合に必須 |
| `LLM_OPENAI_BASE_URL` / `LLM_OPENAI_API_KEY` | ロールが `openai_compat:` の場合に必須 |
| `SUMMARY_BATCH_MAX_ITEMS` / `SUMMARY_BATCH_WORKERS` | メンテナンス要約バッチの規模・並列度 |
| `MAINTENANCE_AUTO_APPROVE` | 既定は`0`（毎回確認）。`1`ならローカル更新を自動許可（有料要約・Mistralは常に明示許可） |

### クラウド送信ポリシー

クラウドへ本文を送るかどうかは**機能フラグそのものが決めます**（`LLM_*`ロール、
`PDF_AI_TOC_FAST_PATH_ENABLE`、`PDF_MISTRAL_TOC_QUEUE_ENABLE`）。既定はすべて`0`で、
有効なのに必要な鍵が無ければ起動時に停止します。

クラウド利用の有無は、機能フラグと実行時の明示承認で管理します。

詳細は
[LLMとプライバシー](llm-and-privacy.md) を参照してください。

### PDFルーティング

`PDF_AI_TOC_FAST_PATH_ENABLE=1` はCustomで明示的に選ぶ有料機能です。AI目次推定は
DB構築フェーズでも本文を送信し得るため、DBだけを先に検証するサーバー構築では
課金方針を確認してから有効にします。

| 変数 | 用途 | 既定 |
|---|---|---|
| `PDF_STRUCTURE_RECOVERY_ENABLE` | `1`でPDFの見出しと文書構造を復元する。`0`では構造化エンジンを呼ばず平文として索引化 | `0` |
| `PDF_STRUCTURE_ENGINE_SHORT` | ページ境界未満のPDFに使うエンジン（`docling`／`granite`／`mistral`） | `docling` |
| `PDF_STRUCTURE_ENGINE_LONG` | ページ境界以上のPDFに使うエンジン（`docling`／`granite`／`mistral`） | `docling` |
| `PDF_STRUCTURE_ENGINE_PAGE_BOUNDARY` | 短いPDFと長いPDFを分けるページ数 | `30` |
| `GRANITE_VENV_PYTHON` | Granite専用virtualenvのPython。アダプタと専用環境の両方がある場合だけ選択可能 | `tmp/granite_docling_venv/bin/python` |
| `PDF_AI_TOC_FAST_PATH_ENABLE` | `1`で、outlineのないclean-text PDFの冒頭テキストからAI目次を推定し、全文見出し照合gate合格時だけDoclingを省略する。クラウド送信のため明示opt-in | `0` |
| `PDF_AI_TOC_MIN_PAGES` | AI目次推定を試す最小PDFページ数 | `30` |
| `PDF_AI_TOC_SAMPLE_PAGES` | AI目次推定へ渡す冒頭PDFページ数 | `20` |
| `PDF_AI_TOC_MIN_COVERAGE` | 推定見出しのうち全文で順序付き再発見できる必要割合 | `0.90` |
| `PDF_AI_TOC_MIN_STRUCTURED_CHUNK_RATIO` | 本文アンカー適用後に構造パスを持つ必要があるチャンク割合 | `0.80` |
| `PDF_AI_TOC_DOCLING_REFERENCES_ENABLE` | `1`で、AI目次採用PDFの参照/文末注セクションのページ範囲だけDoclingで再解析し1エントリ=1チャンクで本文へmerge。fail-closed | `0` |
| `PDF_MISTRAL_TOC_QUEUE_ENABLE` | `1`で、短い／長いPDFのいずれかに`mistral`を指定した際のBatch queueを有効化。本フラグ無効時はDoclingへ退避 | `0` |
| `PYMUPDF_NATIVE_OUTLINE_ANOMALY_RATIO_MAX` | 内蔵outlineがある場合に偽陽性として許容するscanned/corruptedページの上限割合 | `0.02` |
| `PDF_POPPLER_TEXT_FALLBACK` | `1`で、custom fontをUnicode復号できないページをローカル`pdftotext`で再抽出し品質改善時だけ採用 | `1` |
| `PDF_OCR_FALLBACK` | `1`で、フォント復号に失敗したページだけをTesseractで再読取する。画像PDF全体の主OCRではない | `1` |
| `PDF_OCR_LANG` / `PDF_OCR_DPI` | Tesseractの言語と描画解像度。日本語データ検出時は`jpn+eng`を自動選択 | 自動 / `300` |

Customで `PDF_AI_TOC_FAST_PATH_ENABLE=1` を有効にすると、AI目次fast pathが資料本文を
LLMへ送り、フェーズ1のDB構築中にも課金が発生し得ます。料金を発生させずDBだけを検証したい
場合は、フェーズ1の間このフラグを`0`にしておき、フェーズ2の監査後に必要な資料だけを
明示的に再処理してください。AI目次fast pathは`PDF_AI_TOC_FAST_PATH_ENABLE`が唯一のゲートです。
AIの印刷ページ番号は採用せず、本文で再発見した見出しのページとreading orderだけを構造境界に
使います。

### ローカルOCR・クラウドOCR

| 変数 | 用途 | 既定 |
|---|---|---|
| `RAPIDOCR_DPI` / `RAPIDOCR_MIN_CONFIDENCE` | 固定レイアウトEPUB・明示re-OCR向けの英/他言語OCR（通常PDFルートでは使わない） | `200` / `0.45` |
| `NDLOCR_BIN` / `NDLOCR_DPI` / `NDLOCR_TIMEOUT_SEC` | 日本語ローカルOCR（NDLOCR-Lite）。Customで検出するか、公式GitHubの検証済み`1.0.0`タグから任意導入し、実行ファイルの絶対パスを保存 | — / `200` / `14400` |
| `MISTRAL_OCR_API_KEY` / `MISTRAL_OCR_MODEL` / `MISTRAL_OCR_BASE_URL` | Mistral OCR（クラウド・Batch API）。専用queue採用に使用 | — |
| `MISTRAL_OCR_BATCH_MAX_INPUT_BYTES` | Batch入力のbase64 JSONL概算上限。大容量PDFを一括uploadせず、残りは次回Batchへ回す | `104857600`（100 MiB） |
| `MISTRAL_OCR_BATCH_UPLOAD_WORKERS` | 分割したBatch入力を並列uploadする最大数 | `3` |

`Maintenance-Widget.command` は有料API処理をauto-approveしません。階層要約とMistral Batchは、
`MAINTENANCE_AUTO_APPROVE`の値にかかわらず該当質問で`y`を明示した場合だけ実行します。
Batch完了後にWidgetを再起動すると、
保存済み結果を回収し、ページcoverage・文字量・言語・構造などの品質gateを通過した資料だけを
V3へ採用します。採用済みBatchは状態ファイルに記録され、再実行しても二重採用しません。

CustomセットアップではTesseract本体と日本語言語データを検出します。不足時に同意すると、
Homebrewの`tesseract`／`tesseract-lang`をウィザードから導入します。Homebrewがない環境では
自動導入せず、手動コマンドを案内します。

### GROBID enrichment（任意・英語学術論文）

| 変数 | 用途 | 既定 |
|---|---|---|
| `GROBID_ENRICHMENT_ENABLE` | `1`で、独立workerから英語の`journalArticle`/`conferencePaper`/`preprint`をGROBID reference enrichment対象にする。埋め込みtransaction内では呼ばない。`0`のままworkerを実行すると、黙って何もせずに終わらずexit code 2で停止する | `0` |
| `GROBID_URL` | ローカルGROBID REST serviceのbase URL | `http://127.0.0.1:8070` |
| `GROBID_TIMEOUT_SEC` | 1資料あたりのHTTP timeout秒 | `120` |

### サーバーDB構築と階層要約

DBのゼロ構築・再構築は `Setup.command` から行います（真の初回は確認なし、既存DBがある
場合は常に`REBUILD`入力確認付きの案内。プロファイル変更時は必須、それ以外は任意）。
構築後に案内されるDB監査（Zotero/原本/DB整合性の
3監査、非破壊）が、現在のDB世代に結び付いた合格gateを作成します。有料の階層AI要約
（`Maintenance-Widget.command`の差分バッチ、または項目選択時の`SUMMARIZE`確認を伴う全件一括生成）は
このgateがなければ実行できず、監査後にDBが変わると次の要約実行前に停止します。

### 構造・要約を手動で構築する場合

本番collectionはV3固定です。構造の再構築は手動実行できますが、要約用gateは必ず
`uv run python scripts/run_db_audit.py`（Zotero・原本・DBの3監査）を通して作成します。
`audit_v3_cutover.py --new-only` 単独ではgateを作れません。

```bash
uv run python scripts/rebuild_document_structure.py --all --collection zotero_paragraphs_v3
uv run python scripts/run_db_audit.py
uv run python scripts/build_structure_summaries.py --all --collection zotero_paragraphs_v3 \
  --mode llm --limit 10 --embed --database-gate data/quality/server_database_gate.json
```

いずれも`--limit N`で分割でき、`--dry-run`で対象確認、`--retry-failed`で再開、
`--force`で差分スキップを無視して再生成できます。

## メンテナンス時のAI要約

```dotenv
SUMMARY_BATCH_MAX_ITEMS=20
SUMMARY_BATCH_WORKERS=10
```

`Maintenance-Widget.command` の要約更新は、文書構造V3の葉からbottom-upに要約を生成し、
LLM要約は `__sum_node` 検索索引へ反映します。有料要約は
`audit_v3_cutover.py --new-only`の合格レポートが現在のDB世代と一致するときだけ実行できます。
現行fingerprint一致分はLLM呼び出しゼロでskipします。

事例を探す場合は、原文を直接検索する `rag_search(search_mode="case")` を使います。
