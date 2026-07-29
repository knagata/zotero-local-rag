# 環境設定

[READMEへ戻る](../README.md)

通常はセットアップウィザードで `.env` を管理します。

```bash
uv run scripts/setup_wizard.py
uv run scripts/setup_wizard.py --status
```

手動設定のひな形は [`.env.example`](../.env.example) です。`.env`と`.env.policy`はGitに追加されません。

## セットアップのプリセット

初期設定は `Setup.command` が行います。**機能フラグの既定はすべて0**で、ウィザードが
選択に応じて明示的に書き込みます。有効なのに必要な鍵が無い場合は、機能名と不足鍵名を
挙げて起動時に停止します（黙って縮退しません）。

| | 1 最小 | 2 ローカル | 3 フル |
|---|---|---|---|
| PDF構造化 | しない（平文チャンク） | する | する |
| 構造化エンジン（境界未満） | — | Docling | Docling |
| 構造化エンジン（境界以上） | — | Docling / Granite | Mistral OCR |
| LLM機能（AI目次・階層要約・クエリ拡張・参照抽出・OCR監査） | なし | なし | あり |
| Citation Network | S2鍵があれば | S2鍵があれば（推奨） | あり |
| **課金** | **なし** | **なし** | あり |

3つの選択軸は独立しています（エンジンの選択だけはPDF構造化が前提）。プリセットは
その既定値セットにすぎないので、「PDFは平文でよいがEPUBの階層要約は欲しい」のような
組み合わせも選べます。

### 構造化エンジンの性格

| エンジン | 費用 | 速度 | 精度 | 備考 |
|---|---|---|---|---|
| Docling | 無料 | 早い | 標準 | 表・数式は最良 |
| Granite | 無料 | 約2.3倍遅い | 総合で最良 | 専用venv必須・Apple Silicon限定 |
| Mistral OCR | **有料**（頁単位） | 早い | 長編スキャンで最良 | Batch queue経由。既存チャンクは採用時まで不変 |

スキャンPDFは局所OCR（日本語=NDLOCR-Lite、他=RapidOCR）でテキスト化してから、
上記エンジンで構造化します。**NDLOCR-Liteは日本語資料に強く推奨**しますが、外部バイナリ
（`NDLOCR_BIN`）が必要なため任意です。未導入でもRapidOCRで動作します。

## Core

| 変数 | 用途 | 既定 |
|---|---|---|
| `FEATURE_LEVEL` | 選んだプリセットの記録（`core`／`citation`／`llm`）。表示用で、機能の可否は個々のフラグが決める | `core` |
| `ZOTERO_DATA_DIR` | Zoteroデータフォルダ | `~/Zotero` |
| `CHROMA_DIR` | ベクトル索引のディレクトリ | `data/chroma` |
| `EMB_PROFILE` | `fast`または`bge` | `fast` |
| `EMB_MODEL` | 埋め込みモデルのパスまたはID | プロファイルから選択 |
| `EMB_DEVICE` | `cpu`、`mps`、`cuda` | 環境から選択 |
| `HF_HUB_OFFLINE` | `1`でHugging Faceへの接続を禁止 | 任意 |
| `CHROMA_HNSW_SYNC_THRESHOLD` | HNSW索引を永続化する頻度。大きくすると再取り込みが高速化する | `100` |

`FEATURE_LEVEL`は管理用で、セキュリティ境界ではありません。

## V3データプレーン（現行本番）

本番の検索・取り込みはV3へ切り替え済みです。次の5値はカットオーバー後に設定されており、
旧値へ戻せば設定のみでロールバックできます。

| 変数 | 用途 | 現行の値 |
|---|---|---|
| `INGEST_STRUCTURED_V3_ENABLE` | 構造化取り込み（zone付与・文書構造）を有効化 | `1` |
| `HIERARCHICAL_SEARCH_V2_ENABLE` | 要約ノードから子孫leafへ検索をルーティング | `1` |
| `CHROMA_COLLECTION` | active Chroma collection | `zotero_paragraphs_v3` |
| `MANIFEST_PATH` | active manifest | `data/manifest_v3.json` |
| `LEXICAL_DB_PATH` | active FTS（語彙索引） | `data/lexical_v3.sqlite3` |

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
| `MAINTENANCE_AUTO_APPROVE` | `1`ならローカル更新を自動許可。有料要約・Mistralは常に明示許可 | `1` |

### クラウド送信ポリシー

クラウドへ本文を送るかどうかは**機能フラグそのものが決めます**（`LLM_*`ロール、
`PDF_AI_TOC_FAST_PATH_ENABLE`、`PDF_MISTRAL_TOC_QUEUE_ENABLE`）。既定はすべて`0`で、
有効なのに必要な鍵が無ければ起動時に停止します。

資料単位の除外タグ・`*_ALLOW_CLOUD_ALL`・`MISTRAL_OCR_FALLBACK_ENABLE`は2026-07-27に
撤去しました。**Zoteroライブラリの資料はRAGで使う時点でチャンクがクラウドへ送られる**ため、
資料単位で「クラウド不可」を表明する仕組みは実態と噛み合っていませんでした。

詳細は
[LLMとプライバシー](llm-and-privacy.md) を参照してください。

### PDFルーティング

| 変数 | 用途 | 既定 |
|---|---|---|
| `PDF_AI_TOC_FAST_PATH_ENABLE` | `1`で、outlineのないclean-text PDFの冒頭テキストからAI目次を推定し、全文見出し照合gate合格時だけDoclingを省略する。クラウド送信のため明示opt-in | `0` |
| `PDF_AI_TOC_MIN_PAGES` | AI目次推定を試す最小PDFページ数 | `30` |
| `PDF_AI_TOC_SAMPLE_PAGES` | AI目次推定へ渡す冒頭PDFページ数 | `20` |
| `PDF_AI_TOC_MIN_COVERAGE` | 推定見出しのうち全文で順序付き再発見できる必要割合 | `0.90` |
| `PDF_AI_TOC_MIN_STRUCTURED_CHUNK_RATIO` | 本文アンカー適用後に構造パスを持つ必要があるチャンク割合 | `0.80` |
| `PDF_AI_TOC_DOCLING_REFERENCES_ENABLE` | `1`で、AI目次採用PDFの参照/文末注セクションのページ範囲だけDoclingで再解析し1エントリ=1チャンクで本文へmerge。fail-closed | `0` |
| `PDF_MISTRAL_TOC_QUEUE_ENABLE` | `1`で、スキャンPDF（OCR層なし、またはOCR層品質gate不合格）のうち30頁以上をMistral OCR専用batch候補として保留する。30頁未満・本フラグ無効時はDocling | `0` |
| `PYMUPDF_NATIVE_OUTLINE_ANOMALY_RATIO_MAX` | 内蔵outlineがある場合に偽陽性として許容するscanned/corruptedページの上限割合 | `0.02` |
| `PDF_POPPLER_TEXT_FALLBACK` | `1`で、custom fontをUnicode復号できないページをローカル`pdftotext`で再抽出し品質改善時だけ採用 | `1` |

`.env` には現行本番として `PDF_AI_TOC_FAST_PATH_ENABLE=1` / `PDF_AI_TOC_MIN_PAGES=30` /
`PDF_MISTRAL_TOC_QUEUE_ENABLE=1` / `PDF_AI_TOC_DOCLING_REFERENCES_ENABLE=1` が常設されています。
AI目次fast pathは`PDF_AI_TOC_FAST_PATH_ENABLE`が唯一のゲートです（資料単位の除外タグは
2026-07-27に撤去）。
deferし、30頁未満・queue無効時はDoclingへ戻ります。AI目次の見出しcoverage不合格は
既存のqueue/Docling policyに従います。
AIの印刷ページ番号は採用せず、本文で再発見した見出しのページとreading orderだけを構造境界に
使います。

### ローカルOCR・クラウドOCR

| 変数 | 用途 | 既定 |
|---|---|---|
| `RAPIDOCR_DPI` / `RAPIDOCR_MIN_CONFIDENCE` | 固定レイアウトEPUB・明示re-OCR向けの英/他言語OCR（通常PDFルートでは使わない） | `200` / `0.45` |
| `NDLOCR_BIN` / `NDLOCR_DPI` / `NDLOCR_TIMEOUT_SEC` | 日本語ローカルOCR（NDLOCR-Lite） | — / `200` / `14400` |
| `MISTRAL_OCR_API_KEY` / `MISTRAL_OCR_MODEL` / `MISTRAL_OCR_BASE_URL` | Mistral OCR（クラウド・Batch API）。専用queue採用に使用 | — |
| `MISTRAL_OCR_BATCH_MAX_INPUT_BYTES` | Batch入力のbase64 JSONL概算上限。大容量PDFを一括uploadせず、残りは次回Batchへ回す | `104857600`（100 MiB） |
| `MISTRAL_OCR_BATCH_UPLOAD_WORKERS` | 分割したBatch入力を並列uploadする最大数 | `3` |

`Maintenance-Widget.command` は有料API処理をauto-approveしません。階層要約とMistral Batchは、
`MAINTENANCE_AUTO_APPROVE`の値にかかわらず該当質問で`y`を明示した場合だけ実行します。
Batch完了後にWidgetを再起動すると、
保存済み結果を回収し、ページcoverage・文字量・言語・構造などの品質gateを通過した資料だけを
V3へ採用します。採用済みBatchは状態ファイルに記録され、再実行しても二重採用しません。

### GROBID enrichment（任意・英語学術論文）

| 変数 | 用途 | 既定 |
|---|---|---|
| `GROBID_ENRICHMENT_ENABLE` | `1`で、独立workerから英語の`journalArticle`/`conferencePaper`/`preprint`をGROBID reference enrichment対象にする。埋め込みtransaction内では呼ばない。`0`のままworkerを実行すると、黙って何もせずに終わらずexit code 2で停止する | `0` |
| `GROBID_URL` | ローカルGROBID REST serviceのbase URL | `http://127.0.0.1:8070` |
| `GROBID_TIMEOUT_SEC` | 1資料あたりのHTTP timeout秒 | `120` |

### 構造・要約を手動で構築する場合

active collectionへ切り替え済みの環境では通常 `Maintenance-Widget.command` に任せますが、
別collectionへ明示構築する場合は両CLIへ同じcollectionを渡します。

```bash
uv run python scripts/rebuild_document_structure.py --all --collection zotero_paragraphs_v3
uv run python scripts/audit_v3_cutover.py --new-only --new-collection zotero_paragraphs_v3 \
  --output data/quality/server_database_gate.json
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

構造化事例DBは廃止済みのため、事例生成・品質確認は実行しません。事例を探す用途は原文を
直接検索する `rag_search(search_mode="case")` が担います。
