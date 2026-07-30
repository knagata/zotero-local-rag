# Tasks

正本: `SPEC.md`。実装・検証の永続的な履歴はこのファイルに残す。
`evaluations/`のローカル生成レポートはZotero識別子や絶対パスを含み得るため追跡しない。

## Active

### 2026-07-30 全コードレビュー: 検出された欠陥の修正

`src/`全体＋Setup/Maintenance-Widgetから呼ばれるscriptsを5並列agentでレビューし、
明確かつ重大な瑕疵を検出（詳細はセッション記録参照）。利用制限による中断に備え、
1件直すごとにここへ進捗を反映してコミットする。

- [x] **F1: `reocr_adoption.py`のChromaロールバックフラグ順序**（2026-07-30）
  `adopt_prepared_reocr`が`collection.delete()`成功後・`upsert()`失敗時に
  ロールバックせず、チャンクが永久消失する。src/reocr_adoption.py:154-203
- [x] **F2: `docling_extract.py`の破損ページzoneタグ付け**（2026-07-30）
  `_patch_pages_with_docling_ocr`が`corrupted_unresolved`プレースホルダーに
  `zone="body"`を付けて索引化し、検索・要約にゴミ文字列が混入する。
  src/docling_extract.py:721-736
- [x] **F3: `rag_mcp_server.py`の`search_items` FFIデッドロック＋品質フィルタ欠落**（2026-07-30）
  `rag_search`/`_hierarchical_search_v2`にある「クエリ埋め込み事前計算」共通化を
  `search_items`にも適用し、`retrieval_policy_allowed`フィルタも揃える。
  src/rag_mcp_server.py:1630-1741
- [x] **F4: `verify_against_source.py`の読み取り失敗fail-open**（2026-07-30）
  ソースPDFが読めない場合、監査结果から除外されて`passed: true`のまま通過する。
  scripts/verify_against_source.py:132-161
- [x] **F5: `index_from_zotero.py`の削除失敗の握りつぶし**（2026-07-30）
  `_delete_by_attachment_keys(strict=False)`の例外が呼び出し側の空`except`で消え、
  削除失敗でも成功扱いでmanifestから孤立キーが消える。
  src/index_from_zotero.py:230-246,1669-1674
- [x] **F6: `rag_mcp_server.py`のグローバルChromaクライアント状態の排他制御**（2026-07-30）
  `_reset_col`/`_col`がロックなしで並行呼び出しされ得る。src/rag_mcp_server.py:232-508
- [ ] **F7: `run_mistral_ocr_batch.py`の`create_job`冪等性チェックポイント**
  課金ジョブ作成後に状態保存前でクラッシュすると再実行で重複課金が起こり得る。
  scripts/run_mistral_ocr_batch.py:191-211
- [ ] **F8: `item_vectors.py`のアイテム単位キャッシュ無効化**
  再取込後も古い埋め込みベクトルがキャッシュされ続け、`related_items`が古い結果を返す。
  src/item_vectors.py:44-99
- [ ] **F9: OCR系extractorの`ocr_pages`/`missing_pages`集計を共通化**
  yomitoku/rapidocrが実際の出力に関わらず全ページ主張する古いバグを、
  mistral/ndlocrで既に直っている集計ロジックに揃える。
  src/yomitoku_extract.py:277, src/rapidocr_extract.py:216
- [ ] **F10: `db_relations.py`の`resolve_work()`曖昧一致の決定性**
  `ORDER BY`なしのクエリで同点候補があると非決定的に同定される。
  src/db_relations.py:830-853, 2606-2644
- [ ] **F11: `chunk_store.py`のCHROMA_DIRパス解決バイパス**
  `DEFAULT_CHROMA_DIR`が`resolve_configured_path`を経由せず生の環境変数から
  直接計算される潜在バグ。src/chunk_store.py:16, scripts/audit_v3_cutover.py

### Phase 0: ゼロ再構築前の取り込み完全性修正

依存順: `Z0` → (`Z1`, `Z2`, `Z3`) → `Z4` → `Z5` → `Z6`。相互に独立する
extractor修正は並行実装する。修正完了まで埋め込み・正本DB再構築は行わない。

- [x] **Z0: source coverageの共通契約と回帰fixtureを定義する**（2026-07-29）
  - PDF page / EPUB spine / HTML EOFを
    `expected_units / attempted_units / text_units / blank_units / failed_units / truncated`
    で表現する。
  - 説明不能な欠落を含む部分成功をcanonicalへ渡さない純粋validatorを追加する。
- [x] **Z1: PDFページ例外・反復ヘッダ除去の欠落経路を閉じる**（2026-07-29）
  - `load_page`/text抽出例外を真正な空ページと分離し、再試行またはfail-closedにする。
  - `page`の前ページ参照を防ぎ、除去で全滅したページを成功扱いしない。
- [x] **Z2: NDLOCRの実出力ページcoverageを正しく報告する**（2026-07-29）
  - render済み、JSON出力済み、本文あり、missingを別々に記録する。
  - 欠落JSONを全ページ処理済みと偽装せずlocal OCR gateで拒否する。
- [x] **Z3: HTML/EPUBの部分欠落を検出・拒否する**（2026-07-29）
  - HTMLのサイズ打切りとcase-insensitiveなstyle/script閉じタグを修正する。
  - EPUBをOPF spine期待集合と照合し、章例外・image-only/mixed spineを記録する。
  - 少量DOMテキストだけでfixed-layout OCR fallbackを回避できないようにする。
- [x] **Z4: インジェスト成功判定をcoverage validatorへ一本化する**（2026-07-29）
  - `pages_without_chunks`を記録するだけで`success`になる経路を廃止する。
  - extractor固有のqualityを共通coverageへ正規化し、不完全結果は書き込まない。
- [x] **Z5: clean rebuildを旧成果物から独立させ、保存境界を整理する**（2026-07-29）
  - `--rebuild`ではlegacy OCR/chunk/manifest/summaryを再利用しない。
  - `extract → validate → prepare → write → verify`へ限定的に分割する。
  - 書込み前後のID集合・attachment/item identity不変条件を一箇所へ集約する。
- [x] **Z6: 構造・検索・citationの到達性と構築前gateを完成する**（2026-07-29）
  - node/zone/policy同期、複数添付item、検索後段filterのunderfillを修正する。
  - citation mapperをactive collectionへ固定し、世代変更時にcacheを破棄する。
  - Zotero/manifest/Chroma/FTS/source coverage/node到達性の全gate合格を再構築条件にする。
- [x] **Z7: 実資料の非書込みスモークテストを行う**（2026-07-29）
  - PDF 2件・HTML 2件・EPUB 2件を、埋め込み・DB更新なしで抽出する。
  - 構造化EPUBはDOM本文が取れた時点で直行し、表紙・ロゴ・挿絵をOCRしない正本workflowを
    再確認。短いtitle/colophon spineをchunk下限で全損させず、画像のみのspineは
    `ignored_image_spines`として明示する。OCRは全ページ画像の固定レイアウトEPUBだけに限定。
  - PDF 1件・HTML 2件は直接抽出合格。234ページPDFは表紙1ページだけを未回収として
    fail-closedに検出し、部分抽出を成功扱いしないことを確認。
  - 埋め込み・canonical DB再構築は未実施。
- [x] **Z8: AI・OCR外部経路を実資料で非書込み検証する**（2026-07-29）
  - AI目次: outlineなし312ページPDFでaccepted、本文coverage 100%、構造化率99.34%。
    p9/p25/p261のIntroduction/Chapter 1/References見出し位置を原画像と照合。
  - AI構造要約: 英語2節は全8文のevidence検証を通過（discard 0）。日本語資料は3 nodeを
    実LLMで生成し、主題・日韓差・階級/政治・復讐/管理の論点を目視確認。DB保存関数は隔離。
  - Mistral Batch: 実スキャン1ページで1/1頁、7,595文字、12 chunks、原画像5箇所と読順一致。
    固定レイアウトEPUB `CAXCWCQB` 130頁も実Batch完走（job
    `b4d15390-6b6b-4d17-9da1-38e95f0c65c1`）。短いOCR頁の全損を修正し、空応答の図版頁は
    `quality_uncertain`なnon-text markerにして、309 chunks・130/130 spine coverage合格。
  - スキャンPDF Docling: 抽出後の`torch.mps.empty_cache()`がSIGSEGVする実バグを修正。
    同一persistent workerで2回連続、各13 chunks・1/1頁coverage合格。
  - 固定レイアウトEPUBのRapidOCR/Docling部分出力を文字数だけで採用しないstrict page-text
    gateを追加。Rapid空10頁のうち9頁に可読文字があることを目視し、Mistral fallbackへ送る。
  - 最終目視QA後の全テスト `870 passed, 2 subtests passed`。外部成果物はscratchのみで、埋め込み・
    canonical DB・manifestへの書込みは未実施。

### サーバー再構築の確定手順

- [x] **V3唯一化とサーバーDB構築を完了する**（2026-07-30、直後の項目で置き換え）
  - 旧collection・旧manifest・旧FTSへのruntime rollbackを廃止し、V3以外の設定をfail-closedにする。
  - ~~`Setup.command` / `setup_wizard.py --server` は設定だけを行い、DB構築・埋め込み・有料APIを実行しない。~~
    （2026-07-30 ユーザー判断で撤回。Setup自身がDB構築を案内・実行する形に変更。詳細は次項。）
  - ~~`Server-Database-Workflow.command` を `1) DB再構築 → 2) Zotero/原本/DB監査 →
    3) 有料階層要約 → 4) 要約監査` の順に別実行する。~~（同ファイルは廃止。次項参照。）
  - DB世代gateがなければ有料要約のAPI呼出しを開始しない、という制約自体は維持
    （`scripts/run_db_audit.py`が作成するgateに変更なし）。Custom設定でAI目次fast pathは
    DB構築時でも課金し得るため、必要に応じて明示的に無効化する。
- [x] **運用スクリプトを`Setup.command`/`Maintenance-Widget.command`の2本へ集約する**（2026-07-30）
  - `Server-Database-Workflow.command`を廃止。理由: 「初回構築」と「その後の運用」しか
    実際には無く、この2つには元々自然な置き場（Setup/Maintenance）があった。危険な操作
    （`--rebuild`・全件要約の一括課金）はファイルを分けて守るのではなく、個別の入力確認
    （`REBUILD`/`SUMMARIZE`手打ち）で守る形に変更（ユーザー判断）。
  - フェーズ2（Zotero照合＋原本照合＋DB完全監査＋gate作成）を`scripts/run_db_audit.py`に
    1本化。Setup（構築直後）・Maintenance-Widget（日常項目）の両方がこれを呼ぶ。
  - `Setup.command`: 設定保存後に`db_lifecycle.existing_database_state()`でDB有無を判定し、
    真の初回は`REBUILD`確認を省略、既存DBがある状態でのプロファイル変更時は確認を必須のまま
    維持（`db_lifecycle.run_rebuild()`/`run_audit()`を呼ぶ。DB構築・監査ロジック自体は
    `src/db_lifecycle.py`に切り出し、対話UIだけ`scripts/setup_wizard.py`側に残した。
    実行主体は`uv run src/index_from_zotero.py --rebuild`等へのサブプロセス呼び出しで、
    フェーズ1〜2のように別の`.command`ファイルへシェルアウトすることはなくなった、の意）。
  - `Maintenance-Widget.command`: 項目「DBを監査する」を追加。非破壊・無料なので既存の
    差分更新・引用ネットワーク更新と同じ既定onの区分に入れ、既定値は前回合格gateの有無で
    自動判定（無い/失効していれば既定yes、最新なら既定skip）。旧フェーズ3・4
    （全件要約一括生成・要約監査）も「全件要約を一括生成する」項目としてオプトイン統合
    （`SUMMARIZE`手打ち確認、既定off、auto-approve対象外）。
  - `docs/`・`README.md`・`TASKS.md`内の旧ファイル参照を更新。

### Phase 3: V3並行再取込とカットオーバー監査を完走する

- [x] ~~**V3全件再取込を小分けで実行する**~~ (2026-07-28)
  - manifest 586件（EPUB 200 / PDF 342 / HTML 44）。全件cutover監査が521/521で通過している
    ことが完了の根拠（監査は全legacy itemのV3カバレッジを要求する）。
  - `INGEST_STRUCTURED_V3_ENABLE=1`で`--limit`を使い、item単位transactionで再開可能なバッチとして実行する。
  - 各バッチ後にmanifest件数、V3 Chroma件数、FTS件数、artifact失敗/再試行件数を照合する。
  - OCR由来PDFは既存テキストをV3境界へ再構成し、再OCRは専用queueへ分離する。
  - テストbatch: 3件成功（EPUB 1・PDF 2）。ingestion code path・metadata stamping確認済み。
  - **`--source-type` filter 追加**: 重いscanned PDFに阻まれないようepub→html→pdfの順で処理可能に。
  - **RapidOCR semaphore leak**: macOSでDoclingのRapidOCRがsemaphoreをリーク、`--limit 3`以下推奨。
  - **flushごとmanifest保存**: クラッシュ時の進捗保全を追加。
  - 2026-07-23: AI目次fast pathの初期本番rolloutを含むPDF 3件×4 batch（計12件）が
    extraction失敗0・各回HNSW query smoke test成功で完了。走査位置は62/342添付まで進行。
    完了済み添付はmtime/fingerprint一致でskipされ、再埋め込みされていない。
  - 2026-07-23 17:23時点: supervisor batch 4が実行中。走査位置132/342、
    `VJJ9HRZ6`はAI目次coverage 100%で908 chunksを抽出し、埋め込み書込み中。
    RapidOCR安定性のため引き続き`--limit 3`で実行する。
  - 引き継ぎ資料: `dev-notes/current/66_v3_batch_handover.md`
- [x] ~~**AI目次gate不合格PDFをMistral OCR専用バッチへ送るルートを作成する**~~（2026-07-27。スキャンPDFのOCR層なし／stage-2不合格も30頁基準で同じ非正本Batch defer契約へ統合）
  - AI目次fast path不合格時は通常PDFバッチ内でDoclingへ自動fallbackせず、既存index・
    manifestを変更しないままスキップする。
  - item/attachment、AI目次の不合格reason・diagnostics、ページ数、言語、source fingerprintを
    永続的な候補状態へ記録し、通常再開時に同じPDFを無限再試行しない。
  - 候補状態からMistral OCR用JSON queueを決定的に生成し、item指定・limit・dry-run・再開を
    サポートする。通常の`--reocr-candidates`と混同しないreason/provenanceを保持する。
  - （当時の仕様）資料単位の除外タグを確認した。現在このタグ機構は撤去済みで、クラウド利用は
    機能フラグと明示承認で管理する。
  - Mistral結果は既存canonicalへ即時上書きせず、ページcoverage・文字量・言語・構造・
    hallucination gateを比較してからitem単位で明示採用する。
  - 候補生成、通常batchでのskip、再実行時skip、cloud policy拒否、Mistral失敗、品質gate不合格、
    明示採用・rollbackをfixtureで検証する。
  - 2026-07-23: 通常batchでのdefer、artifact status永続化、source stat一致時skip、専用JSON
    queue生成、`target_engine=mistral_ocr` dispatchを実装。
  - 2026-07-23: 公式Mistral Batch API（JSONL upload → `/v1/ocr` job → status再開 →
    output回収）を実装。結果はsource fingerprint、全ページcoverage、ページ重複、最低文字量、
    gibberish、反復artifactを検査し、合格分だけ非正本の採用queueへ出力する。採用queueは
    既存`--reocr-candidates`経路でV3正本へ書けるため、OCR APIの二重呼出しは発生しない。
  - 2026-07-23: 合成PDF 1件の実API smokeを実施。Batch用file uploadは成功したが、
    job作成がMistral APIの`402 Payment Required`で拒否されたため、非同期完走確認は
    アカウントのBatch利用・課金状態を解消後に再開する。upload後/job作成前の失敗でも
    `input_file_id`を保存し、再試行時に二重uploadしないよう修正済み。
  - 2026-07-23: 課金有効化後に同じ合成PDFで再試験し、`RUNNING → SUCCESS`、成功1・失敗0、
    output回収、品質gate合格まで実APIで確認。Batch応答は本文Markdown 8,156文字を返す一方
    `blocks=[]`だったため0チャンクになる問題を検出し、Markdown block fallbackを追加。
    保存済み実応答の再生で14チャンク・見出し1件を生成し、関連テスト19件成功。
  - 中断された`H5B2EANF`の部分書込み（Chroma 1,024 / FTS 896）を新ルートで清掃し、
    Chroma 0 / FTS 0 / inflight解除を確認。210ページのMistral候補としてqueue化済み。
  - 残り: 実候補でBatchを送信・回収し、品質gateレポートを確認してから明示採用する。
    canonical書込み失敗時のrollback fixtureも追加する。
    詳細: `dev-notes/current/72_pdf_routing_mistral_queue.md`
  - 2026-07-27: 通常PDFのRapidOCR/NDLOCR先行routeを廃止。スキャンPDFはOCR層なし、または
    stage-2品質gate不合格なら、30頁未満はDocling、30頁以上は明示有効なBatch queueへ
    deferする。混在レイアウトとRapidOCRの短行チャンク爆発を避けるため。固定レイアウトEPUBと
    明示re-OCRは変更しない。
- [x] ~~**GROBIDを学術論文PDF向け構造・引用抽出器としてpilot評価する**~~ (2026-07-23)
  - 対象はZotero `itemType`が`journalArticle` / `conferencePaper` / `preprint`のPDFに限定し、
    書籍・既存処理済みPDFの再埋め込み条件にはしない。
  - 固定3〜5資料で現行Doclingと同じPDFを処理し、見出し・段落順、参考文献分割、書誌field、
    本文中引用marker→参考文献link、処理時間、失敗率を比較する。
  - GROBID TEIを既存V3 block/node/reference契約へ変換するadapter案と、サービス停止・timeout・
    malformed/inconsistent TEI時に現行PDF routeへ戻るfail-closed条件を確認する。
  - 採用条件を満たした場合のみ、新規・未処理の対象種別PDFでGROBIDをpipeline候補に追加する。
    既存埋め込みはそのまま維持する。
  - GROBID 0.9.0/OpenJDK 21を隔離環境で実測。英語3件は見出し・参考文献・本文中引用linkが
    有望、日本語1件は見出し/参考文献0で不採用。全4件HTTP 200、1.27〜8.32秒。
  - 結論: canonical chunk抽出は置換せず、英語学術論文の構造・参考文献・引用linkを補強する
    optional enrichmentとして採用候補にする。詳細: `dev-notes/current/73_grobid_scholarly_pdf_pilot.md`
- [x] ~~**GROBID英語学術論文enrichmentをpipelineへ追加する**~~ (2026-07-23)
  - Zotero `itemType`が`journalArticle` / `conferencePaper` / `preprint`かつ英語の新規・未処理PDF
    だけを対象にし、既存処理済み資料を再埋め込みしない。
  - local REST client、timeout/health check、TEI parserを隔離adapterとして実装し、header候補、
    section evidence、`biblStruct`、本文`ref[type=bibr]` linkを既存V3契約へ変換する。
  - GROBID停止、invalid TEI、body/section空、引用link異常時は現行pipelineを継続し、embeddingを
    blockしない。日本語その他の言語にはrouteしない。
  - TEI fixtureによるunit test、service integration test、同一source再開・provenance testを追加する。
  - 埋め込みtransactionから分離した`run_grobid_enrichment.py`を実装。既定dry-run、item/limit、
    PDF SHA-256+processor versionによる再開skip、失敗非retryableを実装した。
  - GROBID TEIのreference、著者、年、DOI、container、本文citation marker linkをparseし、
    既存`reference_review_queue`へreview-onlyでstageする。works graphへ直接書かない。
  - 実論文`FRIZY8BS`で9 references、4/4 citation linksを保存し、同fingerprint再実行がservice
    停止中でもskipされることを確認。関連19テスト成功。
- [x] ~~**GROBID enrichmentの運用方法を確定する**~~ (2026-07-28)
  - **実バグ発見**: 唯一の入口`scripts/run_grobid_enrichment.py`が実行直前に
    `os.environ["GROBID_ENRICHMENT_ENABLE"]="1"`で**flagを無条件に上書き**していた。
    flagは装飾でしかなく、「有効なのにserviceが停止している」という最も起きやすい
    運用ミスをどこも検出しなかった。上書きを廃止し、flag offまたは
    `verify_enabled_features()`が問題を返す場合は1件も処理せずexit 2。
  - `feature_gates`へ登録。GROBIDは鍵を持たないので「設定済み」＝「起動中」であり、
    判定はhealth probe（`/api/isalive`、timeout 2秒、プロセス内memo化）しかない。
    `FEATURE_REQUIREMENTS`は**有効な機能しか`available()`を呼ばない**ので、既定0の環境では
    HTTPが1回も出ない（テストで固定）。診断snapshotにもprobeを入れない。
  - 運用: launchd常駐ではなく手動起動。health checkはworker起動時のみ。実行は埋め込み直後
    ではなく`--limit`分割の全件backfill 1回＋以後は手動追い込み（SHA-256と
    `grobid:0.9.0-crf`のskipで差分のみ）。詳細は`dev-notes/current/84_grobid_and_epub_fallback.md`。
  - 以下は当初の設計方針（維持）:
  - 基本方針: GROBIDをS2引用更新pipelineのローカル前処理として位置付ける。
    `PDF埋め込み → GROBID抽出 → reference照合/審査 → S2同定・外部引用network更新`の順にする。
  - GROBIDの参考文献原文・本文中引用位置は一次証拠として保持し、S2のpaper ID・正規化書誌・
    被引用情報で上書きしない。GROBID↔S2対応にはconfidence、照合根拠、resolver versionを保存する。
  - DOI一致を最優先し、DOIなしはtitle・author・yearで候補検索する。高信頼一致だけ自動確定し、
    曖昧一致・複数候補・書誌矛盾は`reference_review_queue`へ残す。
  - 実行タイミングを、埋め込みbatch直後・日次maintenance・全件完了後の一括backfillのどれに
    するか決定する。埋め込みtransactionとは分離し、GROBID障害で索引作成を止めない。
  - GROBID serviceの起動・停止・health checkを手動、Maintenance UI、launchd等のどこで管理するか
    決定し、OpenJDK/GROBID version・モデル・設定fingerprintを固定する。
  - 新規・未処理だけを対象にするか、既存の英語学術論文も再埋め込みなしで段階的backfillするか、
    item/limit/優先順位と所要時間を見積もって決定する。
  - `reference_review_queue`へ入った候補の審査、承認後のworks graph反映、本文citation markerと
    既存chunk IDの対応付け、未対応markerの扱いを運用手順として定義する。
  - timeout、invalid/低品質TEI、0 references、service停止、version変更時の再試行・skip・rollback・
    ログ保持期間を決め、通常の`--retry-failed`と混同しない専用再試行手順を文書化する。
  - 日本語その他の非対象言語を誤処理しない監査と、削除済みZotero itemに属するGROBID派生候補の
    purge方針を決定する。
- [x] ~~**非標準・固定レイアウトEPUBのフォールバックを実装する**~~ (2026-07-28・実装済みを確認)
  - 実データ検証: manifest上のEPUB **200件すべてが索引済み、チャンク0件は0件**。
    engine内訳は`epub_dom` 192 / `epub_dom_leaf_fallback`のみ3・混在2 /
    `ndlocr-lite_epub_fixed_layout` 2 / `rapidocr_epub_fixed_layout` 1。
  - note 74の5件は全て回収済み（XASKBF88 138 / EI7JN68R 183 / CAXCWCQB 1,110 /
    CCHED4LQ 1,474 / BMKYYFCZ 1,088 チャンク）。固定レイアウト3件は`epub:spineN` locatorへ復帰。
  - **feature gateには載せない**: 外部リソース不要で、失敗時は既存経路へ落ちる。
    local OCR側に既にavailability判定があるため二重switchになる。
  - 残差は`CAXCWCQB`と`4CTI73FQ`のstructure_v3が`flat_fallback`である点のみ。本文欠落ではない。
  - 以下は当初の実装方針（達成済み）:
  - `XASKBF88` / `EI7JN68R`はSAGE系のleaf `div`本文を抽出するDOM fallbackを追加し、
    約70,990 / 87,819文字の本文を構造付きで回収する。
  - `CAXCWCQB` / `CCHED4LQ` / `BMKYYFCZ`は130 / 268 / 391ページ画像の固定レイアウト
    EPUBとして判定し、OPF spine順の画像を既存PDF OCR routerへ渡す。
  - 元EPUB・派生ページ・OCRページのfingerprint/locator対応、RTL、EPUB目次→spine対応、
    全ページcoverage、gibberish/repetition gateを保持する。
  - 実行中PDF batch完了後にコード変更し、まず該当5件だけ再処理する。既存成功EPUBの
    再埋め込みは行わない。詳細: `dev-notes/current/74_epub_fallback_design.md`
  - 2026-07-23: OCRのないPDFと固定レイアウト画像EPUBに共通のlocal OCR routerを追加。
    英語・その他はRapidOCR、日本語はNDLOCR-Liteの1種類ずつに固定し、全ページcoverage・
    最低文字量・gibberish・反復・timeout gate不合格時だけDoclingへfallbackする。
    EPUBは派生PDFのOCR後に元の`epub:spineN` locatorへ決定的に戻す。
  - 2026-07-23: 画像EPUB 3件（`CAXCWCQB` / `CCHED4LQ` / `BMKYYFCZ`）を
    `data/quality/image-epub-local-ocr-queue-v3.json`へ再取込対象として登録。
    V3埋め込み担当へ、既存inflightの整合性確認後にこの3件をitem単位で処理するよう追加済み。
- [x] ~~**Granite-Docling MLXを英語スキャン向け隔離workerとして追加評価する**~~ (2026-07-23)
  - 固定英語4資料pilotは平均+0.094997。ただし表・数式は-0.060000。
  - 50ページ英語スキャンはGranite 171.258秒、標準75.049秒でGraniteが約2.28倍遅い。
    速度目的の全面移行は行わず、標準Doclingの構造gate失敗時の選択的fallback候補とする。
  - 本番venvとはTransformers要件が衝突するため、専用venv/processから構造化JSONを返す。
  - 長文hallucination、末尾欠落、表routeを確認してからfallback実装を判断する。
  - 50ページ連続試験は50/50ページを両方式で完走し、末尾ページまで出力を確認。ただし
    Graniteは標準Doclingより約2.28倍遅く、長文ground truthなしでは追加文字の正否も
    確定できないため、現時点ではfallback workerを実装せず評価完了とする。
  - 表・数式fixtureでは標準Doclingを明確に下回るため、この資料種別をGraniteへrouteしない。
  - 再現テスト3件成功。将来、標準Doclingの構造gate失敗が実運用で蓄積した場合のみ、
    隔離worker実装を新規タスクとして起票する。
  - 詳細: `dev-notes/current/68_granite_docling_mlx_bakeoff.md`
- [x] ~~**失敗itemを理由別に回収する**~~ (2026-07-21)
  - `failed/retryable`を`--retry-failed`で再実行する。
  - `empty / blocked / degraded`を混同せず、未解消item keyとreason codeを一覧化する。
  - 新規script `scripts/list_artifact_status.py`: --unresolved-only / --retryable-only / --item で照会。
- [x] ~~**全件cutover auditを実行する**~~ (2026-07-28)
  - `passed: true` / 521件中失敗0 / global failures 0。同日朝の時点では428件が失敗していた
    （`stale_structure_fingerprint` 423 + `missing_v3_item` 5）。
  - 削除済み5件（Zoteroが404を返す）は`--exclude-item`で除外する。この5件はlegacyにのみ残る。
  - 構造状態: recovered 257 / exact 171 / flat_fallback 112 / unavailable 18。
  - **訂正**: 本日の監査結果JSONはリポジトリに保存していない（scratchpad出力のみ）。
    監査JSONはZotero識別子を含むローカル生成物としてGit追跡対象外にした。
    再現するには`--exclude-item`5件付きで再実行すること。
  - **未了分は下の2項目へ分離した**（監査の実行と、その結果の目視レビューは別作業）。
- [x] **EPUB・outline PDF・無構造PDF・論文を各3件以上目視確認する**（2026-07-29）
  - EPUB: `5PND4GEQ`（通常DOM）、`4CTI73FQ`（縦組leaf fallback）、
    `CAXCWCQB`（固定レイアウト130頁）を開始・中間・終端で照合。前2件の旧DBのspine順/
    見出し欠落は現コードで解消済み。固定レイアウトのRapidOCRに単語連結・読順・誤読を確認し、
    Latin単語境界欠落gateを追加してRapidOCR→Docling→Mistral queueへfail-closedにした。
  - outline PDF: `BPCMHQ66`、`X2CBUIHI`、`AA6WGTQU`を計20頁以上目視。狭い段間と片側1 block
    の二段組順、同一頁内TOC遷移、BOM/節番号付きReferences、zero-width文字による文字化け誤判定、
    頁番号と結合した反復footerを修正。再抽出で全件page coverage合格。
  - 無構造PDF: `AHPZ6SCS`、`AE28ZUN8`、`IMBFR28G`を確認。前2件はAI/metadata見出しと
    本文・References・Indexが一致。スキャン`IMBFR28G`の旧OCR破損を確認し、`scanned_no_text`
    provenanceを再OCR候補に残すgateを追加。
  - 論文: `FRIZY8BS`、`BPCMHQ66`、`XHMT88V7`の各3頁（先頭・中間・終端）を原画像と照合。
    全頁coverageは合格し、二段組修正後は左右列順・References遷移・日本語本文/図版captionが一致。
  - 目視QAで見つけた修正を含む全テスト: `870 passed, 6 warnings, 2 subtests passed`。
    埋め込み・canonical DB・manifestへの書込みは未実施。
- [x] **サーバー再構築と有料階層要約を段階実行に分離する**（2026-07-29）
  - `Server-Database-Workflow.command`を追加し、DBゼロ再構築、DB完全監査、階層AI要約、
    要約・要約索引監査を同一runでは選べない4フェーズに分離。
  - `audit_v3_cutover.py --new-only`の合格レポートをmanifest、pipeline config、
    Chroma/FTS chunk ID、全文書構造のfingerprintへ結び付けた。監査後にDBが変わると
    `build_structure_summaries.py --mode llm`はAPI呼出し前に停止する。
  - `audit_structure_summaries.py`でsummary status、source/prompt fingerprint、空/meta要約、
    DBのsearchable要約IDと`__sum_node` IDの完全一致を検証。
  - 日常MaintenanceでもDeepSeek要約とMistral Batchをauto-approve対象外に変更。
  - V3唯一化・セットアップ再設計・外部監査gate封印後の全テスト
    `906 passed, 6 warnings, 2 subtests passed`。
- [x] ~~**active collection切替の実行手順を確定する**~~ (2026-07-23)
  - 全件監査pass後に、active Chroma / FTS / manifestをV3へ同時切替した。
  - `.env`で`CHROMA_COLLECTION=zotero_paragraphs_v3`、`MANIFEST_PATH=data/manifest_v3.json`、
    `LEXICAL_DB_PATH=data/lexical_v3.sqlite3`、V3 ingestion/hierarchical retrievalを有効化。
  - 当時はlegacy collection / manifest / FTSを一世代保持した。このrollback方針は撤回し、
    現在はV3バックアップまたは原本からの再構築を用いる。

### 削除item・孤児レコードの整理（2026-07-27 完了）

未解消3件の調査から、purge経路の実バグ2件が判明した。**未解消は0件になった。**

- [x] ~~**削除済みitemのpurge**~~ (2026-07-27)
  - `scripts/purge_orphans.py`（dry-run既定）。Zoteroへ**item単位で直接照会し404のみを削除根拠**とし、
    それ以外のエラーは削除しない（ネットワーク断を「削除済み」と読まないためfail-closed）。
  - 削除5件。`RTVZVXT8`の**1,897チャンクをChroma・FTS双方から除去**し、構造5・ノード5・
    台帳18行・イベント69行を整理。実行前に対象を`data/backups/purged_orphans_20260727.jsonl`
    （1,897チャンク）と`_rows.json`（326行）へ退避済み。
  - 結果: Chroma 477,678 / FTS 477,678 で一致、HNSW query正常。
- [x] ~~**バグ: purgeの候補集合が`item_citation_status`限定だった**~~ (2026-07-27)
  - 1回目の`--apply`でChroma/FTSは消えたのに台帳が0件のままで発覚。citation mappingを
    通っていないitemは不可視で、**Codexが追加したV3テーブルの削除処理に一度も到達していなかった**。
  - 候補集合を「本関数がpurgeする全テーブルの和」へ修正。回帰テスト3件追加。
- [x] ~~**バグ: live集合が狭く、生存itemを削除しかけていた**~~ (2026-07-27)
  - `{a.parentItemKey for a in attachments if a.parentItemKey}`は**親を持たない添付**と
    **note-onlyのitem**を取りこぼす。citation表のみ触っていた頃は無害だったが、
    purgeがV3テーブルへ拡張された結果、次のフルスコープ実行で`FSIXT5VE`（note 42チャンク、
    Zoteroに現存）の構造・台帳が削除されるところだった。
  - `orphan_cleanup.live_item_keys()`で「パイプラインが実際に書き込むキーの導出」と一致させ、
    notes列挙に失敗した場合はpurge自体を中止する。
- [x] ~~**note-onlyitemを`excluded`へ**~~ (2026-07-27)
  - noteは仕様上canonical構造の対象外なのに`blocked/no_chunks`と記録され、未解消リストに
    永久に残っていた。`excluded/note_only_item`へ変更（`rebuild_document_structure.py`・
    `build_structure_summaries.py`）。索引全走査で該当は`FSIXT5VE`1件のみと確認し適用済み。
- [x] ~~**再parent化による台帳の孤児**~~ (2026-07-27)
  - 単独PDFは添付キーで追跡されるが、後から親itemへ紐付けると以降は親キーで記録され旧行が残る
    （`AJSX4LFZ`→`Q56RQ6H6`）。通常の運用操作なので再発する。
  - `orphan_cleanup.stale_identity_keys()`＋`db_relations.drop_stale_identity_rows()`で、
    **両キーが手元にある取込時に**旧identityを退役させる。チャンクは触らない（内容は生存）。

### セットアップウィザードの再設計（正本: `dev-notes/current/80_setup_wizard_profiles.md`）

軸は**課金の有無**であって外部送信ではない（RAGとして使う時点でチャンクはアシスタントへ渡る）。
A（PDF構造化）・B（課金LLM）は独立、C（構造抽出エンジン）はA=する のときのみ意味を持つ。

- [x] ~~**1. `audit_disabled` の導入**~~ (2026-07-27)
  - `replacement_required()` が未測定の層を一律で置換対象にしていたため、**LLMを切ると
    全スキャンPDFが再OCRへ回る**（＝LLMを止めると作業が増える）状態だった。
  - 「測れなかった」と「測らないと決めた」を区別。`DISABLED_REASON` を新設し、
    監査無効時は取込側で明示的に記録する。
- [x] ~~**2. `src/feature_gates.py`（既定0＋鍵なしエラー）**~~ (2026-07-27)
  - 当初`auto`導出で実装したが、**既定はすべて0**へ変更（ウィザードが明示的に書くため
    賢い既定は設定を読みにくくするだけ）。
  - **有効なのに鍵が無いのはエラー**。`verify_enabled_features()`が機能名と不足鍵名を挙げ、
    `index_from_zotero`起動時に停止する。黙って縮退させない
    （「有効にしたのに何も起きない」を防ぐ）。実環境で7件の不整合検出を確認済み。
  - `MISTRAL_OCR_FALLBACK_ENABLE`を**撤去**。全経路が「明示的な操作」または
    「queue登録のみ（実送信は別コマンド）」に守られており、二重確認でしかなかった。
    `mistral_ocr_fallback_available()`→鍵のみ見る`mistral_ocr_available()`へ置換。
  - 所有者の`.env`には全機能を明示的に`=1`で記載（既定0でも現行動作が変わらないよう）。
    バックアップ: `.env.bak_20260727_gates`。
- [x] ~~**3. `PDF_STRUCTURE_RECOVERY_ENABLE`（選択A）**~~ (2026-07-27)
  - `0`でDocling/Granite/Mistralを一切呼ばない。頁パッチ3種・AI目次・エスカレーション・
    局所OCR連鎖のDocling fallbackをすべてゲート。EPUB/HTMLのDOM構造化は対象外。
  - 実資料で平文経路（60チャンク・Docling不使用）を確認し、通常設定へ復旧済み。
- [x] ~~**4. `PDF_STRUCTURE_ENGINE_SHORT/LONG/PAGE_BOUNDARY`（選択C）**~~ (2026-07-27)
  - ハードコードの「短→Docling／長→Mistral queue」を置換。境界ごとに
    docling/granite/mistralを独立指定。境界は`PDF_AI_TOC_MIN_PAGES`と別設定
    （前者は「目次を持つ長さか」、後者は「どちらがコストを負うか」で問いが違う）。
  - mistralはBatch queue経由（queue無効時はDoclingへ退避し、資料が未構造化で終わらない）。
  - **未実装エンジンの選択は起動時に停止**。graniteはvenvとアダプタの両方を要求する
    （ベイクオフのvenvが残っており、venvだけ見ると呼べないエンジンを選べてしまう）。
- [x] ~~**5. Graniteアダプタ**~~ (2026-07-27)
  - `src/granite_worker.py`＋`scripts/granite_runner.py`。`mlx-vlm`と現行Doclingは
    transformers要件が衝突し同一venv不可のため、**forkではなくJSONを介したsubprocess**。
    資料ごとに新プロセス（モデル読込は数秒でGranite自体の実行時間に対して小さく、
    MLXバッファのリークがバッチ全体に蓄積しない）。
  - 正規化とチャンク化は再実装せず、`extract_chunks_from_pdf_with_docling`へ
    VlmPipelineコンバータを注入して**Doclingと同一の経路**を通す。
  - Granite失敗時はDoclingへfallback（品質のための選択であって必須要件ではないため、
    資料が未構造化で終わらないようにする）。
  - 実資料検証: 11頁を63秒で92チャンク抽出。テスト11件。
- [x] ~~**6. ウィザードのプリセット選択・詳細調整・既存構成の維持**~~ (2026-07-27)
  - 3プリセット（最小／ローカル／フル）＋「現状を維持」＋任意のエンジン個別調整。
  - 鍵が入力されなかった場合は該当機能を`0`に落とす（次回起動が止まる設定を残さない）。
  - 状態表示は編集中のconfigに対して検証を実行（`.env`の読込を一時停止するので、
    未保存の鍵が見えたり、削除した鍵が残って見えたりしない）。
- [x] ~~**6b. Minimal／Custom方式と独立した構造化設定へ再設計**~~ (2026-07-29)
  - 3プリセットをMinimal／Customへ置換。CustomはCitation Network、PDF構造化、
    AI目次・OCR監査・クエリ拡張・階層要約・参考文献抽出を一項目ずつ確認する。
  - PDF構造化ではページ境界未満／以上のエンジンを独立選択し、利用可能な環境では
    `Granite / Granite`を含む任意のDocling／Granite／Mistral組み合わせを許可する。
  - S2・Mistral・DeepSeekを有効に選んだ場合は対応キーを必須入力とし、入力文字が
    画面に表示されないことをプロンプト直前に明示する。
  - セットアップウィザードと全`.command`の利用者向け案内・質問・エラーを日本語化。
    製品名・環境変数名・確認入力語など、操作上そのまま示す必要がある識別子は維持する。
  - Granite未導入時も選択肢から隠さず、選択後の確認に応じてApple Silicon用の専用venvと
    固定依存をウィザードが導入・検証する。失敗または拒否時は該当区分をDoclingへ戻す。
  - CustomでNDLOCR-Liteを検出し、未導入なら公式GitHubの検証済み`1.0.0`タグから
    約450MBの隔離`uv tool`として任意導入する。検証した実行ファイルの絶対パスを
    `NDLOCR_BIN`へ保存する（NDLOCR-LiteはPyPI未公開のためレジストリ名では指定しない）。
  - Tesseractは画像PDF全体の主OCRではなく、フォント復号失敗ページ専用の補助であることを
    ウィザードと文書に明記する。Customでは本体と日本語言語データを検出し、不足時は
    明示同意後にHomebrewから`tesseract`／`tesseract-lang`を導入・検証する。
- [x] ~~**7. ドキュメント反映**~~ (2026-07-27)
  - `docs/configuration.md`にプリセット表とエンジンの性格比較、`llm-and-privacy.md`に
    既定0と課金対象の明示、`.env.example`に全フラグの説明。

### 取込ゲートの仕様確定（正本: `dev-notes/current/79_embedding_gates.md`）

2026-07-26にユーザーが場合分けを確定し、**同日実装完了**（全526テストpass・新規44件）。
仕様と実装記録の正本は79。較正実測18件の結果も同ファイルに記録。

- [x] ~~**U0-a: スキャン数頁をサンプリングしてLLMに品質判定させる**~~ (2026-07-26)
  - `src/ocr_layer_audit.py`。必須要件はすべて較正実測から導かれたもの:
    プロンプトに字面の例を入れない（報告83例中**23%が捏造**＝プロンプト例の反響。
    `借景と坪庭`は6例全て捏造でsevere判定）／報告語が本文にliteralに存在するか
    **機械検証して棄却**／分母は自前で数える（モデル推定は2,000語→6,452字と振れる）／
    `verdict`は使わない（1.0%でsevere、3.75%でminor）／`ocr_misrecognition`のみ計上。
  - `cloud_text_allowed()`でfail-closed、source_fingerprintでキャッシュ（再取込は無料）。
  - **Producerによる無条件再OCRは撤回**: `ABBYY FineReader 11`の*Elementary Structures*は
    0.25%＝良質、新しい`Acrobat Pro DC 17 Paper Capture`の*Japan-ness*は1.75%＝劣化。
    「旧世代OCR＝悪い」は成立しない。事前情報として記録するのみ。
- [x] ~~**U0-b: `report_chunk_quality` MCPツールと報告ループ**~~ (2026-07-26)
  - `chunk_quality_reports`テーブル（スコープは**チャンク本文hash**＝再抽出で自動retire）、
    MCPツール2種、サーバinstructionsへ報告義務を追記。
  - `candidate_assessment`と`list_reocr_candidates.py`へ配線し**ループが閉じた**:
    読者が気づく→報告→再OCR候補に浮上→再OCR。重み6点/件（上限3件分）で決定的指標より高い
    — 直接証拠であり、文字数・gibberish率が健全に見える劣化を捕まえる唯一の信号だから。
- [x] ~~**U1: 文書種別（テキスト層の信頼度）の判定**~~ (2026-07-26)
  - `src/pdf_provenance.py`。純テキスト頁の有無で判定（**標本は先頭を厚く**取る。前付は
    文書先頭にしかないため均等散布では画集の前付を引き当てられない）。
    E2cのfigure付与をborn-digital限定に、ゲートの`document_scanned`を種別ベースへ。
  - **設計修正**: ゲートが却下するのは段2が`degraded`と測定したときだけ。当初の
    「スキャン由来なら一律却下」はスキャン67件を全部再処理してしまい仕様の段3と矛盾した。
    未測定（`unverified`）は却下しない（測れなかったことは問題の証拠ではない）。
- [x] ~~**U2: スキャンPDFの部分OCR失敗に頁単位Docling**~~ (2026-07-26)
  - `_scan_pages_needing_repair`＋`patch_corrupted_pages_with_docling`。回復しない頁は
    `figure`ではなく`corrupted_unresolved`。既マーカー頁は再試行しない（ループ防止）。
  - **設計修正**: legacy OCR再利用経路が種別判定・段1・段2・頁修復を全部迂回していた。
    再利用テキストは定義上OCR出力で最も検証が必要なのに逆になっていた。
    `_audit_reused_ocr_chunks`を追加し、degradedなら再利用を放棄して通常抽出へ落とす。
- [x] ~~**段1: 決定的な抽出欠陥検出**~~ (2026-07-26)
  - 孤立1字率>5%（ラテン語優位テキストのみ）と合字脱落。実測で完全分離
    （`Writing culture`47.1% / `Writings on Art`4.5% ⇔ 他は≤3.1%、ft脱落226件⇔他≤6件）。
    **これらはOCRでなく抽出の問題**なので対処は再抽出であり再OCRではない。
  - 合字コードポイント（ﬁ/ﬂ）は`clean_extracted_text`のNFKCが既に分解済みで対応不要と確認。
- [x] ~~**U3: 破損データをタグ付きで保持し、opt-in検索モード**~~ (2026-07-26)
  - zone→policy表に`corrupted`追加、gibberish頁を保持、`rag_search(include_corrupted)`。
  - 遡及なし（以後の取込分のみ）。閾値は絶対値、figureの本文ヒットは許容。
- [x] ~~**U4: 行き止まりで失敗させず品質タグ付きで埋め込む**~~ (2026-07-26)
  - `_adopt_with_quality_uncertain`。U3の`corrupted`と違い**既定で検索可能**
    （テキストが疑わしいだけで既知の破損ではなく、注記付きで見つかる方が不在よりよい）。
- [x] ~~**U5: 実データへの適用**~~ (2026-07-28)
  - キュー92件すべて処理済み。うち31件は**2026-07-27のMistral OCR batch（37/37採用）で
    既に完了していた**。runnerには採用後に自分のキューを更新する手順が無いため、
    `deferred`ラベルだけが作業より長く生き残っていた（送信済みを未送信と読み違える原因になった）。
    DBの`artifact_processing_status`は31件すべて`success` / `processor_version=mistral_ocr`。
  - 検証: `H2SMKUJY`（*Remediation*）のV3チャンク1,909件に`righrs`/`srorage`は0件、
    `rights reserved`が21件、`storage and retrieval`が1件。`extraction_engine=mistral_ocr`。
    **追加のAPI送信は不要だった。**
  - キューの31件を`completed_via_mistral_batch`へ実測に基づいて更新済み。
- [x] ~~**U6: 構造の再構築とcutover監査の通過**~~ (2026-07-28)
  - 423件の構造がチャンク入れ替え（V3再抽出・Mistral採用）で`stale_structure_fingerprint`に
    なっていた。構造再構築は決定的でLLMを使わない（397件で29秒）。
  - **cutover監査 `passed: true` / 521件中失敗0 / global failures 0。** 朝の時点では428件が
    失敗していた。削除済み5件（Zoteroが404を返す）は`--exclude-item`で除外する。
  - 要約は再利用が効かない。既存3,916件はper-node fingerprintを持たない旧世代で、
    旧行の再利用は「文書全体のfingerprintが不変」が条件だが、**変わったからこそ再構築して
    いる**ので原理的に満たされない。これはバグではなく安全規則が正しく働いた結果で、
    要約が指す本文はもう存在しないため再生成が正しい。23,020回のLLM呼び出しで再生成。
  - `--workers 40`で1時間13分・失敗0。20では67.8要約/分、40では476要約/分（CPU使用率6%の
    I/Oバウンドなので並列度がそのまま効く）。末尾は親要約が子に依存して直列化する。
  - **注意（自分の誤診の記録）**: 作業中「legacyが配信されている」と報告したが誤り。
    cutoverは2026-07-23に完了済みで`.env`に`CHROMA_COLLECTION=zotero_paragraphs_v3`がある。
    検証スクリプトで`load_dotenv_native()`を呼ばずに`chunk_store`を使ったため、
    環境変数が空になり`embedder_config.json`（legacyのまま）へフォールバックしていた。
    **`active_collection_name()`を確認するコードは必ず`.env`を読んでから呼ぶこと。**
- [x] ~~**U7: MIN_LEAF_CHARSを400→1,000へ**~~ (2026-07-28)
  - 25,858件の実測で、要約長は本文長にほぼ依存しない（400-1,000字帯で中央値248字、
    20,000字超で448字）。したがって圧縮率は本文長だけで決まる: 400-1,000帯で0.354、
    20,000字超で0.009。716字の本文を248字で置き換えても数文しか節約できず、
    当たれば結局本文を開くので、要約レイヤーは経路を減らすどころか一つ増やす。
  - 既存の1,000字未満の要約4,385件を削除（`data/backups/small_leaf_summaries_20260728.json`）。
    チャンクは別コレクションなので本文の検索性は不変。

- [x] ~~**U5a: truncated応答からのsalvageが機能していなかった**~~ (2026-07-28)
  - `salvage_items()`にはパーサの**メッセージ**が渡されていた。メッセージは既に捨てられた
    文字列へのoffsetを指すだけなので、salvageは常に空を返し、切り詰められた監査は全て
    `llm_unavailable`で終わっていた。しかも**OCRが悪い資料ほど項目リストが長く切れやすい**ので、
    監査は本来検出すべき資料でこそ失敗していた。
  - `InvalidLLMResponse`に`raw`（復号できなかった本文そのもの）を持たせて解決。回帰テスト2件。
- [x] ~~**U5b: manifestがorphan purgeの対象外だった**~~ (2026-07-28)
  - 削除された添付のmanifest行が永久に残り、cutover監査は**それ以降に監査した全itemで**
    `manifest_chroma_attachment_mismatch`をglobal failureとして報告し続けていた。
  - `stale_manifest_keys()`を追加し`purge_orphans.py`から呼ぶ。**ファイル不在とチャンク不在の
    両方**を条件にする（外付けドライブ未接続を削除と誤認しないため）。テスト5件。
  - 実行結果: `C66HF59V`（分割版PDFで、完全版`6SRYJ3Y3`が別途索引済み）1行を削除。
    監査のglobal failureは0になった（`passed:false`は残るが、これは未移行の
    legacy 526件によるもので別件）。

- [x] ~~**U8: zone誤分類で資料が1件まるごと検索から消えていた**~~ (2026-07-28)
  - `excavating.ai.html`（Crawford & Paglen、65,000字）の全ノードが`zone=index` /
    `retrieval_policy=exclude`。`retrieval_policy_allowed()`はRRF統合後の全ヒットに
    適用されるので、階層検索だけでなく通常の`rag_search`からも**完全に不可視**だった。
  - 原因は **`<main class="Index">`** ただ1つ。Squarespaceがランディングページの
    レイアウト名に使う語で、書籍の索引とは無関係。`_semantic_tokens()`が祖先を遡るため、
    `<main>`配下の790要素すべてがこれを継承していた。
  - 対処1: 走査を`main`の手前で止める（`_CONTENT_ROOTS`）。`<main>`は定義上その文書の
    主要コンテンツなので、そこに付いたzone語は全体に等しく掛かり、何も区別しない。
  - 対処2: `_reclaim_fully_excluded_document()`。**内容を持つ葉がすべて除外zoneなら
    body へ戻す。** zone判定は文書の一部についての推測であって、全体に当たった時点で
    構造の記述をやめて文書を消している。自分の索引だけで構成された文書は存在しない。
    `corrupted`は対象外（意図的な除外であって誤分類ではない）。テスト3件。
  - 再取込・再構築後、検索1〜5位をこの資料が占める（直前まで0件）。
  - 同資料の237,831字が`endnote`（本文50,611字）である件は**調査のうえ対処しないと決めた**
    (2026-07-28)。見出しパスの引きずりではない: 資料自身が`<h1>END NOTES</h1>`（DOM位置95%）
    の下にこれらを置いており、分類は資料の構造に忠実である。注[1]だけで12,717字という
    書き方が異例なだけで、Squarespaceの1ブロックに注と散文が混在しているため、zone層では
    分離できない（分離するには`<p>`内部を解析する content 層の処理が要る）。
  - **1件のために機構を足さない根拠**: 1万字超の530件を走査し、傍題(paratext)が60%を
    超えるのはこの1件のみ。系統的な型ではない。また`endnote`は`exclude`ではなく
    `explicit_only`なので、U8の「完全に不可視」とは深刻度が違う。
- [x] ~~**U9: Mistral batch採用後にキューが更新されない**~~ (2026-07-28)
  - deferralは「これから送る」という将来についての主張だが、送信・採用後にそれを retire
    する手順が無く、ラベルが作業より長く生き残っていた。31件が未送信に見え、**同じ資料に
    二重課金する寸前**だった（実際にユーザーはこの誤報告を受けて入金した）。
  - `--reconcile-adopted`を追加。判断の権威はキューではなく台帳で、
    `artifact_processing_status`が`success`かつ`processor_version=mistral*`の行だけを閉じる。
    台帳に記録が無いdeferralは触らない（「未採用」と「採用済みだが未記録」を、作業を失う
    方向に混同しないため）。Docling成功でクラウドdeferralを閉じないこともテストで固定。

### Phase 4準備: 設計レビュー指摘の修正（backfill前・正本: `dev-notes/current/76_v3_design_review_refactoring_plan.md`）

- [x] ~~**R1（P1・pilot前必須）: 要約の差分スキップを実装する**~~ (2026-07-23)
  - `build_structure_summaries()`に`_summaries_are_current()`を追加。既存node要約の
    source_fingerprint/prompt_versionが現行構造と一致し、artifact statusがstale/failedで
    なく、`mode=="llm"`ならllm要約が1件以上ある場合に`status=skipped_current`で
    LLM呼び出しゼロで抜ける。`force=True`でこの判定を飛ばす（死にパラメータを生かした）。
  - `get_all_document_node_summaries`に`item_key`フィルタを追加。CLIに`--force`を配線。
  - 回帰テスト2件追加（同一fingerprint 2回目でLLM呼数0 / extractive既存+`--mode llm`再生成）。
- [x] ~~**R1b（V3構造再構築時の要約入力再利用）**~~ (2026-07-27)
  - `replace_document_structure()`がFK cascade前のLLM node要約を隔離cacheへ退避し、
    `build_structure_summaries()`がleaf/parentの実プロンプト入力（タイトル＋本文segment、または
    子要約列）のSHA-256一致時だけ新nodeへ再利用する。node IDの変更自体は再生成理由にしない。
  - 旧行にper-node fingerprintが無い場合はdocument全体source fingerprint一致を追加条件にして
    安全側へ倒す。変更leafは再利用せず、将来の行はper-node fingerprintで他leafへの影響を局所化する。
  - cacheは検索対象外で、再利用時も現行source fingerprint/provenanceを保存し、旧reduction partsは
    空に置換する。構造置換後のLLM API呼出し0をfixtureで検証済み。
- [x] ~~**R3（P1・pilot前必須）: 要約索引更新の失敗分離とartifact_type分離**~~ (2026-07-23)
  - `--embed`を「1件でも失敗なら全skip」から`succeeded`item分のみ実行へ変更。exit codeと
    JSONで失敗件数を報告する。
  - node要約索引の状態を`embeddings`から専用`artifact_type="summary_index"`へ分離。
    `_ARTIFACT_TYPES`とCHECK制約に追加し、既存DB向けに表を作り直す後方互換マイグレーションを
    実装（旧行を保全して`summary_index`受入れを検証済み）。
  - `--all`時は`item_keys=None`で呼び、stale差分削除経路を生かす。
  - 【インシデントと修正】初版migrationは移行先CHECKに`cases`を含めず、grandfatherされた
    `cases` 526行でINSERTがCHECK違反となり、実DBで一度migrationが失敗。全2805行は
    `artifact_processing_status_old`へ退避され消失は無かったが、liveテーブルが空になった。
    (1) 移行先CHECKに`cases`を追加（新規書込みは`_ARTIFACT_TYPES`がPython側で拒否）、
    (2) `live空 + _old有データ`を検知して自動修復する起動時パスを追加。実DBは2805行へ復旧・
    検証済み（`summary stale 559`・`embeddings success 559`・`cases 526`等がレビュー値と一致）。
    バックアップ`data/relations.db.bak_20260723_232657`を保持（不要なら削除可）。
- [x] ~~**R2（P1・pilot前推奨）: V3要約のlegacy依存を`summary_core.py`へ分離する**~~ (2026-07-23)
  - 6関数＋その私的ヘルパー・正規表現・SUMMARY_ONLY_SCHEMA（計12関数＋8定数）を新設
    `src/summary_core.py`へ移動。`build_summaries.py`は`summary_core`からの再importで
    互換維持（外部の`build_summaries.X`参照を保持）。`build_structure_summaries.py`は
    `summary_core`から直接importしてlegacy依存を解消（Phase 6 attic化の前提が成立）。
  - プロンプト文字列は不変（移動のみ）。テストのDeepSeekClientパッチ対象を`summary_core`へ
    修正。全425テストが決定順・ランダム順ともにpass。
  - 2026-07-29追記: Phase 6対応で`build_summaries.py`からDB writer・CLI・旧要約索引生成を
    撤去した。現在はモデル比較用のread-only互換helperだけを公開し、V3要約の保存と索引化は
    `build_structure_summaries.py`へ一本化している。
- [x] ~~**R5（P2）: 親要約の還元モデル役を仕様通りstandardに統一**~~ (2026-07-24)
  - ユーザー判断: 仕様§5.1通り。`_parent_summary`に`reduce_role`パラメータを追加し、
    親ノード（章・item root）の還元を`standard`役に（単一グループも含め）。leaf segmentの
    結合呼び出しは`reduce_role="cheap"`でleaf=cheapを維持。複数グループ時の中間passは
    context分割用にcheapのまま、最終synthesisのみreduce_role。回帰テスト追加（親=standard・
    leaf=cheapの役割を検証）。
- [ ] **R4+R10（P3・大規模）: PDFルーティング抽出と`index_from_zotero.py`分割**
  - 本番ルーティングは`EngineRegistry`未使用の手書き400行ネスト。`src/pdf_routing.py`へ
    純粋関数として抽出し、`main_async`（約1,200行）を`ingest_routing/flush/scope/locks`へ分割。
  - backfill・Phase 5完了後に1モジュールずつ独立実施。
- [x] ~~**R17（P1・運用）: Maintenance-Widgetの要約stepを承認前全件backfillから保護**~~ (2026-07-23)
  - widget step 2を`data/quality/summary_backfill_approved`マーカー存在チェックで保護。
    マーカー無し時は`--limit 10`のpilotのみ実行し、全559 itemが走らないようにした。
    承認後（マーカー作成）はR1差分スキップにより`--all`が現行fingerprint一致分をskipする。
  - [x] 追記（2026-07-24）: LLM要約バッチ処理を標準機能化。
    `scripts/build_structure_summaries.py`に`--workers N`（ThreadPoolExecutor、item単位並列、
    DBはWALモードへ切替）を追加し、5並列で実測約3.3倍高速化（実データ検証・失敗0）。
    widget側は承認後もバッチ上限＋並列で実行するよう変更（1回のメンテナンスで全件を
    ブロッキングしない設計）。バッチ件数・並列度は`SUMMARY_BACKFILL_BATCH_SIZE`
    （既定10）・`SUMMARY_BACKFILL_WORKERS`（既定5）で調整可能（`.env.example`に文書化）。
    大規模一括backfillは別途`scripts/detached_summary_backfill.py --workers N --limit N`
    （setsid+caffeinateでharness非依存にデタッチ実行、`--workers`/`--limit`とも引数化）。
    テスト2件追加（既定バッチ列・env上書き）、全442テストpass。
- [x] ~~**R18（P2・要ユーザー判断）: 承認済みPDFルートflagを.envへ常設する**~~ (2026-07-24)
  - ユーザー判断: 両方許可。`.env`に`PDF_AI_TOC_FAST_PATH_ENABLE=1`・`PDF_AI_TOC_MIN_PAGES=30`・
    `PDF_MISTRAL_TOC_QUEUE_ENABLE=1`を常設（section 4b）。これでwidgetのライブラリ差分更新で
    追加される30ページ以上の無構造PDFはAI目次fast pathを既定ルートにし、gate不合格分は
    Mistral OCR専用queueへ退避する（queue生成のみ）。
  - `MISTRAL_OCR_FALLBACK_ENABLE`（同期クラウド送信の別機能）は0のまま維持した。
    資料単位タグによる許可機構は後に撤去し、現在は機能フラグと明示承認を使う。
- [x] ~~**R19（P3）: widget末尾に状態台帳サマリ表示stepを追加（仕様§8）**~~ (2026-07-23)
  - widget末尾にread-onlyの`list_artifact_status.py --unresolved-only`を追加。step 1/2の
    「文書構造v2」ラベルをV3へ修正。Mistral queue候補件数の表示は
    `list_mistral_toc_candidates.py`が`--output`必須（queue書込みの副作用あり・read-only
    でない）ため見送り、artifact statusのblocked集計で代替可能。
    `rebuild_document_structure.py`の`--changed-only`scopeは将来課題として残す。
- [ ] **参照・注の取り込み時境界保持（R6の前提条件）** 正本: `dev-notes/current/77_reference_note_boundary_preservation.md`
  - [x] Part A（済 2026-07-24）: `html_extract.py`に`_entry_texts()`を追加し、
    `extract_dom_blocks`/`extract_leaf_container_blocks`を zone-aware に。参照/注zoneの
    ブロックを`<br/>`境界でエントリ分割（1エントリ=1ブロック、インライン要素は非境界）。
  - [x] Part B（済・確認のみ）: 追加改修不要。分割後は`merge_short_chunk_records`の`"\n\n"`
    結合で復元可能（`_split_reference_lines`で3件復元を確認）。PDF(Docling)は
    `_PRESERVE_SHORT_LABELS`で既に保持済み。
  - [x] Part A修正（済 2026-07-24）: `<br/>`だけでなく`<li>`/`<dd>`/複数`<p>`境界にも対応
    （`<aside><ol><li>`内包でblob化する標準EPUB向け）。非破壊検証で`2VRVDVMA`が
    旧参照の99%回収・Jaccard 0.907。非標準SAGE 2件は旧DOM抽出が壊滅で新経路が優位。
  - [x] Part D（済）: Docling連続参照が独立チャンクを維持する回帰テスト。
  - [x] Part E1（済 2026-07-24）: **AI目次/PyMuPDF PDF経路がzone未付与**だった問題
    （ユーザー指摘）。`apply_anchors`に`_KIND_ZONE`を追加し参照/注セクション配下へ
    zone付与。全430テストpass。
  - [x] Part C（済 2026-07-24）: footnote/endnoteの本文リンクを決定論化。`_entry_records`が
    `element_ids`/`noteref_targets`を記録、`merge_short_chunk_records(union_keys=…)`で結合を跨ぎ
    保持、`_resolve_note_citations`が`(chapter_index, note_id)→citing_chunk_id`（同一spine限定・
    scalar）を付与。`2VRVDVMA`で40チャンクにciting付与・`validate_metadata`通過。
    残: **別ファイル文末注**（note bodyが別spine）はhrefファイル部→spine index解決が必要で未対応（安全側で未リンク）。
  - [x] Part E2a（済 2026-07-24）: `chunk_reference_extractor`にflatten番号付き文末注/参照の
    連番マーカー分割を追加（単一参照内の数字では過分割しない）。
  - [x] Part E2b（済 2026-07-24・ユーザー選択=案1）: 参照/文末注セクションのページ範囲だけ
    Doclingで再解析し本文へmerge。`extract_reference_sections_with_docling`＋
    `_reference_section_pages`/`_splice_reference_chunks`、flag`PDF_AI_TOC_DOCLING_REFERENCES_ENABLE`
    （既定off・fail-closed）。全437テストpass。脚注（page-bottom散在）は原理的にレイアウト解析必須で
    案1対象外。
  - [x] Part E2c（済 2026-07-25・ユーザー提案＋2回の追加バグ修正）: 一部ページのみscannedな
    PDFで、全document Mistral OCR送信が過剰だった問題を修正。
    `patch_scanned_pages_with_docling`＋`recompute_scanned_quality_after_patch`実装。
    `PDF_SCANNED_PAGE_PATCH_ENABLE=1`／`PDF_SCANNED_PAGE_PATCH_MAX_PAGES=30`で`.env`に
    **常設済み**。
    - 追加バグ1: `insert_pdf`ベースのsub-PDF構築が画像xrefを破壊しレンダリング結果が白紙化。
      `_build_scanned_page_subset`（直接レンダリング→白紙なら埋込み画像を直接抽出し再描画）
      で解決。
    - 追加バグ2（ユーザー指摘）: 装飾的ポスター文字等は標準OCRで検出不可。無理に通すと
      ゴミデータ汚染のリスクがあるため、**「Doclingが試行して0件」も解決済み扱い**にする
      設計へ変更（`patch_scanned_pages_with_docling`が`(chunks, attempted_pages)`を返す）。
    - 実証: `4E9SC474`（ポスター図版含む）でscanned_ratio gateを正しく突破、
      `7XEH85I4`は`extraction_engine=pymupdf`でMistral不要のまま1743チャンク成功。
    - 追加バグ3（59件バッチ実行中に発見）: `pdf_extract.py`のPyMuPDFレイアウト
      フォールバック経路が既存に持っていた潜在バグ（`source_block_indices=[]`が
      マージされず単独チャンクとして残る）が、`_upsert_in_subbatches`のChroma
      upsertで`ValueError`となりバッチ全体を停止させた（scanned page patch機能とは
      無関係の既存バグが今回のバッチ規模で顕在化）。`_upsert_in_subbatches`の
      upsert直前に空リストメタデータ値を汎用的に除去する防御的サニタイズを追加して解決
      （個別呼び出し元を全て直すのではなく共通upsert関数で一括対応）。回帰テスト2件追加。
    全450テストpass。59件バッチ処理を再実行中（バグ修正版）。
  - [x] Part E2c追加バグ4（済 2026-07-26・ユーザー指摘）: Doclingが「本文を検出した」と
    主張してもOCR誤認識ゴミ（単語連結等）が混じるケースをM5TQ4HLZで確認。
    `_looks_like_scanned_patch_ocr_noise`（文字数下限なし＋スペース密度異常検知）を追加し、
    ノイズ判定チャンクは`block_type="figure"`扱いへ差し戻す。
  - [x] `pymupdf_fast_path_rejection_reason`の目次必須ゲート撤廃（済 2026-07-26・
    ユーザー決定）: `missing_usable_outline`を撤廃。目次無しでもテキスト品質が良ければ
    ローカルでそのまま採用（M5TQ4HLZ: 127頁無目次カタログが目次無しのみでMistralへ
    回されていた問題を修正）。
  - [x] Part E2d（済 2026-07-26・ユーザー指示: 破損頁の修復可能性調査＋実装）:
    `corrupted_pages`（`analyze_text_quality`のテキスト品質判定＝フォントエンコーディング
    不整合 or OCR/言語的ノイズ、PDF構造破損ではない）も、レンダリング自体は正常
    （グリフ描画はToUnicode/CMapに非依存）なため、スキャン頁と同じ`_build_scanned_page_subset`
    機構でDocling再OCRにより修復できることを実データ（`AX5CRKJ6`）で実証。
    - `docling_extract._patch_pages_with_docling_ocr`共通ヘルパー化＋
      `patch_corrupted_pages_with_docling`追加。未解決ページは
      `block_type="corrupted_unresolved"`マーカー（スキャン頁の`"figure"`と区別）。
    - `pdf_extract.recompute_corrupted_quality_after_patch`（姉妹関数）。
    - `index_from_zotero.py`: スキャン頁パッチ直後に挿入。破損頁は既にチャンク化済みのため
      パッチ適用ページの既存チャンクを削除してから置き換え。
      `PDF_CORRUPTED_PAGE_PATCH_ENABLE=1`で`.env`に常設。
    - **修復優先の方針明確化**（ユーザー指示）: `pymupdf_fast_path_rejection_reason`の
      `scanned_ratio`/`corrupted_ratio`はE2c/E2dパッチ後の**未解決残余**であり生の検出値
      ではないことをdocstringに明記。許容チェックはパッチが解決しきれなかった残余への
      最終フォールバックという位置づけを明示。
    - **許容閾値の非対称解消**（ユーザー指示）: 目次あり2%／目次なし0%だった非対称を
      両方2%（`PYMUPDF_NATIVE_OUTLINE_ANOMALY_RATIO_MAX`）へ統一。
    - **可視化**（ユーザー指示）: 2%許容内で見逃された残存破損/スキャン頁も
      `[RAG QUALITY WARNING]`レポートに「N unresolved …page(s) (within tolerance)」として表示。
    - 全458テストpass（新規6件）。
  - [x] E2d追加バグ修正（済 2026-07-26・レビュー指摘）: `recompute_corrupted_quality_after_patch`
    が`extraction_failure_pages`/`extraction_failure_ratio`/`content_corruption_pages`を
    再計算していなかった（`corrupted_pages`は両者の合算のため、extraction_failure型の破損頁を
    パッチ修復しても古いratioが`pymupdf_fast_path_rejection_reason`の無条件却下を発火させ続ける）。
    内訳リストも試行済み頁除去・再計算するよう修正、パッチ後にfast pathゲートを通過する
    回帰テストを追加。現行ライブラリに該当item無し（将来向けの正しさ修正）。
  - [x] 取り込み分岐監査P1〜P7の実装（済 2026-07-26・ユーザー承認）: 正本
    `dev-notes/current/78_ingestion_gate_audit.md`冒頭の実装状況ブロック参照。
    - P1: AI目次「見出しなし」判定（insufficient_*）は同一ファイル（mtime/size不変）の
      再取込で再試行しない（manifestキャッシュ）。整列失敗はartifact_status=degraded
      （ai_toc_alignment_failed）で構造化候補として可視化。
    - P2: local OCR全滅後のDocling無ゲート再実行を廃止し、ページ数下限なしのMistral queue
      退避を追加。残るDoclingエスカレーションにgibberish/repeat_artifactsの最低限ゲート。
    - P3: 頁スコア/文書ratioの1変数2役を分離（TEXT_QUALITY_SCANNED/CORRUPTED_RATIO_THRESHOLD
      新設、未設定時は旧変数から継承、値は現状維持）。
    - P4: AI目次の残余ratioゲートを0許容→2%（PYMUPDF_NATIVE_OUTLINE_ANOMALY_RATIO_MAX）統一。
    - P5: E2c/E2dパッチ後チャンクを(page, reading_order)で再ソート。
    - P6: queue退避countsにfast_path_reason/ai_toc_rejection_reasonを併記。
    - P7: queueフラグのAI-TOCフラグとのAND結合を解消。
    - 全475テストpass（新規17件: `tests/test_ingestion_gate_routing.py`新設ほか）。
    - 優先度低4項目（min_pages二重チェック・cloud policy照会キャッシュ・reason改名・
      閾値コメント明文化）は未実施（別途判断）。
  - [x] E2c/E2dページパッチのOOM修正（済 2026-07-26・再取込26/60件目クラッシュ対応）:
    AX5CRKJ6（破損頁208/413）の208頁一括200dpi sub-PDF＋Docling一括処理がOOM（SIGKILL・
    tracebackなし）で取込ジョブごと停止。`_patch_pages_with_docling_ocr`を有界バッチ処理
    （`PDF_PAGE_PATCH_BATCH_PAGES`既定24頁、バッチ間でgc+torchキャッシュ解放、
    1バッチ失敗でも残り継続・失敗頁はfigure/corrupted_unresolvedマーカー化、
    chunk idに`b{batch}`挿入で衝突防止）へ変更。回帰テスト3件追加、全478テストpass。
  - [ ] **【今後の検討事項】脚注（脚注・page-bottom）の構造化**: AI目次/PyMuPDF経路では
    脚注はページ下部・小フォント・見出し無しで散在し、見出しベースのzone付与・境界検出が
    原理的に不可能（チャンク化時に行幾何も消失）。E2b案1（参照/文末注セクションのページ範囲
    Docling）では脚注はカバーできない。選択肢: (i) 脚注が主体のPDFはDoclingへルーティング
    （脚注は全ページ散在でページ範囲切出し不可）、(ii) PyMuPDFの`get_text("dict")`幾何
    （小フォント＋ページ下部＋区切り線）で脚注zoneを自前検出。要方式検討・費用対効果評価。
  - [x] 再取込の正式な仕組み（済 2026-07-24）: `index_from_zotero.py`に`--force-reparse`を追加。
    抽出コード変更後にscope指定（`--item`/`--limit`/`--source-type`必須）でunchanged skipを
    バイパスして再抽出する。`pipeline_fingerprint`は埋め込み互換性のみで抽出コード非包含のため、
    コード変更では自動再取込されない設計への明示的opt-in。manifest手術は不要。
  - [x] 検証3件をDBへ再取込（済 2026-07-24）: `2VRVDVMA`（1件目はmanifest手術、後にforce-reparse
    実証）・`3E4ELASY`/`3N4BZVBU`（force-reparse）。DB版検証で `2VRVDVMA` bibliography 121→304
    チャンク・参照480件・Jaccard 0.907・citing_chunk_id 40件。SAGE 2件は旧DOM壊滅で新経路優位。
    manifest/DBバックアップ保持（`data/manifest_v3.json.bak_*`）。
  - [x] 本番反映（済 2026-07-24）: 全197 text-EPUBを`--force-reparse`で再取込（inflight/pending
    空のクリーン完了・失敗0）。DB検証で新境界＋citing_chunk_id反映を確認。
    速度対策として`sync_threshold`を100→10000（env化）し高速化。
    Chromaバックアップ`data/chroma.bak_20260724_103049`保持。
  - [x] **R6再挑戦→採用（済 2026-07-24）**: `citation_mapper.map_item_local_references`を
    `chunk_reference_extractor.extract_references_from_chunks`（bibliography/endnote/footnote
    zone、Part Cの`citing_chunk_id`で注の本文リンクを決定論解決）へ切替。二重パース廃止。
    `epub_reference_extractor.py`は非推奨バナーを付け本番から退役（検証ツールのみ参照）。
    DB版検証: 2VRVDVMA 480参照・Jaccard 0.907。全441テストpass。
- [x] ~~**R6（P2）: EPUB参照抽出の二重パース廃止を検証 → 採用不可**~~ (2026-07-24)
  - `src/chunk_reference_extractor.py`（`zone=bibliography`チャンク→参照候補）と検証スクリプト
    `scripts/compare_epub_reference_extraction.py`を実装し、実EPUB 3件で旧DOM抽出と比較。
  - 結果（新bib / 旧bibliography / 一致）: 2VRVDVMA 131/443/0、3E4ELASY 78/0/0、3N4BZVBU 147/5/5。
    Jaccard 0.0 / 0.0 / 0.034 と極めて低い。
  - **採用不可の理由**: (1) V3 bibliographyチャンクは参照境界が保証されずEPUBによってblob化する、
    (2) 旧DOM抽出とV3のzone分類（bibliography/footnote/endnote）が食い違い、旧が拾う
    endnote/footnote参照を再現できない。**二重パースは廃止せず現状維持**。
    `citation_mapper.py`は未変更（回帰リスク無し）。将来、取り込み時にbibliography境界を
    1参照=1行で保持する改善後に再挑戦する（新module・検証scriptは証跡として保持）。
- [x] ~~**R15（P3）: 事例DB残置レコードの退避を実行**~~ (2026-07-24)
  - ユーザー判断で`scripts/retire_case_database.py --apply`を実行。case_annotations 2027・
    case_evidence 2067・artifact_processing_status(cases) 526・events 537・Chroma
    `zotero_paragraphs__cases` 2027を`data/backups/cases-20260723T154105.917922Z`へ退避後に削除。
    退避スクリプトのartifact_processing_status再構築CHECKへ`summary_index`を追加しR3と整合
    （casesはCHECKから除去）。DB検証: cases 0行・summary_index保持・孤児テーブル無し・
    再マイグレーション不要・全426テストpass。
- [x] ~~**R7（P2）: `purge_removed_items`へV3構造・artifact状態を追加**~~ (2026-07-23)
  - `document_nodes`（node_id参照のsummaries/chunks/partsをcascade削除）・
    `document_structures`・`artifact_processing_status`・`artifact_processing_events`の
    削除済みitem行削除を追加。`counts`へ4キー追加。回帰テスト追加。実DB purgeは未実行。
- [x] ~~**R8（P2）: 階層検索応答の書誌情報を修正**~~ (2026-07-23)
  - R8-1: `candidate_items`のtitle/yearをパラグラフchunk metadata（実item書誌）から解決。
    node（章）見出しをitem titleとして返す誤り・year常時欠落を解消。
  - R8-2: `item_summary_snippet`に`item_summary_provenance`（legacy/none）を付与し、
    backfill前のlegacy要約混在をMCP利用側が識別できるようにした。
- [x] ~~**R14（P3）: 階層検索のleaf解決をSQLite直引きへ**~~ (2026-07-23)
  - `get_node_descendant_leaf_ids`を追加し、Chroma metadata往復（数千chunk hydrate）を
    廃止して候補nodeの子孫leaf node_idをSQLiteから直接解決。回帰テスト更新。
- [ ] **R11/R12/R13（P3・任意）: モジュール分割とルーティング設定一元化**
  - `db_relations.py`（3,191行）/`rag_mcp_server.py`（2,222行）のドメイン分割、
    `RoutingConfig`一元化。
- [x] ~~**R16（P3）: ドキュメント鮮度バナー**~~ (2026-07-23)
  - `docs/architecture.md` / `docs/claude-guide.md`の冒頭に「現行正本はdev-notes/current
    64・75」の注意バナーを追加。本文更新はPhase 6の`SPEC.md`改訂時。

### Phase 4: 階層要約をbackfillする

- [x] ~~**AI文書構造プロファイルのgold pilotを実行する**~~ (2026-07-23)
  - 冒頭20ページpilotでは英語書籍 title F1 0.9143、日本語書籍0.9836で有望。
  - PDF bookmarkの階層は正解とは限らないため、title・parent・level・印刷ページ・PDFページを人手確認する。
  - first-10 / first-20 / `目次全体+本文5ページ`を比較し、全文blockへの適用後にparent-edge・節境界・chunk所属を採点する。
  - AI出力はcandidate profileとし、捏造見出しをcanonical構造へ自動採用しない。
  - 問題のある縦書きPDF`4UL3USRQ`と、スキャンが歯抜けの`Q3JJQ9M8`は除外。縦書き枠は未選定とし、目次全項目・印刷ページ連続性・章末まで確認できた通常書籍だけを追加する。
  - `6YUXNA5V`『勉強の哲学』もユーザー判断でgold採点対象から除外。既存AI pilot結果は参考記録としてのみ保持する。
  - 入力済みgold 4資料のfirst-20 AI採点はmicro title F1 0.9683、level正解率0.9907。`E4PJ686A`は60/60完全一致。
  - 長い書籍のPDF page完全一致は13〜20%のため、AIのpage hintを正本採用せず、印刷ページ→PDFページを決定論的に照合する。
  - `IMBFR28G.csv`は見出しにUnicode置換文字173個があり採点不能。AI目次を直接正本採用せず、
    全文で決定論的に再照合する安全gateが実装・実資料検証済みのため、人手再入力と5件目の
    採点は費用対効果が低いと判断し中止した。
  - 詳細結果: `dev-notes/current/70_ai_outline_gold_first20_results.md`
  - 異常PDFの`BXPH3DBZ`は除外し、正常な二段組スキャン論文`IMBFR28G`（脚注・参考文献あり）へ差し替える。
  - 詳細: `dev-notes/current/69_ai_outline_sampling_pilot.md`
  - 正常4資料のfirst-20結果を回帰基準として保存し、first-10／適応sampleの追加比較は行わない。
    AI目次のprompt・採用gateを大きく変更する場合だけ、新しい正常資料でpilotを再開する。
- [x] ~~**AI目次→全文照合による長文PDF fast pathを実装し、初期本番検証する**~~ (2026-07-23)
  - clean text・outlineなしPDFだけを対象に、冒頭20ページから目次粒度の候補を生成し、
    全文で見出しを順序付き再発見できた場合だけPyMuPDF chunksへ構造を付与する。
  - 30ページ未満、スキャン・破損、抽出失敗pageあり、cloud policy確認不能、本文見出し
    coverage 90%未満、構造付きchunk 80%未満はfail-closedでDoclingへ戻す。
  - `83Z5HCN9`（312ページ）は本文見出しcoverage 100%、1,356 chunks、抽出7.3秒で
    本番V3へ格納。別の長文資料もcoverage 100%、抽出3.6秒で成功。
  - 14・21ページ論文がAIを呼ばずDoclingへ進むこと、4回の3件batchが失敗0であること、
    関連15テストと各batch後のHNSW query smoke testを確認済み。
  - feature自体の既定はoff。PDF再取込batchでは`PDF_AI_TOC_FAST_PATH_ENABLE=1`と
    `PDF_AI_TOC_MIN_PAGES=30`を明示して有効化する。
  - 実装詳細: `dev-notes/current/71_ai_toc_fast_path_implementation.md`
- [~] **V3 10件LLM要約pilot** — 全件backfill実施により目的を失った (2026-07-28)
  - pilotは「全件をやるべきか」を判断するための材料集めだったが、構造再構築が423件の要約を
    無効化したため、全件再生成が選択ではなく前提になった。実測は本番実行から得ている
    （23,020呼び出し / 1時間13分 / workers=40で失敗0）。
  - **未回収の論点が1つある**: 親要約のcheap/standard品質比較（R5＝還元モデル役の決定）は
    行っていない。現状は`_deepseek_model`の既定のまま。
- [x] ~~**承認後にLLM要約を全件backfillする**~~ (2026-07-28)
  - 22,118件（生成23,020回、うち1,000字未満の4,385件は後から削除）。
  - 差分再開は`_summaries_are_current()`が担当し、再実行はLLM呼び出し0で済む。
  - **計画と違えた点**: exact/recovered優先・flat_fallbackはsegment限定という区別はせず、
    対象423件を一律に処理した。構造が総入れ替えになったので優先度を付ける意味が無かった。
- [x] ~~**V3 node要約索引を構築する**~~ (2026-07-28)
  - `zotero_paragraphs_v3__sum_node` 12,280件。内訳: 要約22,118 − extractive 215（`searchable=0`、
    索引に入らないことをDBで確認済み）− 単一子の親9,623（子と同内容のため抑制）。
  - **未確認**: summary collection障害時のdirect fallback。下のR9の試験で扱う。

- [x] ~~**注(endnote)を通常検索の対象にする**~~ (2026-07-28・ユーザー決定)
  - 変更前は`endnote`/`footnote`とも`explicit_only`で、クエリに「注」「出典」等の語
    （`_EXPLICIT_NOTE_INTENT`の正規表現）が無ければ**9,259チャンクが検索に出なかった**。
  - 人文系では注に実質的な議論が入る。同日調べた`excavating.ai`は注[1]だけで12,717字あり、
    「テーマ外文献に埋まった事例を探す」用途では、これを語の偶然に依存して隠すのは損失が大きい。
  - `endnote`→`normal`（8,381チャンク＋201ノードを移行）。`footnote`は書誌情報中心のため
    `explicit_only`のまま。`summary_policy`は両方`exclude`を維持（注が章要約に混ざるのは別問題）。
  - **`zone`が検索結果に一切出ていなかったので`meta`へ追加した。** これが無いと注が本文と
    区別できないまま返り、そのまま引用される。
  - **注意**: `ZONE_POLICIES`の変更は取込時にしか効かない。`retrieval_policy`はチャンクの
    メタデータとDB行に焼き込まれるので、既存分は個別更新が要る。

- [x] ~~**U10: 説明不能な本文喪失2種を修正**~~ (2026-07-28)
  - outlier一覧のレビューから発見。**4件は外れ値の閾値に掛かっていなかった**（ratio 0.81〜0.95）
    ので、集計だけ見ていれば見落としていた。U8と同じ「集計では見えない」構図。
  - **巻末註の丸ごと喪失 4件**（`ZPIPI7G4` 212–251頁、`Q9N8ZL82` 142–173頁、
    `Y8D4PPNN` 251–292頁、`FUDKAS3M` 221–247頁の一部）。`_splice_reference_chunks`が
    参照ページの注チャンクを**先に全削除**し、Doclingの再解析結果で置き換える。ガードが
    「置換が空でないこと」だけだったので、**1チャンクでも残れば40頁の削除が確定**した。
    置換の目的は境界の改善であって文字数の削減ではないため、置換後の文字数が削除分の
    50%を下回るなら置換を中止して原本を残す（`MIN_REFERENCE_SPLICE_RATIO`）。テスト2件。
  - **本文13頁の喪失 1件**（`4CY8EIIB`）。`merge_short_chunk_records`の境界キーが
    `(page, reading_order)`で、`reading_order`は記録ごとに一意。つまり**ページ内で
    2つのチャンクが結合することが構造上ありえず、短い断片を救済するはずのmergeが
    最初から機能していなかった**。`_merge_layout_blocks`が1行=1ブロック（約34字）に
    崩れたページでは全行が`HARD_MIN_CHARS`を下回り、merge内部の下限適用で全滅した。
  - 境界キー自体を変えると**全PDFのチャンク粒度が変わる**ので採らず、
    「ページに本文があったのに結果が0チャンクになった場合だけ」章単位の境界で
    やり直す（`_reclaim_fully_excluded_document`と同じ原則: **規則がページを丸ごと
    消すなら、それは規則の誤適用**）。既存の粒度は不変。テスト1件。
  - **未了**: 修正は今後の取込にしか効かない。該当5件の再取込が必要。
- [ ] **U11: itemKeyを持たない1,774チャンク（17添付）**
  - V3コレクション内に`itemKey`も`chunk_scheme`も`zone`も`retrieval_policy`も持たない
    チャンクが1,774件ある。V3移行時にlegacyから採用した分（`adopted_existing_chunks: 242342`）
    のうち、メタデータのスタンプが漏れたもの。
  - 影響: 既定`normal`なのでベクトル検索には出るが、**itemに紐づかないため
    item単位の監査・引用・階層ルーティングからは見えない**。cutover監査が
    `passed: true`を出したのは、監査がitem単位で回るためこれらを一度も見ないから。
  - `document_nodes`も0件。該当添付の再取込が必要。

- [x] ~~**P3-1: 主要PDF経路のzone未判定を解消**~~ (2026-07-28)
  - `pdf_extract.py`はzoneを`corrupted`しか付与せず、paratext判定はAI-TOC/Docling
    経由の資料限定だった。実測: 約1,060万字・29,147チャンク・217アイテムが本来
    paratext（文献リスト・索引・後注）なのに`zone=body`のまま検索に出ていた。
  - 見出し→zoneの語彙が`html_extract.py`・`docling_extract.py`・`pdf_toc_recovery.py`の
    3ファイルに独立して存在し（P3-5）、しかも末尾の見出ししか見ない実装だったため
    「Notes」配下の「Chapter 3」がbody扱いになる欠陥（P3-4、実測2,715チャンク）もあった。
  - `src/heading_zone.py`を新設し単一定義に統合。`classify_heading_path()`は祖先を
    outermost-firstで辿り、最初に一致したparatext zoneを返す。
  - `pdf_extract.py`は既に`chapter_detect.py`経由で`structure_path`/`chapter`/`section`を
    持っていたので、それをそのままこの分類器に渡すだけで済んだ（`corrupted`は優先）。
    合成PDF（PyMuPDF目次に"Bibliography"見出し）で実証: 該当頁の全チャンクが
    `zone=bibliography`になることを確認。
  - `docling_extract.py`の`_heading_zone`は独自語彙・`fullmatch`だったため
    "Notes to Chapter 3"型が全滅していた。共有関数への委譲に置換。
  - `html_extract.py`の`_zone_for_element`は末尾の見出しだけを見ていたため
    祖先を辿る形に修正（P3-4を解消）。既存の正規表現定数は`heading_zone`への
    エイリアスとして残し、既存テストとの互換を維持。
  - テスト: `heading_zone`単体18件、PDF実抽出1件（合成PDFで実証）、
    HTML祖先辿り1件。いずれも修正前に失敗することを確認済み。

- [x] ~~**P1-08: linked_url添付の無言消失を解消**~~ (2026-07-28)
  - `linked_url`（外部URLへのポインタでZoteroがファイル実体を持たない）が
    通常添付として扱われ、ローカル解決・Local APIダウンロードが両方失敗した末に
    `except Exception: continue`で無言消失していた（`DEBUG_ZOTERO_LOCALAPI=1`の
    ときしか見えない）。`75QYJJYK`がZotero上の適格添付585件中manifestに無い唯一の1件。
  - `linkMode`を早期に見て明示的にスキップ・常時警告出力するように変更。
    一般の解決失敗も同様に常時警告するよう変更（同じ握り潰しパターン）。
- [x] ~~**P1-10: ファイル差し替え検出に内容指紋を追加**~~ (2026-07-28)
  - skip判定が`mtime`と`size`の一致だけを見ており、同じ場所への差し替えが
    たまたま同じサイズなら永久に検出できなかった。manifestに差し替えを
    検知する情報が一切保存されていなかった。
  - `src/manifest.py`に`content_signature()`を追加。全文読み込みは全ファイルへの
    I/Oコストが高すぎるため、先頭・末尾各1MiB＋サイズのハッシュに留める
    （前付け・奥付が変われば通常検出できる）。
  - `mtime`/`size`が一致した場合のみ計算（本当に変わったファイルへの余分なI/Oを
    発生させない）。指紋を持たない旧行は、それだけを理由に再解析を強制しない
    （以後の書き込みから記録する）。
  - 判定を`_source_content_unchanged()`として純粋関数に切り出し、5テストで固定。

- [x] ~~**P1-06: パイプライン指紋の無条件スタンプに監査痕跡を追加**~~ (2026-07-28)
  - `bind_manifest_pipeline(adopt_existing=True)`は既存資料に指紋を無条件でスタンプし、
    以後の再解析skip判定を永久に「一致」させていた。**指紋を押した行為自体が、
    それが正しいかを検証する機会を奪う自己封印**。次回の再構築（新collection）でも
    `ensure_pipeline_config`が新規config作成＝`created_pipeline=True`となり再発する。
  - 挙動は変えず（既存の安定動作を壊さない）、`pipeline_fingerprint_adopted: true`を
    追加して信頼採用と実測確認を区別できるようにした。
- [x] ~~**P1-11b: Zoteroを真とする照合チェックを新設**~~ (2026-07-28)
  - `scripts/verify_zotero_reconciliation.py`。Zotero側の適格添付一覧
    （`list_pdf_attachments`の生データを共有関数`classify_attachment_source_type`で
    分類）とmanifestを比較し、欠落を報告する。読み取り専用。
  - 実行結果: 25件欠落。うち24件は`imported_file`で、`dateAdded`が全て本日06:07〜06:08
    （作業中にユーザーがZoteroへ追加した分、まだ取込未実行）。残り1件が`linked_url`の
    `75QYJJYK`（P1-08で既に経路修正済み、再取込待ち）。**バグではなく、チェックが
    正しく機能した結果**。

- [x] ~~**P3-9: `HARD_MIN_CHARS`を言語対応に**~~ (2026-07-28)
  - 他の長さ定数（`MIN_CHUNK_CHARS`/`MAX_CHARS`/`TARGET_CHARS`）は全てCJK版を持つのに、
    mergeの最終フィルタ`HARD_MIN_CHARS`（既定40）だけが言語盲のまま
    `merge_short_chunk_records`内部で固定参照されていた。
  - `HARD_MIN_CHARS_CJK`（既定20、他の定数と同じ比率で半分）を追加し、
    `merge_short_chunk_records`に`hard_min_chars`引数を追加（既定値は従来通りなので
    後方互換）。5経路（pdf_extract・html_extract×2箇所・docling_extract・
    mistral_ocr_extract・rapidocr_extract）全てに配線。各経路は元々CJK判定
    （`is_cjk`/`no_space`）を`min_chars`/`max_chars`の選択に使っており、同じ判定を
    そのまま渡すだけで済んだ。

- [x] ~~**P5: `PersistentClient`構築がHNSWキャッシュを不必要に無効化**~~ (2026-07-29)
  - `rag_mcp_server.py`の`_col()`は`chroma.sqlite3`+manifestのmtime合計が上がったら
    indexerが書き込んだと判断し`_reset_col()`（~2GBのHNSWセグメントを破棄・再mmap）を
    呼んでいた。しかし`chromadb.PersistentClient`を**構築するだけ**で（読み取り専用の
    監査スクリプトでも）`chroma.sqlite3`のmtimeが上がることを確認済み——このチェックは
    「indexerが新規ベクトルを書いた」と「誰かが読み取り用にクライアントを開いただけ」を
    区別できていなかった。P5監査で測定されたp95約20秒・最悪30〜90秒のクエリ遅延の
    直接原因。
  - 修正: mtime上昇を「疑い」のトリガーとしては維持しつつ、`_reset_col()`実行前に
    実際のembedding行数（`chroma.sqlite3`を`mode=ro`の生SQLiteで直接読む
    `_collection_row_count()`。行数確認自体がmtimeを乱さないよう`chromadb`クライアントは
    経由しない）で裏付けを取るよう変更。行数が変化していなければ誤検知と判断し
    `_reset_col()`をスキップするが、`_COL_INIT_MTIME`はその場で新しい値へ更新し、同じ
    誤検知シグナルが以後のクエリ毎に再トリガーし続けることを防ぐ。行数取得自体が
    失敗した場合（DB読めない等）はfail-safeでリセットを実行する。
  - `tests/test_col_staleness_corroboration.py`を追加（5件）。修正前のコードに戻すと
    全件失敗することを確認済み。全779件のテストスイートが通過。

- [x] ~~**P2-01: Docling merge前破棄でページ全損**~~ (2026-07-29)
  - `docling_extract.py`の`split_long_paragraph`直後にあった
    `if len(part) < HARD_MIN_CHARS: continue`が、`_merge_docling_chunks`（同一
    block_type/zone/heading の連続片を結合してから`hard_min_chars`で最終判定する）
    に断片が届く前に個別に破棄していた。1ブロック1行のようなページでは全断片が
    しきい値未満になり、結合の材料が一つも残らずページ全体が消えていた
    （pdf_extract.py側で既に直した劣化ページ復帰と同型の欠陥）。
  - 事前フィルタを撤去し、空文字列のみ弾くよう変更。断片は全て`_merge_docling_chunks`
    へ渡り、そこで結合・最終しきい値判定を行う。
  - 副次的に発見: `tests/test_pdf_extract_layout.py`で、`if __name__=="__main__"`
    ガードの後ろに追記された2テスト（P3-1zone確認・P3-9 CJK確認として本日
    「確認済み」と報告していたもの）が、実際にはガード直後の別関数の**本体内に
    ネストして定義され、一度も収集・実行されていなかった**ことが判明。クラスへ
    戻して復活させ、両方とも現行コードで通過することを確認。`tests/test_no_orphaned_test_methods.py`
    をこの形（`if __name__`ガード内だけでなく、任意の関数内にネストした`test_*`定義）
    まで検出するよう拡張。

- [x] ~~**P2-02: 反復ヘッダ除去でページ消失**~~ (2026-07-29)
  - `pdf_extract.py`の反復行・反復prefix除去で`layout_records`が空になった場合の
    `continue`が2箇所とも無言でページを捨てており、`empty_pages`にも
    `low_text_pages`にも記録が残らなかった。`quality_info["repeated_header_dropped_pages"]`
    を追加し、除去によって空になったページ番号を記録するよう修正。

- [x] ~~**P2-04: Mistral OCR網羅偽装**~~ (2026-07-29)
  - `mistral_ocr_extract.py`の`quality["ocr_pages"]`が応答内容に関わらず常に
    `range(1, page_count+1)`だった。応答から欠落したページ・ブロック0件のページを
    `ocr_pages`から除外し、差分を`missing_pages`として記録するよう修正。

- [x] ~~**P2-08: ローカルOCR Doclingフォールバック型バグ**~~ (確認のみ・既に解決済み)
  - コミット`3b9e2d3`（本セッション前半）で既に修正済みと確認。テストも存在。

- [x] ~~**P2-09/P2-14: gibberish判定・default_qualityが健全体を装う**~~ (2026-07-29)
  - `html_extract.py`のHTML/EPUB両抽出関数、6箇所の早期return全てが
    `default_quality`（健全な空1ページ文書に見える形）をそのまま返しており、
    「読めなかった」「DOMブロック0件」「gibberish判定」「EbookLib未導入」
    「EPUB読込失敗」のどれで諦めたのか記録が残らなかった。各returnに
    `failure_reason`を追加。`local_ocr_pipeline.py:175`は既に十分な失敗記録
    （`local_ocr`内のattempts/gate詳細）を持っており対象外と確認。

- [x] ~~**P2-13: 抽出ステータスが部分欠落を表現できない**~~ (2026-07-29)
  - `extraction=success`が添付単位の二値で、583添付がsuccessのまま539ページの
    欠落を抱えていた実測（原本比較でのみ発見）。純粋関数`_pages_without_chunks()`
    を追加し、`mark_artifact_status`の`counts`に`pages_without_chunks`・
    `chars_out`を記録するよう修正。

- [x] ~~**P2-15: 原本比較の装飾グリフ誤検出**~~ (2026-07-29)
  - `source_page_chars()`が`len(text)`をそのまま使っており、「•」約1,000個の
    装飾的な羅列が`MIN_SOURCE_PAGE_CHARS`を超えて本文と誤認されていた
    （VU5FWYMC p1）。`_meaningful_char_count()`（alnum文字のみ数える）を追加し、
    閾値ではなく文字種で判定するよう変更。

- [x] ~~**P6-3: item非所属チャンクが監査の目に届かない**~~ (2026-07-29)
  - `audit_v3_cutover.py`のitem単位比較はlegacy item keyを反復するため、
    itemKeyを持たないチャンクは一度も評価されない（1,774件実測）。
    `chunk_store.list_chunk_ids_without_item()`（LEFT JOIN、itemKey行が
    そもそも無いケースも拾う）を追加し、グローバルゲートの
    `chunks_without_item`失敗として配線。
  - P6-1（node_id実在確認）・P6-4（zone/retrieval_policy不到達検査）は
    確認の結果、既に本セッション前半の code-review 対応で
    `dangling_node_ids`・`unretrievable_documents`として実装・配線済みだった。
  - P6-6（添付単位の構造欠落がitem単位で隠れる）は`document_structures`が
    item_keyのみでキーされている設計そのものに起因し、P4-1/P4-8の
    fingerprint・再構築設計と絡むため、単純な不変条件追加では閉じない。
    今回は見送り、P4-1/P4-8と合わせて設計判断が必要と再確認。

### Phase 5: 階層検索を既定へ切り替える

- [ ] **V3 active切替後の検索統合試験を実行する**
  - leaf限定、policy filter、3経路RRF、direct fallback、provenanceを実データで確認する。
  - LLM query expansionを除くローカル検索p95を記録する。
  - 注意（R9）: この前提は解消した。`__sum_node`は12,280件で埋まっており(2026-07-28)、
    **summary routing経路を含む本試験が実施可能**になった。手動クエリ2本
    （英語・日本語）で要約レイヤーが応答することは確認済みだが、統合試験は未実施。
- [x] **階層検索V2を唯一の本番検索経路にする** (2026-07-29)
  - `rag_mcp_server.py`から旧検索へのflag rollbackを撤去し、V3階層検索へ固定した。
  - 障害時も旧経路へは戻さず、V3バックアップまたは原本から復旧する。

### Phase 6: legacyデータ面の物理撤去

- [~] **legacy runtime routeを物理削除する**
  - [x] Citation Graphのブラウザ内要約生成・再生成・手動編集と
    `POST/PUT /api/node/summary`を撤去。保存済みV3階層要約の閲覧・品質報告だけに限定した
    （2026-07-29、`fcf11b6`）。
  - [x] `src/build_summaries.py`のlegacy DB writer/CLI/旧要約索引生成、
    `scripts/build_deepseek_summaries.py`、`scripts/run_grounding_gate.py`、
    `db_relations`のlegacy summary writer公開関数を撤去した
    （2026-07-29、`fcf11b6`）。
  - [x] V3 node要約の空文字保存を拒否し、Citation Graph outlineのsummary parts取得を
    node単位N+1から一括SQLへ変更。`setup_wizard.py`の`.env`更新を権限0600のランダム一時
    ファイル＋atomic replaceへ変更した。全906テスト、Ruff、構文・差分検査に合格
    （2026-07-29、`fcf11b6`）。
  - [ ] 旧chunk/FTS、`__sum_item`、`__sum_section`、legacy summary table、
    `insight_generation_status`の物理データ・残存read-only互換コードを撤去する。
  - 復旧は旧面への切替ではなく、V3バックアップまたは原本からの再構築とする。
- [x] ~~**`SPEC.md`を実装結果へ更新する**~~ (2026-07-27)
  - V3構造、DeepSeek要約、事例機能廃止、Gold QAの位置付け、採用したPDFルーティングを反映した。

### 継続監査

- [ ] **V3 cutover後の文字量比outlierを資料単位でレビューする**
  - 2026-07-28の全件監査で**27件**（21件は2026-07-23時点の数）。うち5件はZoteroから削除済みの
    item（`ratio=0.0`）なので実質22件。監査JSONはローカルで再生成し、Gitには保存しない。
  - `CAXCWCQB`の旧PyMuPDF由来scanned警告はRapidOCR品質結果と分離し、警告来歴の整理を行う。
  - 目的は**説明不能な減少が0件であることの確認**。ざっと見た限り大半はV3側の改善で、例えば
    `2N3HG369`はlegacy 8チャンクに対しV3 2,481チャンク（取れていなかったものが取れた）。

- [ ] **決定的fixtureと全テストを各cutover候補で実行する**
  - coverage、zone、境界跨ぎ0、bottom-up scope、OCR gate、状態遷移、検索fallbackを維持する。

## Review Findings（2026-07-22 修正完了）

詳細監査: `dev-notes/current/67_v3_embedding_batch_review.md`

> PID 56797 のV3 PDFバッチと子workerの停止を確認後に修正した。
> 既存埋め込みの全件再計算は不要。初回再開時に保存済みV3 chunkから構造をbackfillする。

- [x] ~~F1（バグ）~~: `list_artifact_status.py` の `message` SQL → `COALESCE` に修正済み
- [x] ~~F2（設計不整合）~~: `unresolved`=failed+blocked, `informational`=empty/degraded に分離済み
- [x] ~~**F3（効率）**~~: 実行中PDFバッチのプロセス環境で`PDF_OCR_FALLBACK=0`を確認済み（二重OCR回避、2026-07-22）。
- [x] ~~F4（軽微）~~: `counts` が `unresolved`/`informational` 出力に含まれることを確認済み
- [x] ~~**F5（軽微）**~~: read専用ツールをSQLite URI `mode=ro`へ変更。
- [x] ~~**F6（高・再開整合性）**~~: `post_index_pending` checkpointと再開時の構造backfillを実装（再埋め込み不要）。
- [x] ~~**F7（高・埋め込み来歴）**~~: V3専用pipeline config、model-state fingerprint、互換性拒否、manifest fingerprintを実装。
- [x] ~~**F8（高・cutover gate）**~~: manifest/Chroma/FTS集合、構造version/fingerprint、pending状態をgate化。noteは検索対象のannotationだがcanonical構造から除外。
- [x] ~~**F9（中・部分PDF）**~~: timeout部分成功をdegraded/retryable化し、ページcoverageと未承認truncated gateを追加。
- [x] ~~**F10（中・commit整合性）**~~: V3 delete/FTS失敗をfatal化し、attachment別ID集合と実vector queryを検証。
- [x] ~~**F11（中・run固定）**~~: run code hashを保存し、実行中の変更を次write前に拒否。
- [x] ~~**F12（低）**~~: numpy等の配列型からembedding dimensionを取得できるよう修正。
- [x] ~~**F13（局所修復）**~~: stale HNSW sentinel自動清掃、元のHNSW設定復元、
  Chromaを正本としたFTS・構造・HNSWの差分修復CLIを追加。manifest coverage不一致は
  自動採用せずattachment単位の再取込へ送る。

## Waiting On Human Decision

- [x] ~~**LLM要約全件backfillの承認**~~ (2026-07-28)
  - 承認を得て実行済み。ただし提示した規模の見積もりは実測と乖離した（当初7,006要約と伝え、
    着地は23,020呼び出し）。小さい資料5件からの外挿だったのが原因で、ノード数の分布は
    中央値39・平均84と大きく偏っている。**この種の外挿は中央値と分布を見てから出すこと。**
- [x] ~~**Mistral OCRフォールバックの本番有効化判断**~~ (2026-07-27)
  - 判断不要になった。`MISTRAL_OCR_FALLBACK_ENABLE`はフラグごと撤去済み。Mistral OCRへ至る
    経路はすべて明示的な操作（`--reocr-candidates`への投入、`--submit`）の後ろにあるか、
    後で別途送るための候補を記録するだけで、通常のingestionが同期的に呼ぶことはない。
    既に明示的な操作の前に2枚目のスイッチを置いても、防ぐはずの障害（鍵があるのに黙って
    何もしない）を生むだけだった。

## Done

- [x] ~~V3 zone/policy/provenance/input scopeの基礎スキーマを追加~~ (2026-07-21)
- [x] ~~EPUB/HTMLのDOMブロック抽出、構造境界付きチャンク結合、表・zone保持を実装~~ (2026-07-21)
- [x] ~~事例MCP公開とMaintenance事例生成・レビューを停止~~ (2026-07-21)
- [x] ~~事例DBの通常経路を撤去し、復元可能なdry-run既定退避CLIを追加~~ (2026-07-21)
- [x] ~~構造V3要約をMaintenanceの単一生成経路へ変更~~ (2026-07-21)
- [x] ~~policy-aware bottom-up要約、30k縮約、scope、prompt分離、索引更新を実装~~ (2026-07-21)
- [x] ~~PDF抽出エンジン共通契約とPyMuPDF・Docling・NDLOCR-Lite基本アダプタを追加~~ (2026-07-21)
- [x] ~~Phase 2の固定採点・レポート・実資料候補選定ツールを実装~~ (2026-07-21)
- [x] ~~OCRベイクオフ8資料を選定し、manifestをfreezeして全8 annotationを作成~~ (2026-07-21)
- [x] ~~PyMuPDF・Docling・NDLOCR-Liteを固定8資料で完走し、比較結果と資料別ルーティング案を保存~~ (2026-07-21)
- [x] ~~V3 Chroma/FTS/manifest隔離、item/limit/dry-run/retry-failed付き再取込を実装~~ (2026-07-21)
- [x] ~~Zotero Local API待ちを解消し、所蔵EPUBでV3実データpilotを実行~~ (2026-07-21)
- [x] ~~EPUB pilot `U8RR2K2S`を1,263 chunks・90 leaves・coverage 100%・重複0・structure recoveredで取込~~ (2026-07-21)
- [x] ~~pilotの親item key `U8RR2K2S`とattachment key `ZFJT72IV`の対応を確認し、collection / manifest / ledgerが同じ資料を指すことを確認~~ (2026-07-21)
- [x] ~~EPUB pilotのcutover auditを実行し、body/back_matter/colophon/tocのzone分布を保存~~ (2026-07-21)
- [x] ~~EPUB pilotでextractive node要約160件を生成し、検索非対象であることを確認~~ (2026-07-21)
- [x] ~~再OCR候補、決定的gate、`--force-adopt --item`、補償付きV3正本採用を実装~~ (2026-07-21)
- [x] ~~leaf限定検索、policy filter、3経路RRF、direct fallback、provenanceを実装~~ (2026-07-21)
- [x] ~~V3 cutover全件監査CLI（coverage・zone・文字量差分・構造状態）を実装~~ (2026-07-21)
- [x] ~~YomiTokuアダプタ・エンジンを追加し8資料ベイクオフで実測（構造化能力でDoclingを上回る、CC BY-NC-SA非商用）~~ (2026-07-21)
- [x] ~~DeepSeek-OCRを実機検証の結果Apple Silicon処理負荷が実用外として候補・実装・モデル重みごと除去~~ (2026-07-21)
- [x] ~~Mistral OCRアダプタ・エンジンを追加し実APIで検証（8資料平均で最高スコア、bbox+type応答schemaを実機確認）~~ (2026-07-21)
- [x] ~~ベイクオフ採点バグ（markdown装飾・短anchor希釈）を修正し全エンジンを再採点、回帰テスト追加~~ (2026-07-21)
- [x] ~~Mistral OCR資料単位フォールバックをingestionへ実装（三重opt-inゲート、既定無効）~~ (2026-07-21)
- [x] ~~`EngineRegistry.select()`をDocling既定＋PyMuPDF fast-path gate（outline+埋込テキストのみ）へ実装~~ (2026-07-21)
- [x] ~~PDF解析エンジンと資料別ルーティングをユーザー承認（Docling既定、YomiToku/Mistralはopt-in、フォールバック粒度=資料単位）~~ (2026-07-21)
- [x] ~~.env/.env.exampleをカテゴリ別に整理し、実使用変数の過不足を解消~~ (2026-07-21)
- [x] ~~Doclingの固定30分タイムアウトが大型資料の処理済み分を全破棄する問題を修正~~ (2026-07-22)
  - Docling自身の`PdfPipelineOptions.document_timeout`（1700秒）を有効化。超過時はDocling内部でページ取得を打ち切りつつ、既に変換済みのページはそのまま保持して`PARTIAL_SUCCESS`を返す（例外にならない）。外側の`DoclingWorker`の強制killより先にこちらが働くため、正当に時間のかかる大型資料でも作業を失わない。
  - `extract_chunks_from_pdf_with_docling`は`quality_info["truncated_by_timeout"]`で部分成功を明示する。
  - 外側`DoclingWorker.timeout_sec`は1800→3600秒に緩和し、「本当にハングした場合」専用のbackstopへ役割変更（単一ページがネイティブOCRコード内でPythonに制御を返さず固まるケースなど、Docling自身のループ内チェックでは検知できないケース向け）。
  - 当初検討していたCPU使用率ベースのliveness detection（Sonnet subagent, task #10）はこの修正で不要となり中止（`docling_worker.py`に変更は未反映のまま、実装差し替え済み）。
  - 既存テスト18件（test_docling_extract.py / test_extraction_engine.py / test_docling_worker.py）は全通過。モックの一つに`ConversionStatus.SUCCESS`を追加。
