# Current development handoff

Updated: 2026-08-15

## 2026-08-15 Citation Graph title repair

- Citation GraphのChroma metadata取得は、同一itemの章名も混在する`title`を文字列`MAX`で
  選ばず、反復数最大の書誌値を使う。Citation対象576 itemのread-only監査で旧規則の
  誤選択399 itemを確認し、報告2 itemはどちらも正しい書名になることを実DBで照合した。
  active DB、構造、要約の更新は不要。文書構造のattachment rootは複数添付識別用の
  ファイル名なので、書名と混同して構造fingerprintを書き換えない。

## Objective

運用開始前に、現行の抽出・取り込み経路を挙動不変で整理し、決定的テスト網を広げる。
その後に現在のBGE-M3と現行コードでV3データプレーンをclean rebuildし、監査済みDBを
検索品質・クラスタ・引用同定・flat PDF復元の評価基準にする。

## 2026-08-13 M5 completion

- clean rebuildと既存Mistral cache 48件の採用は完了。active V3はmanifest 618添付、
  `hnsw_validated=true`、inflight/failed/deferred 0。Zotero required 618、原本欠落頁0、
  orphan/dangling/unretrievable 0、cutover監査593/593合格。
- Zotero reconciliationはmanifest済みEPUB siblingの現在のattachment-local tagも個別照会し、
  `rag:exclude`ならPDFをrequiredから外さない。取込側のusable sibling規則と一致し、実照合618/618合格。
- rebuild後に検出したunchanged structureのChroma node metadata未同期を修正し、2,857件から0へ
  戻した。Zotero監査はattachment-local excludeと、committed EPUB siblingを持つ
  `rag:prefer-epub` PDFを取込と同じ規則で扱う。
- 警告検証でGranite失敗の内因を不正crop座標と再現。例外chainをrunner境界で保持する。
  崩壊doctagはDB書込み前に検査し、本文を保つ反復suffix除去、少数bbox反転の限定正規化、暗色scanner
  matte頁だけの一時crop再試行を追加した。再試行PDFは非crop頁を元PDF objectのまま保持する。
  最小失敗資料は保持版でGranite 5/5頁・41 chunkまで回復。対象8件だけの選択再取込は完了し、Granite
  1件、Docling安全fallback 7件。failed/deferred/inflight 0、manifest 618、HNSW validated、3監査合格。
  Docling workerはtimeout/crash交換時にkillまでescalateして必ずreapする。
- EPUB部分coverage 69冊・276 spineのうち270は100字以下の非画像補助wrapperだった。狭いblank
  判定の非書込み再抽出で67冊がcomplete、残る2冊6 spineはfail-closed。active DBは監査済み旧抽出
  のままで、本文chunkは同一。次にこのmetadata-only差分を採用する場合もattachment scopeを明示する。
- PDF部分coverage 32件・163頁は主に画像頁。本文と画像artifactを分ける設計なしに自動採用しない。
  RapidOCR空結果274回はsubprocessからcountを返す仕組みを作るまでログ抑制しない。

## 2026-08-14 structure repair

- PDF内蔵outlineがないMistral資料をstructure-only Maintenanceが再読すると、OCR由来の
  `structure_path`を消してflat fallbackへ落とす経路を修正した。content-addressed raw OCR cacheを
  ローカル再解析し、保存済みchunk ID集合との完全一致かつ全chunk mapping時だけmetadataを再投影する。
  mismatchは採用せず、API call・rechunk・re-embeddingは行わない。
- 影響資料1件は書込み前dry-runと対象backup後、2,904 source chunkだけをitem scopeで修復した。
  `exact` 1,217 nodes / 1,104 leavesとなり、Citation Graphへ112 visible nodesが返る。更新前後の
  Chroma ID・本文は同一。ローカルbackupは`data/backups/pre-structure-repair-*-20260814/`。
- 続けてflat fallback 199件をread-only inventoryし、Mistral 23件中、cache・ID完全一致・全chunk
  mapping・exact/recoveredの安全ゲートを通る14件だけをitem scopeで修復した。9件はflatのまま、
  非Mistral 176件は未変更。対象25,897保存行のID・本文は同一で、visible outline空は0。
  現分布はexact 174 / recovered 233 / flat fallback 185。対象backupは
  `data/backups/pre-flat-mistral-structure-repair-20260814/`。
- AI TOCが全見出しをlevel 2から返すと先頭見出しが文書全体の偽親になるため、最小levelを1へ
  正規化した。Abstract後の本文だけがdeeperな誤応答も、連続runを同じ幅だけ上げて機能領域を閉じる。
  採用済み47件を全件dry-runし、改善が明確な21件だけをbackup後に更新、重複が表面化する1件は保留。
  対象21,520保存行のID・本文は同一、構造はdry-runと一致、再埋め込みなし。backupは
  `data/backups/pre-ai-toc-level-repair-20260814/`。
- EPUB fragment TOCが正規pathを切り替えた後もspine内の古いDOM見出しstackを持ち越し、文書タイトルや
  章見出しを各sectionへ重複挿入する経路を修正した。209 itemのread-only再解析でpath差分のある8 item
  だけをbackup後に更新し、3 itemはsource/chunk完全一致またはusable TOC gate不合格で未変更。
  対象4,154保存行のID・本文は同一、再埋め込みなし。全canonical treeでAbstract・概要・摘要rootの
  別見出し子孫は0。backupは`data/backups/pre-epub-toc-boundary-repair-20260814/`。

## 2026-08-14 summary trust metadata

- 階層要約監査は索引8,074/8,074件一致だが、6 itemが`degraded/no_llm_summary`で停止した。
  余剰status 1件は既知のnote-only除外で正常。全件再生成は行わない。
- summary-onlyの文単位根拠検証統計をinput scopeへ保存し、完全合格を`accepted`、根拠確認済み文が
  残る部分合格を`candidate`として採用する。Citation Graphはverified/limited/legacy/unavailableを
  明示する。既存合格要約は検証記録なしのまま保持し、有料API callとactive DB書換えは未実行。
- coverage付き既定suiteは1,636 passed / 7 skipped / 5 deselected。要約生成本体は243行へ縮小し、
  function-size上限も同時に下げた。
- 2026-08-15にuser承認後、失敗6 itemだけを`--force --embed`で再生成した。5件は初回、
  約119万字が1章へ集約された1件は2回目で成功。全6件が`accepted`で根拠確認文の棄却0。
  要約索引は8,080/8,080件、全体の階層要約監査は合格。全件再要約は行っていない。

## 2026-08-15 summary maintenance progress

- Maintenance Widgetの全件要約`SUMMARIZE`確認を、処理開始時ではなく項目4の選択直後へ移した。
  後段処理後の入力待ちは発生しない。
- 要約CLIはitem完了数/総数・status内訳と、API送信/応答/失敗/処理中件数を逐次表示する。
  正常応答では根拠確認済み文数/生成文数も出し、並列worker間の出力はlockで整列する。
- coverage付き既定suiteは1,640 passed / 7 skipped / 5 deselected。`summary_core.py`の未到達文
  上限は28から18へ下げた。

## Read first

1. `CLAUDE.md`を最初から最後まで読む。
2. `SPEC.md`のactive V3 data planeと、`TASKS.md`の2026-08-10節を読む。
3. `docs/post-refactor-followups.md`で、実データ判断が必要な残件を確認する。
4. `git status --short`と`git log -3 --oneline`で、ユーザー変更と現在のコミットを確認する。

## Current repository state

- PDF取込の最初のseamとして明示NDLOCR/Docling overrideを
  `_extract_pdf_override`へ分離し、実extractorを呼ばない決定的テストで固定した。
  部分成功とartifact statusの純粋契約も追加した。
- 通常PDF抽出後のgateを純粋な`_pdf_gate_plan`と環境adapterへ分離し、
  5 actionとfast-path/queue照会の短絡を決定的fixtureで固定した。
- Mistral deferのartifact status・manifest・既存chunk保持・inflight清掃を
  `_defer_pdf_to_mistral`に隔離し、実DBを使わない統合fixtureで固定した。
- OCR layer auditの無効/cache/一時障害/原本障害/標本不足を
  `_audit_pdf_ocr_layer`に隔離し、LLMを呼ばないfixtureで固定した。
- born-digital PDFの画像頁patchを`_patch_born_digital_scanned_pages`に隔離し、
  読順splice・失敗時保持・文字0件のfigure解決をfixtureで固定した。
- scan-derived OCRの欠落頁を`scanrepair` namespaceで置換する責務を
  `_repair_scan_derived_pages`に隔離し、blank再計算と失敗時保持を固定した。
- corrupted text pageのgarbled chunk全置換・読順splice・quality再計算を
  `_patch_corrupted_text_pages`に隔離し、worker失敗時の元データ保持を固定した。
- OCR layerの無いscanのMistral deferとDocling/Granite置換を
  `_run_initial_scan_replacement`に隔離し、provenance・audit非適用・失敗時の状態を固定した。
- PyMuPDFが0 chunkだが頁がある非scan-replacement PDFのDocling fallbackを
  `_extract_empty_pdf_with_docling`に隔離し、provenance・試行状態・失敗・0頁短絡を固定した。
- 劣化・未検証OCR layerのMistral deferとDocling/Granite置換を
  `_run_post_audit_scan_replacement`に隔離し、audit継承と失敗時の状態を固定した。
- generic gateのDocling正常採用と不採用/例外時のPyMuPDF fallback保持を
  `_escalate_pdf_to_docling`に隔離し、quality-uncertain理由の追記を固定した。
- AI TOCの同一source no-structure継承、accepted/rejectedのchunkとdiagnostics更新を
  `_recover_pdf_outline_with_ai_toc`に隔離し、実LLM無しで固定した。
- 同一sourceの確定済みAI TOC no-structure判定を、対象限定の`--refresh-ai-toc`＋
  `--force-reparse`で安全に再評価できるようにした。item/attachment scopeとfeature enableを
  必須にし、PDF以外・parser/OCR override・quality/retry modeとの曖昧な併用は拒否する。
  discovery後にもscopeが非空かつPDF-onlyか検証し、EPUB/HTMLやmixed itemを正本write前に拒否する。
  通常の負キャッシュ節約とattachment commitは不変で、manifest/fingerprint手修正は不要。
- 5種類のPDF gate action実行を`_dispatch_pdf_gate_action`に集約し、
  disabledタグとlocal-exhausted時のDocling非再実行を追加固定した。
- 通常PDFのpre-gate sequenceを`_prepare_pdf_for_structure_gate`と`_PdfPreGateState`へ
  呼出し順不変で集約し、`_extract_pdf_chunks`を113行まで縮小した。
- M1は完了。最初のseamとして、強制Mistral採用以外の通常HTML/EPUB/PDF dispatchを
  `_extract_attachment_by_source_type`へ分離した。合成fixtureで各extractorの排他選択と
  PDF deferred結果の保持を固定した。次に抽出済みchunk・status・manifest情報から
  attachment 1件分の`PendingAttachmentCandidate`を副作用なしで組み立て、全fieldを
  `PendingIndexBatch.add_attachment`で一括stageするようにした。
- M3の隔離scale pilotは完了。`scripts/embedding_scale_pilot.py`がfake/real BGEを明示的に分け、
  active V3とのpath重複を拒否する。BGE-M3/MPSの300合成chunkはbatch 128・sync 10000で
  `4.59 chunks/s`、peak RSS `900.53 MB`、close/reopen後300 IDs一致、HNSW query成功。
  batch 8の部分中断再開と、途中upsert失敗後のvector/FTS補償も成功した。
  `--output`はactive relations DB本体とSQLite sidecarもpilot開始前に拒否する。
  計測開始時は新規または空のdata planeを必須とし、既存pilot planeのID skipをthroughputとして
  誤報しない。同一呼出し内の意図的な中断・resumeだけが既存IDを利用する。
- M2は完了。`pdf_extract.py`の既定suite到達は73%だが、フェーズ別ではfallback/OCR 6%、
  finalize/error 28%に留まり、実資料baselineもcorrupted/OCR routeを対象外としている。そこで
  chunk生成・OCR本体は動かさず、最終chunk ID検査とtext-defect付与だけを純粋な
  `_finalize_pdf_output`へ分離した。`extract_chunks_from_pdf`は614行から607行になった。
  今回のrebuild世代では現行heuristic、OCR route、chunk本文・IDを凍結し、P3 flat-PDF候補は
  実資料評価を要する次世代変更へ送る。clean rebuildは旧chunkを流用せず全件再抽出する。
- M4の元preflightはruntime `4b58d6b`で完了し、dry-run inventoryは
  590 items / 616 attachments（PDF 364、EPUB 208、HTML 44）、canonical writeなし。
  `data/backups/pre-clean-rebuild-20260810-4b58d6b`へ8.67GBのrollback snapshotを作り、別copyから
  Chroma 513,683、FTS 513,684、manifest 614、lexical/relations quick check `ok`を確認した。
  既知の孤立FTS 1行は修復していない。backup後も空き70.6GB、indexing lockは残っていない。
  backup destinationはsource Chroma自身・配下を実パス比較でcopy前に拒否する。
- userの明示変更によりlocal `.env`はPDF構造復元、AI TOC、Mistral queueが1。AI TOC採用後の
  参照節を再解析するlocal Docling flagも1。OCR layer audit、query expansion、LLM summaries、
  LLM reference extractionは0のまま。queue生成は外部送信しないが、Mistral Batchの実送信・採用と
  適格PDFのAI目次は別途有料call承認を要する。
  `docs/embedding-rebuild-readiness.md`がgo/no-go、実行入口、rollbackの記録である。
- 150行超の関数は22件から16件へ減少。最大は
  `index_from_zotero._index_library`（1,092行）、次が`pdf_extract.extract_chunks_from_pdf`（607行）。
- 既定suiteは1,599 passed / 4 skipped / 5 deselected。実資料取込baselineは5 passed。
  baseline子プロセスはlocal `.env`に関係なくhosted ingestion feature 6種を強制offにする。
  初回slow差分のPDFは全3頁でAI目次の最小30頁未満、AI TOC statusも無かったため有料callは
  発生していない。active V3への書込みもなく、防壁追加後の再検証は全件hosted-offでpass。
  抽出・取り込み・構造を変更したら
  `uv run pytest -m slow`も必要。
- coverage予算の`pdf_extract.py`は、M2以前の207からCI Linux実測205へ下げている。macOSでは
  同じ`not slow and not corpus` selectorが202だが、その値を全環境へ要求するとLinux CIだけが
  失敗するため、クロスプラットフォーム上限は205とする。製品コードや抽出挙動の差ではない。
- M4.4で、clean rebuildの完全性検査をHNSW ready公開より前へ移し、失敗時の
  `hnsw_validated=false`をMCP read pathが永続的に拒否するようにした。後段の
  `rebuild_document_structure.py`も非dry-run全体で共有indexing lockを持つ。indexer code
  fingerprintは`sha256:f3f1ebbac3e383b0c94d22c696caaad45c62d7eabc8b169d0b96a9ae01f5eec5`。
- M4.5は完了。`environment_with_saved_dotenv`が保存済み`.env`/`.env.policy`を親環境へ上書きし、
  indexer・structure rebuild・audit childへ明示的に渡す。親long=Granite/queue=0より保存済み
  long=Mistral/queue=1が両rebuild childで勝つ回帰テストを追加し、対象54件、compileall、fatal Ruff、
  差分検査に合格した。
- userが修正前のclean rebuildを33/618で停止し、差分継続も353/619で停止した。active V3は
  manifest 352件、inflight 1件、`hnsw_validated=false`の未完成世代で、indexing lockは無い。
  MCPは拒否する。古いOCR routeで生成した部分世代は再利用せず、M5は最初からclean rebuildする。
- M5直前に旧complete-generation snapshotの欠落を検出した。現activeは既に検索不可なので、
  `data/backups/pre-m5-incomplete-20260811`へ未完成世代を診断用に一式退避した。Chroma/FTS
  272,007行、manifest 352件、SQLite quick checkはいずれも`ok`。正常世代のrollbackではない。
- user承認後のM5初回実行は8/619で停止した。監視ログのOCR layer audit `measured`を契機に、
  保存済み`.env`でOCR layer auditがreadinessと逆に1だったことを検出し、auditだけ0へ戻した。
  この部分世代は再利用せずclean rebuildを再度最初から行う。query expansion、LLM summaries/
  reference extraction、Mistral OCRはuser確認により1でよく、保存設定は1。再実行childは開始時環境を
  固定しているため前3機能は0のまま完走させる。Mistral queue deferred 1件、Batch送信・採用なし。
- review残件のP1は`run_db_audit.py`のreport出力先保護。現在の設定は全て`data/quality/`で安全だが、
  任意設定でactive DBを指せるためfollow-upへ記録した。埋め込み開始条件には含めず、今回未修正。
  残るP3はS2 author-less識別とflat PDF 3形状で、どちらも実データを読んでから方針を決める。

## Existing V3 database assessment

現DBは未完成世代なので品質評価・検索の正本にしてはいけないが、コード整理や合成テストを
続ける妨げにはならない。

- active planeは`zotero_paragraphs_v3` / `manifest_v3.json` / `lexical_v3.sqlite3`。
- manifestは352添付、inflight 1件、`hnsw_validated=false`。MCP read pathはこの世代を拒否する。
- 修正前のstale OCR routeで一部を生成したため、差分継続やfingerprint手修正で採用しない。
- 旧complete-generation snapshotは現存しない。診断用snapshot
  `data/backups/pre-m5-incomplete-20260811`も未完成かつ検索不可なので品質正本ではない。
  次のclean rebuildでのみ世代全体を置換し、個別storeを混ぜない。

## Next work, in order

`TASKS.md`の「2026-08-10 埋め込み開始までのマイルストーン」を進捗の正本とする。
M0〜M4.5は完了。修正前に開始・停止されたactive V3は未完成かつ検索不可なので再利用しない。
次はユーザーの別承認後、`Setup.command`からclean rebuildを最初から行う。

1. **M1（完了）**: 通常attachment dispatchとpending候補の組立を決定的fixtureで固定した。
2. **M2（完了）**: 最終出力確定だけを純粋関数へ分け、現行抽出・chunk境界を今回の世代として
   凍結した。同じ世代内でchunk本文・IDを変える未解決変更はない。
3. **M3（完了）**: 暫定bulk設定は`UPSERT_BATCH_SIZE=128`、
   `CHROMA_HNSW_SYNC_THRESHOLD=10000`。300合成chunkで`4.59 chunks/s`・peak RSS
   `900.53 MB`を確認し、中断再開・補償・reopen・実queryも通した。
4. **M4（完了）**: 対象commit、dry-run inventory、現V3 backup、容量、lock、埋め込み設定、
   有料機能無効、rollback点をgo/no-go表で確定し、`REBUILD`入力前で停止した。
5. **M4.1（完了）**: 対象限定のAI TOC負キャッシュrefreshを実装・検証し、runtimeを
   `docs/embedding-rebuild-readiness.md`へpinしてReady to Embedへ戻した。
6. **M4.2（完了）**: pilot/backupの稼働データ防壁とrefresh解決後scopeを修正・検証し、
   runtimeをreadinessへpinしてReady to Embedへ戻した。
7. **M4.3（完了）**: pilotの開始data planeを新規/空に限定し、既存planeの測定誤報を拒否。
   実装runtimeをreadinessへpinしてReady to Embedへ戻した。
8. **M4.4（完了）**: 不完全clean generationをHNSW readyにせず、MCPもmanifest gateで拒否。
   後段structure writerを共有lockで排他し、埋め込み開始に直接必要なreview指摘を解消した。
9. **M4.5（完了）**: Setupの全DB lifecycle childで保存済みconfigを親processの古い値より優先し、
   stale Granite/queue設定を使った事故の回帰テストを追加した。
10. **M5（別承認）**: M4.5完了と明示承認後だけclean rebuildを実行する。完了後に
   `uv run python scripts/run_db_audit.py`でZotero・原本・DBの3監査を通す。

## Decision boundaries

- 合成fixture、純粋関数化、挙動不変の責務分離は自律的に進めてよい。
- 抽出heuristic、OCR route、chunk境界、S2 author-less条件、flat PDF昇格条件、固定8クラスタの
  変更は生成物・実資料を読み、精度trade-offが見えた時点でユーザーへ判断材料を示す。
- 現DBへの増分取り込み、fingerprintの手修正、全件rebuild、有料LLM/OCRは実行しない。
  実資料baselineはhosted featureを強制offにして検証する。
  rebuildは上記の挙動不変リファクタ完了後に行う。
- 実データを書き換えない診断では、`.env`を`load_dotenv_native(ROOT)`で読んでからactive planeを
  解決する。識別子、書名、絶対パスを追跡ファイルへ保存しない。

## Required handoff discipline

変更ごとに`TASKS.md`と品質予算を同じコミットへ含める。完了または方針変更時にはこのファイルも
更新し、新しいセッションが古いDB診断や開始点を前提にしないようにする。
