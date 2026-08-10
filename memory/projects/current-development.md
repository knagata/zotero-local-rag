# Current development handoff

Updated: 2026-08-11

## Objective

運用開始前に、現行の抽出・取り込み経路を挙動不変で整理し、決定的テスト網を広げる。
その後に現在のBGE-M3と現行コードでV3データプレーンをclean rebuildし、監査済みDBを
検索品質・クラスタ・引用同定・flat PDF復元の評価基準にする。

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
- userの明示変更によりlocal `.env`はPDF構造復元とAI TOCが1。AI TOC採用後の参照節を再解析する
  local Docling flagも1。OCR layer audit、query expansion、LLM summaries、LLM reference
  extraction、Mistral queueは0のまま。M5を実行すると適格PDFのAI目次で有料LLMを呼び得るが、
  M5承認前には実行しない。
  `docs/embedding-rebuild-readiness.md`がgo/no-go、実行入口、rollbackの記録である。
- 150行超の関数は22件から16件へ減少。最大は
  `index_from_zotero._index_library`（1,092行）、次が`pdf_extract.extract_chunks_from_pdf`（607行）。
- 既定suiteは1,599 passed / 4 skipped / 5 deselected。実資料取込baselineは5 passed。
  baseline子プロセスはlocal `.env`に関係なくhosted ingestion feature 6種を強制offにする。
  初回slow差分のPDFは全3頁でAI目次の最小30頁未満、AI TOC statusも無かったため有料callは
  発生していない。active V3への書込みもなく、防壁追加後の再検証は全件hosted-offでpass。
  抽出・取り込み・構造を変更したら
  `uv run pytest -m slow`も必要。
- P1/P2の既知製品欠陥はない。残るP3はS2 author-less識別とflat PDF 3形状で、どちらも
  実データを読んでから方針を決める。

## Existing V3 database assessment

現DBを品質評価の正本にしてはいけないが、コード整理や合成テストを続ける妨げにはならない。

- active planeは`zotero_paragraphs_v3` / `manifest_v3.json` / `lexical_v3.sqlite3`。
- manifestは614添付。保存済みpipeline fingerprintは全entryで一致し、adopted entryは0。
  HNSW validated、post-index pendingなし、inflightなし。
- Chromaは513,683 IDs、FTSは513,684 IDs。FTS側に短い孤立行が1件あり、現状態では
  cutoverのID一致gateを通らない。
- 保存時と現在はいずれもBGE-M3・1024次元・正規化あり。保存済みmodel-state/pipeline
  fingerprintは現在環境と不一致なので、既存collectionへの増分取り込みはfail-closedで拒否される。
- ただし保存済み16文書を現在のBGE-M3で再計算した対ベクトルcosineは全件1.0。
  実ベクトル空間は一致している強い証拠だが、fingerprintを現在値で上書きして「検証済み」に
  してはならない。
- pipeline fingerprintは抽出コードを含まない。現在のchunkが最新抽出規則で作られた保証はなく、
  抽出コード変更後の既存資料には`--force-reparse`が必要。

## Next work, in order

`TASKS.md`の「2026-08-10 埋め込み開始までのマイルストーン」を進捗の正本とする。
到達点は全件埋め込みの実行ではなく、`Setup.command`の`REBUILD`確認直前で止める
M0〜M4.2は完了。M4.3のpilot再利用拒否も実装・検証済みで、実装commitをreadinessへ
pinする記録commitだけが残る。pin完了まではM5をHOLDする。

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
7. **M4.3（pin待ち）**: pilotの開始data planeを新規/空に限定し、既存planeの測定誤報を拒否。
   実装commitをreadinessへpinしてReady to Embedへ戻す。
8. **M5（別承認）**: 明示承認後だけclean rebuildを実行する。完了後に
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
