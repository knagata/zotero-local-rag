# Current development handoff

Updated: 2026-09-29

## 2026-09-29 re-OCR adoption audit completeness

- 再OCR採用はprepared chunkから共通`source_coverage`を再構成・検証してmanifestへ保存する。不完全な
  coverageは正本更新前に拒否する。構造メタデータ付与後の段落からfingerprintを確定し、リスト値を
  通常取込・SQLite監査と同じJSON scalarでChromaへ保存する。
- 既存の対象資料を保存済み48/48頁・540 chunk結果から再採用した。coverageはexpected/attempted/textが
  すべて48、索引・構造・manifestのfingerprintは一致。DB監査はZotero required 639/639、原本欠落・
  orphan・dangling・unretrievable 0、cutover 614/614、failed 0で合格した。

## 2026-09-28 vertical Japanese PDF reading order

- 縦組みPDFは座標の横組みsortを使わず、line directionで判定したページのcontent stream順を保持する。
  縦列本文を見出し境界まで結合し、block numberで詳細geometryと対応づける。実資料は203/203頁・
  503 chunkとなり、報告箇所を263字の連続段落で確認した。
- 差し替え後の新添付を対象限定で再取込し、通常同期で削除済み旧添付を退役した。新添付はChroma・FTS
  とも503件、旧添付はともに0件。報告3 queryは段落・資料検索の双方で対象を返す。構造は更新済み、
  階層要約はsource変更により`stale`であり、有料再生成は別途行う。

## 2026-09-28 compact CJK retrieval and targeted re-OCR

- `MIN_RETURN_CHARS=200`が、縦組みPDFの正常な30〜40字日本語source regionをsemantic検索と
  `search_items`から除外していた。`ja`/`zh`は既定30字（`MIN_RETURN_CHARS_CJK`）とし、FTS直接一致は
  長さfloorだけで捨てない。報告された3 queryはいずれも対象限定段落検索5件、資料検索1件を返す。
- 対象限定NDLOCR-Lite再抽出は48/48頁、540 chunk、coverage 1.0、文字量比0.959でgate通過。
  queue runnerの採用関数キーワード不一致を直して正本へ採用し、構造`exact`、要約`stale`となった。
- 文字化け資料の除外同期は本文・FTS・構造・要約を削除済み。Remote MCPは検索修正版で再起動済み。

## 2026-09-28 summary freshness excludes empty structures

- 管理画面の階層要約件数が、本文・node 0件の`rag:exclude` itemも空の構造ヘッダとstale summary状態だけで
  1件と数えていた。要約監査を1件以上の構造nodeを持つitemに限定し、本文起点の実バッチ候補と一致させた。

## 2026-09-28 Laokoon RAG exclusion

- ZoteroのLaokoon PDF添付へ`rag:exclude`を付与し、除外同期済み。manifest、本文・要約Chroma、FTS、
  構造node、要約、添付artifactはいずれも0件。構造ヘッダは`unavailable`・空source fingerprintで
  検索対象を持たない。未解決artifact全体は0件。送信済みMistral Batchの結果はRAGへ採用しない。

## 2026-09-28 Remote MCP embedding device correction

- local `.env`はBGE-M3の絶対パスを明示していたため、`EMB_PROFILE=bge`でも明示モデル分岐の
  CPU既定値を選んでいた。macOSのBGEはモデルパス明示時もMPSを既定にし、local設定にも
  `EMB_DEVICE=mps`を明記した。MPS availabilityはtrue、関連59テスト合格、Remote MCP再起動後の
  local/public OAuth healthはいずれも正常。埋め込みモデル・次元・fingerprintは変更していない。

## 2026-09-28 readable sparse OCR pages

- 80字未満という長さだけで章扉・短い章末を未解決スキャン頁に数えていた。十分な文字率、制御文字なし、
  文字化けなしを満たす短文は正常なOCR文字層として保持し、空画像・短いラベル・記号ノイズは従来どおり
  疑わしい頁とする。文書全体の2%許容値は変更しない。対象PDFは再判定で未解決scan頁7→0、対象限定
  再取込後にextraction success、203頁・2,362 chunk・90,585字、未解決artifact全体は2→1となった。

## 2026-09-27 coverage budget platform correction

- macOSにだけある任意依存で到達した既存分岐までcoverage予算を縮め、Linux CIで5モジュールが
  悪化判定された。該当5件だけをCI実測上限へ戻し、今回の新規・変更コードの改善値は維持した。

## 2026-09-27 evidence handoff boundary

- `rag_search`の各段落結果へpath-freeなversion付き`evidence_reference`を追加。別リポジトリの
  Research Workspace MCPがZotero本文を複製せず、根拠位置と引用文を案件へ保存できる境界とする。

## 2026-09-26 read-only external text discovery

- stdio/Remote共通MCPへ`search_open_texts`と`inspect_open_text_candidate`を追加。NDL、J-STAGE、
  Open Library、`CINII_APP_ID`設定時のCiNiiを対象に文字資料だけを候補提示し、後者はローカルZoteroの
  title重複と書誌不足を確認する。外部ファイル取得、Zotero登録、候補永続化、索引更新はしない。
- 学位、掲載先、所属、引用数から品質スコアを作らず、観察可能な注意点だけを返す。引用データ欠落は
  `unknown_not_penalized`として減点しない。
- 実利用でCiNii/J-STAGEの年フィールド欠落、複数語の0件、NDL 429を確認。CiNii
  `prism:publicationDate`、J-STAGE `pubyear`を正規化し、`auto/all/any/phrase`検索、source別の
  戦略表示、NDL直列化・5分cache・429限定再試行を追加した。DOI/NAID/NDL IDを保持し、重複確認にも使う。

## 2026-09-26 admin confirmation keyboard submit

- 階層要約・Mistral OCRの確認ダイアログはformの`submit`を単一経路とし、入力欄のEnterと実行ボタンを
  同じ`dialog.close("confirm")`へ接続した。キャンセルは`type=button`で分離し、asset versionを更新した。
- Mistral Batchのstatus照会は終端状態だけでなく`queued`/`running`/`in_progress`もstateのphaseへ保存する。
  管理画面の現在状態にremote phase、完了数/総数、最終照会時刻を表示する。admin jobの完了は照会処理の
  完了であり、OCR batch完了ではないため、両者を画面上で区別する。
- 抽出失敗でmanifest未到達の添付は、Zoteroから削除されても旧Artifact行が残った。全件同期のstale候補を
  manifestだけでなくattachment-scoped台帳との和から作り、inventoryにないキーを既存の大量削除guard下で
  清掃する。PDF置換済み資料の旧EPUB failed行は対象限定で削除済み。
- Mistral Batch全体が`SUCCESS`でも、資料単位の504は通常の`output_file`ではなく`error_file`へ置かれる。
  collectorは両方を結合して解析し、408/409/425/429/5xxだけをretryableとして記録する。maintenanceは
  retryable reportのitemだけで新しいBatchを作る。Laokoon 1件の504結果を回収し、対象限定で再送済み。

## 2026-09-25 generic-title S2 identity repair

- 一般的な章タイトル`Introduction`が、著者なしのS2候補`An Introduction to the Bootstrap`へ
  タイトル類似度だけで誤採択され、被引用9,000件の照合を開始した。S2候補に著者がないtitle-search経路は、
  完全タイトルまたは副題を除いた主タイトルとの正規化一致を必須とする。著者欠落の正しい完全一致は維持する。
- 実行中のCitation jobを停止し、当該資料へ部分保存された誤引用7,144件を対象限定で削除して台帳を戻した。
  修正版のitem限定再実行は`not_found`で正常完了し、誤引用は0件となった。

## 2026-09-25 Mistral maintenance queue rollover

- ブラウザのMistral OCR jobは、2026-08の旧stateが`collected`かつ採用済みだと、新たな
  `awaiting_mistral_ocr_batch`を確認せず即時成功していた。採用済みstateではArtifact台帳をread-onlyで
  再確認し、待機候補があればbatch CLIの`--submit`へ渡して新しいstateを準備・送信する。待機0件なら
  外部APIを呼ばない。実データではPDF 2件（314頁・194頁）が候補・sourceとも解決可能。

## 2026-09-03 admin maintenance progress

- 階層要約の前段にある全資料の構造差分確認は、実測済みの直前回で約4分35秒かかったが、完了まで
  進捗を出さなかった。10件ごとの確認済み件数・変更数・失敗数をstderrへ逐次表示する。
- 30分ごとの読み取り専用`update_check`は実行記録を保持しつつ、ブラウザの実行履歴から除外する。
  更新状況欄の結果・確認日時・期限切れ判定は従来どおり表示する。
- Zotero追加直後のEPUB 1件が、実在しない`file://`先をLocal APIから返して確認全体を失敗させた。
  実在するfile redirectは復号して利用し、欠落はfail-openで原本欠落件数へ残す。通常の索引処理は
  引き続きfail-closed。更新状況には最新確認失敗の警告と、階層要約・要約索引監査を追加した。
  4種類の更新状況はデスクトップで4列1行、640px以下では1列表示とする。クイック実行が即失敗した
  場合も同じ原本欠落が第1stepをfail-closedで止めたもので、原本の再配置・再添付が必要。
- ブラウザの固定job catalogへ有料`mistral_ocr`を追加。`MISTRAL`確認後、保存済みphaseに応じて
  submit/status/collectを1段階進め、回収済みなら品質gate合格分だけをV3へ採用して構造を更新する。
  開始確認語は有料の`summary_batch`と`mistral_ocr`だけに限定し、その他は直接開始する。

## 2026-09-02 documentation audience split

- `README.md`と`docs/`を人間の利用者向けに限定し、実装構造、変更理由、回帰防止、コード変更後の
  再処理注意を`developer/`へ移した。開発者・コーディングエージェントの入口は
  `developer/README.md`、正本規則は引き続き`CLAUDE.md`とする。
- 旧`citation_graph/README.md`と開発文書5件を`developer/`へ移動し、利用者向け文書に残っていた
  退役機能の経緯、内部データプレーン契約、将来の移行計画を除いた。

## 2026-09-02 authenticated remote Citation Graph

- 従来の`citation_graph/server.py`と`http://localhost:7234`を変更せず、Google OAuth付きの独立proxy
  `citation_graph/remote_proxy.py`を追加。既存Remote MCPのGoogle資格情報・許可メールを共用し、OAuth
  state、PKCE、Google ID token検証、HTTPS限定署名sessionを通過したブラウザだけをloopbackのGraphへ
  転送する。FunnelはRemote MCPと分離したHTTPS 8443を想定する。
- Citation Graph本体とOAuth proxyを独立LaunchAgentとして管理する
  `scripts/manage_citation_graph_service.py`を追加。ログイン時起動、異常終了時10秒throttle付き再起動、
  install/status/restart/uninstall、local/public health確認を行い、秘密値はplistへ複製しない。
- OAuth公開URLの`/admin/`にメンテナンス画面を追加。manifest/gate/indexing lock/artifact、固定job、
  step/log/履歴を表示し、差分取込＋構造、DB監査、構造、10件要約、Citation更新だけを独立runnerで
  実行する。単一job、same-origin POST、確認語、actor記録、PID/job/token検証停止、再起動耐性を持ち、
  任意command、破壊的rebuild、force、全件有料要約は対象外。管理画面はCitation Graph本体と同じ
  パレット、タイポグラフィ、カード密度、状態色を使い、CSS assetは即時反映のため保存しない。
  日常運用向けのクイック実行は、索引差分、構造・目次差分、監査、Citation差分を順に実行し、
  失敗時は後続工程へ進まない。階層要約は含めない。読み取り専用の更新状況確認jobは、Zoteroと
  manifest、構造dry-run、Citation台帳を照合し、未更新件数・対象キー・確認日時を画面へ保存表示する。
  専用LaunchAgentがログイン時と30分ごとにこの確認jobを開始し、他job実行中は次回へ延期するため、
  画面アクセスは重い照合を待たず保存済み結果を即時表示する。確認語のないread-only jobは
  確認ダイアログを経ず直接開始し、確認語の必要な書込みjobだけ入力フォームを出す。launchdが
  bootout直後のbootstrapへ一時EIOを返す場合は短時間リトライする。再登録失敗時は旧plistとloaded状態を
  Citation Graph 3 Agent・Remote MCPとも復元する。
- 更新状況の索引判定は添付に加えてZotero note versionもmanifestと照合し、Citation判定は`--all`と
  同じ全書誌itemを対象にDOI・ISBN差分も数える。原本欠落・構造失敗・Citation errorは緑の0件に
  埋めず赤い要確認とする。45分超は期限切れ、書込みjob中は再確認待ちとし、正常終了後に確認jobを自動起動。
- browser adminのDB監査はlaunchdの限定PATHで`uv`を再探索せず、runnerを起動済みの
  `sys.executable`で3監査を実行する。実DB再監査はZotero 622/622、原本欠落・orphan・dangling・
  unretrievable 0、cutover 597/597で合格し、`server_database_gate.json`を再作成した。
- 直近reviewで、DB監査が開始時に旧gateを無効化するのに確認なしだった点、Zoteroで削除した
  DOI/ISBNを差分・台帳同期が扱わなかった点、自動再確認が全`RuntimeError`を競合扱いした点を修正。
  DB監査は`AUDIT`必須、識別子は空値も正確に同期し、再確認は専用競合例外だけを延期する。
- GitHub Actionsのcoverage予算に新規adminモジュール3件を登録し、`Check nothing lost its tests`
  の新規未到達文エラーを解消。`checkout@v5` / `setup-uv@v8.3.2`へ更新し、強制Node.js 24実行時の
  Node.js 20警告も解消する。v9はGitHub側のタグ解決に失敗したため採用しない。

## 2026-08-31 authenticated remote MCP

- 既存のstdio MCPを変更せず、同じtool registryをFastMCP Streamable HTTPで公開する独立入口
  `src/rag_mcp_http_server.py`を追加。Google OAuth資格情報・公開HTTPS origin・確認済みGoogle
  email allowlistを必須とし、HTTP待受はloopback限定。Tailscale Funnelから`/mcp`を公開する。
- 2026-09-01にmacOS LaunchAgent管理を追加。ログイン時起動、異常終了時の再起動、10秒throttle、
  `data/`ログ、install/status/restart/uninstallとlocal/public OAuth health確認を提供する。

## 2026-09-01 storage cleanup

- 検索不能な`pre-m5-incomplete`診断世代と、後発rollbackに包含済みの2026-08-14対象backup
  4件を削除。次に現行active V3を`data/backups/current-v3-full-20260901`へ一式backupし、作成時と
  独立verifyの両方でChroma/lexical 522,761件、manifest 622添付、SQLite quick check `ok`を確認。
  置換された旧`pre-h2s-ai-toc-20260815-83ef402`も削除した。OCR cache、BGE-M3、Granite環境は
  再生成・再取得コストがあるため保持した。`uv cache prune`は稼働中MCPがcacheを使用中だったため、
  安全ロックを尊重して見送った。

## 2026-08-15 generated PDF bookmark rejection

- 全頁に`00000___<32 hex>`形式の内部ID bookmarkを持つPDF 1件が、167偽chapter・
  156偽章要約を生んでいた。PDF outlineの過半数がこの形またはファイル名なら、
  chapter/path/treeの全経路で使用不能とする共通ゲートを追加。outlineあり原本131 PDFの
  read-only走査で影響は該当1件だけ。対象dry-runは584 chunk完全割当の`flat_fallback`
  23 node / 21 leaf、偽見出し0。
- 明示承認後、15,035,319,759 byteの検証済みbackupを作り、対象1添付だけAI TOCを再実行。
  9 anchor（coverage 1.0）、584 chunk、`recovered` 22 node / 11 leafへactive DBを更新した。
  正本9章を確認し、要約対象8章は全て`llm/accepted/searchable`。511字の`About this Title`は
  最小長規則でskip。DB監査593 item失敗0、原本欠落0、要約索引7,932 ID完全一致で合格。

## 2026-08-15 visible Herdr command panes

- `CLAUDE.md`に、CodexがHerdr管理下にいるときは長時間のtest/coverage/build/server/monitorを
  同一workspaceの専用タブで実行する規則を追加した。ユーザのフォーカスを移さず、同一cwd、
  タスク内のタブ再利用、終了時のタブID/ペーンID/結果報告を必須とし、Herdr外では通常実行へfallbackする。

## 2026-08-15 GitHub SSH preference

- GitHubのfetch/pushはSSH remoteを優先する。HTTPSで認証・権限エラーの場合はSSH認証を確認し、
  `git@github.com:<owner>/<repo>.git`へ切り替える。Claudeのrepo指示とCodex global指示へ反映済み。

## 2026-08-27 documentation refresh

- README・日常運用・Citation NetworkのSetup/Maintenance説明を現行実装へ同期。項目6は退役済みで、
  relation reportは`review_relation_reports.py`の明示実行が必要。M5 readinessは完了前の判断記録で
  再実行手順ではないと明記し、CLAUDEの監査済み現況値も更新した。

## 2026-08-27 limited summary embedding scope

- Maintenanceの10件差分要約が全件`skipped_current`でも、`--all`扱いで約7,932要約を再埋め込みする
  不具合を修正。limited runは変更itemだけ、全件currentなら埋め込み0件。limitなしの明示的な
  `--all`だけ全索引reconciliationを行う。limitはcurrent除外後に適用し、後方のstale itemへ到達する。
  要約候補の走査・選択件数と、要約索引の埋め込み件数進捗も追加した。

## 2026-08-15 Citation Graph title repair

- Citation GraphのChroma metadata取得は、同一itemの章名も混在する`title`を文字列`MAX`で
  選ばず、反復数最大の書誌値を使う。Citation対象576 itemのread-only監査で旧規則の
  誤選択399 itemを確認し、報告2 itemはどちらも正しい書名になることを実DBで照合した。
  active DB、構造、要約の更新は不要。文書構造のattachment rootは複数添付識別用の
  ファイル名なので、書名と混同して構造fingerprintを書き換えない。

## Objective

監査済みactive V3を検索品質・クラスタ・引用同定・構造復元の正本として維持し、個別に
報告された表示・抽出・構造品質の問題を対象限定で診断・修復する。

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
3. `developer/post-refactor-followups.md`で、実データ判断が必要な残件を確認する。
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
  `developer/embedding-rebuild-readiness.md`がgo/no-go、実行入口、rollbackの記録である。
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
- **M5前の履歴:** 修正前のclean rebuildを33/618で停止し、差分継続も353/619で停止した。当時の
  active V3はmanifest 352件、inflight 1件、`hnsw_validated=false`でMCP拒否だった。この世代は
  再利用せず、その後のM5 clean rebuildで置換済み。
- M5直前に旧complete-generation snapshotの欠落を検出した。当時のactiveは検索不可だったため、
  未完成世代を診断用に一式退避した。Chroma/FTS 272,007行、manifest 352件、SQLite quick checkは
  いずれも`ok`だったが正常世代のrollbackではなく、2026-09-01の容量整理で削除した。
- user承認後のM5初回実行は8/619で停止した。監視ログのOCR layer audit `measured`を契機に、
  保存済み`.env`でOCR layer auditがreadinessと逆に1だったことを検出し、auditだけ0へ戻した。
  この部分世代は再利用せずclean rebuildを再度最初から行い、完了した。query expansion、LLM summaries/
  reference extraction、Mistral OCRはuser確認により1でよく、保存設定は1。再実行childは開始時環境を
  固定しているため前3機能は0のまま完走させる。Mistral queue deferred 1件、Batch送信・採用なし。
- review残件のP1は`run_db_audit.py`のreport出力先保護。現在の設定は全て`data/quality/`で安全だが、
  任意設定でactive DBを指せるためfollow-upへ記録した。埋め込み開始条件には含めず、今回未修正。
  残るP3はS2 author-less識別とflat PDF 3形状で、どちらも実データを読んでから方針を決める。

## Existing V3 database assessment

active V3はM5 clean rebuildと後続の対象修復を完了した監査済み世代であり、検索・品質評価の
正本として使用できる。

- active planeは`zotero_paragraphs_v3` / `manifest_v3.json` / `lexical_v3.sqlite3`。
- 2026-09-01 backup時点でChroma/lexical 522,761件、manifest 622添付。
- 直近の全DB監査は593 item中失敗0、原本欠落頁・orphan・dangling・unretrievableはいずれも0。
- H2SDTWFQ修復後の要約索引監査は期待/実在7,932 ID一致。現在の復旧点は検証済みの
  `data/backups/current-v3-full-20260901`。

## Next work, in order

`TASKS.md`を進捗の正本とする。M0〜M5、監査、対象構造修復は完了している。次の作業は
監査済みactive V3を維持しながら、個別に報告された表示・構造品質の問題を対象限定で診断する。

1. **M1（完了）**: 通常attachment dispatchとpending候補の組立を決定的fixtureで固定した。
2. **M2（完了）**: 最終出力確定だけを純粋関数へ分け、現行抽出・chunk境界を今回の世代として
   凍結した。同じ世代内でchunk本文・IDを変える未解決変更はない。
3. **M3（完了）**: 暫定bulk設定は`UPSERT_BATCH_SIZE=128`、
   `CHROMA_HNSW_SYNC_THRESHOLD=10000`。300合成chunkで`4.59 chunks/s`・peak RSS
   `900.53 MB`を確認し、中断再開・補償・reopen・実queryも通した。
4. **M4（完了）**: 対象commit、dry-run inventory、現V3 backup、容量、lock、埋め込み設定、
   有料機能無効、rollback点をgo/no-go表で確定し、`REBUILD`入力前で停止した。
5. **M4.1（完了）**: 対象限定のAI TOC負キャッシュrefreshを実装・検証し、runtimeを
   `developer/embedding-rebuild-readiness.md`へpinしてReady to Embedへ戻した。
6. **M4.2（完了）**: pilot/backupの稼働データ防壁とrefresh解決後scopeを修正・検証し、
   runtimeをreadinessへpinしてReady to Embedへ戻した。
7. **M4.3（完了）**: pilotの開始data planeを新規/空に限定し、既存planeの測定誤報を拒否。
   実装runtimeをreadinessへpinしてReady to Embedへ戻した。
8. **M4.4（完了）**: 不完全clean generationをHNSW readyにせず、MCPもmanifest gateで拒否。
   後段structure writerを共有lockで排他し、埋め込み開始に直接必要なreview指摘を解消した。
9. **M4.5（完了）**: Setupの全DB lifecycle childで保存済みconfigを親processの古い値より優先し、
   stale Granite/queue設定を使った事故の回帰テストを追加した。
10. **M5（完了）**: clean rebuild、Zotero・原本・DBの3監査、HNSW公開を完了した。

## Decision boundaries

- 合成fixture、純粋関数化、挙動不変の責務分離は自律的に進めてよい。
- 抽出heuristic、OCR route、chunk境界、S2 author-less条件、flat PDF昇格条件、固定8クラスタの
  変更は生成物・実資料を読み、精度trade-offが見えた時点でユーザーへ判断材料を示す。
- 現DBへの書込みは対象を明示し、dry-runとbackupを先行する。fingerprintは手修正しない。
  全件rebuildと有料LLM/OCRは、その都度ユーザーの明示承認後だけ実行する。実資料baselineは
  hosted featureを強制offにして検証する。
- 実データを書き換えない診断では、`.env`を`load_dotenv_native(ROOT)`で読んでからactive planeを
  解決する。識別子、書名、絶対パスを追跡ファイルへ保存しない。

## Required handoff discipline

変更ごとに`TASKS.md`と品質予算を同じコミットへ含める。完了または方針変更時にはこのファイルも
更新し、新しいセッションが古いDB診断や開始点を前提にしないようにする。

# 2026-09-29 CI coverage予算のLinux差分

- GitHub Actions run `36441132574` はテスト成功後のcoverage ratchetで失敗した。
- 新規の未到達防御分岐2件（PyMuPDF詳細辞書の例外fallback、re-OCR構造fingerprint不一致）
  は直接テストを追加した。
- macOSでは追加後に旧上限へ戻る一方、Linuxでは既存の任意依存・環境分岐により
  `pdf_extract.py` がさらに4文、`text_utils.py` が1文未到達になるため、その差だけ上限を補正した。

# 2026-09-29 承認付きZotero収録

- `propose_zotero_import`は候補と重複検査結果を24時間固定し、保存しない承認句を返す。外部書込みなし。
- `approve_zotero_import`は承認句の完全一致、単回claim、直前重複再検査、Zotero Local APIの
  native runtime authorizationを経てmetadataまたはPDFを追加する。
- PDFはsource別HTTPS allowlist、250MB、`%PDF-`を検証し、Zoteroの3段階upload後に
  `index_from_zotero.py --item ... --force-reparse`で対象だけ索引化する。
- proposal状態は既定で`data/zotero_import_proposals.json`（gitignore対象のdata配下）へ保存する。
- 新機能初回CIで`update_citations.py`の既存11文が実行順依存で未到達になったため、skip、
  DOI解決成功、mapper例外の3経路を直接テストへ固定した。coverage上限は緩めていない。
- `import_mode=epub`を追加。allowlist内の`.epub`直接URLだけを受け入れ、ZIP内`mimetype`を
  検証して`application/epub+zip`添付として保存し、PDF同様に対象限定で索引化する。
# 2026-09-29 Maintenance進捗バー

- 管理画面の実行中ジョブに進捗バーを追加。ログの`PROGRESS x/y`または`進捗 x/y`を優先し、
  取得不能時はstep index/totalと開始後経過時間から各step内10〜90%を概算する。
- 概算は「経過時間からの目安」とストライプで区別し、完了状態以外は99%を超えない。
