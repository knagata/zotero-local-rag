# Current development handoff

Updated: 2026-08-10

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
- 150行超の関数は22件から17件へ減少。最大は
  `index_from_zotero._index_library`（1,104行）、次が`_extract_pdf_chunks`（685行）。
- 既定suiteは1,521 passed / 4 skipped / 5 deselected。実資料取込baselineは5 passed。
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

1. 次は通常PyMuPDF結果からのPDF gate decisionをfakeで固定する。
   `structure_recovery`、local OCR試行、chunk有無、page minimum、Mistral queue可否から
   `disabled / defer / local_exhausted / docling_escalation / keep`を選ぶ部分は純粋関数化できる。
2. Mistral deferのartifact payload・manifest保存・既存chunk保持を、extractorと実DBを
   呼ばず統合テストする。経験則・threshold・公開出力は変えない。
3. その後にOCR audit outcome、born-digital/scan-derived/corrupted頁patchの順で、
   テストが届いた責務だけを`_extract_pdf_chunks`から抽出し、1 seamずつ対象テストと
   function-size ratchetを更新する。低coverage部分を機械的に移動して「検証済み」と扱わない。
4. `_extract_pdf_chunks`の挙動不変分割が終わってから`_index_library`を同じ方法で分割する。
5. 抽出・chunk境界を一度固定した時点でV3をバックアップし、`Setup.command`の`REBUILD`から
   clean rebuildする。保存済みfingerprintの手修正や孤立FTS 1行だけの先行修復はしない。
6. rebuild後に`uv run python scripts/run_db_audit.py`を実行し、Zotero・原本・DBの3監査を
   通す。その世代だけを実データ品質判断に使う。

## Decision boundaries

- 合成fixture、純粋関数化、挙動不変の責務分離は自律的に進めてよい。
- 抽出heuristic、OCR route、chunk境界、S2 author-less条件、flat PDF昇格条件、固定8クラスタの
  変更は生成物・実資料を読み、精度trade-offが見えた時点でユーザーへ判断材料を示す。
- 現DBへの増分取り込み、fingerprintの手修正、全件rebuild、有料LLM/OCRは実行しない。
  rebuildは上記の挙動不変リファクタ完了後に行う。
- 実データを書き換えない診断では、`.env`を`load_dotenv_native(ROOT)`で読んでからactive planeを
  解決する。識別子、書名、絶対パスを追跡ファイルへ保存しない。

## Required handoff discipline

変更ごとに`TASKS.md`と品質予算を同じコミットへ含める。完了または方針変更時にはこのファイルも
更新し、新しいセッションが古いDB診断や開始点を前提にしないようにする。
