# 手動校訂の安全性と検証

利用手順は [資料デバッグ](../docs/document-review.md)。ルート登録は
`citation_graph/debug_routes.py`、読取りは `citation_graph/debug_reader.py`。
下書き/履歴は `data/debug_review.sqlite3` へ追記し、canonical側を変更しない。
履歴復元も楽観ロック付きの新revisionを作る。原本hashが異なる履歴は拒否する。

反映は `scripts/apply_debug_review.py` を固定引数で管理jobから起動する。
`citation_graph/debug_apply.py` は共通writer lockと現監査を要求する。
`data/debug_rollback/current.json.gz` は対象itemの差分を保持する復旧journalで、
変更前本文・変更本文のベクトル・item限定のrelations・manifestを含む。
128MiBの未圧縮上限を超えたら書込み前に拒否する。完全DBのコピーは増やさない。

通常例外時はjournalから補償復旧し、追加されたChroma metadataキーもnull更新で除去する。
プロセス強制終了・復旧失敗時はjournalを保全し、inflight/未検証gateを解除せず原因を調べる。
元本文とベクトル、関連構造、全文索引、manifestの復旧を確認して全監査を再実行する。
journalだけを根拠に監査なしで検索を公開しない。

本文修正は変更IDのみローカル埋め込み、目次修正は既存builderで再構築する。
兄弟添付は保持し、ノートを構造更新に含めない。階層要約/要約索引はstaleにする。
原本差替え・force-reparse後の修正の自動再適用は行わない。

検証は `tests/test_debug_review.py`、`tests/test_debug_routes.py`、
`tests/test_debug_apply.py`。native Chromaの小さい隔離DBで、反映途中の失敗時に
本文・ベクトル・追加metadata・FTSが正確に戻ることを確認する。
本番DBへの校訂をテストとして実行せず、有料機能を起動しない。

## 画面と分類編集

画面全体はviewport固定とし、本文・目次・注/参考文献・差分・履歴をタブで切替える。
紙面は初期fit、拡大時のみviewer内部をスクロールする。左右端のoverlay矢印は
スクロール領域の外へ固定し、入力中の左右キーをページ移動に使用しない。
`tests/debug_review_dom.cjs` は3サイズのfit計算・拡大時の全領域到達・タブ/ページ移動・
直接目次追加・注分類・保存競合を検証する。実ブラウザ目視とは区別する。

`citation_graph/debug_annotations.py` がページ指定を既存チャンクへ解決する。
未抽出ページの目次下書きは許可するが、反映時は拒否する。既存anchorとpageの食い違いも拒否。
分類は既存ZONE_POLICIESのみ許可し、上限500チャンク、前後分類も履歴へ保存する。
構造builderが明示分類を変更する場合は書込み前に拒否し、保護規則を迂回しない。
参考文献/注に関係する本文・実効zoneの変更ではreferencesをstale、
item_citation_statusをpendingへ戻す。同台帳もrollback journalのitem限定snapshotに含める。
Citation Networkの再処理は自動起動せず、従来の明示操作・課金同意を維持する。

## PDF直接表示と先読み

`citation_graph/debug_pdf.py` は認証済みPDFのGET/HEAD/single Rangeを配信する。
初回inspectの原本stat・canonical SQLite/WAL・manifest statと文書versionを最大4件保持し、
ページ移動/Range配信ではZotero上の現在パスとstat世代を照合する。検査中の世代変更を拒否する。
配信は同一FDから読み、読取り前後の原本変更時は中断する。任意パス・multipart Rangeは受けない。
manifestの索引時mtime/size/content_signatureと現原本が不一致、または対象inflightなら表示を拒否する。
下書き保存/反映は引き続きfresh inspectを行い、表示キャッシュを更新許可の根拠にしない。

`citation_graph/static/debug-pdf.js` はPDF.jsの文書handleを再利用し、canvasとTextLayerを重ねる。
前後の3ページを上限に描画を保持し、canvasは各400万pixel以下、source切替えで破棄する。
PDFは必要部分をRange取得し、全PDF自動先読みは無効化。チャンクも前後3ページのPromiseを保持し、
表示前のcheap state検査を省略しない。遅い旧描画が新ページへ上書きされないようrequestを検証する。

同梱SDKは `citation_graph/static/pdfjs/assets.json` の版・配布元・integrity・各file hashを正本とする。
公式npm配布物から必要なlegacy build・worker・viewer CSS・CMaps・標準font・WASM/ICCとlicenseを保存する。
更新は `uv run python scripts/update_review_pdfjs.py`。版/integrityを変更する場合はSDK APIと
`tests/test_debug_pdf.py` の固定版検証を同時に更新する。CDNや外部ビューアへ資料を送らない。
worker/font/CSPと日本語PDFの表示確認を行い、第三者コードを独自修正せず追跡する。

検証は `tests/test_debug_pdf.py`（認証/Range/HEAD/配信中差替え/世代排他/SDK hash）、
`tests/pdf_review_dom.cjs`（text layer/handle再利用/fit/zoom/描画上限/先読み/遅い描画の拒否）と
`tests/debug_review_dom.cjs`（チャンク先読みと編集UI）。SDK本体の実PDF解析/描画と
実ブラウザの文字選択・位置確認は、mock操作テストと分けて記録する。

同梱配布物はbyte-for-byteで保持するため、`.gitattributes`で同梱ディレクトリだけ
upstream由来のwhitespace検査を除外する。自作コードの検査は維持し、SDKは全file hashで検証する。
