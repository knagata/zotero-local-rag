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
