# 開発・保守

[READMEへ戻る](../README.md)

## テスト

```bash
uv run pytest -q
uv run python -m compileall -q src scripts tests
```

### 構造復元ベースライン

フラットPDFの見出しから木を復元する規則は、どれも実際の蔵書に合わせて調整した
経験則で、良し悪しは実物の本にどう効くかでしか決まらない。この判断をこれまで
「復元できた件数」でやってきたが、その数字は誤った木も正しい木も1件と数える。
実際、**目次領域が本文を171ページ分飲み込んだ版のほうが、その前の版より件数が
良かった**。口述史料のインタビュー質問3件が小見出しに昇格した変更も、件数上は
無害に見えた。

そこで84冊の復元結果を `tests/baselines/structure_recovery.json` に記録し、
`tests/test_structure_recovery_corpus.py` が毎回引き直して照合する。

```bash
uv run python scripts/build_structure_recovery_baseline.py          # 差分を見る
uv run python scripts/build_structure_recovery_baseline.py --write  # 採択する
```

差分は**読んでから**採択する。件数が増える変更でも、ある本が自分の巻末注に章を
明け渡した結果ということがある。

この網が守らない範囲は3つ。**コーパスに無い形式**（部ごとに章番号がリセットされ
る本は84冊に含まれない）は `tests/test_source_structure_refresh.py` の手書き
テストが受け持ち、そちらがCIで強制される部分になる。**記録時点で既に誤っていた
木**はスナップショットでは検出できず、ベースライン自体を検査するのは
`test_no_recovered_tree_repeats_a_boundary`（1つの境界を二度主張しない）だけ。
**ライブラリが無い環境**では照合をスキップする（入力行の凍結は31,068行・2.5MBで
リポジトリ最大ファイルの5倍になり、縮めるには普通の本文行を捨てるしかない。直近
の不具合はまさにそこに潜んでいた）。

## 引用関係レポート

```bash
uv run python scripts/review_relation_reports.py
uv run python scripts/review_relation_reports.py --list
```

Disableは元のS2データを削除せず、`relation_reports` の安定キーで検索・グラフから除外します。

## 主なディレクトリ

| 場所 | 内容 |
|---|---|
| `src/` | MCPサーバー、索引、検索、引用、要約 |
| `scripts/` | 管理・評価・修復CLI |
| `tests/` | 回帰テスト |
| `docs/` | 利用者・開発者向け文書 |
| `data/` | DB、索引、モデル、ログ、評価結果。Git管理外 |

内部構造は [アーキテクチャ](architecture.md) を参照してください。

Citation Graphへ階層要約・文書構造を統合する画面仕様は、[階層要約・構造ビュー設計](citation-insights-ui-design.md) を参照してください。

## 参考文献の審査キュー

```bash
uv run python scripts/review_references.py list --status pending --limit 20
uv run python -m src.reference_quality_report --status pending
```

参考文献の抽出結果は、原文根拠を再検証してから適用します。修復・分類用CLIの詳細は各コマンドの `--help` を参照してください。

英語学術論文のGROBID enrichmentは本文埋め込みとは独立して実行します。既定はdry-runです。

```bash
# 対象判定のみ
.venv/bin/python scripts/run_grobid_enrichment.py --limit 5

# ローカルGROBID service起動後、reference review queueへ保存
.venv/bin/python scripts/run_grobid_enrichment.py --limit 5 --apply
```

対象はZotero種別が`journalArticle`、`conferencePaper`、`preprint`の英語PDFだけです。
同じPDF SHA-256とGROBID versionで成功済みなら再処理せず、GROBID障害は埋め込み・manifest・Chromaへ影響しません。

実行前に`.env`で`GROBID_ENRICHMENT_ENABLE=1`を設定し、ローカルGROBID serviceを起動してください。
flagが`0`のとき、またはflagが`1`なのにserviceが応答しないときは、workerが処理を始める前にexit code 2で停止します
（他のoptional機能と同じく`src/feature_gates.py`が判定します）。

## 検索品質評価

```bash
uv run python scripts/eval_retrieval.py data/quality/gold_qa.jsonl --k 10
# V3のLLMノード要約を生成・埋め込み後に、階層検索経路も含めて比較する
uv run python scripts/eval_retrieval.py data/quality/gold_qa.jsonl --k 10 --include-hierarchical-v2
```

## ローカルデータとバックアップ

重要な運用データ（現行はV3データプレーン）:

- `data/relations.db`
- `data/chroma/`（active collection `zotero_paragraphs_v3` と `zotero_paragraphs_v3__sum_node`）
- `data/lexical_v3.sqlite3`
- `data/manifest_v3.json`

旧collection・`manifest.json`・`lexical.sqlite3`は本番経路では使用しません。復旧が必要な場合は、
現在のV3データ面をバックアップから復元するか、原本からServer workflowのフェーズ1を再実行します。

大規模修復前だけ `data/backups/` にV3スナップショットを作成します。修復後にDB整合性と回帰テストを確認し、古い中間バックアップを削除します。`data/`はGitやSoftware Updateでは同期されないため、必要なら利用者側の暗号化バックアップへ保存してください。

## ドキュメント方針

- `README.md`は短い入口として保つ
- 詳細は目的別の `docs/*.md` に置く
- 同じ設定説明を複数文書へコピーせず、リンクする
- CLIやファイル名を変更したら関連リンクを検索・更新する
