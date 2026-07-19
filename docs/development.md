# 開発・保守

[READMEへ戻る](../README.md)

## テスト

```bash
uv run python -m unittest discover -s tests -q
uv run python -m compileall -q src scripts tests
```

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

Citation Graphへ階層要約・構造化事例を統合する画面仕様は、[階層要約・事例ビュー設計](citation-insights-ui-design.md) を参照してください。

## 参考文献の審査キュー

```bash
uv run python scripts/review_references.py list --status pending --limit 20
uv run python -m src.reference_quality_report --status pending
```

参考文献の抽出結果は、原文根拠を再検証してから適用します。修復・分類用CLIの詳細は各コマンドの `--help` を参照してください。

## 検索品質評価

```bash
uv run python scripts/eval_retrieval.py data/quality/gold_qa.jsonl --k 10
# v2のLLMノード要約を生成・埋め込み後に、旧経路との比較も出力する
uv run python scripts/eval_retrieval.py data/quality/gold_qa.jsonl --k 10 --include-hierarchical-v2
```

## ローカルデータとバックアップ

重要な運用データ:

- `data/relations.db`
- `data/chroma/`
- `data/lexical.sqlite3`
- `data/manifest.json`

大規模修復前だけ `data/backups/` にスナップショットを作成します。修復後にDB整合性と回帰テストを確認し、古い中間バックアップを削除します。`data/`はGitやSoftware Updateでは同期されないため、必要なら利用者側の暗号化バックアップへ保存してください。

## ドキュメント方針

- `README.md`は短い入口として保つ
- 詳細は目的別の `docs/*.md` に置く
- 同じ設定説明を複数文書へコピーせず、リンクする
- CLIやファイル名を変更したら関連リンクを検索・更新する
