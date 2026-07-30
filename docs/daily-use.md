# 日常の使い方

[READMEへ戻る](../README.md)

## まとめて更新する

macOSでは `Maintenance-Widget.command` をダブルクリックします。

```bash
bash Maintenance-Widget.command
```

`MAINTENANCE_AUTO_APPROVE=1`でも、有料APIを使う階層要約とMistral OCR Batchは自動許可されません。

1. ライブラリ差分更新（＋文書構造V3の差分更新）
2. DBの監査（Zotero本体・原本との突き合わせ。非破壊・読み取り専用。要約の実行に必要）
3. 要約の差分更新（DeepSeek AI要約。DB監査合格後のみ、既定off・少量バッチ）
4. 全件要約の一括生成（DeepSeek課金・`SUMMARIZE`入力確認・DB監査合格後のみ、既定off・重い処理）
5. Citation Network更新
6. 報告された品質・引用関係の確認
7. Mistral OCR Batchの送信、または完了済み結果の回収・品質確認・採用（任意）

項目2（監査）は無料・非破壊なので、直近の合格gateが無いか失効している場合は既定でyesに
なります（Enterだけで実行）。合格gateが最新なら既定でskipします。項目3・4・7は有料または
クラウド送信を伴うため既定offで、該当質問で`y`を入力した場合だけ実行します。前段が失敗した
場合は、古いデータで後続処理をしないよう自動停止します。

## DBをゼロから構築・再構築する

DBのゼロ構築・再構築は`Setup.command`から行います。設定保存後、初回構築の場合はもちろん、
既存DBがある場合も（設定を何も変えていなくても）毎回再構築するかどうかの案内が出ます。

- 真の初回（まだ何も構築されていない）は、破棄するデータが無いので確認なしで進められます。
- 既存DBがある場合は、設定を変更したかどうかにかかわらず毎回「再構築しますか？」と案内が出ます。
  プロファイルを変更した場合は既存の埋め込みが使えなくなるため`REBUILD`の入力が必須、
  それ以外の設定（PDF構造化やLLM機能など）だけ変えた場合は任意なのでEnterでスキップできます。

構築が終わると、続けてDB監査（非破壊）の実行も案内されます。ここでスキップした場合や、
以後の運用でDBが変わった場合の監査は`Maintenance-Widget.command`の項目2から実行できます。
階層AI要約（項目3・4）は監査合格後にのみ実行できます。

DB構築時のAI目次推定を有効にしている場合は、構造復元のためDeepSeekを呼ぶことがあります。
DB構築を完全に外部APIなしで試す場合は`PDF_AI_TOC_FAST_PATH_ENABLE=0`にしますが、その結果は
AI目次なしの別仕様になるため、最終DBでは通常設定に戻して再構築・再監査してください。

## 個別にCLI実行する

```bash
# ライブラリ差分更新
uv run src/index_from_zotero.py --progress

# 文書構造V3の差分更新
uv run python scripts/rebuild_document_structure.py --all

# 要約（LLM）。全件backfillは承認後、通常は --limit で限定実行
uv run python scripts/build_structure_summaries.py --all --mode llm --limit 10 --embed \
  --database-gate data/quality/server_database_gate.json

# Citation Network
uv run src/update_citations.py --all

# 報告された品質・引用関係を確認
uv run python scripts/triage_quality_reports.py
uv run python scripts/review_relation_reports.py
uv run python scripts/review_summary_quality_reports.py

# 未解決の処理状態を確認（read-only）
uv run python scripts/list_artifact_status.py --unresolved-only
```

## 基本的な検索の頼み方

### 特定の資料を探す

```text
Zoteroにあるマルセル・モースの資料を一覧にして
```

### テーマから関連資料を探す

```text
Zoteroから「贈与と互酬性」に関係する資料を探して
```

### 原文の根拠を探す

```text
この資料の中で、修復について論じている段落を前後の文脈付きで示して
```

### 複数資料を比較する

```text
関連資料を絞り込んでから、共通点と相違点を原文根拠付きで比較して
```

詳しいツール選択は [Claude利用ガイド](claude-guide.md) を参照してください。

## 引用グラフを開く

`Show-Citation-Graph.command`をダブルクリックします。先にCitation Network更新を済ませてください。画面の見方は [Show Citation Networkガイド](show-citation-network.md) を参照してください。

## アプリケーションを更新する

`Software-Update.command`をダブルクリックします。`.env`、`data/`、`.venv/`、`.claude/`は保持されます。更新後はClaude Desktopを再起動してください。

## 高度な抽出・再処理

- Citationの再開や参照回収: [Citation Network](citation-network.md)
- LLM要約・参考文献抽出: [LLMとプライバシー](llm-and-privacy.md)
- OCRやエラー対応: [トラブルシューティング](troubleshooting.md)

抽出コードを変更した後に既存資料をやり直す場合は、scopeを指定して再取り込みします（`pipeline_fingerprint` は抽出コードを含まないため自動では再取り込みされません）。

```bash
# 特定itemだけ再抽出
uv run src/index_from_zotero.py --force-reparse --item ABCDEFGH

# 親item内の特定添付だけを再抽出（queue worker向け。兄弟PDFは処理しない）
uv run src/index_from_zotero.py --force-reparse --item ABCDEFGH --attachment IJKLMNOP --source-type pdf

# 種別を絞って再抽出（--item / --limit / --source-type のいずれか必須）
uv run src/index_from_zotero.py --force-reparse --source-type epub --limit 20
```
