# 日常の使い方

[READMEへ戻る](../README.md)

## まとめて更新する

macOSでは `Maintenance-Widget.command` をダブルクリックします。

```bash
bash Maintenance-Widget.command
```

`MAINTENANCE_AUTO_APPROVE=1` が既定のため、五項目すべて（クラウド送信を伴うMistral OCR Batchを含む）が
自動許可されます。確認式に戻すには `MAINTENANCE_AUTO_APPROVE=0` を指定します。

1. ライブラリ差分更新（＋文書構造V3の差分更新）
2. 要約の差分更新（DeepSeek AI要約。未承認時は限定pilotバッチ）
3. Citation Network更新
4. 報告された品質・引用関係の確認
5. Mistral OCR Batchの送信、または完了済み結果の回収・品質確認・採用（任意）

通常はEnterを6回押すだけで開始できます。実行しない項目だけ `n` を入力します。Mistral Batchを実行する場合だけ5番目で`y`を入力してください。初回はBatchを送信して完了後の再起動を案内します。次回、完了済みなら結果を回収し、品質gate合格分だけをV3へ採用します。未確認レポートがある場合は、その後に個別にDisable、Keep、保留を選びます。Enterは安全側の保留です。前段が失敗した場合は、古いデータで後続処理をしないよう自動停止します。実行後に未解決の処理状態サマリ（Mistral queue候補・失敗・truncated等）が表示されます。

## 個別にCLI実行する

```bash
# ライブラリ差分更新
uv run src/index_from_zotero.py --progress

# 文書構造V3の差分更新
uv run python scripts/rebuild_document_structure.py --all

# 要約（LLM）。全件backfillは承認後、通常は --limit で限定実行
uv run python scripts/build_structure_summaries.py --all --mode llm --limit 10 --embed

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
