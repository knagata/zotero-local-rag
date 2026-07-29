# 日常の使い方

[READMEへ戻る](../README.md)

## まとめて更新する

macOSでは `Maintenance-Widget.command` をダブルクリックします。

```bash
bash Maintenance-Widget.command
```

`MAINTENANCE_AUTO_APPROVE=1`でも、有料APIを使う階層要約とMistral OCR Batchは自動許可されません。

1. ライブラリ差分更新（＋文書構造V3の差分更新）
2. 要約の差分更新（DeepSeek AI要約。DB監査合格後のみ、既定off）
3. Citation Network更新
4. 報告された品質・引用関係の確認
5. Mistral OCR Batchの送信、または完了済み結果の回収・品質確認・採用（任意）

階層要約とMistral Batchは、該当質問で`y`を入力した場合だけ実行します。前段が失敗した場合は、
古いデータで後続処理をしないよう自動停止します。

## サーバーでDBをゼロから構築する

`Server-Database-Workflow.command`を使い、次のフェーズを別々に起動します。

1. DBをゼロから構築（階層要約なし）
2. DB完全監査と、現在のDB世代に結び付いた合格gateの作成
3. 階層AI要約と要約索引の構築（明示的な課金確認あり）
4. 要約本文・fingerprint・処理状態・要約索引IDの完全監査

フェーズ3はフェーズ2の合格gateがなければ開始できません。監査後にmanifest、チャンクID、
FTS IDまたは文書構造が変わった場合も、古いgateを拒否して停止します。

ここで分離するのは「階層要約」です。DB構築時のAI目次推定を有効にしている場合は、
構造復元のためDeepSeekを呼ぶことがあります。DB構築を完全に外部APIなしで試す場合は
`PDF_AI_TOC_FAST_PATH_ENABLE=0`にしますが、その結果はAI目次なしの別仕様になるため、
最終DBでは通常設定に戻して再構築・再監査してください。

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
