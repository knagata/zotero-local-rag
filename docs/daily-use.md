# 日常の使い方

[READMEへ戻る](../README.md)

## まとめて更新する

macOSでは `Maintenance-Widget.command` をダブルクリックします。

```bash
bash Maintenance-Widget.command
```

次の四項目がすべて既定で有効です。

1. ライブラリ差分更新
2. ローカル抽出型要約更新
3. Citation Network更新
4. Citation GraphまたはClaudeから報告された引用関係の確認

通常はEnterを5回押すだけで開始できます。実行しない項目だけ `n` を入力します。未確認レポートがある場合は、その後に個別にDisable、Keep、保留を選びます。Enterは安全側の保留です。前段が失敗した場合は、古いデータで後続処理をしないよう自動停止します。

## 個別にCLI実行する

```bash
# ライブラリ差分更新
uv run src/index_from_zotero.py --progress

# ローカル抽出型要約
uv run python -m src.build_summaries

# Citation Network
uv run src/update_citations.py --all

# 報告された引用関係を確認
uv run python scripts/review_relation_reports.py
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

- Citationの再開やEPUB参照回収: [Citation Network](citation-network.md)
- LLM要約・参考文献抽出: [LLMとプライバシー](llm-and-privacy.md)
- OCRやエラー対応: [トラブルシューティング](troubleshooting.md)
