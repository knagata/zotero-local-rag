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
4. 全件要約の一括生成（DeepSeek課金・選択時の`SUMMARIZE`入力確認・DB監査合格後のみ、既定off・重い処理）
5. Citation Network更新
6. 品質報告のAI判定（退役済み。現在は実行されません）
7. Mistral OCR Batchの送信、または完了済み結果の回収・品質確認・採用（任意）

項目2（監査）は無料・非破壊なので、直近の合格gateが無いか失効している場合は既定でyesに
なります（Enterだけで実行）。合格gateが最新なら既定でskipします。項目3・4・7は有料または
クラウド送信を伴うため既定offで、該当質問で`y`を入力した場合だけ実行します。前段が失敗した
場合は、古いデータで後続処理をしないよう自動停止します。

項目4の`SUMMARIZE`確認は、`y`で選んだ直後、ほかの項目選択と最終開始確認より前に行います。
そのためライブラリ更新や埋め込み処理の完了後に追加入力を待つことはありません。要約実行中は
itemバッチの完了数に加え、APIの送信・正常応答・失敗・処理中件数と、応答ごとの根拠確認済み
文数をTerminalと保存ログへ逐次表示します。選択されたitemがすべて最新なら要約索引の再埋め込みも
skipします。変更がある場合は変更itemだけを更新し、埋め込み件数も逐次表示します。

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

# 文書構造だけを検証（本文チャンク・埋め込みは変更しない）
uv run python scripts/rebuild_document_structure.py --all --dry-run

# 文書構造だけを差分更新（EPUB原本の目次を再読込し、再埋め込みしない）
uv run python scripts/rebuild_document_structure.py --all

# ゾーン方針（ZONE_POLICIES）を変更したあとの反映。retrieval_policy は
# ノードとチャンクメタデータに焼き込まれているため、コード変更だけでは
# 既存資料に反映されない。--force で全件の構造とメタデータを再同期する
uv run python scripts/rebuild_document_structure.py --all --force

# 要約（LLM）。全件backfillは承認後、通常は --limit で限定実行
uv run python scripts/build_structure_summaries.py --all --mode llm --limit 10 --embed \
  --database-gate data/quality/server_database_gate.json

# Citation Network
uv run src/update_citations.py --all

# 報告内容をCLIから個別に確認（Maintenance項目6からは実行されません）
uv run python scripts/triage_quality_reports.py
uv run python scripts/review_relation_reports.py
uv run python scripts/review_summary_quality_reports.py

# 未解決の処理状態を確認（read-only）
uv run python scripts/list_artifact_status.py --unresolved-only

# flat_fallback資料を原因別に診断（read-only、原本・索引を変更しない）
uv run python scripts/diagnose_flat_structures.py --format markdown \
  --output data/flat_structure_diagnostics.md
```

## Zoteroタグで添付ファイルをRAGから除外する

- 添付ファイル自体に `rag:exclude`: その添付だけを除外します。
- 親アイテムに `rag:prefer-epub`: 正常に抽出・索引登録できたEPUB添付が同じ親にある場合だけ、PDF添付を除外します。

タグを付けた後に通常の「ライブラリ差分更新」を実行すると、既存の本文・埋め込み・語彙索引から
対象添付を削除し、親資料の階層構造も更新します。Zotero内の原本ファイルは削除しません。
タグを外すと次回の差分更新で再び取り込み対象になります。EPUBが見つからない、抽出に失敗する、
またはEPUB自体が `rag:exclude` の場合、`rag:prefer-epub` はPDFを残します。PDFの削除はEPUBの
索引登録がmanifestへ正常にコミットされた後に行われます。

埋め込み設定を変更済みで通常同期の安全ゲートが止まる端末でも、除外だけは再埋め込みなしで
反映できます（タグ解除後の再取り込みは通常同期が必要です）。

```bash
uv run src/index_from_zotero.py --sync-rag-exclusions-only --progress
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

# 同一PDFの「見出しなし」判定も再評価してAI目次をやり直す
uv run src/index_from_zotero.py --refresh-ai-toc --force-reparse \
  --item ABCDEFGH --attachment IJKLMNOP --source-type pdf

# 種別を絞って再抽出（--item / --limit / --source-type のいずれか必須）
uv run src/index_from_zotero.py --force-reparse --source-type epub --limit 20
```

`--refresh-ai-toc`は、同一mtime/sizeのPDFに保存された確定済みno-structure判定だけを
再利用せず、通常の再抽出・再埋め込み経路でAI目次を再実行します。対象外への課金と再処理を
避けるため、`--force-reparse`と`--item`または`--attachment`が必須です。通常は兄弟添付を
巻き込まない`--attachment`を使ってください。AI目次とPDF構造復元が有効である必要があり、
parser/OCR override、`--check-quality`、`--retry-failed`とは併用できません。
ページ数、native outline、本文品質など既存のAI目次適格条件は変更しません。
Zoteroから解決した対象が0件または非PDFを含む場合も、再埋め込み前に停止します。親itemに
EPUB/HTMLの兄弟添付がある場合は、上例のように`--attachment`または`--source-type pdf`で
PDFだけへ絞ってください。
