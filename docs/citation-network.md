# Citation Network 更新ガイド

Citation Network更新は、Zotero資料の被引用情報と、資料本文から抽出した参照文献を取得・照合してローカルDBへ保存します。参照文献は取り込み時に境界保持したV3チャンク（参考文献・脚注・巻末注のzone）から復元します。

## 通常の更新

macOSでは `Maintenance-Widget.command` をダブルクリックし、不要な処理を `n` で外して実行します。Citation Network更新だけを実行する場合は、一・二・四項目を `n`、三項目をEnterで選択します。

CLIから直接実行する場合:

```bash
# 未処理・エラー分を更新（通常はこれ）
uv run src/update_citations.py --all

# 特定アイテムだけ更新
uv run src/update_citations.py --item ITEMKEY

# 特定アイテムを再処理
uv run src/update_citations.py --item ITEMKEY --force

# 全アイテムを再処理（通常は不要）
uv run src/update_citations.py --all --force
```

`--all`は既に完了したアイテムをスキップします。途中終了やAPI障害で `error` になったアイテムは、次回の通常更新で再試行されます。

## 参照の追加回収

一冊に大量の参照がある場合、S2問い合わせ上限を超えた候補は `skipped` として保留されます。通常更新後も保留が残る場合だけ、上限を増やして再解決します。

```bash
uv run src/update_citations.py --resume-skipped

# 一冊あたりの問い合わせ上限を明示
uv run src/update_citations.py --resume-skipped --epub-budget 300
```

## 処理内容

1. Zotero Local APIから書誌と添付情報を取得
2. DOI/ISBNがない場合、OpenAlexでDOI候補を検索
3. Semantic Scholarから被引用論文を取得
4. V3チャンク（参考文献・脚注・巻末注のzone）から参照文献候補を復元
5. 参照文献をSemantic Scholar等で照合
6. 埋め込み類似度で引用・参照箇所をチャンクへ対応付け
7. `data/relations.db`へ保存

OpenAlexの一致はタイトル類似度で検証されます。`ZOTERO_USER_ID`と`ZOTERO_API_KEY`の両方がある場合だけ、解決したDOIをZotero Web APIへ書き戻します。

## S2データの訂正

Semantic Scholarの引用・参照関係は原則としてそのまま利用します。全件をLLMや原文照合で再検証することはしません。

誤った関係を見つけた場合は、Citation GraphのエッジまたはClaudeのMCPツールから報告できます。報告時点では関係は消えません。`Maintenance-Widget.command` の確認工程で人間が判断します。

- Disable: 誤関係としてグラフ・MCP検索・推薦から除外
- Keep: 正しい関係として維持
- Enter: 判断せず保留

DisableはDB行の削除ではなく、所蔵アイテムキーとS2 paper IDの安定した組み合わせで保存されます。Citation Networkを再取得しても復活しません。

## ステータス

### アイテム単位

| 値 | 意味 | 次回の通常更新 |
|---|---|---|
| `pending` | 未完了または処理中 | 再処理 |
| `s2_done` | S2完了、参照処理が未完了 | 参照抽出から再開 |
| `mapped` | 正常完了 | スキップ |
| `not_found` | S2に該当論文なし | スキップ |
| `error` | 429や通信障害等 | 再処理 |

### 参照単位

| 値 | 意味 | 対応 |
|---|---|---|
| `matched` | S2論文へ照合済み | 不要 |
| `not_found` | S2に候補なし | 不要 |
| `ambiguous` | 一意に決められない | 保留 |
| `skipped` | 問い合わせ上限超過 | `--resume-skipped` |
| `error` | API障害 | 通常更新または `--resume-skipped` |

## Semantic Scholarのレート制限

全プロセスが `data/s2_rate.lock` を共有し、リクエスト間隔を調整します。429は指数バックオフで最大3回再試行されます。連続失敗時は5分間のサーキットブレーカーが働きます。

Citation Networkには `S2_API_KEY` が必要です。

[Semantic Scholar APIページ](https://www.semanticscholar.org/product/api)の「Request an API Key」からキーを申請してください。キーなしの共有枠は実用上の制限が厳しいため、このプロジェクトでは必須として扱います。

`.env`で設定します。

```dotenv
S2_API_KEY=...
```

## よくある状況

### 429が出た

実行中は自動的に待機・再試行します。最終的に失敗しても `error` として残るため、時間を置いて通常更新を再実行してください。全件強制更新は不要です。

### `not_found`が多い

異常とは限りません。書籍、人文系、日本語文献はSemantic Scholarに収録されていないことがあります。確実でない候補へ自動接続しない設計です。

### 処理を途中で止めた

現在処理中のアイテムは未完了状態で残り、次回の通常更新で再処理されます。完了済みアイテムはスキップされます。

### Zoteroから資料を削除した

統合commandのライブラリ更新時に、削除済みアイテムの索引と引用データも整理されます。
