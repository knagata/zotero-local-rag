# Zotero Local RAG

Zoteroに保存したPDF・HTML・EPUBを、Claude Desktopなどから検索・参照できるローカルRAGです。

- 本文を段落単位で検索
- 関連資料や前後の文脈を取得
- ローカル要約を作成
- 必要に応じて引用ネットワークやLLM機能を追加

基本機能はローカルで動作します。外部APIやLLMは、使いたい機能に合わせて後から設定できます。

## まず使ってみる

必要なもの:

- Zotero Desktop
- Python 3.10
- [uv](https://docs.astral.sh/uv/)

macOSでは [Setup.command](Setup.command) をダブルクリックします。ターミナルから実行する場合:

```bash
uv run scripts/setup_wizard.py
```

ウィザードで次の項目を設定します。

1. Zoteroデータフォルダ
2. 埋め込みモデル（通常は `fast` で十分です）
3. 使いたい機能の段階
4. Claude Desktopへの接続
5. 初回インデックス作成

完了後、Claude Desktopを再起動し、次のように依頼できます。

```text
Zoteroから「修復的司法」に関係する資料を探して
```

詳しい手順は [初回セットアップ](docs/getting-started.md) を参照してください。

## 三段階の機能

| 段階 | 追加設定 | できること |
|---|---|---|
| Core | 埋め込みモデル | Zotero検索、本文検索、文脈取得、目次、ローカル要約 |
| Citation Network | インターネット接続 + S2 APIキー | 被引用・参照文献、引用グラフ、未所蔵文献候補 |
| LLM-assisted | APIキーまたはCLIログイン | 高品質要約、クエリ拡張、参考文献抽出・整理 |

Citation NetworkにはSemantic ScholarのAPIキーが必要です。

詳しい違いは [機能と必要な設定](docs/features.md) を参照してください。

## 普段の更新

macOSでは [Maintenance-Widget.command](Maintenance-Widget.command) をダブルクリックします。

次の三処理が既定で選択されているため、通常はEnterを4回押すだけです。

1. Zoteroライブラリの差分更新
2. ローカル抽出型要約の更新
3. Citation Networkの更新

不要な項目だけ `n` を入力して除外できます。ログはTerminalへ表示されます。

詳しくは [日常の使い方](docs/daily-use.md) を参照してください。

## プライバシー

Core機能はローカルで処理できます。LLMへ本文を送る機能は、送信ポリシーが設定されていない場合は安全のため停止します。

推奨設定は、Zoteroの `private` や `no-cloud` タグを送信対象から除外する方法です。詳しくは [LLMとプライバシー](docs/llm-and-privacy.md) を参照してください。

## ドキュメント

| 目的 | ガイド |
|---|---|
| 初めて設定する | [初回セットアップ](docs/getting-started.md) |
| 普段の更新・検索 | [日常の使い方](docs/daily-use.md) |
| 何ができるか確認する | [機能と必要な設定](docs/features.md) |
| Citation Networkを使う | [Citation Network](docs/citation-network.md) |
| LLMと送信制御を設定する | [LLMとプライバシー](docs/llm-and-privacy.md) |
| `.env`を手動設定する | [環境設定](docs/configuration.md) |
| エラーを解決する | [トラブルシューティング](docs/troubleshooting.md) |
| Claudeでの検索方法を詳しく見る | [Claude利用ガイド](docs/claude-guide.md) |
| 内部構造を確認する | [アーキテクチャ](docs/architecture.md) |

## 開発者向け

```bash
uv run python -m unittest discover -s tests -q
```

詳細は [開発・保守](docs/development.md) を参照してください。

## ライセンス

[MIT License](LICENSE)
