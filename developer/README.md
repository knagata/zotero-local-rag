# 開発者・コーディングエージェント向け資料

このディレクトリは、実装、変更時の注意、設計判断、検証手順、移行記録をまとめる場所です。
利用者向けのセットアップ・設定・日常操作はルートの`README.md`と`docs/`に置き、ここへは
利用者が通常の運用で読む必要のない情報だけを置きます。

## 作業開始時に読むもの

1. [`CLAUDE.md`](../CLAUDE.md) — コーディングエージェントが毎回守る作業規則
2. [`SPEC.md`](../SPEC.md) — 実装契約の正本
3. [`memory/projects/current-development.md`](../memory/projects/current-development.md) — 現在の引継ぎ
4. [`TASKS.md`](../TASKS.md) — 完了事項と変更履歴

## 資料

| 資料 | 内容 |
|---|---|
| [開発・保守](development.md) | テスト、品質予算、実データ検証、バックアップ、文書規則 |
| [アーキテクチャ](architecture.md) | データプレーン、抽出・検索・引用パイプライン、内部モジュール |
| [Citation Graph実装](citation-graph.md) | OAuth proxy、管理job、静的assetの実装上の注意 |
| [Citation Graph画面設計](citation-insights-ui-design.md) | 階層要約・構造ビューの設計契約 |
| [再構築readiness記録](embedding-rebuild-readiness.md) | 完了済みM5移行の判断・監査記録 |
| [変更後フォローアップ](post-refactor-followups.md) | 未解決の実装課題と再評価条件 |

## 文書の境界

- `README.md`と`docs/`: 人間の利用者が、導入・設定・検索・更新・障害対応に必要な情報
- `developer/`: 実装方法、内部契約、変更理由、回帰防止、移行履歴
- `TASKS.md`と`memory/`: 開発履歴とセッション引継ぎ
- `dev-notes/`と`evaluations/`: Git追跡外の実データ調査

利用者向け文書へ内部関数名、過去実装との比較、コード変更後だけ必要な再処理、テスト予算の
説明を追加しないでください。利用者の操作に影響する場合は、内部理由ではなく「何を設定・実行すると
何が起きるか」を記載します。
