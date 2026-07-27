# Citation Graph 階層要約・構造ビュー設計

状態: `64_ingestion_redesign_audit_and_plan.md` に基づく現行仕様

## 目的

Citation Graphで資料を選択したまま、Zotero概要、文書構造、階層要約、処理状態を確認する。

## タブ

- `概要`: Zotero概要と資料全体の要約
- `構造`: V3の章・節・segmentツリーと各ノード要約
- `節要約`: 文書順の要約と根拠チャンク
- `処理状態`: extraction、structure、summary、references、embeddingsの状態

構造化事例カードと専用APIは提供しない。本文中の具体例は事例DBを介さず、
`rag_search(search_mode="case")`で原文チャンクを直接検索する。

## 表示規則

- `zone`、summary/retrieval/citation policyをノード単位で表示する。
- 要約の`input_scope_json`から採用・除外した子ノードと理由を確認できるようにする。
- `body`の子孫だけを親要約へ含め、脚注・巻末注・文献・奥付等の除外状態を明示する。
- 原文根拠はチャンクID、ページ・locator、抽出エンジンprovenanceとともに表示する。
- 要約品質報告はitem/node summaryだけを対象とし、即時削除せず品質triageへ送る。

旧事例ビューの画面仕様は
`dev-notes/archive/citation-insights-ui-design-legacy.md`へ退避している。
