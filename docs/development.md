# 開発・保守

[READMEへ戻る](../README.md)

## テスト

```bash
uv run pytest -q
uv run python -m compileall -q src scripts tests
```

既定の実行は `slow` マーカーの付いたテストを除外します。現在それに当たるのは下の
取込ベースライン5件で、実測で全体205秒のうち140秒（68%）を占めていました。蔵書・
Zotero・埋め込みモデル・手元生成のベースラインを要するためCIでは元々スキップされて
おり、既定から外しても強制される検査層は変わりません。**抽出・取込・構造復元に
関わる変更をしたら明示的に走らせてください。**

```bash
uv run pytest -m slow
```

マーカーの宣言は `pyproject.toml`。宣言・除外・付与・この記述が食い違っていないことは
`tests/test_default_test_selection.py` が検査します（綴りを間違えたマーカーは除外され
ず、症状は「実行がまた遅い」だけなので気付けません）。

### 取込ベースライン

取込ループは当時1,856行あり（うち1,389行が添付1件分のforループ本体）、**それを呼ぶ
テストは1つも無かった**。現在は `main_async` がロック解放のための薄いラッパで、本体は
`_index_library`（1,111行）。in-processで呼べる網は `tests/test_ingest_end_to_end.py`
にもあり（合成蔵書・数秒・CIで実行）、下のベースラインは実蔵書側を受け持つ。

分解の前にここへ網を張る。extract-method がここで失敗するときは、じわじわずれるので
はなく**外側で更新されるはずのローカルが更新されなくなる**という壊れ方をする。抽出
した関数への単体テストでは見えず、ループを実際に回すものだけが見える。

代表3経路（HTML/PDF/EPUB）を使い捨てデータプレーンへ通し、実行が印字したカウンタ・
マニフェスト行・全チャンク（識別子/block_type/zone/構造、本文はハッシュ）・artifact
状態を記録する。本番のインデックスには触れない。5件で約140秒かかるため `slow`
マーカーを付けており、既定の `uv run pytest` では走らない（上記）。

```bash
uv run python scripts/build_ingestion_baseline.py          # 差分を見る
uv run python scripts/build_ingestion_baseline.py --write  # 採択する
```

**子プロセスの環境は継承せず組み立てる。** `os.environ` をそのまま渡すと、呼び出し側が
持っていた `PIPELINE_CONFIG_PATH` が漏れ込み、一時的な `CHROMA_DIR` と食い違って子が
起動を拒否する。単独では通りフルスイートでのみ落ちるので、ネットの不具合ではなく
「たまに落ちるテスト」に見える。データプレーンが導出する変数は消してから渡す。

**このネットが見ているのはループの28%**（1,075実行行中302行）。5件の添付を、初回取込／
変更なし／品質再読の3パスで流している。当初は3件・1パスで21%だった。

**括り出す前にそのブロックの到達率を見ること。** 見ていない箇所の変更は、ネットが検証
したのではなく異議を唱えなかっただけになる（最初の3件の括り出しが実際そうだった）。

まだ覆えていないのは、画像ページのEPUB（L3629, 2%）と抽出ルーティング本体の大半
（L2794, 15%）。除外した経路とその理由は `scripts/build_ingestion_baseline.py` の
CORPUS 直下に書いてある — **非決定的な出力を持つ資料はネットに入れない**。破損PDFの
修復はOCRを通り、同じ入力で109チャンクと108チャンクを返した。

```bash
uv run python scripts/measure_ingestion_net_coverage.py
```

**Zoteroが起動していないと実行できない。** 取込は蔵書の一覧をローカルAPI越しに読み、
到達不能なZoteroを「空」とは解釈しない（そう解釈すれば全削除になるため）。閉じている
場合、テストは失敗ではなくスキップする — 赤にしても「Zoteroを開いていない」という
事実を報告するだけでコードの情報ではない。

Zoteroノートは対象外。添付を持たないためこのループを1度も通らず、含めると「何も
起きなかった」という結果を記録して、ループが何をしようと一致し続ける。

**OCR層監査の値は揮発として除外している。** 候補提案にLLMを使うため、同じPDFでも実行
ごとに `ocr_layer_error_rate` が 0.00807 / 0.01066 と変わり、`ocr_layer_needs_review`
が false / true で反転する。設計通りだが、リファクタの影響判定には使えない。

### 構造復元ベースライン

フラットPDFの見出しから木を復元する規則は、どれも実際の蔵書に合わせて調整した
経験則で、良し悪しは実物の本にどう効くかでしか決まらない。この判断をこれまで
「復元できた件数」でやってきたが、その数字は誤った木も正しい木も1件と数える。
実際、**目次領域が本文を171ページ分飲み込んだ版のほうが、その前の版より件数が
良かった**。口述史料のインタビュー質問3件が小見出しに昇格した変更も、件数上は
無害に見えた。

そこで84冊の復元結果を `tests/baselines/structure_recovery.json` に記録し、
`tests/test_structure_recovery_corpus.py` が毎回引き直して照合する。

このベースラインは**Git追跡対象外**。84冊分のZotero item_key・書名・本文から抽出
した章見出しを含み、蔵書そのものの目録に近い。このリポジトリは公開されているので、
`evaluations/` と同じ理由で追跡しない。手元に無ければ照合テストはスキップされるので、
まず `--write` で生成する。

```bash
uv run python scripts/build_structure_recovery_baseline.py          # 差分を見る
uv run python scripts/build_structure_recovery_baseline.py --write  # 採択する
```

差分は**読んでから**採択する。件数が増える変更でも、ある本が自分の巻末注に章を
明け渡した結果ということがある。

この網が守らない範囲は3つ。**コーパスに無い形式**（部ごとに章番号がリセットされ
る本は84冊に含まれない）は `tests/test_source_structure_refresh.py` の手書き
テストが受け持ち、そちらがCIで強制される部分になる。**記録時点で既に誤っていた
木**はスナップショットでは検出できず、ベースライン自体を検査するのは
`test_no_recovered_tree_repeats_a_boundary`（1つの境界を二度主張しない）だけ。
**ライブラリが無い環境**では照合をスキップする（入力行の凍結は31,068行・2.5MBで
リポジトリ最大ファイルの5倍になり、縮めるには普通の本文行を捨てるしかない。直近
の不具合はまさにそこに潜んでいた）。

## 関数サイズのラチェット

150行を超える関数は `tests/function_size_budget.json` に記録され、
`tests/test_function_size_ratchet.py` が毎回引き直して照合します。**記録された関数が
伸びたら赤、150行超の関数が新しく生まれたら赤、縮んだら「新しい値で記録し直せ」と
言って赤**になります。最後のものが無いと、一度分解したところで上限が古いまま残り、
次の変更がその余白をまた使ってしまいます。

記録時点で23関数（最大は取込ループ本体 `_index_library` の1,111行）。上限は分解が進むたびに下がります。

```bash
uv run python scripts/build_function_size_budget.py          # 差分を見る
uv run python scripts/build_function_size_budget.py --write  # 採択する
```

対象は `src/`・`scripts/`・`citation_graph/`。テストは対象外です（最長67行で、長い
テストは大抵もつれではなく事例の一覧なので）。**この検査はコードの良し悪しを見ません。**
1つの事例一覧として読める200行と、分岐の連鎖の200行を区別できないので、増加を止める
だけで、読みやすさの判断は分解する人に残しています。

## カバレッジのラチェット

モジュールごとの**未実行の文数**を `tests/coverage_budget.json` に凍結し、CIが毎回
検査します。**割合ではなく文数**にしているのは、割合はコードを足すだけで下がるため
です（良く書けたコードでも下がるので、書かない方が得になってしまう）。文数は「誰も
実行しないコードが増えたか、届いていたテストが消えたか」のときだけ増えます。

```bash
uv run coverage run --source=src,citation_graph -m pytest -q
uv run python scripts/build_coverage_budget.py --check   # 悪化していないか
uv run python scripts/build_coverage_budget.py --write   # 採択する
```

記録時点で **88モジュール・5,553文**が未実行。凍結した数字には意味があります —
変更頻度の上位3ファイルが同時にカバレッジ下位でした（`index_from_zotero` 42%、
`citation_graph/server` 37%、`rag_mcp_server` 41%）。**触る場所ほど守られていない**
状態で、それが「いじると驚く」ことの実体です。床はそれ以上悪くならないようにします。

CIはテストとカバレッジを1回の実行で兼ねます。部分的な実行（1ファイルだけ、`-m slow`
だけ）のデータで比較すると全モジュールが悪化と出るので、計測ファイル数が少なすぎる
ときは比較せずに停止します。

## 例外を飲み込む形のラチェット

CIが強制するruffは致命的な4ルールだけですが、それとは別に**欠陥の形をしたルール**の
件数を `tests/lint_budget.json` に凍結し、`tests/test_lint_ratchet.py` が照合します。
**増えたら赤、減ったら「記録し直せ」と言って赤**です。

```bash
uv run python scripts/build_lint_budget.py          # 差分を見る
uv run python scripts/build_lint_budget.py --write  # 採択する
```

対象は `BLE001`（広いexcept）、`S110`/`S112`（except: pass / continue）、
`PLW1510`（`check`無しのsubprocess）、`TRY004`、`F401`。記録時点で309件あります。

**既存の309件を今日直すのではなく、増えないようにするのが目的です。** 全部を
`# noqa` で黙らせると「誰も検査しない主張」が309個できるだけなので、件数を凍結して
既存を見えるまま残します。実際、2026-08-09に直した欠陥はほぼ全部この形でした —
1つのストアが失敗しただけで他3つと失敗記録を放棄した補償、「スキャンではない」と
読まれた分類失敗、「そのチャンクは無い」と報告されたChromaエラー、「引用文脈なし」
として返されたDB障害。

**正しいexceptもあります。** 描画できないページを「白紙ではない」と返す
`_rendered_page_is_visually_blank` のように、fail-closedで理由が書いてあるものは
そのままです。減らすときは1件ずつ現物を読んでから。

## 現状を数字で見る

「あとどれくらい残っているか」を記憶で答えると毎回違う答えになるので、コードと予算
ファイルとレジスタから毎回引き直します。

```bash
uv run python scripts/show_project_health.py
uv run python scripts/show_project_health.py --coverage  # スイートを計測付きで回す（分単位）
```

**目標は「指摘ゼロ」ではありません。** 変更を安全に入れられること — 機械的な層が後退を
拒否し、読まないと判断できない部分が読める量に収まっていること — が目標です。

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

Citation Graphへ階層要約・文書構造を統合する画面仕様は、[階層要約・構造ビュー設計](citation-insights-ui-design.md) を参照してください。

## 参考文献の審査キュー

```bash
uv run python scripts/review_references.py list --status pending --limit 20
uv run python -m src.reference_quality_report --status pending
```

参考文献の抽出結果は、原文根拠を再検証してから適用します。修復・分類用CLIの詳細は各コマンドの `--help` を参照してください。

英語学術論文のGROBID enrichmentは本文埋め込みとは独立して実行します。既定はdry-runです。

```bash
# 対象判定のみ
.venv/bin/python scripts/run_grobid_enrichment.py --limit 5

# ローカルGROBID service起動後、reference review queueへ保存
.venv/bin/python scripts/run_grobid_enrichment.py --limit 5 --apply
```

対象はZotero種別が`journalArticle`、`conferencePaper`、`preprint`の英語PDFだけです。
同じPDF SHA-256とGROBID versionで成功済みなら再処理せず、GROBID障害は埋め込み・manifest・Chromaへ影響しません。

実行前に`.env`で`GROBID_ENRICHMENT_ENABLE=1`を設定し、ローカルGROBID serviceを起動してください。
flagが`0`のとき、またはflagが`1`なのにserviceが応答しないときは、workerが処理を始める前にexit code 2で停止します
（他のoptional機能と同じく`src/feature_gates.py`が判定します）。

## 検索品質評価

```bash
uv run python scripts/eval_retrieval.py data/quality/gold_qa.jsonl --k 10
# V3のLLMノード要約を生成・埋め込み後に、階層検索経路も含めて比較する
uv run python scripts/eval_retrieval.py data/quality/gold_qa.jsonl --k 10 --include-hierarchical-v2
```

## ローカルデータとバックアップ

重要な運用データ（現行はV3データプレーン）:

- `data/relations.db`
- `data/chroma/`（active collection `zotero_paragraphs_v3` と `zotero_paragraphs_v3__sum_node`）
- `data/lexical_v3.sqlite3`
- `data/manifest_v3.json`

旧collection・`manifest.json`・`lexical.sqlite3`は本番経路では使用しません。復旧が必要な場合は、
現在のV3データ面をバックアップから復元するか、原本からServer workflowのフェーズ1を再実行します。

大規模修復前だけ `data/backups/` にV3スナップショットを作成します。修復後にDB整合性と回帰テストを確認し、古い中間バックアップを削除します。`data/`はGitやSoftware Updateでは同期されないため、必要なら利用者側の暗号化バックアップへ保存してください。

## ドキュメント方針

- `README.md`は短い入口として保つ
- 詳細は目的別の `docs/*.md` に置く
- 同じ設定説明を複数文書へコピーせず、リンクする
- CLIやファイル名を変更したら関連リンクを検索・更新する
