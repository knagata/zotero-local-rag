# zotero-local-rag

Zotero蔵書に対するローカルRAG。人文・社会科学系、和書と書籍が中心。

**目的は要約ではない。** ユーザーが資料を横断的・網羅的に閲覧し、求める文献に到達し、
潜在的な関心に偶然出会うこと。遠読（階層要約・Citation Graph）と精読（引用に基づく
高精度検索）をシームレスに支えること。AI利用による資料離れの是正が動機なので、
ユーザーと資料の距離を広げる方向の変更は、それ自体が目的に反する。

## 正本

| | |
|---|---|
| `SPEC.md` | 実装契約。矛盾したらこちらが優先 |
| `TASKS.md` | 実装・検証の永続的な履歴。1件直すごとに反映してコミット |
| `docs/` | 利用者向け・開発手順 |
| `dev-notes/`, `evaluations/` | **gitignore済み**。Zotero識別子や絶対パスを含むため追跡しない。テストが読むデータをここに置かない |

## セッション引継ぎ

`memory/projects/current-development.md`が存在するときは、このファイルの次に読む。これは
現在地と開始手順の作業メモであり、実装契約の`SPEC.md`や履歴の`TASKS.md`を上書きしない。
完了・方針変更・測定値更新時は同じコミットで更新し、古い開始点を次セッションへ渡さない。

## 対話

日本語で。

## コマンド

```bash
uv run pytest -q
uv run python -m compileall -q src scripts tests citation_graph
uv run ruff check --select E9,F63,F7,F82 src scripts tests citation_graph
```

CI はこの3つに `uv run python -c "import src.cli"` を加えたものを回す。
テストは CI では計測付き（`uv run coverage run --source=src,citation_graph -m pytest -q
-W error::RuntimeWarning`）で、その結果を `scripts/build_coverage_budget.py --check` が
検査する。Python は 3.10 固定。

既定の実行は `slow` マーカーの付いたテストを除外する。取込ベースライン
（`tests/test_ingestion_baseline.py`、実測205秒中140秒）がそれで、蔵書・Zotero・
埋め込みモデル・手元生成のベースラインを要するためCIでは元々走っていない。
**抽出・取込・構造復元に触ったら明示的に走らせる。**

```bash
uv run pytest -m slow
```

マーカーの宣言と除外・付与・この記述の対応は `tests/test_default_test_selection.py`
が検査する。綴りを間違えたマーカーは除外されず既定の実行に戻るだけなので、
機械に押し戻させる。

### 長時間コマンドはHerdrで見えるようにする

CodexがHerdr管理下（`HERDR_ENV=1`）で動いているとき、全件テスト、coverage、build、
サーバー、監視などの継続時間が長いコマンドは、現在のタブに新しいHerdrペーンを
`--no-focus`で作り、リポジトリの作業ディレクトリを引き継いで実行する。同じタスクでは
そのペーンを再利用し、終了時にペーンIDと結果を報告する。数秒で終わるread-only診断や
ファイル確認は直接実行でよい。Herdr外では通常の実行手段にfallbackする。

## 作業上の規則

### 経験則を変えたら、生成物を読む。集計値で判断しない

抽出・構造復元・引用同定の規則はどれも蔵書に合わせた経験則で、良し悪しは実物にどう
効くかでしか決まらない。「何件成功したか」は誤った出力も正しい出力も1件と数えるので、
改善と悪化を区別しない。実際、目次領域が本文を171ページ分飲み込んだ版のほうが、その
前の版より件数が良かった。

### 真偽テストは都度アップデートする

スナップショット（`tests/baselines/`）は現在の挙動を固定するだけで、それが正しいことは
保証しない。記録時点で誤っていた出力はそのまま固定される。だから**規則を変えるたびに、
スナップショットの採択だけで済ませず、スナップショットに依存しない真偽テストを追加・
更新する**。「1つの境界を二度主張しない」のように、コーパスの中身と無関係に成り立つ
性質を書く。これが唯一、間違ったベースラインを検出できる層になる。

現状の対応関係:

| 層 | 例 | CIで強制 |
|---|---|---|
| 真偽テスト（性質） | `test_no_recovered_tree_repeats_a_boundary` | ○ |
| 手書きの規則テスト | `tests/test_source_structure_refresh.py` | ○ |
| 実コーパス照合 | `tests/test_structure_recovery_corpus.py` | × (蔵書とベースラインが要る) |
| 取込の特性化 | `tests/test_ingestion_baseline.py` | × (同上＋埋め込みモデル) |

ベースライン `tests/baselines/` は**追跡対象外**。書名と章見出しを含み、このリポジトリ
は公開されているため。手元で `--write` して使う。

構造復元のベースライン更新:

```bash
uv run python scripts/build_structure_recovery_baseline.py          # 差分を見る
uv run python scripts/build_structure_recovery_baseline.py --write  # 採択する
```

差分は読んでから採択する。件数が増える変更でも、ある本が自分の巻末注に章を明け渡した
結果ということがある。

### 記録は「腐らない形」で書く。更新は同じコミットで

「ドキュメントを更新すること」という心構えでは効かない。記録が腐るのは、**誰も読み
返さず、何もそれに反論しないから**。だから書くときに形を選ぶ。

- **二重に書かない。** 事実の家は1つ（`SPEC.md` / `docs/` / `pyproject.toml` /
  `.gitignore`）。他所からは参照するだけにする。写した瞬間から両者は独立に古くなる。
- **反論されうる形で書く。** コマンドはCIから写す（乖離すればCIが落ちる）。パスや
  ファイル名は`tests/test_documentation_references.py`が実在を検査する。数値は
  再生成できるスクリプトを添える。何にも検査されない断定は、そのぶん寿命が短い。
- **非追跡の場所に一次情報を置かない。** `.claude/`と`dev-notes/`はgitignore配下で、
  コードと一緒にレビューされない。実際`.claude/STATE.md`は2か月古い説明を配り続けた。
- **ファイルを動かしたら、それを名指す文書を同じコミットで直す。** 別コミットに
  回した分は戻ってこない。

`CLAUDE.md`は毎セッション読み込まれるので、ここに古い記述があると全員に配られる。
古くなりうる記述を足す前に、それが何によって検査されるかを決める。

### プッシュしたらCIの結果を見る。手元の緑はCIの緑ではない

CIはこのリポジトリで唯一「必ず実行される」検査層で、そこが赤いなら守られていない。
プッシュ後に必ず確認する。

```bash
gh run list --limit 1
gh run view <id> --log-failed
```

**赤でないことは緑ではない。** 2026-08-06のrunはランナー確保に失敗し、テストを1件も
実行しないまま終わっていた。誰も見ていなかったため2日分の変更が未検証で積まれ、次に
実際に走ったとき7件が同時に落ちた。`success` を目で確認すること。

**手元で通ってもCIで落ちる。** その7件は3種類とも、手元にあってCIに無いものに依存して
いた: インデックス済み蔵書、埋め込みモデル、gitignore配下のファイル（`.claude/`）。
最後のものは「手元では通る記述」を検出するために書いたテスト自身が、手元では通ることで
落ちた。**CIに無いものを要するテストは、失敗ではなくスキップさせる**（`tests/
test_structure_recovery_corpus.py` のskipifが例）。要否の判定は、静的なリストを別途
保守するのではなく、gitなど元の情報源に問う。

### 巨大関数はラチェットで押し戻す

150行超の関数は `tests/function_size_budget.json` に記録済みで、伸ばすと落ちる。
新しく150行超を作っても落ちる。縮めたら `--write` で採択して上限を下げる
（採択しないと落ちる）。都度の判断で分解し続けるのではなく、機械に押し戻させる。
手順は `docs/development.md`。

### テストが届かない行を増やさない

モジュールごとの未実行文数は `tests/coverage_budget.json` に凍結済み。増やすと **CIが
落ちる**。割合ではなく文数なので、テスト付きのコードはいくら足しても増えない。減ったら
採択して床を下げる。

```bash
uv run coverage run --source=src,citation_graph -m pytest -q
uv run python scripts/build_coverage_budget.py --check   # 悪化していないか
uv run python scripts/build_coverage_budget.py --write   # 採択する
```

### 例外を飲み込む形は増やさない

広いexcept・`except: pass`・`check`無しsubprocess の件数は
`tests/lint_budget.json` に凍結済み。増やすと落ちる。**既存の309件は残っている**が、
新規は通らない。直したら `--write` で採択して上限を下げる。手順は `docs/development.md`。

### 語彙は単一定義

見出しの機能語彙は `src/heading_zone.py`、構造語彙は `src/heading_structure.py` だけに
置く。どちらも、複数の抽出器が各自のコピーを持って食い違った結果として作られた。
モジュール内にローカルな正規表現を足さない。言語を増やすのはパターンの拡張であって
分岐の追加ではない。

### 破壊的なgit操作の前に確認する

`git checkout -- <file>` / `reset --hard` / `stash` の前に `git status` を見て、
未コミットの作業を退避する。このセッションで実際に修正内容を消している。

### 抽出コードを変えたら再取込が要る

`--force-reparse` を明示しない限り、既存チャンクは作り直されない。スコープ指定
（`--item` 等）が必須。

## 検証環境

蔵書は約589アイテム・513,683チャンク。埋め込みは BGE-M3 / MPS（`EMB_PROFILE=bge`）。
実データを見るスクリプトは `load_dotenv_native(ROOT)` を先に呼ぶこと。呼ばないと
プロファイルが既定値になり、測定結果を取り違える。
