# Show Citation Networkガイド

[READMEへ戻る](../README.md)

Show Citation Networkは、Zoteroの所蔵資料と、それらを引用する外部資料・所蔵資料が参照する文献との関係を可視化するローカルブラウザー画面です。

## 開く前に

Show Citation Network自体は引用情報を更新しません。まず `Maintenance-Widget.command` でCitation Network更新を実行してください。

必要なもの:

- `data/relations.db` に保存済みの引用・参照データ
- 画面表示時に表示用ライブラリを読み込むためのインターネット接続
- Citation Networkを更新するための `S2_API_KEY`

## 起動と終了

macOSで [Show-Citation-Graph.command](../Show-Citation-Graph.command) をダブルクリックします。ターミナルからは次のように起動できます。

```bash
uv run citation_graph/server.py
```

グラフの計算中にブラウザが開きます。表示サーバーは自分のMac内の `127.0.0.1` だけで待ち受けます。通常のURLは `http://localhost:7234` で、使用中の場合は次の空きポートが選ばれます。

終了するときは、起動したターミナルで `Control+C` を押します。ブラウザのタブを閉じるだけではサーバーは終了しません。

## 外部コンピュータからGoogleログインで開く

通常のローカルサーバーを起動したまま、別ターミナルで公開専用proxyを起動します。

```bash
uv run python -u citation_graph/remote_proxy.py
tailscale funnel --bg --https=8443 http://127.0.0.1:7244
```

`.env`の`CITATION_GRAPH_PUBLIC_URL`は
`https://<Macの名前>.<tailnet>.ts.net:8443`とし、32文字以上のランダムな
`CITATION_GRAPH_SESSION_SECRET`を設定します。生成例:

```bash
uv run python -c 'import secrets; print(secrets.token_urlsafe(48))'
```

Google Cloud Consoleの同じウェブアプリケーションOAuthクライアントへ、
`https://<Macの名前>.<tailnet>.ts.net:8443/auth/callback`を承認済みリダイレクトURIとして追加します。
許可ユーザーはRemote MCPと同じ`REMOTE_MCP_ALLOWED_GOOGLE_EMAILS`です。公開URLからの画面・API操作は
すべて認証対象ですが、従来の`http://localhost:7234`には影響しません。

### 自動起動と状態確認

動作確認後はCitation Graph本体、OAuth proxy、更新状況の定期確認をmacOS LaunchAgentとして登録します。

```bash
uv run python scripts/manage_citation_graph_service.py install
uv run python scripts/manage_citation_graph_service.py status
```

Graphとproxyはログイン時に起動し、異常終了時は10秒以上空けて再起動します。更新状況はログイン時と
30分ごとに読み取り専用で確認し、別の管理jobが実行中なら次回へ延期します。秘密値はplistへ複製せず
`.env`から読みます。運用コマンド:

```bash
uv run python scripts/manage_citation_graph_service.py restart
uv run python scripts/manage_citation_graph_service.py uninstall
```

ログは`data/citation-graph*.log`です。`status`では`update_check_launch_agent=loaded`も確認します。
`uninstall`はLaunchAgentだけを削除し、Tailscale Funnelの
8443設定は変更しません。

### ブラウザで処理状況を確認・更新する

Google OAuth公開URLへ`/admin/`を加えると管理画面を開けます。

```text
https://<Macの名前>.<tailnet>.ts.net:8443/admin/
```

表示内容:

- manifestの添付・ノート件数、HNSW検証、inflight、最終更新
- DB監査gate、indexing lock、failed／blocked artifact数
- 読み取り専用照合で確認した添付・ノート索引、文書構造・目次、全Zotero書誌itemの
  Citation Network未更新件数と対象キー。原本欠落・構造失敗・Citationエラーは赤い要確認表示
- 実行中jobのstep、進捗ログ、停止操作
- proxy再起動後も残る直近20件の実行履歴

実行できる処理:

| 処理 | 内容 | 確認語 |
|---|---|---|
| 更新状況を確認 | Zotero・索引・構造・Citation台帳を変更せず照合し、未更新件数を表示 | なし |
| クイック実行 | 索引・文書構造・目次の差分更新、DB監査、Citation Network更新を順番に実行 | `QUICK` |
| ライブラリ差分更新 | Zotero差分取込後に文書構造・目次を差分更新 | `UPDATE` |
| DB監査 | Zotero・原本・索引を照合。開始時に前回gateを無効化し、合格時に再作成 | `AUDIT` |
| 文書構造・目次の差分更新 | 原本を再確認し、変更資料だけ更新。再埋め込みなし | `STRUCTURE` |
| 階層要約の差分更新 | 構造更新・監査後、DeepSeekで最大10件・10並列・要約索引反映 | `SUMMARIZE` |
| Citation Network更新 | 未処理・エラー分の引用・参照関係を更新 | `CITATIONS` |

クイック実行は各工程が成功した場合だけ次へ進み、途中で失敗するとそこで停止します。階層要約は含みません。
「更新状況を確認」は原本とノートの追加・変更・削除、保存チャンクに対する構造fingerprint、
全Zotero書誌itemとCitation台帳・DOI・ISBN（削除を含む）を照合し、外部APIへの更新送信や正本DBへの書き込みは
行いません。結果には確認日時が付き、
次回確認まで保存されます。LaunchAgent設定後は30分ごとに自動更新され、必要なら画面から即時確認もできます。
確認から45分を超えると期限切れを表示します。索引・構造・要約・Citationの書込みjobを開始すると
「再確認待ち」になり、正常終了後は読み取り専用確認を自動実行して表示を更新します。
確認語が「なし」の読み取り専用処理はクリックすると直接開始し、確認ダイアログを表示しません。
同時に実行できるjobは1件です。階層要約は有料APIを使うため、毎回確認語が必要です。停止も`STOP`の
入力が必要で、保存したPID・job ID・tokenと実際のprocess commandが一致するときだけprocess groupへ
通知します。任意command、DB全削除・`--rebuild`、構造`--force`、全件有料要約は管理画面から実行できません。

状態は`data/admin_jobs/<job-id>.json`、出力は同名`.log`へ600権限で保存されます。開始者と停止者の
Googleメールも監査用に記録します。OAuth proxyが再起動してもjob processは継続し、OS再起動等で
processが失われた記録は次回表示時に`interrupted`へ変更されます。

## グラフの見方

- Zoteroアイテム: 自分のライブラリにある資料
- 被引用元: 選択した資料を引用している外部資料
- 参照先: 選択した資料が引用・参照している資料
- ノードの大きさ: その資料の被引用数
- エッジの太さ: 保存された関係の強さ

ノードにマウスを置くと書誌情報を確認できます。クリックするとその資料と直接つながるノードが強調され、被引用元と参照先を区別できます。空白部分または同じノードをクリックすると選択を解除できます。

ノード選択後にエッジをクリックすると、保存されている引用コンテキストとページ情報が「概要/コンテキスト」タブに表示されます。元データに本文の根拠がない場合は表示されません。

## 誤った関係を報告する

エッジを開き、「誤りを報告」を押して具体的な根拠を入力します。例えば「元PDFの参考文献一覧に存在しない」「DOIが別の著作を示す」などです。分野が違うように見えるという理由だけでは報告しません。

報告しただけではエッジは非表示になりません。管理者が
`uv run python scripts/review_relation_reports.py`を実行し、Disableを選んだ場合だけ
除外されます。Disableされた関係は、Citation Networkの再取得後も復活しません。

## 主な操作

| やりたいこと | 操作 |
|---|---|
| 資料を探す | 右側の資料一覧でタイトル・著者を検索 |
| 所蔵資料だけ見る | 「Zotero所収のみ」を有効化 |
| Zoteroコレクションで絞る | 左側のコレクションフィルタを選択 |
| 著名な資料に絞る | 最低被引用数のスライダーを調整 |
| 書誌・概要を見る | ノードまたは資料一覧の行をクリック |
| 引用文脈を見る | ノードを選択し、強調されたエッジをクリック |
| 全体を移動・拡大縮小する | ドラッグで移動、スクロールでズーム |

## 引用ネットワークと意味マップ

画面上部で二種類の表示を切り替えられます。

- 引用ネットワーク: 実際の引用・参照関係を表示
- 意味マップ: 所蔵資料の埋め込みを二次元に配置し、内容が近い資料を近く表示

意味マップはZotero所蔵資料だけを表示します。UMAP、t-SNE、PCA、MDSから配置方法を選べます。近さは探索の手がかりであり、研究上の関係を証明するものではありません。

## 概要・要約・原文コンテキスト

- Zotero所蔵資料の概要は、Zoteroが起動中ならローカルAPIから読み込めます。
- Zotero所蔵資料を選ぶと、右側の「概要/コンテキスト」に「概要」と「節要約」が表示されます。
- 「節要約」では文書順の要約を開き、必要な節だけ原文チャンクを確認できます。
- 具体例や経験的な記述は、原文検索と周辺コンテキストで確認します。
- 外部資料の概要を開くと、CrossrefまたはSemantic Scholarへ識別子を送って取得します。
- 階層要約は一括処理で生成され、この画面では閲覧と問題報告だけを行います。
- 引用コンテキストの翻訳は `AZURE_TRANSLATOR_KEY` を設定した場合だけ利用でき、対象文がAzure Translatorへ送信されます。

節要約に原文との不一致があれば「問題を報告」から具体的な根拠を入力します。報告直後には内容を削除せず、「報告済み・判定待ち」として残します。メンテナンス時の品質確認で誤りと確定したものだけが検索対象から可逆的に除外されます。

節要約が未生成の場合は、`Maintenance-Widget.command` で要約更新を実行してください。

通常のグラフ操作で `relations.db` の内容を外部サービスへ送信することはありません。ただし、ブラウザは表示用JavaScriptとフォントをCDNから読み込みます。概要取得と翻訳の送信先と内容は上記のとおりです。送信制御の詳細は [LLMとプライバシー](llm-and-privacy.md) を参照してください。

## 書誌情報の修正

Zotero所蔵資料の編集ボタンはZoteroの対象アイテムを開きます。Zotero側で修正し、次回のライブラリ更新で反映させてください。

外部資料のタイトル、著者、年、DOI、ISBNは画面から補正できます。この修正は `relations.db` のローカル上書きとして保存され、Zoteroや外部サービスの元データは書き換えません。DOIまたはISBNが他ノードと重複する場合は警告が表示されます。

## CLIで表示範囲を絞る

```bash
# 上位100件のZotero資料に絞る
uv run citation_graph/server.py --top 100

# 特定資料だけ表示
uv run citation_graph/server.py --item ITEMKEY

# 参照先ノードを隠す
uv run citation_graph/server.py --no-refs

# 自動でブラウザを開かない
uv run citation_graph/server.py --no-open
```

その他の選択肢は次で確認できます。

```bash
uv run citation_graph/server.py --help
```

`--min-cc` はサーバーが読み込む外部資料の最低被引用数で、既定値は10です。画面内のスライダーは、読み込み済みの資料をさらに絞り込みます。被引用数の小さい資料も見たい場合は `--min-cc 0` で起動してください。

## うまく表示されないとき

### 「No citation data found」と表示される

`Maintenance-Widget.command` でCitation Network更新を実行してから再起動してください。

### 追加した資料が出ない

グラフは起動時の保存データを表示します。Citation Network更新後に、起動中のサーバーを `Control+C` で止め、再度起動してください。

### ブラウザが自動で開かない

ターミナルに表示された `http://localhost:...` をブラウザで開いてください。

### ノードが少ない

画面内の最低被引用数フィルタ、コレクションフィルタ、「Zotero所収のみ」を確認してください。それでも少ない場合は `--min-cc 0` で起動し、Citation Networkの `not_found` や保留状態も確認します。
