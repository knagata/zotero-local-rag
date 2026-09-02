# Citation Graph

Citation Graphの画面、ローカルHTTPサーバー、read-only insight serviceをまとめた
パッケージです。

```bash
uv run citation_graph/server.py
```

このコマンドは従来どおり`127.0.0.1`だけで待ち受け、ローカルでは認証なしで使えます。
`remote_proxy.py`は別プロセスのGoogle OAuth公開入口で、認証済み・許可メールのリクエストだけを
ローカルサーバーへ転送します。Google資格情報はRemote MCPと共用し、秘密値は`.env`から読みます。

```bash
uv run python -u citation_graph/remote_proxy.py
uv run python scripts/manage_citation_graph_service.py status
```

自動起動、Funnel、Google callback、ログ、運用コマンドの詳細は
[Show Citation Networkガイド](../docs/show-citation-network.md#外部コンピュータからgoogleログインで開く)を
参照してください。

OAuth公開URLの`/admin/`はメンテナンス管理画面です。`admin_jobs.py`の固定job catalogだけを
独立processで実行し、`data/admin_jobs/`へ状態とログを保存します。proxyが再起動してもjobは継続し、
再起動で孤立した記録は`interrupted`として回収します。更新状況は専用LaunchAgentがログイン時と30分ごとに
同じ単一job制御を通して確認するため、アクセス時は保存済み結果を即座に表示できます。ローカルの無認証7234には
管理APIを載せません。

## 画面の構成

| 置き場所 | 内容 |
|---|---|
| `static/app.css` · `static/app.js` | 画面のスタイルと動作のほぼ全て |
| `server.py` の `_build_sigma_html()` | HTMLの外枠と凡例（件数を埋め込む部分）だけ |
| `remote_proxy.py` | Google OAuth、許可メール、署名session、ローカルserverへの転送 |
| `admin_jobs.py` | 管理画面の固定job catalog、単一実行、状態・ログ・安全な停止 |
| `admin_routes.py` | OAuth管理画面・状態API・開始／停止API |
| `static/admin.*` | OAuth配下の処理状況・メンテナンス画面 |

**CSS/JSをPython文字列へ戻さないでください。** 以前は4,100行のCSS/JSが
`_build_sigma_html()` の文字列リテラル内にあり、構文チェッカもリンタもエディタも
それを見ることができませんでした。その状態で `esc()` を `fetch().then()` の内側で
定義し `.catch()` から呼ぶというバグが出荷され、ブラウザで `ReferenceError` に
なるまで誰も検出できませんでした（2026-08-03）。`static/` にあれば
`node --check citation_graph/static/app.js` で検証できます。

パレット（`_PALETTE`）だけはPython側を単一の真実の源とし、`:root` のCSS変数と
`window.__RAG_THEME__` としてサーバーが注入します。

画面は実行時に生成され、グラフデータはローカルDBからAPI経由で取得します。
書誌情報や引用文脈を埋め込んだ静的HTMLは生成・コミットしません。

ランタイムDB、配置キャッシュ、生成レポートはすべて`data/`に保存され、Gitでは
ディレクトリ全体を除外します。
