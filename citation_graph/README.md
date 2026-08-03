# Citation Graph

Citation Graphの画面、ローカルHTTPサーバー、read-only insight serviceをまとめた
パッケージです。

```bash
uv run citation_graph/server.py
```

## 画面の構成

| 置き場所 | 内容 |
|---|---|
| `static/app.css` · `static/app.js` | 画面のスタイルと動作のほぼ全て |
| `server.py` の `_build_sigma_html()` | HTMLの外枠と凡例（件数を埋め込む部分）だけ |

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
