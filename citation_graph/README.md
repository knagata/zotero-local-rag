# Citation Graph

Citation Graphの画面、ローカルHTTPサーバー、read-only insight serviceをまとめた
パッケージです。

```bash
uv run citation_graph/server.py
```

画面は実行時に生成され、グラフデータはローカルDBからAPI経由で取得します。
書誌情報や引用文脈を埋め込んだ静的HTMLは生成・コミットしません。

ランタイムDB、配置キャッシュ、生成レポートはすべて`data/`に保存され、Gitでは
ディレクトリ全体を除外します。
