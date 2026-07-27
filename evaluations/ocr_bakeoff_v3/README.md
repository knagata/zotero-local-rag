# PDF解析エンジン・ベイクオフ V3

`64_ingestion_redesign_audit_and_plan.md` §4.0・§4.5・§9.2 Phase 2の評価基盤。PDFのリポジトリ保存は行わない。既定はローカルのみで、モデル取得・クラウド送信は発生しない。唯一の例外は`mistral_ocr`（クラウド）で、`MISTRAL_OCR_API_KEY`設定と`OCR_BAKEOFF_ALLOW_CLOUD=1`の両方を明示しない限り`unavailable`のままであり、意図せず送信されることはない。

## 人間が用意するもの

1. `manifest.json` の8カテゴリに対応する所蔵PDFを選ぶ。
2. 各 `path_env` に絶対パスを設定する。パスをmanifestへ書かない。
3. `annotation.example.json` を基に `annotations/<sample_id>.json` を作る。各regionのanchorはページ内で一意な短い原文とし、見出し階層・読み順・zone・表/キャプション・locator/bbox要件を目視で記録する。
4. PDFとannotationを固定した時点で `manifest.json` の `frozen` を `true` にする。

annotationsは正解ラベルだけなのでコミット可能。著作物本文の長い転記は避け、anchorとexpected_textは照合に必要な最小範囲にする。

`pages`は目視固定した1始まりの評価ページである。実行時に`tmp/ocr_bakeoff_v3/sources/`へ一時PDFを作り、全エンジンへ同一ページだけを渡す。元PDFは変更しない。

## CLI

```bash
.venv/bin/python scripts/run_ocr_bakeoff.py --list
.venv/bin/python scripts/run_ocr_bakeoff.py --dry-run --engine pymupdf --sample en_two_column
.venv/bin/python scripts/run_ocr_bakeoff.py --engine pymupdf --engine docling
```

既定出力は `tmp/ocr_bakeoff_v3/`。`report.json`、`report.md`、実行時の正規化raw JSONを生成する。dry-runはPDFを処理しない。

ローカル候補は `pymupdf`、`docling`、`ndlocr_lite`、`yomitoku`（lite/CPU）。利用可否検査はモデルをimport・downloadせず、依存パッケージ、実行ファイル、メモリを確認する。
YomiTokuはCC BY-NC-SA 4.0（非商用）ライセンスであり、個人研究用途に限定して利用する。実測結果は`results/comparison.json`・`results/routing_proposal.md`を参照（日本語縦書き・注参考文献カテゴリで首位）。
DeepSeek-OCRは実機検証の結果、Apple Silicon上での処理負荷（速度・メモリ）が実用に見合わないため候補から除外した（2026-07-21判断）。

クラウド候補は `mistral_ocr` のみ。ローカル抽出器で構造化が明確に失敗した資料だけに使う任意fallback用途で、既定の常時送信経路にはしない。`MISTRAL_OCR_API_KEY`（`.env`）を設定し、かつベイクオフ実行時に `OCR_BAKEOFF_ALLOW_CLOUD=1` を明示しない限り `unavailable` のままになる。`no-cloud` タグの資料はこの二重ゲートとは別に `EngineRegistry.select(no_cloud=True)` で常に拒否される。

実測済み（2026-07-21）: 8カテゴリ平均0.848342で全エンジン中最高（詳細は`results/routing_proposal.md`・`results/comparison.md`）。応答は`pages[].blocks[]`にbbox（`top_left_x/y`,`bottom_right_x/y`）とtype（header/footer/title/text/caption/table/image）を持つことを実機で確認し、`src/mistral_ocr_extract.py`はこの構造を直接消費する。スコアが最高でも既定エンジンはDoclingのままとし、Mistral OCRはopt-inフォールバックに限定する。

## 固定採点

構造を主軸に、見出し階層22%、読み順22%、zone分類18%、表/キャプション保持12%、locator/bbox復帰10%、ツリー整合10%、文字精度6%。欠落した正解regionは不正解とし、anchor照合・ペア順序・完全一致の階層/zoneを決定的に採点する。処理時間とメモリ条件はレポートへ記録するが、品質スコアには混ぜない。
