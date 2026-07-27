# PDF解析エンジンの採用案

## 結論（2026-07-21改訂: Mistral OCR実測とスコアラー修正を反映）

**Mistral OCR（クラウド）が8カテゴリ平均で最高スコア（0.848342）となった。** ただしクラウド送信を
伴うため、既定エンジンには**しない**。V3のローカル既定は引き続き **Docling** とし、Mistral OCRは
「ローカル抽出器の構造化が明確に失敗した資料」「no-cloud対象外かつ高重要度の資料」に限定した
opt-inフォールバックとして位置付ける。日本語縦書きと注・参考文献はYomiToku/Mistral OCRが同程度に
Doclingを上回るため、この2カテゴリはローカル優先でYomiTokuをopt-in候補にする（クラウド送信を避けたい
場合の選択肢として）。PyMuPDFは高速preflight、NDLOCR-Liteはくずし字等の特殊資料向け任意fallback。

YomiTokuは**CC BY-NC-SA 4.0（非商用）ライセンス**（本プロジェクトは非商用のため利用可）。
Mistral OCRは**クラウド送信**であり、`no-cloud`資料には絶対に使わず、資料単位の送信許可を要する。

日本語縦書きはYomiToku/Mistral OCRの追加で0.370→0.666まで改善したが、まだ全カテゴリ中で最も
低い部類であり自動採用の合格水準には届かない。縦書き資料は引き続き `degraded` として再OCR候補へ
送り、元PDFへのlocatorと抽出品質を明示する。

## スコアラー修正（2026-07-21・重要）

Mistral OCRの`no_outline`サンプルを検証中に、`src/ocr_bakeoff.py`の領域対応付け
（`_match_regions`）に2つの実バグを発見し修正した。

1. **markdown装飾による完全一致の破損**: Mistral OCRは強調構文（`*text*`）を含むmarkdownを返すため、
   正解anchorとブロックテキストが実質同一でも`*`が挿入された時点で完全一致（`anchor in text`）が
   壊れていた。
2. **フォールバック時のratio希釈**: 完全一致に失敗した際、短いanchor（数十字）を長い複数文の
   ブロック全文（数百字）に対して`SequenceMatcher.ratio()`で比較すると、無関係な残りの文章の
   長さで比率が希釈され、実際には正しい対応でも0.6の閾値を下回っていた（実測: 正しい対応で
   ratio≈0.09）。

修正: 対応付け専用に`*_`\`#`を除去する正規化を追加し、フォールバック指標をanchor自身の長さに
対する「連続一致部分の割合」（`find_longest_match`ベース）に変更した。`text_accuracy`スコアの
計算自体（採点用の`normalize_text`）は変更していない。

**影響**: 既存キャッシュ済みraw出力を再抽出なしで再採点した結果、Docling・PyMuPDFは無変化
（バグの影響を受けるサンプルがなかった）。**NDLOCR-Liteの平均は0.542092→0.633549へ上方修正**
（`embedded_text_pair`・`scanned_pair`で対応付け漏れがあった）。Mistral OCRの初回集計0.803は
本書の0.848342へ修正。回帰テストを`tests/test_ocr_bakeoff.py`に追加済み。

## 実測根拠（修正後、全エンジン共通の採点ロジックで再計算）

| Engine | 平均score | 平均処理時間 | 平均peak RSS |
|---|---:|---:|---:|
| **Mistral OCR** | **0.848342** | 2.20秒 | 76.4 MB |
| YomiToku | 0.767856 | 7.91秒 | 2680.4 MB |
| Docling | 0.751806 | 2.68秒 | 1385.5 MB |
| NDLOCR-Lite | 0.633549 | 2.92秒 | 156.0 MB |
| PyMuPDF | 0.482233 | 0.02秒 | 55.9 MB |

- 固定8カテゴリを各エンジンで完走した。各カテゴリ1資料の小規模pilotである。
- **Mistral OCRの強み**: スキャン資料 0.973176（次点Docling 0.753176・YomiToku 0.751770。
  クラウドOCRの解像度処理が有利）。英語二段組 0.975102、日本語横書き 0.973221、埋込みテキスト
  0.972940でYomiTokuと並び首位。処理時間・メモリともに全エンジン中最軽量（HTTP呼び出しのみで
  ローカルモデル推論が不要なため）。
- **Mistral OCRの弱み**: 表・数式 0.911921でDoclingの0.970144に明確に劣る（表を含む一部の
  cellでtext_accuracyが下がる）。`no_outline`は0.724829でNDLOCR-Lite・Docling・YomiTokuと僅差
  最下位（差0.01程度、1資料のみの評価なので確定的ではない）。
- Mistral OCRの応答は`pages[].blocks[]`にbbox（`top_left_x/y`, `bottom_right_x/y`）と
  type（header/footer/title/text/caption/table/image）を持つことを実機で確認した
  （`src/mistral_ocr_extract.py`はこの構造を直接消費する。事前のmarkdown-onlyという想定は誤りだった
  ため実装済みコードを実測結果に合わせて書き直した）。`confidence_scores`は今回のサンプルでは
  常にnullだった。
- YomiTokuは日本語縦書き 0.666019・注参考文献 0.589565でDoclingを明確に上回る
  （Mistral OCRとほぼ同点）。表・数式は0.487921でDoclingに劣る。
- PyMuPDFは最軽量（0.02秒・56MB）だが表構造・見出し検出の欠如でスコアは最も低い。

詳細値は `comparison.json`（v2）と `comparison.md` を正本とする。YomiToku/Mistral OCR行の実測は、
元の`OCR_BAKEOFF_*_PDF`環境変数がセッション限りで消えていたため、`run_ocr_bakeoff.py`が最初の
実行時に生成した1ページ抽出済みキャッシュを全エンジン共通の入力として再利用した。

## 提案ルーティング

| 入力状態 | engine | 採用条件 |
|---|---|---|
| EPUB / HTML | DOM extractor | 現行V3経路を維持 |
| PDF embedded text・単純layout | PyMuPDF preflight | 文字coverage、gibberish、reading-order、structure gateがすべて合格した場合だけ正本候補 |
| PDF embedded text・複雑layout | Docling | 二段組、表・数式、outlineなし、見出し復元が必要な資料 |
| PDF partial / scanned | Docling既定、高重要度はMistral OCR | Mistral OCRがスキャンで明確に優位（0.973 vs 0.753）。no-cloud資料はDoclingのみ |
| 日本語横書き | Docling | pilot合格。通常gateを適用 |
| 日本語縦書き | YomiToku（lite/CPU）または Mistral OCR、Doclingと併記 | 自動採用禁止。改善はしたが合格水準未達のため`degraded`として再OCR・目視候補へ送る |
| 注・参考文献が密な資料 | YomiToku opt-in（ローカル優先）、Mistral OCRも同水準 | heading_path/zoneの復元がDocling比で優れることを確認済み。個別評価件数が1件のためgate通過を条件にopt-in |
| 古典籍・くずし字等 | NDLOCR-Lite | 専用sampleでDoclingを上回ることを確認した資料種別だけopt-in |
| ローカル抽出器の構造化が明確に失敗 | Mistral OCR（クラウド） | 任意fallback。実測済み・8カテゴリ平均で最高スコア。no-cloud資料には使わない |
| no-cloud | 上記ローカル経路のみ（Mistral OCR除外） | クラウドへ送信しない |

## 採用後に実装する設定

1. `EngineRegistry.select()`を入力品質とlayout特性による明示的routerへ置換する。
2. PyMuPDF preflightの合格条件を決定的quality gateとして固定する。
3. Docling失敗時はPyMuPDFへ黙って降格せず、`degraded`とfallback理由をartifact ledgerへ記録する。
4. 日本語縦書き・注参考文献はYomiTokuをopt-in候補として`EngineRegistry`に登録済み
   （`YomitokuEngine`、`src/yomitoku_extract.py`）。自動採用は禁止のまま、8カテゴリを
   各3資料以上へ拡張した再測定で安定して優位が確認できてから既定へ昇格を検討する。
5. 8カテゴリを各3資料以上へ拡張してもルーティングを再現できるか、Phase 3の目視監査と併せて確認する。
6. YomiToku・Mistral OCRの表・数式スコアが低い原因（表セルの文字列化方法）を切り分け、
   `src/yomitoku_extract._table_text` / `src/mistral_ocr_extract._page_blocks_from_response`の
   table実装に改善余地があるか見直す。
7. Mistral OCRを`MistralOCREngine`（`src/mistral_ocr_extract.py`）として登録済み・**実測完了**
   （2026-07-21）。cloud=Trueかつ`MISTRAL_OCR_API_KEY`設定＋`OCR_BAKEOFF_ALLOW_CLOUD=1`の明示
   opt-inがない限りベイクオフでも`unavailable`のままとし、通常実行で黙って送信されない。
   応答schema（`pages[].blocks[]`にbbox+type、markdownは補助）は実機で検証済み。8カテゴリ平均で
   最高スコアだが、**クラウド送信そのものは既定経路にしない**——資料単位の送信許可・no-cloud除外・
   高重要度資料は人が採用判定、という原則（§今回の採用範囲外）は維持する。
8. `src/ocr_bakeoff.py`のスコアラー修正（対応付けのmarkdown除去・partial-match化）に伴う回帰
   テストを`tests/test_ocr_bakeoff.py`へ追加済み。今後のraw再採点は必ず修正後のコードで行う。

## 今回の採用範囲外

- DeepSeek-OCRは実機検証（Apple Silicon）の結果、処理負荷が実用に見合わず候補から除外した（2026-07-21判断）。以後の比較対象に含めない。
- Google Document AI / Azure Document Intelligence Layoutは未着手。
- クラウドとローカルの結果が不一致でも自動上書きせず、高重要度資料は人が採用結果を選ぶ
  という原則（`64_ingestion_redesign_audit_and_plan.md` §4.5-5）を維持する。Mistral OCRが
  スコア最高でも、既定エンジンをクラウドへ切り替えることは意味しない。
- 1カテゴリ1資料の結果だけで、日本語縦書きや古典籍の品質保証は行わない（YomiToku/Mistral OCR
  追加後も同様）。
