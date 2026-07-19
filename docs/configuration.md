# 環境設定

[READMEへ戻る](../README.md)

通常はセットアップウィザードで `.env` を管理します。

```bash
uv run scripts/setup_wizard.py
uv run scripts/setup_wizard.py --status
```

手動設定のひな形は [`.env.example`](../.env.example) です。`.env`と`.env.policy`はGitに追加されません。

## Core

| 変数 | 用途 | 既定 |
|---|---|---|
| `FEATURE_LEVEL` | ウィザードの設定段階を示す管理用マーカー | `core` |
| `ZOTERO_DATA_DIR` | Zoteroデータフォルダ | `~/Zotero` |
| `CHROMA_DIR` | ベクトル索引 | `data/chroma` |
| `EMB_PROFILE` | `fast`または`bge` | `fast` |
| `EMB_MODEL` | 埋め込みモデルのパスまたはID | プロファイルから選択 |
| `EMB_DEVICE` | `cpu`、`mps`、`cuda` | 環境から選択 |
| `HF_HUB_OFFLINE` | `1`でHugging Faceへの接続を禁止 | 任意 |

`FEATURE_LEVEL`は管理用で、セキュリティ境界ではありません。

## Citation Network

| 変数 | 用途 |
|---|---|
| `S2_API_KEY` | Semantic Scholar APIキー。Citation Networkでは必須 |
| `ZOTERO_USER_ID` + `ZOTERO_API_KEY` | 解決したDOIのZotero書き戻し。任意 |
| `CINII_APP_ID` | CiNii Research v2。任意 |
| `CROSSREF_MAILTO` | Crossref polite pool。任意 |

## LLM

| 変数 | 用途 |
|---|---|
| `LLM_CHEAP` | 大量処理（要約、クエリ拡張） |
| `LLM_STANDARD` | 通常処理と一次フォールバック |
| `LLM_REVIEW` | 品質確認と最終フォールバック |
| `DEEPSEEK_API_KEY` | DeepSeek API |
| `LLM_OPENAI_BASE_URL` | OpenAI互換サーバー |

送信制御は [LLMとプライバシー](llm-and-privacy.md) を参照してください。

## メンテナンス時のAI要約

```dotenv
SUMMARY_BATCH_MAX_ITEMS=20
SUMMARY_BATCH_WORKERS=10
CASE_BATCH_MAX_ITEMS=10
CASE_SELECTOR_SAMPLES=2
CASE_BATCH_WORKERS=10
CASE_MAX_PER_SECTION=5
```

`Maintenance-Widget.command` の要約更新は、ローカル抽出型要約、DeepSeek AI要約、構造化事例の順に差分実行します。事例候補はFlashによる複数回の和集合から集め、Proで順位・状態を判定します。根拠のある候補は `confirmed`、`partial`、`candidate` として段階保存されます。定時スケジューラやCodex利用枠の確認は行いません。
