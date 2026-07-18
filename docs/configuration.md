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
| `LLM_DEFAULT` | 共通モデル |
| `LLM_EXPAND` | クエリ拡張モデル |
| `LLM_SUMMARY` | 要約モデル |
| `LLM_EXTRACT` | 参考文献抽出モデル |
| `DEEPSEEK_API_KEY` | DeepSeek API |
| `LLM_OPENAI_BASE_URL` | OpenAI互換サーバー |

送信制御は [LLMとプライバシー](llm-and-privacy.md) を参照してください。

## 夜間実行

```dotenv
NIGHTLY_ENABLE=1
NIGHTLY_START_TIME=03:30
NIGHTLY_LAUNCH_MODE=terminal
NIGHTLY_MAX_HOURS=5
NIGHTLY_MAX_ITEMS=20
NIGHTLY_MIN_WEEKLY_REMAINING_PERCENT=20
```

macOSへ登録・確認:

```bash
scripts/install_nightly_launchd.sh
scripts/install_nightly_launchd.sh --check
```

Documents配下などmacOSの保護対象にある場合は `NIGHTLY_LAUNCH_MODE=terminal` を使います。
