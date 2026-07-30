#!/usr/bin/env python3
import argparse
import json
import getpass
import importlib.util
import os
import platform
from contextlib import contextmanager
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import db_lifecycle
from src.v3_data_plane import (
    V3_COLLECTION, V3_LEXICAL_NAME, V3_MANIFEST_NAME, resolve_configured_path,
)
from src.embedder import ensure_embedding_model

ENV_GROUPS = (
    ("基本設定: ローカル検索と索引", (
        "FEATURE_LEVEL", "ZOTERO_DATA_DIR", "CHROMA_DIR", "EMB_PROFILE",
        "EMB_MODEL", "EMB_DEVICE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
        "INGEST_STRUCTURED_V3_ENABLE", "HIERARCHICAL_SEARCH_V2_ENABLE",
        "CHROMA_COLLECTION", "MANIFEST_PATH", "LEXICAL_DB_PATH",
    )),
    ("引用ネットワークと書誌情報", (
        "S2_API_KEY", "ZOTERO_USER_ID", "ZOTERO_API_KEY", "CINII_APP_ID",
        "CROSSREF_MAILTO",
    )),
    ("LLM支援機能", (
        "LLM_CHEAP", "LLM_STANDARD", "LLM_REVIEW",
        "DEEPSEEK_API_KEY", "DEEPSEEK_THINKING", "DEEPSEEK_REASONING_EFFORT",
        "ANTHROPIC_API_KEY", "LLM_OPENAI_BASE_URL", "LLM_OPENAI_API_KEY",
        "SUMMARY_EXCLUDE_TAGS", "EXTRACT_EXCLUDE_TAGS",
        "SUMMARY_ALLOW_CLOUD_ALL", "EXTRACT_ALLOW_CLOUD_ALL",
        "SUMMARY_BATCH_MAX_ITEMS", "SUMMARY_BATCH_WORKERS",
    )),
    ("ローカルOCR", (
        "NDLOCR_BIN", "NDLOCR_DPI", "NDLOCR_TIMEOUT_SEC",
    )),
)


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if "=" in stripped and not stripped.startswith("#"):
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def write_env_file(path: Path, values: dict[str, str]) -> None:
    """Write a predictable, grouped .env while preserving unknown settings."""
    rendered = [
        "# scripts/setup_wizard.pyにより生成されました。",
        "# 秘密情報を含む.envはGitの追跡対象外です。",
    ]
    written: set[str] = set()
    for title, keys in ENV_GROUPS:
        rows = [f"{key}={values[key]}" for key in keys if values.get(key, "") != ""]
        if rows:
            rendered.extend(["", f"# {title}", *rows])
            written.update(key for key in keys if values.get(key, "") != "")
    extras = sorted(
        key for key, value in values.items() if key not in written and value != ""
    )
    if extras:
        rendered.extend(["", "# その他の既存設定"])
        rendered.extend(f"{key}={values[key]}" for key in extras)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write("\n".join(rendered) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _set_optional_secret(config: dict[str, str], key: str, prompt: str) -> None:
    current = bool(config.get(key))
    suffix = " [設定済み・Enterで維持・'-'で削除]" if current else " [任意]"
    print("  入力したAPIキーは画面に表示されません。")
    value = getpass.getpass(prompt + suffix + ": ").strip()
    if value == "-":
        config.pop(key, None)
    elif value:
        config[key] = value


def _set_required_secret(config: dict[str, str], key: str, prompt: str) -> None:
    """Require a secret while allowing Enter to retain an existing value."""
    print("  入力したAPIキーは画面に表示されません。")
    while True:
        current = bool(config.get(key))
        suffix = " [設定済み・Enterで維持]" if current else " [必須]"
        value = getpass.getpass(prompt + suffix + ": ").strip()
        if value:
            config[key] = value
            return
        if current:
            return
        print(f"{key}は選択した機能に必要です。値を入力してください。")


#: Minimal is the only bundled baseline. Custom edits these independent values
#: directly; FEATURE_LEVEL is descriptive and never a runtime feature gate.
V3_DATA_PLANE = {
    "INGEST_STRUCTURED_V3_ENABLE": "1",
    "HIERARCHICAL_SEARCH_V2_ENABLE": "1",
    "CHROMA_COLLECTION": V3_COLLECTION,
    "MANIFEST_PATH": f"data/{V3_MANIFEST_NAME}",
    "LEXICAL_DB_PATH": f"data/{V3_LEXICAL_NAME}",
}


PRESETS: dict[str, dict[str, str]] = {
    "minimal": {
        **V3_DATA_PLANE,
        "FEATURE_LEVEL": "core",
        "PDF_STRUCTURE_RECOVERY_ENABLE": "0",
        "PDF_STRUCTURE_ENGINE_SHORT": "docling",
        "PDF_STRUCTURE_ENGINE_LONG": "docling",
        "PDF_STRUCTURE_ENGINE_PAGE_BOUNDARY": "30",
        "PDF_AI_TOC_FAST_PATH_ENABLE": "0",
        "OCR_LAYER_AUDIT_ENABLE": "0",
        "QUERY_EXPANSION_ENABLE": "0",
        "LLM_SUMMARIES_ENABLE": "0",
        "LLM_REFERENCE_EXTRACTION_ENABLE": "0",
        "PDF_MISTRAL_TOC_QUEUE_ENABLE": "0",
        "CITATION_NETWORK_ENABLE": "0",
    },
}

LLM_FLAGS = (
    "PDF_AI_TOC_FAST_PATH_ENABLE", "OCR_LAYER_AUDIT_ENABLE", "QUERY_EXPANSION_ENABLE",
    "LLM_SUMMARIES_ENABLE", "LLM_REFERENCE_EXTRACTION_ENABLE",
)

GRANITE_VENV_PYTHON = "tmp/granite_docling_venv/bin/python"
GRANITE_REQUIREMENTS = ("docling==2.102.1", "mlx-vlm==0.6.6")
NDLOCR_REQUIREMENT = (
    "git+https://github.com/ndl-lab/ndlocr-lite.git@1.0.0"
)
DOCLING_EXTRA = "pdf-docling"


def describe_preset(config: dict[str, str]) -> str:
    """Distinguish the zero-cost minimal baseline from an edited setup."""
    values = PRESETS["minimal"]
    return (
        "minimal"
        if all(config.get(key) == value for key, value in values.items())
        else "custom"
    )


def apply_preset(config: dict[str, str], name: str) -> None:
    config.update(PRESETS[name])


def _ask_yes_no(prompt: str, *, current: bool = False) -> bool:
    suffix = "[Y/n]" if current else "[y/N]"
    answer = input(f"{prompt} {suffix}: ").strip().casefold()
    if not answer:
        return current
    return answer in {"y", "yes"}


def enforce_v3_configuration(config: dict[str, str]) -> None:
    """Migrate edited configuration to the sole supported production data plane."""
    config.update(V3_DATA_PLANE)


def _granite_selectable() -> bool:
    try:
        from src.feature_gates import granite_configured
    except ImportError:  # pragma: no cover - src not importable in odd layouts
        return False
    return granite_configured()


def _granite_environment_ready(python_path: Path) -> bool:
    if not python_path.exists():
        return False
    check = (
        "from importlib.util import find_spec;"
        "raise SystemExit(0 if find_spec('docling') and find_spec('mlx_vlm') else 1)"
    )
    try:
        completed = subprocess.run(
            [str(python_path), "-c", check],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def install_granite_environment(
    config: dict[str, str], *, python_path: Path | None = None,
) -> bool:
    """Create the isolated Granite environment selected by the operator."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print("[!] GraniteはApple Silicon搭載Macでのみ利用できます。")
        return False
    uv_path = shutil.which("uv")
    if not uv_path:
        print("[!] uvが見つからないためGranite専用環境を作成できません。")
        return False

    interpreter = python_path or ROOT / GRANITE_VENV_PYTHON
    if not interpreter.is_absolute():
        interpreter = ROOT / interpreter
    # Do not call resolve(): a venv's python is commonly a symlink to the base
    # interpreter, and resolving it would discard the venv's site-packages.
    interpreter = interpreter.absolute()
    environment = interpreter.parent.parent
    print("\nGranite専用環境を準備します。初回はモデル関連パッケージの取得に時間がかかります。")
    commands: list[tuple[str, list[str]]] = []
    if not interpreter.exists():
        commands.append((
            "専用virtualenvを作成",
            [uv_path, "venv", str(environment), "--python", "3.10", "--clear"],
        ))
    commands.extend(
        (
            f"{requirement}をインストール",
            [uv_path, "pip", "install", "--python", str(interpreter), requirement],
        )
        for requirement in GRANITE_REQUIREMENTS
    )
    for label, command in commands:
        print(f"  - {label}しています...")
        try:
            completed = subprocess.run(command, cwd=ROOT)
        except OSError as exc:
            print(f"[!] Granite専用環境の準備に失敗しました: {exc}")
            return False
        if completed.returncode != 0:
            print(f"[!] {label}に失敗しました（終了コード: {completed.returncode}）。")
            return False

    if not _granite_environment_ready(interpreter):
        print("[!] Granite専用環境を検証できませんでした。Doclingへ戻します。")
        return False
    try:
        configured_path = str(interpreter.relative_to(ROOT))
    except ValueError:
        configured_path = str(interpreter)
    config["GRANITE_VENV_PYTHON"] = configured_path
    print("[+] Granite専用環境の準備が完了しました。")
    return True


def _docling_ready() -> bool:
    """Return whether Docling can be imported by the setup environment."""
    return importlib.util.find_spec("docling") is not None


def install_docling() -> bool:
    """Install Docling into the project's uv environment and verify it."""
    if _docling_ready():
        return True
    uv_path = shutil.which("uv")
    if not uv_path:
        print("[!] uvが見つからないためDoclingをインストールできません。")
        return False
    print("\nDoclingをプロジェクト環境へインストールします。初回は時間がかかります。")
    try:
        completed = subprocess.run(
            [uv_path, "sync", "--extra", DOCLING_EXTRA], cwd=ROOT,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[!] Doclingのインストールに失敗しました: {exc}")
        return False
    if completed.returncode != 0:
        print(f"[!] Doclingのインストールに失敗しました（終了コード: {completed.returncode}）。")
        return False
    if not _docling_ready():
        print("[!] Doclingをインポートできません。")
        return False
    print("[+] Doclingのインストールが完了しました。")
    return True


def _find_ndlocr(config: dict[str, str]) -> Path | None:
    configured = str(config.get("NDLOCR_BIN") or "").strip()
    if configured:
        path = Path(configured).expanduser()
        if path.exists():
            return path.absolute()
    discovered = shutil.which("ndlocr-lite")
    return Path(discovered).absolute() if discovered else None


def _ndlocr_executable_ready(executable: Path) -> bool:
    if not executable.exists():
        return False
    try:
        completed = subprocess.run(
            [str(executable), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def install_ndlocr(config: dict[str, str]) -> bool:
    """Install NDLOCR-Lite as an isolated uv tool and record its executable."""
    uv_path = shutil.which("uv")
    if not uv_path:
        print("[!] uvが見つからないためNDLOCR-Liteをインストールできません。")
        return False
    print("\nNDLOCR-Liteを専用環境へインストールします（使用容量の目安: 約450MB）。")
    try:
        installed = subprocess.run(
            [uv_path, "tool", "install", "--force", NDLOCR_REQUIREMENT],
            cwd=ROOT,
        )
    except OSError as exc:
        print(f"[!] NDLOCR-Liteのインストールに失敗しました: {exc}")
        return False
    if installed.returncode != 0:
        print(
            "[!] NDLOCR-Liteのインストールに失敗しました"
            f"（終了コード: {installed.returncode}）。"
            "GitHubへの接続を確認してください。"
        )
        return False

    try:
        bin_result = subprocess.run(
            [uv_path, "tool", "dir", "--bin"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        bin_result = None
    candidates: list[Path] = []
    if bin_result is not None and bin_result.returncode == 0 and bin_result.stdout.strip():
        candidates.append(Path(bin_result.stdout.strip()) / "ndlocr-lite")
    discovered = shutil.which("ndlocr-lite")
    if discovered:
        candidates.append(Path(discovered))
    executable = next(
        (path.absolute() for path in candidates if _ndlocr_executable_ready(path)),
        None,
    )
    if executable is None:
        print("[!] NDLOCR-Liteの実行ファイルを検証できませんでした。")
        return False
    config["NDLOCR_BIN"] = str(executable)
    print(f"[+] NDLOCR-Liteを利用できます: {executable}")
    return True


def configure_ndlocr(config: dict[str, str]) -> None:
    """Detect or optionally install the local Japanese OCR tool."""
    print("\n日本語ローカルOCR")
    executable = _find_ndlocr(config)
    if executable is not None and _ndlocr_executable_ready(executable):
        config["NDLOCR_BIN"] = str(executable)
        print(f"  ✅ NDLOCR-Liteを検出しました: {executable}")
        return
    config.pop("NDLOCR_BIN", None)
    print("  NDLOCR-Liteは日本語の縦書き・旧字体を含む画像資料の再OCRに使用します。")
    print("  無料・ローカル処理で、インストール後の使用容量は約450MBです。")
    if _ask_yes_no("NDLOCR-Liteをインストールしますか？", current=True):
        if not install_ndlocr(config):
            print("[案内] NDLOCR-Liteなしで続行します。必要ならSetup.commandを再実行してください。")
    else:
        print("[案内] NDLOCR-Liteのインストールをスキップしました。")


def _find_tesseract() -> Path | None:
    discovered = shutil.which("tesseract")
    if discovered:
        return Path(discovered).absolute()
    for candidate in (
        Path("/opt/homebrew/bin/tesseract"),
        Path("/usr/local/bin/tesseract"),
    ):
        if candidate.exists():
            return candidate
    return None


def _tesseract_languages(executable: Path) -> set[str]:
    try:
        completed = subprocess.run(
            [str(executable), "--list-langs"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    if completed.returncode != 0:
        return set()
    return {
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip() and not line.startswith("List")
    }


def _find_homebrew() -> Path | None:
    discovered = shutil.which("brew")
    if discovered:
        return Path(discovered).absolute()
    for candidate in (
        Path("/opt/homebrew/bin/brew"),
        Path("/usr/local/bin/brew"),
    ):
        if candidate.exists():
            return candidate
    return None


def install_tesseract(*, language_data_only: bool = False) -> bool:
    """Install Tesseract and Japanese language data through Homebrew."""
    brew = _find_homebrew()
    if brew is None:
        print("[!] Homebrewが見つからないためTesseractを自動インストールできません。")
        return False
    packages = ["tesseract-lang"] if language_data_only else [
        "tesseract", "tesseract-lang",
    ]
    print(f"\nHomebrewで{'・'.join(packages)}をインストールします。")
    try:
        completed = subprocess.run(
            [str(brew), "install", *packages],
            cwd=ROOT,
        )
    except OSError as exc:
        print(f"[!] Tesseractのインストールに失敗しました: {exc}")
        return False
    if completed.returncode != 0:
        print(
            "[!] Tesseractのインストールに失敗しました"
            f"（終了コード: {completed.returncode}）。"
        )
        return False
    executable = _find_tesseract()
    languages = _tesseract_languages(executable) if executable else set()
    if executable is None or "jpn" not in languages:
        print("[!] Tesseract本体または日本語言語データを検証できませんでした。")
        return False
    print(f"[+] 日本語対応のTesseractを利用できます: {executable}")
    return True


def configure_tesseract(*, allow_install: bool) -> None:
    """Detect Tesseract and optionally install it for a Custom setup."""
    print("\n" + "=" * 60)
    print("5. Tesseract OCR（任意・フォント復号に失敗したPDFページの補助）")
    executable = _find_tesseract()
    languages = _tesseract_languages(executable) if executable else set()
    if executable is not None and "jpn" in languages:
        print("   ✅ 日本語対応のTesseractを検出しました。")
        print("      文字化けした日本語PDFページのOCRフォールバックを利用できます。")
        return
    if executable is not None:
        print("   ⚠️  Tesseractはありますが日本語言語データがありません。")
        if allow_install and _ask_yes_no(
            "Homebrewで日本語言語データをインストールしますか？",
            current=True,
        ):
            if install_tesseract(language_data_only=True):
                return
        print("      日本語ページは英語OCRとなり、文字化けする可能性があります。")
        print("      手動インストール: brew install tesseract-lang")
        return
    print("   ℹ️  Tesseractが見つかりません。フォント復号失敗時の補助OCRは利用できません。")
    if allow_install and _ask_yes_no(
        "HomebrewでTesseractと日本語言語データをインストールしますか？",
        current=True,
    ):
        if install_tesseract():
            return
    print("      手動インストール: brew install tesseract tesseract-lang")


def _choose_engine(label: str, current: str) -> str:
    """Choice (C) for one size bucket.

    The three options are presented with their trade-offs rather than a
    recommendation, because which one is right depends on what the library
    holds and whether the owner will pay per page.
    """
    granite_description = "Granite — 無料・約2.3倍低速・総合精度が高い"
    if not _granite_selectable():
        granite_description += "（初回選択時に専用環境を導入）"
    options = [
        ("docling", "Docling — 無料・高速・表と数式に強い"),
        ("granite", granite_description),
        ("mistral", "Mistral OCR — ページ単位課金・高速・長いスキャンに強い"),
    ]
    print(f"\n  {label}")
    for number, (_name, description) in enumerate(options, start=1):
        print(f"     [{number}] {description}")
    names = [name for name, _description in options]
    default = str(names.index(current) + 1) if current in names else "1"
    choice = input(f"  選択 [1-{len(options)}、既定 {default}]: ").strip() or default
    try:
        return options[int(choice) - 1][0]
    except (ValueError, IndexError):
        return options[0][0]


def configure_pdf_engines(config: dict[str, str]) -> None:
    boundary = config.get("PDF_STRUCTURE_ENGINE_PAGE_BOUNDARY", "30")
    entered = input(f"\n  PDFエンジンを切り替えるページ数 [{boundary}]: ").strip()
    if entered.isdigit() and int(entered) > 0:
        boundary = entered
    config["PDF_STRUCTURE_ENGINE_PAGE_BOUNDARY"] = boundary
    print(f"\nPDF構造化に使うエンジンを選択してください（{boundary}ページで区切ります）")
    config["PDF_STRUCTURE_ENGINE_SHORT"] = _choose_engine(
        f"{boundary}ページ未満:", config.get("PDF_STRUCTURE_ENGINE_SHORT", "docling"),
    )
    config["PDF_STRUCTURE_ENGINE_LONG"] = _choose_engine(
        f"{boundary}ページ以上:", config.get("PDF_STRUCTURE_ENGINE_LONG", "docling"),
    )
    # Granite-Docling has its own isolated environment, but the ingestion
    # router deliberately falls back to the regular Docling worker if Granite
    # fails.  Install the fallback as well so that selecting Granite does not
    # produce a misleading "Docling is not installed" warning later.
    uses_docling = bool({
        config["PDF_STRUCTURE_ENGINE_SHORT"], config["PDF_STRUCTURE_ENGINE_LONG"],
    } & {"docling", "granite"})
    if uses_docling and not _docling_ready():
        print("\nDoclingが現在の環境にインストールされていません。")
        if "granite" in {
            config["PDF_STRUCTURE_ENGINE_SHORT"], config["PDF_STRUCTURE_ENGINE_LONG"],
        }:
            print("   Granite失敗時のフォールバックにもDoclingを使用します。")
        install_now = _ask_yes_no(
            "今すぐDoclingをプロジェクト環境へインストールしますか？",
            current=True,
        )
        if not install_now or not install_docling():
            raise SystemExit(
                "Doclingを選択したため、セットアップを中止しました。"
                "インストール後にウィザードを再実行するか、別のPDFエンジンを選択してください。"
            )
    uses_granite = "granite" in {
        config["PDF_STRUCTURE_ENGINE_SHORT"], config["PDF_STRUCTURE_ENGINE_LONG"],
    }
    if uses_granite and not _granite_selectable():
        print("\nGraniteを使うには専用環境の初回導入が必要です。")
        install_now = _ask_yes_no(
            "今すぐGranite専用環境をインストールしますか？",
            current=True,
        )
        installed = install_now and install_granite_environment(config)
        if not installed:
            for key in ("PDF_STRUCTURE_ENGINE_SHORT", "PDF_STRUCTURE_ENGINE_LONG"):
                if config[key] == "granite":
                    config[key] = "docling"
            print("[案内] Graniteを選んだ区分はDoclingへ戻しました。")
    uses_mistral = "mistral" in {
        config["PDF_STRUCTURE_ENGINE_SHORT"], config["PDF_STRUCTURE_ENGINE_LONG"],
    }
    config["PDF_MISTRAL_TOC_QUEUE_ENABLE"] = "1" if uses_mistral else "0"
    if uses_mistral:
        _set_required_secret(config, "MISTRAL_OCR_API_KEY", "Mistral OCR APIキー")


def _current_llm_provider_choice(config: dict[str, str]) -> str:
    specs = " ".join(
        str(config.get(role) or "").casefold()
        for role in ("LLM_CHEAP", "LLM_STANDARD", "LLM_REVIEW")
    )
    if "codex_cli:" in specs:
        return "2"
    if "claude_cli:" in specs:
        return "3"
    if "openai_compat:" in specs:
        return "4"
    return "1"


def configure_llm_provider(config: dict[str, str]) -> None:
    print("\n既定のLLMプロバイダーを選択してください:")
    print("   [1] DeepSeek API: Flash / Pro / Pro（推奨）")
    print("   [2] Codex CLI（ローカルのCodexログインと利用枠を使用）")
    print("   [3] Claude CLI（ローカルのClaudeログインと利用枠を使用）")
    print("   [4] OpenAI互換サーバー（Ollama、LM Studio、vLLMなど）")
    default = _current_llm_provider_choice(config)
    provider = input(f"選択 [1-4、既定 {default}]: ").strip() or default
    for legacy_key in ("LLM_DEFAULT", "LLM_EXPAND", "LLM_SUMMARY", "LLM_EXTRACT"):
        config.pop(legacy_key, None)
    if provider == "2":
        config.update(dict.fromkeys(
            ("LLM_CHEAP", "LLM_STANDARD", "LLM_REVIEW"), "codex_cli:auto"))
    elif provider == "3":
        config.update(dict.fromkeys(
            ("LLM_CHEAP", "LLM_STANDARD", "LLM_REVIEW"), "claude_cli:auto"))
    elif provider == "4":
        config.update(dict.fromkeys(
            ("LLM_CHEAP", "LLM_STANDARD", "LLM_REVIEW"), "openai_compat:local"))
        base = input("OpenAI互換ベースURL [http://localhost:11434/v1]: ").strip()
        config["LLM_OPENAI_BASE_URL"] = base or "http://localhost:11434/v1"
        _set_optional_secret(config, "LLM_OPENAI_API_KEY", "必要な場合のAPIキー")
    else:
        config.update({
            "LLM_CHEAP": "deepseek:deepseek-v4-flash",
            "LLM_STANDARD": "deepseek:deepseek-v4-pro",
            "LLM_REVIEW": "deepseek:deepseek-v4-pro",
        })
        _set_required_secret(config, "DEEPSEEK_API_KEY", "DeepSeek APIキー")


LLM_FEATURE_CHOICES = (
    (
        "PDF_AI_TOC_FAST_PATH_ENABLE",
        "対象PDFの目次をLLMで推定しますか？",
        True,
    ),
    (
        "OCR_LAYER_AUDIT_ENABLE",
        "PDFのOCRテキスト層をLLMで品質監査しますか？",
        True,
    ),
    (
        "QUERY_EXPANSION_ENABLE",
        "検索時にLLMクエリ拡張を使いますか？",
        False,
    ),
    (
        "LLM_SUMMARIES_ENABLE",
        "メンテナンスバッチで階層要約を生成しますか？",
        False,
    ),
    (
        "LLM_REFERENCE_EXTRACTION_ENABLE",
        "参考文献の抽出にLLMを使いますか？",
        False,
    ),
)


def configure_custom_features(config: dict[str, str]) -> None:
    """Configure independent feature axes without imposing a bundled tier."""
    for key, value in PRESETS["minimal"].items():
        config.setdefault(key, value)
    config["FEATURE_LEVEL"] = "custom"

    print("\n引用ネットワーク")
    citation_enabled = _ask_yes_no(
        "Semantic Scholarの引用ネットワークを有効にしますか？",
        current=config.get("CITATION_NETWORK_ENABLE") == "1",
    )
    config["CITATION_NETWORK_ENABLE"] = "1" if citation_enabled else "0"
    if citation_enabled:
        _set_required_secret(config, "S2_API_KEY", "Semantic Scholar APIキー")

    print("\nPDF構造化")
    structure_enabled = _ask_yes_no(
        "PDFから見出しと文書構造を復元しますか？",
        current=config.get("PDF_STRUCTURE_RECOVERY_ENABLE") == "1",
    )
    config["PDF_STRUCTURE_RECOVERY_ENABLE"] = "1" if structure_enabled else "0"
    if structure_enabled:
        configure_pdf_engines(config)
    else:
        config.update({
            "PDF_STRUCTURE_ENGINE_SHORT": "docling",
            "PDF_STRUCTURE_ENGINE_LONG": "docling",
            "PDF_MISTRAL_TOC_QUEUE_ENABLE": "0",
        })

    print("\nLLM支援機能")
    any_llm = False
    for flag, prompt, needs_pdf_structure in LLM_FEATURE_CHOICES:
        if needs_pdf_structure and not structure_enabled:
            config[flag] = "0"
            continue
        enabled = _ask_yes_no(prompt, current=config.get(flag) == "1")
        config[flag] = "1" if enabled else "0"
        any_llm = any_llm or enabled
    if any_llm:
        configure_llm_provider(config)


def configure_feature_level(config: dict[str, str]) -> None:
    """Choose the minimal baseline or configure every feature axis."""
    # Settings removed on 2026-07-27: an item-level cloud veto could not hold,
    # since anything indexed reaches the assistant through search.
    for key in (
        "SUMMARY_ALLOW_CLOUD_ALL", "EXTRACT_ALLOW_CLOUD_ALL",
        "SUMMARY_EXCLUDE_TAGS", "EXTRACT_EXCLUDE_TAGS",
        "OCR_FALLBACK_ALLOW_CLOUD_ALL", "OCR_FALLBACK_EXCLUDE_TAGS",
        "MISTRAL_OCR_FALLBACK_ENABLE",
    ):
        config.pop(key, None)

    existing = describe_preset(config)
    print("\n3. この環境のセットアップ方式を選択してください")
    print("   [1] Minimal（最小）— 追加取得・課金なし。PDFは平文として索引化")
    print("   [2] Custom（カスタム）— 引用ネットワーク、PDFエンジン、各LLM機能を個別設定")
    can_keep = "FEATURE_LEVEL" in config or any(flag in config for flag in LLM_FLAGS)
    if can_keep:
        print(f"   [0] 現在の設定を維持（現在: {existing}）")
    default = "0" if can_keep else "1"
    choice = input(f"選択 [0-2、既定 {default}]: ").strip() or default
    if choice == "0":
        return
    if choice == "2":
        configure_custom_features(config)
    else:
        apply_preset(config, "minimal")


def print_configuration_status(config: dict[str, str]) -> None:
    level = config.get("FEATURE_LEVEL", "旧形式/カスタム")
    print("Zotero Local RAG 設定状態")
    print(f"  設定方式       : {level}")
    print(f"  Zoteroデータ   : {config.get('ZOTERO_DATA_DIR', '~/Zotero')}")
    print(f"  埋め込み       : {config.get('EMB_PROFILE', 'fast')}")
    print(f"  S2 APIキー     : {'設定済み' if config.get('S2_API_KEY') else '未設定'}")
    print(f"  軽量LLM        : {config.get('LLM_CHEAP', '未設定')}")
    print(f"  標準LLM        : {config.get('LLM_STANDARD', '未設定')}")
    print(f"  レビューLLM    : {config.get('LLM_REVIEW', '未設定')}")
    preset = "最小" if describe_preset(config) == "minimal" else "カスタム"
    print(f"  設定種別       : {preset}")
    print(f"  データ面       : {config.get('CHROMA_COLLECTION', '未設定')}")
    if config.get("PDF_STRUCTURE_RECOVERY_ENABLE") == "1":
        boundary = config.get("PDF_STRUCTURE_ENGINE_PAGE_BOUNDARY", "30")
        print(
            f"  PDF構造化      : {boundary}ページ未満 "
            f"{config.get('PDF_STRUCTURE_ENGINE_SHORT', 'docling')} / "
            f"{boundary}ページ以上 {config.get('PDF_STRUCTURE_ENGINE_LONG', 'docling')}"
        )
    else:
        print("  PDF構造化      : 無効（平文として索引化）")
    enabled = [flag for flag in LLM_FLAGS if config.get(flag) == "1"]
    print(f"  LLM機能        : {len(enabled)}/{len(LLM_FLAGS)}件有効")
    # Report a feature switched on without its resource here too, so the
    # mismatch is visible at setup rather than only when a run refuses to start.
    for problem in _configuration_problems(config):
        print(f"  [!] {problem}")


def _configuration_problems(config: dict[str, str]) -> list[str]:
    """Same check the pipeline runs at startup, evaluated against ``config``.

    The .env loader is suspended for the duration: the wizard is reporting on
    the configuration being *edited*, and letting the file on disk leak in
    would show a key the user has not saved yet -- or hide one they just
    removed.
    """
    problems = _v3_configuration_problems(config)
    try:
        from src import feature_gates
    except ImportError:  # pragma: no cover - src not importable in odd layouts
        return problems
    original_loader = feature_gates.load_dotenv_native
    feature_gates.load_dotenv_native = lambda *_a, **_k: None
    try:
        with mock_environment(config):
            return problems + feature_gates.verify_enabled_features()
    finally:
        feature_gates.load_dotenv_native = original_loader


def _v3_configuration_problems(config: dict[str, str]) -> list[str]:
    return [
        f"{key}は{expected!r}である必要があります。旧データ面は退役済みです。"
        for key, expected in V3_DATA_PLANE.items()
        if str(config.get(key) or "") != expected
    ]


@contextmanager
def mock_environment(config: dict[str, str]):
    """Evaluate feature gates against a config dict rather than the live env."""
    saved = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update({key: str(value) for key, value in config.items()})
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def get_claude_config_path() -> Path | None:
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    elif system == "Windows":
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            return Path(appdata) / "Claude" / "claude_desktop_config.json"
    elif system == "Linux":
        return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"
    return None


def configure_claude_mcp(root_dir: Path, chroma_dir: Path, emb_profile: str,
                         env_overrides: dict | None = None) -> bool:
    config_path = get_claude_config_path()
    if config_path is None:
        print("[!] 未対応OSのためClaude設定の場所を特定できません。MCP設定をスキップします。")
        return False

    uv_path = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")

    new_entry = {
        "command": uv_path,
        "args": [
            "--directory",
            str(root_dir),
            "run",
            "python",
            "-u",
            "src/rag_mcp_server.py",
        ],
        "env": {
            "CHROMA_DIR": str(chroma_dir),
            "EMB_PROFILE": emb_profile,
        },
    }
    if env_overrides:
        new_entry["env"].update(env_overrides)

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except json.JSONDecodeError:
            print(f"[!] {config_path}を解析できません。MCP設定をスキップします。")
            return False
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = {}

    config.setdefault("mcpServers", {})

    existing = config["mcpServers"].get("zotero-rag")
    if existing:
        print("\n[zotero-ragの現在のMCP設定]")
        print(f"  実行ファイル : {existing.get('command')}")
        print(f"  引数         : {existing.get('args')}")
        ans = input("MCP設定は既に存在します。上書きしますか？ [y/N]: ").strip().lower()
        if ans != "y":
            print("MCP設定の更新をスキップしました。")
            return False

    config["mcpServers"]["zotero-rag"] = new_entry

    temporary = config_path.with_name(config_path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    temporary.replace(config_path)

    print(f"\n[+] Claude MCP設定を更新しました: {config_path}")
    print(f"    実行ファイル : {uv_path}")
    print(f"    引数         : {new_entry['args']}")
    return True


def offer_initial_db_build(
    root_dir: Path, *, chroma_directory: Path, manifest_file: Path, profile_changed: bool,
) -> bool:
    """Build (and then audit) the V3 DB when this run's config requires it.

    ``chroma_directory``/``manifest_file`` must be resolved by the caller from
    the config just saved (not read from ``os.environ`` here) -- this wizard
    never loads its own ``.env`` back into its process, so reading the
    environment directly would check the default location instead of
    whatever the operator configured (2026-07-30).

    A true first run -- and only a state we can positively prove is empty --
    has nothing to lose, so it skips the typed REBUILD confirmation. Anything
    else keeps it: the confirmation protects an *existing* database, and a
    state we cannot read is not evidence that there is no database. A profile
    change is the case that reaches it in practice (2026-07-30 user decision),
    since that rebuild discards real embeddings.

    There used to be a third .command file (Server-Database-Workflow.command)
    hosting both steps; it added a file to discover without changing either
    risk profile, so its build/audit logic now lives in src/db_lifecycle.py
    and scripts/run_db_audit.py instead (2026-07-30 user decision). The typed
    REBUILD confirmation is carried over from it verbatim -- collapsing the
    files must not quietly weaken the guard they contained. This function
    owns only the interactive confirmation UX; the non-interactive
    classify/rebuild/audit primitives live in src/db_lifecycle.py so this
    wizard file's growing config-collection concerns don't keep absorbing
    unrelated DB-lifecycle logic (2026-07-30).

    Returns ``False`` only when the operator asked for something (a build or
    an audit) that then failed -- the caller uses this to make Setup.command
    exit non-zero instead of printing its usual "safe to close" message.
    Declining an offer is not a failure and returns ``True``.
    """
    db_state = db_lifecycle.existing_database_state(chroma_directory, manifest_file)
    destructive = db_state != db_lifecycle.DB_STATE_EMPTY
    if destructive and not profile_changed:
        return True

    print("\n" + "=" * 60)
    if destructive:
        print("5. DBの再構築（設定変更のため必要）")
        print("   埋め込みプロファイルが変更されたため、既存のV3データベースは")
        print("   このままでは使えません（既存の埋め込みを破棄して作り直します）。")
        if db_state == db_lifecycle.DB_STATE_UNKNOWN:
            print("   [注意] 既存DBの状態を確認できませんでした（manifestが読めない、")
            print("          または設定が解決できません）。中身がある前提で扱います。")
        print("   Chromaコレクション・manifest・語彙索引をすべて削除して作り直します。")
        confirmation = input("続行するには REBUILD と入力してください（中止する場合はEnter）: ").strip()
        if confirmation != "REBUILD":
            print("   スキップしました。準備ができたら手動で次を実行してください:")
            print("     uv run src/index_from_zotero.py --rebuild --progress")
            return True
    else:
        print("5. 初回DB構築")
        print("   まだデータベースが構築されていません。")
        ans = input("今すぐ構築しますか？ [Y/n]: ").strip().lower()
        if ans == "n":
            print("   スキップしました。準備ができたら手動で次を実行してください:")
            print("     uv run src/index_from_zotero.py --rebuild --progress")
            return True

    print()
    build_code = db_lifecycle.run_rebuild(root_dir)
    if build_code != 0:
        print(f"\n[エラー] DB構築が失敗しました（終了コード: {build_code}）。上記の出力を確認してください。")
        return False

    print("\n構築が完了しました。続けてDB監査の実行を推奨します"
          "（Zotero本体との突き合わせを含む、非破壊の読み取り専用チェックです）。")
    ans = input("今すぐDB監査を実行しますか？ [Y/n]: ").strip().lower()
    if ans == "n":
        print("   監査は後でMaintenance-Widget.commandの「DBを監査する」から実行できます。")
        return True
    print()
    audit_code = db_lifecycle.run_audit(root_dir)
    if audit_code != 0:
        print(f"\n[注意] DB監査が不合格でした（終了コード: {audit_code}）。出力を確認してください。")
        return False
    print("\n[合格] DB監査に合格しました。要約生成をご希望の場合はMaintenance-Widget.command"
          "から実行してください（DeepSeek API課金あり）。")
    return True


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Zotero Local RAGを設定します。",
        usage=argparse.SUPPRESS,
        add_help=False,
    )
    parser._optionals.title = "オプション"
    parser.add_argument(
        "-h", "--help", action="help",
        help="このヘルプを表示して終了します",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="秘密情報を表示せずに現在の機能設定を確認します",
    )
    parser.add_argument(
        "--server", action="store_true",
        help="ローカルのClaude Desktop登録を行わずサーバーを設定します",
    )
    args = parser.parse_args(argv)
    root_dir = Path(__file__).resolve().parents[1]
    env_path = root_dir / ".env"

    existing_config = read_env_file(env_path)
    if args.status:
        policy_config = read_env_file(root_dir / ".env.policy")
        print_configuration_status({**policy_config, **existing_config})
        return

    print("=" * 60)
    print("   Zotero Local RAG - V3セットアップ")
    print("=" * 60)

    modify = True
    prev_emb_profile = existing_config.get("EMB_PROFILE", "fast")
    if "ZOTERO_DATA_DIR" in existing_config:
        print("\n[現在の設定]")
        print(f"  ZOTERO_DATA_DIR : {existing_config['ZOTERO_DATA_DIR']}")
        print(f"  EMB_PROFILE     : {prev_emb_profile}")
        print(f"  FEATURE_LEVEL   : {existing_config.get('FEATURE_LEVEL', '旧形式/カスタム')}")
        print(f"  S2_API_KEY      : {'設定済み' if existing_config.get('S2_API_KEY') else '未設定'}")
        print(f"  LLM_STANDARD    : {existing_config.get('LLM_STANDARD', '未設定')}")
        ans = input("\n既存の設定が見つかりました。変更しますか？ [y/N]: ").strip().lower()
        if ans != "y":
            modify = False

    if not modify:
        retired = _v3_configuration_problems(existing_config)
        if retired:
            rendered = "\n".join(f"  - {problem}" for problem in retired)
            raise SystemExit(
                "既存設定が退役済みのデータ面を参照しています。"
                "ウィザードを再実行し、設定の変更を選択してください:\n" + rendered
            )

    if modify:
        default_zotero = os.path.expanduser("~/Zotero")
        if os.name == "nt":
            default_zotero = os.path.expanduser(r"~\Zotero")
        current_zotero = existing_config.get("ZOTERO_DATA_DIR", default_zotero)

        zotero_dir = input(
            f"\n1. Zoteroデータフォルダの場所を入力してください\n"
            f"   （Enterで現在値を維持: {current_zotero}）\n> "
        ).strip()
        if not zotero_dir:
            zotero_dir = current_zotero
        expanded_zotero = Path(zotero_dir).expanduser()
        if not (
            expanded_zotero.is_dir()
            and (expanded_zotero / "zotero.sqlite").is_file()
            and (expanded_zotero / "storage").is_dir()
        ):
            raise SystemExit(
                f"無効なZoteroデータフォルダです: {expanded_zotero}\n"
                "zotero.sqliteとstorage/が必要です。設定は変更していません。"
            )
        existing_config["ZOTERO_DATA_DIR"] = zotero_dir

        print("\n2. 使用する埋め込みモデルのプロファイルを選択してください")
        print("   [1] fast（既定・軽量高速・一般的な文章向け）")
        print("   [2] bge（BGE-M3・高負荷・幅広い多言語文章に対応）")
        current_emb = existing_config.get("EMB_PROFILE", "fast")
        emb_default = "2" if current_emb == "bge" else "1"
        emb_choice = input(
            f"選択 [1または2、Enterで{current_emb}を維持]: "
        ).strip() or emb_default

        if emb_choice == "2":
            existing_config["EMB_PROFILE"] = "bge"
        else:
            existing_config["EMB_PROFILE"] = "fast"

        print("\n埋め込みモデルを確認しています（初回はダウンロードに時間がかかります）…")
        try:
            existing_config["EMB_MODEL"] = ensure_embedding_model(
                existing_config, root_dir,
            )
        except RuntimeError as exc:
            raise SystemExit(
                f"埋め込みモデルを準備できませんでした: {exc}\n"
                "ネットワーク接続を確認して、ウィザードを再実行してください。"
            ) from exc

        configure_feature_level(existing_config)
        if existing_config.get("FEATURE_LEVEL") == "custom":
            configure_ndlocr(existing_config)
        enforce_v3_configuration(existing_config)
        problems = _configuration_problems(existing_config)
        if problems:
            rendered = "\n".join(f"  - {problem}" for problem in problems)
            raise SystemExit(
                "設定が不完全なため保存しませんでした:\n" + rendered
            )
        write_env_file(env_path, existing_config)
        print("\n[+] 設定を.envへ保存しました")

    emb_profile = existing_config.get("EMB_PROFILE", "fast")
    profile_changed = modify and emb_profile != prev_emb_profile
    chroma_dir = Path(existing_config.get("CHROMA_DIR", str(root_dir / "data" / "chroma")))

    if args.server:
        print("\n4. サーバーモード: ローカルのClaude Desktop登録をスキップしました。")
    else:
        print("\n" + "=" * 60)
        print("4. Claude Desktop MCPサーバーの設定")
        ans = input("Claude DesktopのMCP設定へzotero-ragを登録しますか？ [y/N]: ").strip().lower()
        if ans == "y":
            configure_claude_mcp(
                root_dir, chroma_dir, emb_profile, env_overrides=V3_DATA_PLANE,
            )
        else:
            print("\nMCP設定をスキップしました。")
            print("手動設定する場合は、claude_desktop_config.jsonへ次を追加してください:")
            uv_path = shutil.which("uv") or "uv"
            manual_env = {
                "CHROMA_DIR": str(chroma_dir),
                "EMB_PROFILE": emb_profile,
                **V3_DATA_PLANE,
            }
            manual = {
                "zotero-rag": {
                    "command": uv_path,
                    "args": [
                        "--directory",
                        str(root_dir),
                        "run",
                        "python",
                        "-u",
                        "src/rag_mcp_server.py",
                    ],
                    "env": manual_env,
                }
            }
            print(json.dumps(manual, indent=2))

    configure_tesseract(
        allow_install=existing_config.get("FEATURE_LEVEL") == "custom",
    )

    if profile_changed:
        print("\n" + "=" * 60)
        print("   ⚠️  埋め込みモデルのプロファイルが変更されました")
        print(f"   変更前: {prev_emb_profile} → 変更後: {emb_profile}")
        print()
        print("   このプロファイルを使う前にV3コレクションを再構築する必要があります。")

    print("\nセットアップウィザードが完了しました。")
    print(f"設定方式: {existing_config.get('FEATURE_LEVEL', '旧形式/カスタム')}")

    # Resolved from the config just saved, not from os.environ -- this
    # process never loads its own .env back into itself (2026-07-30).
    # resolve_configured_path is the same resolution rule v3_data_plane.py's
    # own chroma_dir()/manifest_path() use, called directly here instead of
    # reimplemented, so both stay consistent (e.g. both expand `~`) rather
    # than risking one getting fixed and the other missed (2026-07-30).
    resolved_chroma_dir = resolve_configured_path(root_dir, chroma_dir)
    manifest_file = resolve_configured_path(
        root_dir, existing_config.get("MANIFEST_PATH", f"data/{V3_MANIFEST_NAME}"),
    )
    # Mirrors v3_data_plane.manifest_path()'s own rejection of a non-canonical
    # filename -- that function can't be called here (it reads MANIFEST_PATH
    # from os.environ, which this process never populates), but the
    # fail-closed contract it enforces for every other V3 entry point must
    # still hold for this one (2026-07-30).
    if manifest_file.name != V3_MANIFEST_NAME:
        raise SystemExit(
            f"設定のMANIFEST_PATHが不正です: {manifest_file}\n"
            f"旧形式のmanifestは廃止されました。ファイル名は{V3_MANIFEST_NAME}である必要があります。"
        )

    db_build_ok = offer_initial_db_build(
        root_dir, chroma_directory=resolved_chroma_dir, manifest_file=manifest_file,
        profile_changed=profile_changed,
    )

    print("\n日常のライブラリ差分更新・DB監査・階層要約の生成（有料API）は")
    print("すべてMaintenance-Widget.commandから行えます。")
    print("設定を変更・拡張する場合は、いつでもSetup.commandを再実行できます。")
    if not db_build_ok:
        # Setup.command checks the process exit code to decide whether to
        # print its "safe to close" message -- a build/audit failure must
        # not be followed by that message from either side (2026-07-30).
        raise SystemExit(2)
    print("このウィンドウは閉じて構いません。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] セットアップを中断しました。")
