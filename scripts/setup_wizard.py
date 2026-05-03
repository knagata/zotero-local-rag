#!/usr/bin/env python3
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


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


def configure_claude_mcp(root_dir: Path, chroma_dir: Path, emb_profile: str) -> bool:
    config_path = get_claude_config_path()
    if config_path is None:
        print("[!] Unsupported OS — could not determine Claude config path. Skipping MCP setup.")
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

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except json.JSONDecodeError:
            print(f"[!] Could not parse {config_path}. Skipping MCP setup.")
            return False
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = {}

    config.setdefault("mcpServers", {})

    existing = config["mcpServers"].get("zotero-rag")
    if existing:
        print(f"\n[Current MCP config for zotero-rag]")
        print(f"  command : {existing.get('command')}")
        print(f"  args    : {existing.get('args')}")
        ans = input("MCP config already exists. Overwrite? [y/N]: ").strip().lower()
        if ans != "y":
            print("Skipped MCP config update.")
            return False

    config["mcpServers"]["zotero-rag"] = new_entry

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Claude MCP config updated: {config_path}")
    print(f"    command : {uv_path}")
    print(f"    args    : {new_entry['args']}")
    return True


def main():
    root_dir = Path(__file__).resolve().parents[1]
    env_path = root_dir / ".env"

    existing_config = {}
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    existing_config[k.strip()] = v.strip()

    print("=" * 60)
    print("   Zotero Local RAG - Setup & Indexer")
    print("=" * 60)

    modify = True
    if "ZOTERO_DATA_DIR" in existing_config:
        print("\n[Current Configuration]")
        print(f"  ZOTERO_DATA_DIR : {existing_config['ZOTERO_DATA_DIR']}")
        print(f"  EMB_PROFILE     : {existing_config.get('EMB_PROFILE', 'fast')}")
        ans = input("\nExisting settings found. Do you want to change them? [y/N]: ").strip().lower()
        if ans != "y":
            modify = False

    if modify:
        default_zotero = os.path.expanduser("~/Zotero")
        if os.name == "nt":
            default_zotero = os.path.expanduser(r"~\Zotero")

        zotero_dir = input(
            f"\n1. Where is your Zotero data directory?\n   (Press Enter for default: {default_zotero})\n> "
        ).strip()
        if not zotero_dir:
            zotero_dir = default_zotero
        existing_config["ZOTERO_DATA_DIR"] = zotero_dir

        print("\n2. Which Embedding Model Profile do you want to use?")
        print("   [1] fast (Default, smaller/faster, good for standard text)")
        print("   [2] bge  (BGE-M3, heavier, supports extensive multilingual text)")
        emb_choice = input("Select [1 or 2, default is 1]: ").strip()

        if emb_choice == "2":
            existing_config["EMB_PROFILE"] = "bge"
        else:
            existing_config["EMB_PROFILE"] = "fast"

        with open(env_path, "w", encoding="utf-8") as f:
            for k, v in existing_config.items():
                f.write(f"{k}={v}\n")
        print("\n[+] Configuration successfully saved to .env")

    emb_profile = existing_config.get("EMB_PROFILE", "fast")
    chroma_dir = Path(existing_config.get("CHROMA_DIR", str(root_dir / "data" / "chroma")))

    print("\n" + "=" * 60)
    print("3. Configure Claude Desktop MCP server")
    ans = input("Register zotero-rag in Claude Desktop's MCP config? [y/N]: ").strip().lower()
    if ans == "y":
        configure_claude_mcp(root_dir, chroma_dir, emb_profile)
    else:
        print("\nSkipped MCP config.")
        print("To set up manually, add the following to claude_desktop_config.json:")
        uv_path = shutil.which("uv") or "uv"
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
                "env": {
                    "CHROMA_DIR": str(chroma_dir),
                    "EMB_PROFILE": emb_profile,
                },
            }
        }
        print(json.dumps(manual, indent=2))

    print("\n" + "=" * 60)
    run_idx = input("Do you want to run the Embedding Indexer now? (Y/n): ").strip().lower()
    if run_idx != "n":
        print("\n[+] Starting Embedding process (this may download models if first time)...")
        print("[+] This process reads your Zotero local database and vectorizes PDFs/HTMLs.\n")

        env = os.environ.copy()
        env.update(existing_config)

        process = subprocess.run(
            ["uv", "run", "src/index_from_zotero.py", "--progress"], env=env, cwd=root_dir
        )

        if process.returncode == 0:
            print("\n[+] Indexing completed successfully!")
        else:
            print(f"\n[!] Indexing failed with exit code {process.returncode}.")
    else:
        print("\nSkipped indexing.")

    print("\nSetup wizard finished. You can close this window.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Setup aborted by user.")
