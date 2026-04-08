#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

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

    print("="*60)
    print("   Zotero Local RAG - Setup & Indexer")
    print("="*60)

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
        if os.name == 'nt':
            default_zotero = os.path.expanduser(r"~\Zotero")
        
        zotero_dir = input(f"\n1. Where is your Zotero data directory?\n   (Press Enter for default: {default_zotero})\n> ").strip()
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
        
        # Save back to .env
        with open(env_path, "w", encoding="utf-8") as f:
            for k, v in existing_config.items():
                f.write(f"{k}={v}\n")
        print("\n[+] Configuration successfully saved to .env")

    print("\n" + "="*60)
    run_idx = input("Do you want to run the Embedding Indexer now? (Y/n): ").strip().lower()
    if run_idx != "n":
        print("\n[+] Starting Embedding process (this may download models if first time)...")
        print("[+] This process reads your Zotero local database and vectorizes PDFs/HTMLs.\n")
        
        # Merge our known config into the current environment so the subprocess inherits it
        env = os.environ.copy()
        env.update(existing_config)
        
        # `uv run` handles dependency isolation automatically
        process = subprocess.run(["uv", "run", "src/index_from_zotero.py", "--progress"], env=env, cwd=root_dir)
        
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
