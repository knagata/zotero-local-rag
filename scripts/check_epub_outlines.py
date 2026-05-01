#!/usr/bin/env python3
"""
manifest.json 内の EPUB ファイルから目次（章構造）を取得して表示するスクリプト。
EPubは toc.ncx (epub2) または nav.xhtml (epub3) に必ず目次が入っている。
"""

import json
import os
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

MANIFEST_PATH = Path(__file__).parent.parent / "data" / "manifest.json"

# XML namespace
NS_NCX = "http://www.daisy.org/z3986/2005/ncx/"
NS_XHTML = "http://www.w3.org/1999/xhtml"
NS_EPUB = "http://www.idpf.org/2007/ops"


def parse_ncx_toc(ncx_text: str) -> list[tuple[int, str]]:
    """toc.ncx (epub2) から目次を取得"""
    try:
        root = ET.fromstring(ncx_text)
        items = []

        def _walk(node, depth=1):
            for nav_point in node.findall(f"{{{NS_NCX}}}navPoint"):
                label_el = nav_point.find(f".//{{{NS_NCX}}}text")
                label = (label_el.text or "").strip() if label_el is not None else ""
                if label:
                    items.append((depth, label))
                _walk(nav_point, depth + 1)

        nav_map = root.find(f"{{{NS_NCX}}}navMap")
        if nav_map is not None:
            _walk(nav_map)
        return items
    except Exception:
        return []


def parse_nav_toc(nav_text: str) -> list[tuple[int, str]]:
    """nav.xhtml (epub3) から目次を取得"""
    try:
        # namespace 付きで parse
        root = ET.fromstring(nav_text)
        items = []

        # epub:type="toc" を持つ <nav> を探す
        def find_toc_nav(node):
            tag = node.tag.split("}")[-1] if "}" in node.tag else node.tag
            if tag == "nav":
                etype = node.get(f"{{{NS_EPUB}}}type") or node.get("epub:type") or ""
                if "toc" in etype:
                    return node
            for child in node:
                result = find_toc_nav(child)
                if result is not None:
                    return result
            return None

        def walk_ol(ol_node, depth=1):
            for li in ol_node:
                tag = li.tag.split("}")[-1] if "}" in li.tag else li.tag
                if tag != "li":
                    continue
                # <a> または <span> のテキストを取る
                label = ""
                for child in li:
                    ctag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                    if ctag in ("a", "span"):
                        texts = [child.text or ""] + [
                            (sub.text or "") + (sub.tail or "") for sub in child
                        ]
                        label = "".join(texts).strip()
                        break
                if label:
                    items.append((depth, label))
                # ネストされた ol
                for child in li:
                    ctag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                    if ctag == "ol":
                        walk_ol(child, depth + 1)

        toc_nav = find_toc_nav(root)
        if toc_nav is not None:
            for child in toc_nav:
                ctag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                if ctag == "ol":
                    walk_ol(child)
        return items
    except Exception:
        return []


def get_epub_toc(epub_path: str) -> list[tuple[int, str]]:
    """EPubから目次エントリのリストを返す [(depth, title), ...]"""
    try:
        with zipfile.ZipFile(epub_path, "r") as zf:
            names = zf.namelist()

            # --- epub3: nav.xhtml を優先 ---
            nav_candidates = [
                n for n in names
                if n.endswith("nav.xhtml") or n.endswith("nav.html")
                or n.endswith("navigation.xhtml")
            ]
            for nav_candidate in nav_candidates:
                text = zf.read(nav_candidate).decode("utf-8", errors="replace")
                toc = parse_nav_toc(text)
                if toc:
                    return toc

            # --- epub2: toc.ncx ---
            ncx_candidates = [n for n in names if n.endswith(".ncx")]
            for ncx_candidate in ncx_candidates:
                text = zf.read(ncx_candidate).decode("utf-8", errors="replace")
                toc = parse_ncx_toc(text)
                if toc:
                    return toc

    except Exception:
        pass
    return []


def check_epub_outlines(manifest_path: Path):
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    files = manifest.get("files", {})

    total_epub = 0
    has_toc = 0
    no_toc = 0
    missing = 0

    results = []

    for key, entry in files.items():
        pdf_path = entry.get("pdf_path", "")
        if not pdf_path.lower().endswith(".epub"):
            continue

        total_epub += 1
        size_mb = entry.get("size", 0) / (1024 * 1024)
        name = Path(pdf_path).name

        if not os.path.exists(pdf_path):
            missing += 1
            continue

        toc = get_epub_toc(pdf_path)
        if toc:
            has_toc += 1
        else:
            no_toc += 1

        results.append({
            "key": key,
            "name": name[:70],
            "size_mb": round(size_mb, 1),
            "toc_count": len(toc),
            "toc": toc,
        })

    # ===== 表示 =====
    print("=" * 65)
    print("📚 EPUB 目次（章構造）調査結果")
    print("=" * 65)
    print(f"EPUB総数:          {total_epub}")
    print(f"  目次あり:        {has_toc}  ({has_toc/total_epub*100:.1f}%)" if total_epub else "")
    print(f"  目次なし:        {no_toc}  ({no_toc/total_epub*100:.1f}%)" if total_epub else "")
    print(f"  ファイル未存在:  {missing}")
    print()

    for r in sorted(results, key=lambda x: -x["toc_count"]):
        toc = r["toc"]
        status = f"✅ {r['toc_count']} エントリ" if toc else "❌ 目次なし"
        print(f"[{r['key']}] {r['name']}  ({r['size_mb']:.1f}MB)  {status}")
        for depth, title in toc[:6]:
            indent = "  " * depth
            print(f"  {indent}{'└' if depth > 1 else '├'} {title[:60]}")
        if len(toc) > 6:
            print(f"    ... 他 {len(toc)-6} エントリ")
        print()


if __name__ == "__main__":
    check_epub_outlines(MANIFEST_PATH)
