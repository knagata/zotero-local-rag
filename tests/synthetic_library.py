"""A library small enough to run in a test and real enough to be worth running.

The ingestion net runs the indexer against the user's own library through a
child process, which is what makes it honest and also what makes it cost 140
seconds, need Zotero open, and stay out of CI. This is the other half: three
documents built here, a source that answers like Zotero's local API, and an
embedder that is arithmetic rather than a model, so the same loop can be called
in-process in a couple of seconds by anyone, anywhere.

It does not replace the net. A synthetic PDF has a text layer, one column, no
scanned pages and no OCR route, so everything downstream of "the text came out"
is exercised here and nothing upstream of it is. What this can check that the
net cannot is the shape of a run: that the four stores agree afterwards, that a
failure leaves nothing half-written, and that a second pass over unchanged
sources decides not to re-index -- the decisions, not the extraction.
"""
from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from zotero_source_localapi import ZoteroAttachment  # noqa: E402

#: Paragraphs long enough to survive the short-chunk merge, in both scripts the
#: library is actually made of.
_BODY = [
    "都市の記憶は建物ではなく通りの幅に残る。歩幅と視線の高さが、そこに住んだ人の"
    "生活の速度を決めていたからである。戦後の区画整理はこの速度を書き換えた。",
    "The second paragraph exists so that the chunker has a boundary to find and "
    "the retrieval layer has more than one thing to rank. It is deliberately "
    "ordinary prose, because ordinary prose is what the library is made of.",
    "第三段落では、引用の体裁を持つ文を置く。田中一郎『都市の記憶』（一九八八年）"
    "は、この主題を最初に扱った日本語の著作である。",
]


def _pdf(path: Path, title: str) -> None:
    """A small text-layer PDF, written with the library the indexer reads it with."""
    import fitz

    document = fitz.open()
    for index, paragraph in enumerate(_BODY):
        page = document.new_page()
        page.insert_text((72, 100), f"{title} — {index + 1}", fontsize=14, fontname="china-s")
        page.insert_textbox(
            fitz.Rect(72, 130, 520, 700), paragraph, fontsize=11, fontname="china-s",
        )
    document.save(path)
    document.close()


def _html(path: Path, title: str) -> None:
    body = "\n".join(f"<p>{paragraph}</p>" for paragraph in _BODY)
    path.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title></head>"
        f"<body><h1>{title}</h1>{body}</body></html>",
        encoding="utf-8",
    )


def _epub(path: Path, title: str) -> None:
    """A minimal but structurally valid EPUB: container, OPF spine, one chapter.

    Written by hand rather than with a builder library so the spine and the
    chapter file stay visible here -- the extractor checks the spine against
    what it managed to read, and a test that cannot see the spine cannot say
    what it was checking.
    """
    chapter = (
        "<?xml version='1.0' encoding='utf-8'?>"
        "<html xmlns='http://www.w3.org/1999/xhtml'><head><title>"
        f"{title}</title></head><body><h1>{title}</h1>"
        + "".join(f"<p>{paragraph}</p>" for paragraph in _BODY)
        + "</body></html>"
    )
    opf = (
        "<?xml version='1.0' encoding='utf-8'?>"
        "<package xmlns='http://www.idpf.org/2007/opf' version='3.0' unique-identifier='id'>"
        "<metadata xmlns:dc='http://purl.org/dc/elements/1.1/'>"
        f"<dc:identifier id='id'>synthetic-{title}</dc:identifier>"
        f"<dc:title>{title}</dc:title><dc:language>ja</dc:language>"
        "</metadata>"
        "<manifest><item id='c1' href='chapter1.xhtml' media-type='application/xhtml+xml'/></manifest>"
        "<spine><itemref idref='c1'/></spine></package>"
    )
    container = (
        "<?xml version='1.0' encoding='utf-8'?>"
        "<container version='1.0' xmlns='urn:oasis:names:tc:opendocument:xmlns:container'>"
        "<rootfiles><rootfile full-path='OEBPS/content.opf' "
        "media-type='application/oebps-package+xml'/></rootfiles></container>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("OEBPS/content.opf", opf)
        archive.writestr("OEBPS/chapter1.xhtml", chapter)


def build_library(root: Path) -> list[ZoteroAttachment]:
    """Write one attachment of each source type and describe them as Zotero would."""
    files = root / "attachments"
    files.mkdir(parents=True, exist_ok=True)
    _pdf(files / "memory.pdf", "都市の記憶")
    _html(files / "streets.html", "Streets and Speed")
    _epub(files / "widths.epub", "通りの幅")
    return [
        ZoteroAttachment(
            attachmentKey="SYNPDF001", parentItemKey="SYNITEM01",
            title="都市の記憶", year=1988, creators=["田中 一郎"],
            pdf_path=str(files / "memory.pdf"), source_type="pdf",
            contentType="application/pdf", filename="memory.pdf",
            language="ja", parentItemType="book",
        ),
        ZoteroAttachment(
            attachmentKey="SYNHTML01", parentItemKey="SYNITEM02",
            title="Streets and Speed", year=2011, creators=["Ada Lovelace"],
            pdf_path=str(files / "streets.html"), source_type="html",
            contentType="text/html", filename="streets.html",
            language="en", parentItemType="webpage",
        ),
        ZoteroAttachment(
            attachmentKey="SYNEPUB01", parentItemKey="SYNITEM03",
            title="通りの幅", year=2003, creators=["佐藤 花子"],
            pdf_path=str(files / "widths.epub"), source_type="epub",
            contentType="application/epub+zip", filename="widths.epub",
            language="ja", parentItemType="book",
        ),
    ]


class SyntheticZoteroSource:
    """Answers the two questions the indexer asks a library.

    Deliberately not a Mock: a Mock agrees with however the caller happens to
    call it, including after the call changes. This has the real signatures, so
    a change to how the indexer asks shows up here as a TypeError rather than
    as a silently empty run.
    """

    def __init__(self, attachments: list[ZoteroAttachment], notes: list[dict[str, Any]] | None = None):
        self.attachments = list(attachments)
        self.notes = list(notes or [])
        self.calls: list[str] = []

    async def list_normalized_attachments(
        self,
        zotero_data_dir: Optional[str],
        pdf_cache_dir: str,
        collection_key: Optional[str] = None,
        require_complete: bool = False,
    ) -> list[ZoteroAttachment]:
        self.calls.append("list_normalized_attachments")
        return list(self.attachments)

    async def list_notes(
        self, collection_key: Optional[str] = None, limit: int = 200, *,
        require_complete: bool = False,
    ) -> list[dict[str, Any]]:
        self.calls.append("list_notes")
        return list(self.notes)


def deterministic_embedding_function(dimensions: int = 32):
    """Vectors from a hash of the text: same input, same vector, no model.

    Good enough for what depends on embeddings in a run -- that every chunk got
    one, that they are stable across passes, and that identical text lands in
    the same place -- and it costs microseconds instead of loading BGE-M3. It
    is not a similarity model and nothing here should assert that near meanings
    land near each other.
    """

    def embed(input=None, texts: list[str] | None = None) -> list[list[float]]:
        # Chroma calls embedding functions with input=; the indexer calls some
        # of them positionally. Accept both rather than discover the difference
        # at the first upsert.
        texts = list(input if input is not None else (texts or []))
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            raw = [
                digest[index % len(digest)] / 255.0 - 0.5
                for index in range(dimensions)
            ]
            norm = sum(value * value for value in raw) ** 0.5 or 1.0
            vectors.append([value / norm for value in raw])
        return vectors

    embed.name = lambda: "synthetic-hash-32"  # Chroma asks embedding functions their name
    return embed
