# src/ocr_layer_audit.py
"""Audit a scan-derived text layer for OCR degradation (note 79, stage 2).

Only scan-derived text needs this: a born-digital PDF's text layer *is* the
typeset result, while OCR output is a lossy derivative whose quality is unknown
until measured. Scan quality is essentially constant across a document -- one
scanner, one setting, one OCR engine, one source copy -- so a few body pages
represent the whole, which is what makes an LLM check affordable here.

Every element of this design was forced by a calibration run over 18 real
documents (recorded in note 79); none of it is defensive boilerplate:

- **The prompt carries no literal example strings.** An earlier version used
  ``righrs``/``srorage``/``wirhout``/``1ike`` as few-shot examples and the model
  echoed them back verbatim for documents where it had little to report. 23% of
  all reported items did not occur in the source at all, and one document was
  called "severe degradation" on entirely fabricated evidence.
- **Every reported item is verified against the sampled text**, and only
  verified items count. This is the same rule the AI-TOC path already follows:
  the model proposes candidates, deterministic reconciliation decides. It
  removed the fabrication completely -- the document above dropped to 0.00%.
- **The denominator is counted here, not asked for.** The model's own word
  estimate moved between 2,000 and 6,452 for the same text.
- **The model's holistic verdict is ignored.** It did not track its own counts:
  ``severe`` at 1.0%, ``minor`` at 3.75%.
- **Only ``ocr_misrecognition`` counts.** Line-break hyphenation and
  accent/curly-quote normalisation showed up as "defects" on healthy
  born-digital control documents.
- **Body pages from the middle 60% are sampled.** Front matter uses decorative
  faces and small caps, so its OCR is systematically worse than the body and
  sampling it biases every document toward "degraded".

Rates do not separate scanned from born-digital cleanly on their own (controls
reached 1.22%), so the acceptance threshold is deliberately conservative: only
clear cases trigger re-OCR automatically, and the middle band is recorded for
the report loop to pick up instead.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping

try:
    from .llm_client import LLMClient, LLMError, get_llm
except ImportError:  # direct execution with src/ on sys.path
    from llm_client import LLMClient, LLMError, get_llm

ACCEPTABLE = "acceptable"
DEGRADED = "degraded"
UNVERIFIED = "unverified"

#: Only this defect class counts toward the rate; see module docstring.
COUNTED_DEFECT = "ocr_misrecognition"

MAX_LISTED_ITEMS = 40
SAMPLE_PAGES = 3
MAX_CHARS_PER_PAGE = 4000
#: The cap has to leave room for a degraded document to actually reach the
#: threshold: 1.5% of a ~2,000-unit sample is ~30 quoted items, and each item
#: is a quoted fragment plus its correction. Roomy enough that a full list of
#: CJK fragments still fits.
AUDIT_MAX_TOKENS = 12000

_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_CJK_CHAR = re.compile(r"[぀-ヿ一-鿿]")

AUDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "as_printed": {"type": "string"},
                    "likely_intended": {"type": "string"},
                    "defect": {"type": "string"},
                },
                "required": ["as_printed", "likely_intended", "defect"],
            },
        },
    },
    "required": ["items"],
}

AUDIT_PROMPT_TEMPLATE = """Audit this text, extracted from a PDF, for defects introduced
by extraction or OCR -- not by the author.

`as_printed` MUST be copied character-for-character from the text below. If you
cannot copy it verbatim from the text, do not report it. Never invent, correct,
normalise or reconstruct a form that is not literally present.

`defect` is one of:
  ocr_misrecognition -- a character was read as a different, visually similar one
  letter_spacing     -- spurious spaces inserted between the characters of a word
  ligature_loss      -- a ligature dropped its characters
  word_boundary      -- two words fused, or one word split
  other

Do NOT report:
- proper nouns, place names, ethnonyms, personal or author names
- regional spelling variants (British vs American, historical orthography)
- foreign-language words, technical or theoretical terminology
- line-break hyphenation
- differences of capitalisation or punctuation style alone
- anything you merely find unfamiliar

If `as_printed` and `likely_intended` would be identical, omit the item.
List at most {max_items} items. If the text is clean, return an empty list.

TEXT:
"""


def audit_prompt(max_items: int = MAX_LISTED_ITEMS) -> str:
    """The audit prompt, asking for only as many items as settle the question."""
    return AUDIT_PROMPT_TEMPLATE.format(max_items=max_items)


#: Kept for tests and callers that only need the wording.
AUDIT_PROMPT = audit_prompt()


def degraded_rate_threshold() -> float:
    """Rate at or above which a text layer is re-OCRed automatically."""
    return float(os.environ.get("OCR_LAYER_DEGRADED_RATE", "0.015"))


def review_rate_threshold() -> float:
    """Rate above which the result is recorded for review but not acted on."""
    return float(os.environ.get("OCR_LAYER_REVIEW_RATE", "0.005"))


#: Below this length the middle-60% window is skipped and every page is used.
#: Excluding front matter protects the measurement in a book; in a short paper
#: it throws away most of an already-tiny document. A two-page article measured
#: ``insufficient_sample`` because the window left exactly one page, and that
#: page was the title page (VRGGXZ4Y, 2026-07-27).
SHORT_DOCUMENT_PAGES = 8


def sample_body_pages(pdf_path: Path, pages: int = SAMPLE_PAGES) -> tuple[str, List[int]]:
    """Return sampled body text and its 1-based page numbers.

    Text-dense pages from the middle 60%, deliberately excluding front matter
    (see module docstring) -- except in a short document, where there is no
    front matter to speak of and the window would discard most of the text.
    """
    import fitz

    with fitz.open(str(pdf_path)) as document:
        page_count = document.page_count
        if page_count <= 0:
            return "", []
        if page_count <= SHORT_DOCUMENT_PAGES:
            low, high = 0, page_count
        else:
            low = int(page_count * 0.2)
            high = max(int(page_count * 0.8), low + 1)
        scored = sorted(
            ((len((document[i].get_text("text") or "").strip()), i)
             for i in range(low, min(high, page_count))),
            reverse=True,
        )
        picked = sorted(index for _size, index in scored[:pages])
        sections = [
            f"\n--- PAGE {index + 1} ---\n"
            f"{(document[index].get_text('text') or '').strip()[:MAX_CHARS_PER_PAGE]}"
            for index in picked
        ]
    return "".join(sections), [index + 1 for index in picked]


def _denominator(text: str) -> int:
    """Count the unit the rate is measured against, without asking the model."""
    cjk = len(_CJK_CHAR.findall(text))
    latin = len(_LATIN_WORD.findall(text))
    return cjk if cjk > latin else latin


def listed_item_cap(denominator: int) -> int:
    """How many items to ask for, given how many would settle the question.

    The cap only has to let a degraded document cross the threshold with
    headroom -- roughly ``denominator * threshold * 2``. Asking for a flat 40
    from a short document makes the reply longer than it needs to be, and every
    extra quoted fragment is another chance for the model to emit a quote or
    backslash that breaks the JSON.
    """
    needed = int(denominator * degraded_rate_threshold() * 2) + 1
    return max(12, min(MAX_LISTED_ITEMS, needed))


_ITEM_OBJECT = re.compile(r"\{[^{}]*\}")


def salvage_items(raw_text: str) -> List[Dict[str, Any]]:
    """Recover whole item objects from a reply that failed to parse.

    The model is asked to quote source text verbatim, so a stray quote or
    backslash in the source can break the surrounding JSON and lose the entire
    measurement -- one malformed escape cost a whole document's audit
    (VRGGXZ4Y, 2026-07-27). Salvaging is safe *here* specifically because every
    item is then checked against the source text: a garbled fragment simply
    fails to verify and drops out, exactly like a fabricated one. That
    reasoning does not transfer to LLM calls whose output is trusted directly.
    """
    salvaged: List[Dict[str, Any]] = []
    for match in _ITEM_OBJECT.findall(raw_text or ""):
        try:
            candidate = json.loads(match)
        except (ValueError, TypeError):
            continue
        if isinstance(candidate, dict) and "as_printed" in candidate:
            salvaged.append(candidate)
    return salvaged


def verify_reported_items(
    items: Any, sampled_text: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split reported items into (verified, rejected).

    An item survives only if it names the counted defect class, quotes text
    that literally occurs in the sample, and actually proposes a change. This
    is what stops fabricated evidence from reaching the rate.
    """
    verified: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, Mapping):
            continue
        as_printed = str(item.get("as_printed") or "")
        intended = str(item.get("likely_intended") or "")
        defect = str(item.get("defect") or "")
        if (
            defect == COUNTED_DEFECT
            and as_printed
            and as_printed != intended
            and as_printed in sampled_text
        ):
            verified.append(dict(item))
        else:
            rejected.append(dict(item))
    return verified, rejected


def classify_rate(rate: float) -> str:
    return DEGRADED if rate >= degraded_rate_threshold() else ACCEPTABLE


#: Recorded when the audit was switched off rather than attempted. Distinct
#: from every failure reason: choosing not to measure is a configuration, not
#: evidence that anything is wrong.
DISABLED_REASON = "audit_disabled"

#: Reasons that say nothing about the text: try again next run instead of
#: spending hours of Docling on a document whose layer may be perfectly good.
_TRANSIENT_REASON_PREFIXES = ("llm_unavailable", "llm_failed")
#: The file itself could not be read. Re-OCR would hit the same wall, so this
#: needs a human to look at the file rather than another extraction attempt.
_FILE_PROBLEM_REASON_PREFIXES = ("sampling_failed",)


def _reason(quality_info: Mapping[str, Any]) -> str:
    return str(quality_info.get("ocr_layer_audit_reason") or "")


def audit_was_transient_failure(quality_info: Mapping[str, Any]) -> bool:
    """Whether the audit failed for a reason unrelated to the text's quality.

    A provider outage or an unparseable reply is not evidence about the OCR
    layer. Replacing the text on that basis would re-OCR the whole document --
    and during an outage, every scanned document in the run at once. The
    verdict is not cached, so the next run measures it properly.
    """
    return _reason(quality_info).startswith(_TRANSIENT_REASON_PREFIXES)


def audit_hit_file_problem(quality_info: Mapping[str, Any]) -> bool:
    """Whether the source file could not be read well enough to sample."""
    return _reason(quality_info).startswith(_FILE_PROBLEM_REASON_PREFIXES)


def audit_sample_too_small(quality_info: Mapping[str, Any]) -> bool:
    """Whether there was too little text to measure anything.

    The text is kept and indexed, flagged as unmeasured, rather than triggering
    a re-extraction: a document this sparse gives a re-OCR nothing to work with
    either (user decision 2026-07-27).
    """
    return _reason(quality_info) == "insufficient_sample"


def audit_was_disabled(quality_info: Mapping[str, Any]) -> bool:
    """Whether the audit was switched off rather than attempted."""
    return _reason(quality_info) == DISABLED_REASON


def replacement_required(quality_info: Mapping[str, Any]) -> bool:
    """Whether the scan-derived text layer should be replaced by re-OCR.

    Only a *measured* verdict of ``degraded`` justifies discarding the existing
    text. Every other outcome is a statement about the audit rather than about
    the text.

    Turning the audit off deserves particular care. It is what happens whenever
    no paid LLM is configured (note 80, choice B), and treating "not measured"
    as "not trustworthy" there would send every scanned PDF in the library to a
    full re-OCR -- so switching the LLM off would *increase* the work done,
    which is the opposite of what the operator asked for. Choosing not to
    measure is a configuration, not a finding.
    """
    quality = str(quality_info.get("ocr_layer_quality") or "")
    if quality == DEGRADED:
        return True
    if quality == ACCEPTABLE:
        return False
    if (
        audit_was_disabled(quality_info)
        or audit_was_transient_failure(quality_info)
        or audit_sample_too_small(quality_info)
        or audit_hit_file_problem(quality_info)
    ):
        return False
    return True


def audit_ocr_text_layer(
    pdf_path: Path, item_key: str, *, client: LLMClient | None = None,
) -> Dict[str, Any]:
    """Measure OCR degradation in a scan-derived text layer.

    Returns metadata for ``quality_info``. ``ocr_layer_quality`` is
    ``acceptable``, ``degraded``, or ``unverified`` when the audit could not
    run -- an unverified layer is never treated as degraded, since failing to
    measure is not evidence of a problem.
    """
    result: Dict[str, Any] = {
        "ocr_layer_quality": UNVERIFIED,
        "ocr_layer_error_rate": None,
        "ocr_layer_verified_count": 0,
        "ocr_layer_rejected_count": 0,
        "ocr_layer_sampled_pages": [],
        "ocr_layer_needs_review": False,
    }

    # No cloud-policy gate here (user decision 2026-07-27). Anything indexed is
    # returned by rag_search and therefore reaches the assistant anyway, so an
    # item that genuinely must not leave the machine cannot be in this index at
    # all -- the check protected nothing it was still in. It also had a failure
    # mode of its own: cloud_text_allowed is fail-closed *including* a failed
    # Zotero tag lookup, so a momentarily unreachable Zotero turned every
    # scanned PDF into "cloud not allowed" and sent it for a full local re-OCR.
    # Whole-document uploads (Mistral OCR and its batch queue) keep their own
    # fail-closed checks; this audit sends three sampled pages.

    try:
        sampled_text, pages = sample_body_pages(pdf_path)
    except Exception as exc:
        result["ocr_layer_audit_reason"] = f"sampling_failed:{exc}"
        return result
    result["ocr_layer_sampled_pages"] = pages

    denominator = _denominator(sampled_text)
    if not sampled_text.strip() or denominator < 200:
        result["ocr_layer_audit_reason"] = "insufficient_sample"
        return result

    # The reply carries up to MAX_LISTED_ITEMS quoted fragments, and in a badly
    # degraded Japanese document those are CJK, which tokenise at roughly one
    # character per token. The shared 4096-token default truncated exactly such
    # a reply mid-JSON (XGEQTPS3, 2026-07-27) -- the worse a document's OCR, the
    # longer the list and the more likely the audit failed to parse, so the
    # measurement gave up precisely on the documents it exists to catch. One
    # retry follows, because a truncated reply is not deterministic.
    llm = client or get_llm("cheap")
    prompt = audit_prompt(listed_item_cap(denominator)) + sampled_text
    last_error: Exception | None = None
    response = None
    salvaged: List[Dict[str, Any]] = []
    for _attempt in range(2):
        try:
            response = llm.generate_json(
                prompt, schema=AUDIT_SCHEMA, timeout=180.0, max_tokens=AUDIT_MAX_TOKENS,
            )
            break
        except LLMError as exc:
            last_error = exc
            # The undecodable body, not the parser's message: the message
            # names an offset into text the client would otherwise drop.
            salvaged = salvage_items(getattr(exc, "raw", "") or str(exc))
            if salvaged:
                break
        except Exception as exc:  # noqa: BLE001 - never block ingestion on the audit
            last_error = exc
            break
    if response is None and not salvaged:
        kind = "llm_unavailable" if isinstance(last_error, LLMError) else "llm_failed"
        result["ocr_layer_audit_reason"] = f"{kind}:{last_error}"
        return result

    verified, rejected = verify_reported_items(
        (response or {}).get("items") if response is not None else salvaged,
        sampled_text,
    )
    rate = len(verified) / denominator
    result.update({
        "ocr_layer_quality": classify_rate(rate),
        "ocr_layer_error_rate": round(rate, 5),
        "ocr_layer_verified_count": len(verified),
        "ocr_layer_rejected_count": len(rejected),
        "ocr_layer_denominator": denominator,
        "ocr_layer_needs_review": rate > review_rate_threshold(),
        "ocr_layer_examples": [
            f'{item["as_printed"]}→{item["likely_intended"]}' for item in verified[:8]
        ],
        "ocr_layer_audit_reason": "measured",
    })
    return result


__all__ = [
    "ACCEPTABLE", "AUDIT_PROMPT", "AUDIT_SCHEMA", "DEGRADED", "UNVERIFIED",
    "audit_hit_file_problem", "audit_ocr_text_layer", "audit_sample_too_small",
    "audit_prompt", "audit_was_transient_failure", "classify_rate",
    "degraded_rate_threshold", "listed_item_cap", "salvage_items",
    "replacement_required", "review_rate_threshold", "sample_body_pages",
    "verify_reported_items",
]
