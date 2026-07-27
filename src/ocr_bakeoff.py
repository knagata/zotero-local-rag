"""Deterministic, engine-neutral scoring and reporting for the PDF V3 bake-off."""
from __future__ import annotations

from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SCORE_WEIGHTS = {
    "heading_hierarchy": 0.22,
    "reading_order": 0.22,
    "zone_classification": 0.18,
    "table_caption_retention": 0.12,
    "locator_bbox_recovery": 0.10,
    "tree_integrity": 0.10,
    "text_accuracy": 0.06,
}


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


_MARKDOWN_DECORATION_RE = re.compile(r"[*_`#]")


def _normalize_for_matching(value: Any) -> str:
    """Region-to-block matching must survive markdown decoration engines add
    around otherwise-verbatim text (Mistral OCR emphasis/heading syntax,
    Markdown-emitting engines generally). ``text_accuracy`` scoring keeps
    using ``normalize_text`` unchanged; only correspondence matching does this."""
    return _MARKDOWN_DECORATION_RE.sub("", normalize_text(value))


def _partial_match_ratio(anchor: str, text: str) -> float:
    """Fraction of ``anchor`` found as one contiguous run inside ``text``.

    ``SequenceMatcher.ratio()`` on the full strings is diluted whenever a
    short anchor is compared against a much longer surrounding block (e.g. a
    one-sentence ground-truth anchor against a multi-sentence paragraph
    block): the denominator includes the block's full length even though
    only the anchor's length is actually being verified. This measures the
    fraction of the *anchor* recovered instead, which is invariant to how
    much unrelated text surrounds it in the candidate block.
    """
    if not anchor:
        return 0.0
    match = SequenceMatcher(None, anchor, text).find_longest_match(0, len(anchor), 0, len(text))
    return match.size / len(anchor)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("version") != "ocr-bakeoff-v3.1":
        raise ValueError("manifest.version must be ocr-bakeoff-v3.1")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("manifest.samples must be a non-empty list")
    ids: set[str] = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ValueError(f"manifest.samples[{index}] must be an object")
        sample_id = str(sample.get("id") or "")
        if not sample_id or sample_id in ids:
            raise ValueError(f"missing or duplicate sample id: {sample_id!r}")
        ids.add(sample_id)
        env_name = str(sample.get("path_env") or "")
        if not re.fullmatch(r"OCR_BAKEOFF_[A-Z0-9_]+_PDF", env_name):
            raise ValueError(f"invalid path_env for sample {sample_id}")
        if any(key in sample for key in ("path", "pdf_path", "source_path")):
            raise ValueError(f"sample {sample_id} embeds a PDF path; use path_env only")
        truth = str(sample.get("ground_truth") or "")
        if not re.fullmatch(r"annotations/[a-z0-9_]+\.json", truth):
            raise ValueError(f"invalid ground_truth for sample {sample_id}")


def validate_truth(truth: Mapping[str, Any]) -> None:
    regions = truth.get("regions")
    if not isinstance(regions, list):
        raise ValueError("truth.regions must be a list")
    ids: set[str] = set()
    for index, region in enumerate(regions):
        if not isinstance(region, Mapping):
            raise ValueError(f"truth.regions[{index}] must be an object")
        region_id = str(region.get("id") or "")
        if not region_id or region_id in ids:
            raise ValueError(f"missing or duplicate region id: {region_id!r}")
        ids.add(region_id)
        if len(str(region.get("text_anchor") or "").strip()) < 4:
            raise ValueError(f"region {region_id} needs a text_anchor of at least 4 characters")
        if not isinstance(region.get("order"), int):
            raise ValueError(f"region {region_id} needs an integer order")


def resolve_sample_path(sample: Mapping[str, Any]) -> Path | None:
    import os

    env_name = str(sample.get("path_env") or "")
    raw = os.environ.get(env_name, "").strip() if env_name else ""
    return Path(raw).expanduser() if raw else None


def _metadata(block: Mapping[str, Any]) -> Mapping[str, Any]:
    md = block.get("metadata")
    return md if isinstance(md, Mapping) else block


def _match_regions(
    regions: Sequence[Mapping[str, Any]], blocks: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    matches: dict[str, int] = {}
    used: set[int] = set()
    for region in regions:
        anchor = _normalize_for_matching(region.get("text_anchor"))
        if not anchor:
            continue
        candidates = [
            (index, _normalize_for_matching(block.get("text"))) for index, block in enumerate(blocks)
            if index not in used
        ]
        exact = [(index, text) for index, text in candidates if anchor in text]
        pool = exact or candidates
        if not pool:
            continue
        # A short anchor compared against a much longer block via full-string
        # ratio() gets diluted by the block's unrelated length; measure the
        # fraction of the anchor recovered as one contiguous run instead.
        index, text = max(pool, key=lambda pair: _partial_match_ratio(anchor, pair[1]))
        if anchor in text or _partial_match_ratio(anchor, text) >= 0.6:
            matches[str(region["id"])] = index
            used.add(index)
    return matches


def _ratio(correct: int, total: int) -> float:
    return 1.0 if total == 0 else correct / total


def _path(md: Mapping[str, Any]) -> list[str]:
    value = md.get("structure_path", [])
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            value = decoded if isinstance(decoded, list) else [value]
        except json.JSONDecodeError:
            value = [part for part in value.split("/") if part]
    return [str(part).strip() for part in value] if isinstance(value, list) else []


def validate_tree(blocks: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    ids: set[str] = set()
    previous_ordinal = -1
    for index, block in enumerate(blocks):
        block_id = str(block.get("id") or "")
        if not block_id:
            errors.append(f"block[{index}] missing id")
        elif block_id in ids:
            errors.append(f"duplicate block id: {block_id}")
        ids.add(block_id)
        ordinal = block.get("ordinal", index)
        if not isinstance(ordinal, int) or ordinal <= previous_ordinal:
            errors.append(f"block[{index}] ordinal is not strictly increasing")
        if isinstance(ordinal, int):
            previous_ordinal = ordinal
        md = _metadata(block)
        if not (md.get("locator") or md.get("page")):
            errors.append(f"block[{index}] missing locator")
        path = _path(md)
        if any(not part for part in path):
            errors.append(f"block[{index}] has empty structure path component")
    return errors


def score_result(truth: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    validate_truth(truth)
    blocks = result.get("blocks", [])
    if not isinstance(blocks, list):
        raise ValueError("result.blocks must be a list")
    regions = truth.get("regions", [])
    if not isinstance(regions, list):
        raise ValueError("truth.regions must be a list")
    matches = _match_regions(regions, blocks)

    heading_regions = [r for r in regions if r.get("heading_path")]
    heading_correct = sum(
        _path(_metadata(blocks[matches[str(r["id"])]])) == [str(v) for v in r["heading_path"]]
        for r in heading_regions if str(r["id"]) in matches
    )
    heading_score = _ratio(heading_correct, len(heading_regions))

    ordered = sorted((r for r in regions if isinstance(r.get("order"), int)), key=lambda r: r["order"])
    pairs = [(a, b) for i, a in enumerate(ordered) for b in ordered[i + 1:]]
    order_correct = sum(
        matches.get(str(a["id"]), 10**9) < matches.get(str(b["id"]), -1)
        for a, b in pairs
    )
    order_score = _ratio(order_correct, len(pairs))

    zoned = [r for r in regions if r.get("zone")]
    zone_correct = sum(
        str(_metadata(blocks[matches[str(r["id"])]]).get("zone", "body")) == str(r["zone"])
        for r in zoned if str(r["id"]) in matches
    )
    zone_score = _ratio(zone_correct, len(zoned))

    special = [r for r in regions if r.get("kind") in {"table", "caption"}]
    special_correct = sum(
        str(_metadata(blocks[matches[str(r["id"])]]).get("block_type", "")).lower() == str(r["kind"]).lower()
        for r in special if str(r["id"]) in matches
    )
    special_score = _ratio(special_correct, len(special))

    located = [r for r in regions if r.get("locator_required") or r.get("bbox_required")]
    locator_correct = 0
    for region in located:
        index = matches.get(str(region["id"]))
        if index is None:
            continue
        md = _metadata(blocks[index])
        locator_ok = not region.get("locator_required") or bool(md.get("locator") or md.get("page"))
        bbox_ok = not region.get("bbox_required") or bool(md.get("bbox"))
        locator_correct += locator_ok and bbox_ok
    locator_score = _ratio(locator_correct, len(located))

    errors = validate_tree(blocks)
    tree_score = 1.0 if not errors else max(0.0, 1.0 - len(errors) / max(1, len(blocks)))

    snippets = [r for r in regions if r.get("expected_text")]
    text_values = []
    for region in snippets:
        index = matches.get(str(region["id"]))
        predicted = blocks[index].get("text", "") if index is not None else ""
        text_values.append(SequenceMatcher(None, normalize_text(region["expected_text"]), normalize_text(predicted)).ratio())
    text_score = sum(text_values) / len(text_values) if text_values else 1.0

    metrics = {
        "heading_hierarchy": heading_score,
        "reading_order": order_score,
        "zone_classification": zone_score,
        "table_caption_retention": special_score,
        "locator_bbox_recovery": locator_score,
        "tree_integrity": tree_score,
        "text_accuracy": text_score,
    }
    total = sum(metrics[name] * weight for name, weight in SCORE_WEIGHTS.items())
    return {
        "score_version": "ocr-bakeoff-v3.1",
        "total_score": round(total, 6),
        "metrics": {name: round(value, 6) for name, value in metrics.items()},
        "weights": SCORE_WEIGHTS,
        "matched_regions": len(matches),
        "total_regions": len(regions),
        "tree_errors": errors,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = ["# PDF解析エンジン ベイクオフ結果", "", "| Sample | Engine | Score | Status |", "|---|---|---:|---|"]
    for run in report.get("runs", []):
        score = run.get("score", {}).get("total_score")
        score_text = f"{float(score):.3f}" if score is not None else "—"
        lines.append(f"| {run.get('sample_id', '')} | {run.get('engine', '')} | {score_text} | {run.get('status', '')} |")
    lines.extend(["", "## 固定重み", ""])
    for name, weight in SCORE_WEIGHTS.items():
        lines.append(f"- `{name}`: {weight:.0%}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "SCORE_WEIGHTS", "load_json", "markdown_report", "resolve_sample_path",
    "score_result", "validate_manifest", "validate_tree", "validate_truth",
]
