from __future__ import annotations

import base64
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import fitz
import httpx

from .mistral_ocr_extract import DEFAULT_BASE_URL, DEFAULT_MODEL
from .reocr_quality import REPEAT_ARTIFACT_RE
from .text_utils import looks_like_gibberish


def source_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"source_mtime": float(stat.st_mtime), "source_size": int(stat.st_size)}


def source_matches(row: dict[str, Any], path: Path) -> tuple[bool, str]:
    current = source_fingerprint(path)
    expected_size = row.get("source_size")
    expected_mtime = row.get("source_mtime")
    if expected_size is not None and int(expected_size) != current["source_size"]:
        return False, "source_size_changed"
    if expected_mtime is not None and abs(float(expected_mtime) - current["source_mtime"]) > 0.001:
        return False, "source_mtime_changed"
    return True, "ok"


def build_batch_request(row: dict[str, Any], pdf_path: Path) -> dict[str, Any]:
    encoded = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
    return {
        "custom_id": str(row["attachment_key"]),
        "body": {
            "document": {
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{encoded}",
            },
            "include_image_base64": False,
        },
    }


def write_batch_jsonl(
    rows_with_paths: Iterable[tuple[dict[str, Any], Path]], output: Path,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for row, path in rows_with_paths:
            handle.write(json.dumps(build_batch_request(row, path), ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_batch_output(text: str) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        custom_id = str(payload.get("custom_id") or "")
        if not custom_id:
            raise ValueError(f"Batch output line {line_number} has no custom_id")
        response = payload.get("response") if isinstance(payload.get("response"), dict) else {}
        status_code = int(response.get("status_code") or 0)
        body = response.get("body")
        if status_code and status_code >= 400:
            results[custom_id] = {"ok": False, "error": body or payload.get("error"), "status_code": status_code}
        elif isinstance(body, dict):
            results[custom_id] = {"ok": True, "body": body, "status_code": status_code or 200}
        else:
            results[custom_id] = {"ok": False, "error": payload.get("error") or "missing response body"}
    return results


def evaluate_ocr_result(result: dict[str, Any], pdf_path: Path) -> dict[str, Any]:
    pages = [page for page in (result.get("pages") or []) if isinstance(page, dict)]
    with fitz.open(str(pdf_path)) as doc:
        expected_pages = doc.page_count
    indices = []
    character_count = 0
    gibberish_pages = 0
    repeat_artifacts = 0
    problem_pages: dict[int, list[str]] = {}
    for fallback_index, page in enumerate(pages):
        try:
            indices.append(int(page.get("index", fallback_index)))
        except (TypeError, ValueError):
            indices.append(fallback_index)
        text = str(page.get("markdown") or "").strip()
        if not text:
            text = "\n".join(
                str(block.get("content") or "").strip()
                for block in (page.get("blocks") or []) if isinstance(block, dict)
            )
        character_count += len(text)
        page_no = indices[-1] + 1
        page_reasons: list[str] = []
        if looks_like_gibberish(text):
            gibberish_pages += 1
            page_reasons.append("gibberish_detected")
        page_repeat_artifacts = len(REPEAT_ARTIFACT_RE.findall(text))
        repeat_artifacts += page_repeat_artifacts
        if page_repeat_artifacts:
            page_reasons.append("repeat_artifacts")
        if page_reasons:
            problem_pages[page_no] = page_reasons
    expected_indices = set(range(expected_pages))
    actual_indices = set(indices)
    reasons: list[str] = []
    if actual_indices != expected_indices:
        reasons.append("incomplete_page_coverage")
    if len(indices) != len(actual_indices):
        reasons.append("duplicate_page_indices")
    minimum_chars = max(80, expected_pages * 20)
    if character_count < minimum_chars:
        reasons.append("insufficient_text")
    # Content anomalies are page-level review signals. They do not discard an
    # otherwise complete document; adoption stamps just those chunks.
    return {
        "passed": not reasons,
        "reasons": reasons,
        "expected_pages": expected_pages,
        "returned_pages": len(pages),
        "character_count": character_count,
        "minimum_chars": minimum_chars,
        "gibberish_pages": gibberish_pages,
        "repeat_artifacts": repeat_artifacts,
        "problem_pages": problem_pages,
    }


class MistralBatchClient:
    def __init__(
        self, api_key: str, *, base_url: str = DEFAULT_BASE_URL, timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout,
        )

    def upload(self, jsonl_path: Path) -> str:
        with jsonl_path.open("rb") as handle:
            response = self.client.post(
                f"{self.base_url}/v1/files",
                data={"purpose": "batch"},
                files={"file": (jsonl_path.name, handle, "application/jsonl")},
            )
        response.raise_for_status()
        return str(response.json()["id"])

    def create_job(
        self, input_file_id: str | list[str], *, model: str = DEFAULT_MODEL, timeout_hours: int = 24,
    ) -> dict[str, Any]:
        response = self.client.post(
            f"{self.base_url}/v1/batch/jobs",
            json={
                "input_files": [input_file_id] if isinstance(input_file_id, str) else input_file_id,
                "model": model,
                "endpoint": "/v1/ocr",
                "timeout_hours": timeout_hours,
            },
        )
        response.raise_for_status()
        return response.json()

    def get_job(self, job_id: str) -> dict[str, Any]:
        response = self.client.get(f"{self.base_url}/v1/batch/jobs/{job_id}")
        response.raise_for_status()
        return response.json()

    def download_file(self, file_id: str) -> str:
        response = self.client.get(f"{self.base_url}/v1/files/{file_id}/content")
        response.raise_for_status()
        return response.text


def save_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
