"""Provider-neutral text and structured-output clients for LLM tasks.

Providers use ``<provider>:<model>`` specifications. A comma-separated value
creates a fallback chain, for example ``codex_cli:auto,claude_cli:sonnet``.
Optional SDKs are imported lazily so local workflows stay lightweight.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, TypeVar

import httpx

try:  # Support both ``python -m src...`` and direct ``python src/...py`` execution.
    from .env_utils import load_dotenv_native
except ImportError:  # pragma: no cover - exercised by direct script execution
    from env_utils import load_dotenv_native


ROLE_DEFAULTS = {
    "cheap": "deepseek:deepseek-v4-flash",
    "standard": "deepseek:deepseek-v4-pro",
    "review": "deepseek:deepseek-v4-pro",
}
TASK_ROLES = {
    "summary": "cheap",
    "expand": "cheap",
    "extract": "standard",
    "review": "review",
}
DEFAULT_MODELS = {
    "gemini": "gemini-3.1-flash-lite",
    "anthropic": "claude-haiku-4-5",
    "deepseek": "deepseek-v4-pro",
    "openai_compat": "gpt-4.1-mini",
    "codex_cli": "auto",
    "claude_cli": "sonnet",
}
RATE_LIMIT_MARKERS = (
    "rate limit", "rate_limit", "too many requests", "quota exceeded",
    "usage limit", "limit reached",
)


class LLMError(RuntimeError):
    """Base error raised by the provider abstraction."""


class ProviderUnavailable(LLMError):
    """The selected provider is not installed or configured."""


class RateLimitReached(LLMError):
    """The provider cannot continue because its current quota is exhausted."""


class InvalidLLMResponse(LLMError):
    """A provider returned output that cannot satisfy the requested contract.

    ``raw`` carries the undecodable text when there was any. A caller that can
    make partial use of a broken reply -- currently only the OCR-layer audit,
    which verifies every recovered item against the source -- needs the text
    itself; the parser's message names a character offset into a string it
    would otherwise discard.
    """

    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw


class LLMClient(Protocol):
    provider: str
    model: str

    def generate_text(
        self, prompt: str, *, max_tokens: int = 2048, timeout: float = 30.0
    ) -> str: ...

    def generate_json(
        self, prompt: str, *, schema: dict[str, Any], timeout: float = 30.0,
        max_tokens: int = 4096,
    ) -> dict[str, Any]: ...


T = TypeVar("T")


def _is_rate_limit(error: BaseException | str) -> bool:
    message = str(error).lower()
    return (
        getattr(error, "status_code", None) == 429
        or any(marker in message for marker in RATE_LIMIT_MARKERS)
        or bool(re.search(r'(?:"status"\s*:\s*|http(?:/\S+)?\s+)429\b', message))
    )


def _retry(
    operation: Callable[[], T], *, attempts: int | None = None, initial_delay: float = 0.5
) -> T:
    """Retry transient provider failures and preserve quota exhaustion."""
    if attempts is None:
        try:
            attempts = int(os.environ.get("LLM_RETRY_ATTEMPTS", "3"))
        except ValueError:
            attempts = 3
    attempts = min(5, max(1, attempts))
    last_error: BaseException | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except (RateLimitReached, ProviderUnavailable, InvalidLLMResponse):
            raise
        except Exception as exc:
            if _is_rate_limit(exc):
                raise RateLimitReached(str(exc)) from exc
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(initial_delay * (2**attempt))
    assert last_error is not None
    if isinstance(last_error, LLMError):
        raise last_error
    raise LLMError(str(last_error)) from last_error


def _extract_json(value: Any) -> dict[str, Any]:
    """Extract a JSON object from SDK objects, CLI envelopes, or text fences."""
    if isinstance(value, dict):
        for key in ("structured_output", "result", "output", "content", "text"):
            nested = value.get(key)
            if isinstance(nested, dict):
                return nested
            if isinstance(nested, str):
                try:
                    return _extract_json(nested)
                except InvalidLLMResponse:
                    pass
        return value
    if not isinstance(value, str):
        raise InvalidLLMResponse(f"Expected a JSON object, got {type(value).__name__}.")

    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise InvalidLLMResponse(
                "The model response did not contain a JSON object.", raw=text,
            ) from exc
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise InvalidLLMResponse(
                f"Invalid JSON response: {exc}", raw=text[start : end + 1],
            ) from exc
    if not isinstance(parsed, dict):
        raise InvalidLLMResponse("The model response was valid JSON but not an object.")
    return parsed


def _validate_schema_subset(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    """Validate the JSON-Schema subset used by this project's LLM contracts."""
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                _validate_schema_subset(value, option, path)
                return
            except InvalidLLMResponse:
                continue
        raise InvalidLLMResponse(f"LLM output does not match any schema at {path}.")

    expected = schema.get("type")
    expected_types = expected if isinstance(expected, list) else [expected]
    type_checks = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "null": lambda item: item is None,
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
    }
    if expected and not any(type_checks[item](value) for item in expected_types):
        raise InvalidLLMResponse(
            f"LLM output has the wrong type at {path}; expected {expected}."
        )
    if isinstance(value, dict) and "object" in expected_types:
        properties = schema.get("properties", {})
        missing = [key for key in schema.get("required", []) if key not in value]
        if missing:
            raise InvalidLLMResponse(
                f"LLM output is missing required fields at {path}: {', '.join(missing)}."
            )
        if schema.get("additionalProperties") is False:
            extras = sorted(set(value) - set(properties))
            if extras:
                raise InvalidLLMResponse(
                    f"LLM output has additional fields at {path}: {', '.join(extras)}."
                )
        for key, child in value.items():
            if key in properties:
                _validate_schema_subset(child, properties[key], f"{path}.{key}")
    if isinstance(value, list) and "array" in expected_types and "items" in schema:
        for index, child in enumerate(value):
            _validate_schema_subset(child, schema["items"], f"{path}[{index}]")


@dataclass
class GeminiClient:
    model: str
    provider: str = "gemini"

    def _client(self):
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ProviderUnavailable("GEMINI_API_KEY is not set.")
        try:
            from google import genai
        except ImportError as exc:
            raise ProviderUnavailable(
                "The gemini provider requires the 'google-genai' package."
            ) from exc
        return genai.Client(api_key=api_key)

    def _generate(
        self, prompt: str, *, max_tokens: int, timeout: float,
        schema: dict[str, Any] | None = None,
    ) -> str:
        def call() -> str:
            from google.genai import types

            config: dict[str, Any] = {
                "max_output_tokens": max_tokens,
                "http_options": types.HttpOptions(timeout=int(timeout * 1000)),
            }
            if schema is not None:
                config.update(
                    response_mime_type="application/json",
                    response_json_schema=schema,
                )
            response = self._client().models.generate_content(
                model=self.model, contents=prompt,
                config=types.GenerateContentConfig(**config),
            )
            text = getattr(response, "text", None)
            if not text:
                raise InvalidLLMResponse("Gemini returned an empty response.")
            return text

        return _retry(call)

    def generate_text(
        self, prompt: str, *, max_tokens: int = 2048, timeout: float = 30.0
    ) -> str:
        return self._generate(prompt, max_tokens=max_tokens, timeout=timeout)

    def generate_json(
        self, prompt: str, *, schema: dict[str, Any], timeout: float = 30.0,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        return _extract_json(
            self._generate(prompt, max_tokens=max_tokens, timeout=timeout, schema=schema)
        )


@dataclass
class AnthropicClient:
    model: str
    provider: str = "anthropic"

    def _client(self, timeout: float):
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise ProviderUnavailable("ANTHROPIC_API_KEY is not set.")
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderUnavailable(
                "The anthropic provider requires the optional 'anthropic' package."
            ) from exc
        return anthropic.Anthropic(api_key=api_key, timeout=timeout)

    @staticmethod
    def _text(response: Any) -> str:
        parts = [getattr(block, "text", "") for block in getattr(response, "content", [])]
        text = "".join(parts).strip()
        if not text:
            raise InvalidLLMResponse("Anthropic returned an empty response.")
        return text

    def generate_text(
        self, prompt: str, *, max_tokens: int = 2048, timeout: float = 30.0
    ) -> str:
        return _retry(lambda: self._text(self._client(timeout).messages.create(
            model=self.model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )))

    def generate_json(
        self, prompt: str, *, schema: dict[str, Any], timeout: float = 30.0,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        def call() -> dict[str, Any]:
            response = self._client(timeout).messages.create(
                model=self.model, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                tools=[{"name": "return_result", "description": "Return the requested structured result.", "input_schema": schema}],
                tool_choice={"type": "tool", "name": "return_result"},
            )
            for block in getattr(response, "content", []):
                if getattr(block, "type", "") == "tool_use":
                    return _extract_json(getattr(block, "input", None))
            raise InvalidLLMResponse("Anthropic did not return the required tool result.")

        return _retry(call)


@dataclass
class OpenAICompatibleClient:
    model: str
    provider: str = "openai_compat"

    def _request(
        self, prompt: str, *, max_tokens: int, timeout: float,
        schema: dict[str, Any] | None = None,
    ) -> str:
        base_url = os.environ.get("LLM_OPENAI_BASE_URL", "").strip()
        if not base_url:
            raise ProviderUnavailable("LLM_OPENAI_BASE_URL is not set.")
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("LLM_OPENAI_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "strict": True, "schema": schema},
            }

        def call() -> str:
            response = httpx.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers, json=payload, timeout=timeout,
            )
            if response.status_code == 429:
                raise RateLimitReached(response.text)
            response.raise_for_status()
            try:
                content = response.json()["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                raise InvalidLLMResponse("Invalid OpenAI-compatible response envelope.") from exc
            if not isinstance(content, str) or not content.strip():
                raise InvalidLLMResponse("OpenAI-compatible provider returned empty content.")
            return content

        return _retry(call)

    def generate_text(
        self, prompt: str, *, max_tokens: int = 2048, timeout: float = 30.0
    ) -> str:
        return self._request(prompt, max_tokens=max_tokens, timeout=timeout)

    def generate_json(
        self, prompt: str, *, schema: dict[str, Any], timeout: float = 30.0,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        result = _extract_json(self._request(
            prompt, max_tokens=max_tokens, timeout=timeout, schema=schema
        ))
        _validate_schema_subset(result, schema)
        return result


@dataclass
class DeepSeekClient:
    """DeepSeek Chat Completions client using its supported JSON mode."""

    model: str
    provider: str = "deepseek"
    thinking: str | None = None
    reasoning_effort: str | None = None
    last_usage: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def _request(
        self, prompt: str, *, max_tokens: int, timeout: float,
        schema: dict[str, Any] | None = None,
    ) -> str:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise ProviderUnavailable("DEEPSEEK_API_KEY is not set.")
        thinking = (
            self.thinking if self.thinking is not None
            else os.environ.get("DEEPSEEK_THINKING", "disabled")
        ).strip().casefold()
        thinking = thinking if thinking in {"enabled", "disabled"} else "disabled"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "thinking": {"type": thinking},
        }
        if thinking == "enabled":
            payload["reasoning_effort"] = (
                self.reasoning_effort if self.reasoning_effort is not None
                else os.environ.get("DEEPSEEK_REASONING_EFFORT", "high")
            ).strip() or "high"
        if schema is not None:
            # DeepSeek supports JSON object mode, not OpenAI's json_schema mode.
            # Include the contract in the prompt, then parse and verify downstream.
            payload["messages"][0]["content"] = (
                prompt + "\n\nReturn JSON matching this schema exactly:\n"
                + json.dumps(schema, ensure_ascii=False)
            )
            payload["response_format"] = {"type": "json_object"}

        def call() -> str:
            response = httpx.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload, timeout=timeout,
            )
            if response.status_code == 429:
                raise RateLimitReached(response.text)
            if response.status_code == 400:
                raise InvalidLLMResponse(
                    f"DeepSeek rejected this request: {response.text[:500]}"
                )
            if response.status_code in {401, 402, 403}:
                raise ProviderUnavailable(
                    f"DeepSeek API is unavailable ({response.status_code}): {response.text[:500]}"
                )
            response.raise_for_status()
            try:
                body = response.json()
                usage = body.get("usage")
                self.last_usage = usage if isinstance(usage, dict) else None
                content = body["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                raise InvalidLLMResponse("Invalid DeepSeek response envelope.") from exc
            if not isinstance(content, str) or not content.strip():
                # DeepSeek documents occasional empty JSON-mode responses.
                # Treat only this provider-specific condition as transient.
                raise LLMError("DeepSeek returned empty content.")
            return content

        return _retry(call)

    def generate_text(
        self, prompt: str, *, max_tokens: int = 2048, timeout: float = 30.0
    ) -> str:
        return self._request(prompt, max_tokens=max_tokens, timeout=timeout)

    def generate_json(
        self, prompt: str, *, schema: dict[str, Any], timeout: float = 30.0,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        result = _extract_json(self._request(
            prompt, max_tokens=max_tokens, timeout=timeout, schema=schema
        ))
        _validate_schema_subset(result, schema)
        return result


@dataclass
class CLIAgentClient:
    model: str
    provider: str

    def _run(
        self, prompt: str, *, timeout: float,
        schema: dict[str, Any] | None = None,
    ) -> str | dict[str, Any]:
        try:
            if self.provider == "codex_cli":
                with tempfile.TemporaryDirectory(prefix="zotero-rag-codex-") as tmp:
                    output_path = Path(tmp) / "response.txt"
                    command = [
                        "codex", "exec", "-",
                        "--sandbox", "read-only", "--ephemeral",
                        "--skip-git-repo-check", "--output-last-message", str(output_path),
                    ]
                    if self.model not in {"auto", "default"}:
                        command.extend(["--model", self.model])
                    if schema is not None:
                        schema_path = Path(tmp) / "schema.json"
                        schema_path.write_text(json.dumps(schema), encoding="utf-8")
                        command.extend(["--output-schema", str(schema_path)])
                    completed = subprocess.run(
                        command, input=prompt, text=True, capture_output=True,
                        timeout=timeout, cwd=tmp, check=False,
                    )
                    output = output_path.read_text(encoding="utf-8") if output_path.exists() else completed.stdout
            elif self.provider == "claude_cli":
                command = [
                    "claude", "--print", "--output-format", "json",
                    "--model", self.model, "--tools", "", "--safe-mode",
                    "--no-session-persistence",
                ]
                if schema is not None:
                    command.extend(["--json-schema", json.dumps(schema)])
                completed = subprocess.run(
                    command, input=prompt, text=True, capture_output=True,
                    timeout=timeout, check=False,
                )
                output = json.loads(completed.stdout) if completed.stdout.strip() else ""
            else:
                raise ProviderUnavailable(f"Unsupported CLI provider: {self.provider}")
        except FileNotFoundError as exc:
            raise ProviderUnavailable(f"The {self.provider} executable was not found.") from exc
        except subprocess.TimeoutExpired as exc:
            raise LLMError(f"{self.provider} timed out after {timeout:g} seconds.") from exc
        except json.JSONDecodeError as exc:
            raise InvalidLLMResponse(f"{self.provider} returned an invalid JSON envelope.") from exc

        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout).strip()
            if _is_rate_limit(message):
                raise RateLimitReached(message)
            raise LLMError(f"{self.provider} exited with {completed.returncode}: {message}")
        return output

    def generate_text(
        self, prompt: str, *, max_tokens: int = 2048, timeout: float = 300.0
    ) -> str:
        del max_tokens
        value = _retry(lambda: self._run(prompt, timeout=timeout))
        if isinstance(value, dict):
            for key in ("result", "text", "content"):
                if isinstance(value.get(key), str):
                    return value[key]
            raise InvalidLLMResponse(f"{self.provider} response contained no text result.")
        if not value.strip():
            raise InvalidLLMResponse(f"{self.provider} returned an empty response.")
        return value.strip()

    def generate_json(
        self, prompt: str, *, schema: dict[str, Any], timeout: float = 300.0,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        return _extract_json(_retry(
            lambda: self._run(prompt, timeout=timeout, schema=schema)
        ))


@dataclass
class FallbackLLMClient:
    clients: list[LLMClient]
    provider: str = "fallback"
    model: str = "multiple"

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        errors: list[str] = []
        for client in self.clients:
            try:
                return getattr(client, method)(*args, **kwargs)
            except (LLMError, ImportError) as exc:
                errors.append(f"{client.provider}:{client.model}: {exc}")
        raise LLMError("All configured LLM providers failed: " + " | ".join(errors))

    def generate_text(
        self, prompt: str, *, max_tokens: int = 2048, timeout: float = 30.0
    ) -> str:
        return self._call("generate_text", prompt, max_tokens=max_tokens, timeout=timeout)

    def generate_json(
        self, prompt: str, *, schema: dict[str, Any], timeout: float = 30.0,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        return self._call("generate_json", prompt, schema=schema, timeout=timeout,
                          max_tokens=max_tokens)


def _parse_spec(spec: str) -> tuple[str, str]:
    provider, separator, model = spec.strip().partition(":")
    provider = provider.lower()
    if provider not in DEFAULT_MODELS:
        supported = ", ".join(DEFAULT_MODELS)
        raise ValueError(f"Unknown LLM provider '{provider}'. Supported providers: {supported}.")
    return provider, model.strip() if separator and model.strip() else DEFAULT_MODELS[provider]


def split_llm_specs(configured: str) -> list[str]:
    """Split a comma-separated provider fallback chain in declared order."""
    return [part.strip() for part in str(configured or "").split(",") if part.strip()]


def _create_client(provider: str, model: str) -> LLMClient:
    if provider == "gemini":
        return GeminiClient(model)
    if provider == "anthropic":
        return AnthropicClient(model)
    if provider == "deepseek":
        return DeepSeekClient(model)
    if provider == "openai_compat":
        return OpenAICompatibleClient(model)
    return CLIAgentClient(model=model, provider=provider)


def get_llm_spec(role: str) -> str:
    """Return the provider specification for one of the three operational roles."""
    load_dotenv_native(Path(__file__).resolve().parents[1])
    normalized = role.strip().lower()
    if normalized not in ROLE_DEFAULTS:
        raise ValueError(f"Unknown LLM role: {role}. Expected cheap, standard, or review.")
    return os.environ.get(f"LLM_{normalized.upper()}") or ROLE_DEFAULTS[normalized]


def get_llm(task: str) -> LLMClient:
    """Resolve a task to the cheap, standard, or review model role."""
    normalized = task.strip().lower()
    role = normalized if normalized in ROLE_DEFAULTS else TASK_ROLES.get(normalized, "standard")
    configured = get_llm_spec(role)
    specs = split_llm_specs(configured) or [ROLE_DEFAULTS[role]]
    clients = [_create_client(*_parse_spec(spec)) for spec in specs]
    if len(clients) == 1:
        return clients[0]
    chain = ",".join(f"{client.provider}:{client.model}" for client in clients)
    return FallbackLLMClient(clients, model=chain)


__all__ = [
    "LLMClient", "LLMError", "ProviderUnavailable", "RateLimitReached",
    "InvalidLLMResponse", "DeepSeekClient", "FallbackLLMClient", "get_llm", "get_llm_spec",
    "split_llm_specs",
]
