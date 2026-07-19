from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.llm_client import (
    CLIAgentClient, DeepSeekClient, FallbackLLMClient, InvalidLLMResponse, LLMError,
    OpenAICompatibleClient, ProviderUnavailable, RateLimitReached,
    _extract_json, _is_rate_limit, _retry, get_llm,
)
from src import llm_client


class FakeClient:
    def __init__(self, provider: str, result=None, error: Exception | None = None):
        self.provider, self.model = provider, "test"
        self.result, self.error = result, error

    def generate_text(self, prompt, **kwargs):
        if self.error:
            raise self.error
        return self.result

    generate_json = generate_text


class LLMClientTests(unittest.TestCase):
    def test_default_llm_is_deepseek_v4_pro(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            llm_client, "load_dotenv_native"
        ):
            client = get_llm("unconfigured_task")
        self.assertIsInstance(client, DeepSeekClient)
        self.assertEqual(client.model, "deepseek-v4-pro")

    def test_extract_json_from_fence_and_cli_envelope(self):
        self.assertEqual(_extract_json("```json\n{\"ok\": true}\n```"), {"ok": True})
        self.assertEqual(_extract_json({"structured_output": {"ok": True}}), {"ok": True})

    def test_extract_json_rejects_array(self):
        with self.assertRaises(InvalidLLMResponse):
            _extract_json("[1, 2]")

    def test_task_setting_preserves_colon_in_model(self):
        with patch.dict(os.environ, {"LLM_EXPAND": "openai_compat:qwen3:14b"}, clear=True):
            client = get_llm("expand")
        self.assertEqual((client.provider, client.model), ("openai_compat", "qwen3:14b"))

    def test_fallback_chain_preserves_order(self):
        with patch.dict(os.environ, {"LLM_SUMMARY": "codex_cli:gpt-5,claude_cli:sonnet"}, clear=True):
            client = get_llm("summary")
        self.assertIsInstance(client, FallbackLLMClient)
        self.assertEqual([item.provider for item in client.clients], ["codex_cli", "claude_cli"])

    def test_fallback_uses_second_provider(self):
        client = FallbackLLMClient([
            FakeClient("first", error=LLMError("down")), FakeClient("second", result="ok")
        ])
        self.assertEqual(client.generate_text("prompt"), "ok")

    def test_retry_converts_rate_limit_without_sleeping(self):
        with self.assertRaises(RateLimitReached):
            _retry(lambda: (_ for _ in ()).throw(RuntimeError("HTTP 429")))

    def test_session_identifier_containing_429_is_not_a_rate_limit(self):
        self.assertFalse(_is_rate_limit("session id: d350e4297eea; model unsupported"))

    def test_codex_auto_omits_explicit_model_flag(self):
        completed = SimpleNamespace(returncode=0, stdout='{"ok": true}', stderr="")
        with patch("src.llm_client.subprocess.run", return_value=completed) as run:
            result = CLIAgentClient("auto", "codex_cli").generate_json(
                "prompt", schema={"type": "object"}
            )
        self.assertEqual(result, {"ok": True})
        self.assertNotIn("--model", run.call_args.args[0])

    def test_retry_transient_error_then_success(self):
        calls = []

        def operation():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("temporary")
            return "ok"

        with patch("src.llm_client.time.sleep"):
            self.assertEqual(_retry(operation), "ok")
        self.assertEqual(len(calls), 2)

    def test_retry_does_not_repeat_configuration_errors(self):
        calls = []

        def operation():
            calls.append(1)
            raise ProviderUnavailable("missing key")

        with self.assertRaises(ProviderUnavailable):
            _retry(operation)
        self.assertEqual(len(calls), 1)

    def test_openai_compatible_json_request(self):
        response = SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": '{"ok": true}'}}]},
        )
        env = {"LLM_OPENAI_BASE_URL": "http://localhost:11434/v1"}
        with patch.dict(os.environ, env, clear=True), patch(
            "src.llm_client.httpx.post", return_value=response
        ) as post:
            result = OpenAICompatibleClient("qwen3:14b").generate_json(
                "prompt", schema={"type": "object"}
            )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(post.call_args.kwargs["json"]["model"], "qwen3:14b")
        self.assertIn("response_format", post.call_args.kwargs["json"])

    def test_deepseek_json_request_uses_native_key_and_json_object_mode(self):
        response = SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {
                "choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
        )
        client = DeepSeekClient("deepseek-v4-pro")
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=True), patch(
            "src.llm_client.httpx.post", return_value=response
        ) as post:
            result = client.generate_json(
                "Return JSON", schema={"type": "object"}
            )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(client.last_usage, {"prompt_tokens": 12, "completion_tokens": 3})
        request = post.call_args.kwargs
        self.assertEqual(request["json"]["model"], "deepseek-v4-pro")
        self.assertEqual(request["json"]["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", request["json"])
        self.assertEqual(request["json"]["response_format"], {"type": "json_object"})
        self.assertEqual(request["headers"]["Authorization"], "Bearer test-key")

    def test_deepseek_is_available_as_provider_spec(self):
        with patch.dict(os.environ, {"LLM_SUMMARY": "deepseek:deepseek-v4-pro"}, clear=True):
            client = get_llm("summary")
        self.assertIsInstance(client, DeepSeekClient)
        self.assertEqual(client.model, "deepseek-v4-pro")

    def test_deepseek_instance_can_override_reasoning_without_mutating_environment(self):
        response = SimpleNamespace(
            status_code=200, raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": '{"ok": true}'}}]},
        )
        with patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "test-key", "DEEPSEEK_THINKING": "disabled",
        }, clear=True), patch("src.llm_client.httpx.post", return_value=response) as post:
            DeepSeekClient(
                "deepseek-v4-pro", thinking="enabled", reasoning_effort="medium",
            ).generate_json("Return JSON", schema={"type": "object"})
        request = post.call_args.kwargs["json"]
        self.assertEqual(request["thinking"], {"type": "enabled"})
        self.assertEqual(request["reasoning_effort"], "medium")

    def test_deepseek_rejects_json_that_does_not_match_schema(self):
        response = SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": '{"wrong": true}'}}]},
        )
        schema = {
            "type": "object", "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"], "additionalProperties": False,
        }
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=True), patch(
            "src.llm_client.httpx.post", return_value=response
        ):
            with self.assertRaises(InvalidLLMResponse):
                DeepSeekClient("deepseek-v4-pro").generate_json("Return JSON", schema=schema)

    def test_claude_cli_disables_tools(self):
        completed = SimpleNamespace(
            returncode=0,
            stdout='{"structured_output": {"ok": true}}',
            stderr="",
        )
        with patch("src.llm_client.subprocess.run", return_value=completed) as run:
            result = CLIAgentClient("sonnet", "claude_cli").generate_json(
                "prompt", schema={"type": "object"}
            )
        self.assertEqual(result, {"ok": True})
        command = run.call_args.args[0]
        self.assertIn("--tools", command)
        self.assertEqual(command[command.index("--tools") + 1], "")


if __name__ == "__main__":
    unittest.main()
