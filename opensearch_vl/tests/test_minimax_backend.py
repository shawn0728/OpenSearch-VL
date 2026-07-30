"""Tests for the MiniMax API backend registration and wiring.

These tests avoid any network access: the HTTP layer is patched so the
request that :class:`MiniMaxClient` would send can be inspected, and the
runner is exercised against a stubbed Anthropic-shaped response.
"""

import os
import sys
import unittest
from unittest import mock

# Make ``import opensearch_infer`` work regardless of the current directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opensearch_infer import auth, config, messages
from opensearch_infer.runners import (
    InferenceConfig,
    MiniMaxRunner,
    build_runner,
)


class RegistryTests(unittest.TestCase):
    def test_models_registered(self):
        for key, api_model, context in (
            ("minimax-m3", "MiniMax-M3", "1000000"),
            ("minimax-m2.7", "MiniMax-M2.7", "204800"),
        ):
            spec = config.MODEL_REGISTRY[key]
            self.assertEqual(spec.family, "minimax")
            self.assertEqual(spec.display_name, api_model)
            self.assertEqual(spec.extra["api_model"], api_model)
            self.assertEqual(spec.extra["context_window"], context)
            self.assertFalse(spec.supports_multi_gpu)

    def test_build_runner_returns_minimax_runner(self):
        runner = build_runner("minimax-m3")
        self.assertIsInstance(runner, MiniMaxRunner)
        self.assertEqual(runner.api_model, "MiniMax-M3")

    def test_region_endpoints(self):
        self.assertEqual(
            config.minimax_region_endpoints("cn_zh")["anthropic_base_url"],
            "https://api.minimaxi.com/anthropic",
        )
        # Unknown regions fall back to the global endpoints.
        self.assertEqual(
            config.minimax_region_endpoints("does-not-exist")["openai_base_url"],
            "https://api.minimax.io/v1",
        )


class MessageConversionTests(unittest.TestCase):
    def test_text_image_and_role_mapping(self):
        contents = [
            {
                "role": "user",
                "parts": [
                    {"text": "hello"},
                    {"image_url": {"url": "https://example.com/a.png"}},
                ],
            },
            {"role": "model", "parts": [{"text": "hi"}]},
        ]
        out = messages.to_minimax_messages(contents)
        self.assertEqual(out[0]["role"], "user")
        self.assertEqual(out[0]["content"][0], {"type": "text", "text": "hello"})
        self.assertEqual(
            out[0]["content"][1],
            {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}},
        )
        # ``model`` maps to the Anthropic ``assistant`` role.
        self.assertEqual(out[1]["role"], "assistant")

    def test_inline_base64_image(self):
        contents = [
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": "AAAA",
                        }
                    }
                ],
            }
        ]
        block = messages.to_minimax_messages(contents)[0]["content"][0]
        self.assertEqual(block["type"], "image")
        self.assertEqual(block["source"]["type"], "base64")
        self.assertEqual(block["source"]["media_type"], "image/png")
        self.assertEqual(block["source"]["data"], "AAAA")


class ClientTests(unittest.TestCase):
    def test_headers_and_request_shape(self):
        client = auth.MiniMaxClient(
            api_key="secret",
            base_url="https://api.minimax.io/anthropic/",
            api_version="2023-06-01",
        )
        headers = client._headers()
        self.assertEqual(headers["Authorization"], "Bearer secret")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        # Trailing slash is stripped so the path stays well-formed.
        self.assertEqual(client.base_url, "https://api.minimax.io/anthropic")

        captured = {}

        class _Resp:
            def raise_for_status(self):
                return None

        def _fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            return _Resp()

        with mock.patch.object(auth.requests, "post", _fake_post):
            client.call(
                model="MiniMax-M3",
                messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                system_instruction="be brief",
                max_tokens=256,
            )
        self.assertEqual(captured["url"], "https://api.minimax.io/anthropic/v1/messages")
        self.assertEqual(captured["json"]["model"], "MiniMax-M3")
        self.assertEqual(captured["json"]["max_tokens"], 256)
        self.assertEqual(captured["json"]["system"], "be brief")

    def test_missing_api_key_raises(self):
        with self.assertRaises(ValueError):
            auth.MiniMaxClient(api_key="")


class RunnerInferTests(unittest.TestCase):
    def test_infer_parses_anthropic_response(self):
        runner = build_runner("minimax-m3")

        class _FakeClient:
            def call(self, model, messages, system_instruction=None, max_tokens=32768):
                assert model == "MiniMax-M3"

                class _Resp:
                    def json(self_inner):
                        return {
                            "content": [{"type": "text", "text": "answer"}],
                            "stop_reason": "end_turn",
                            "usage": {"input_tokens": 3, "output_tokens": 4},
                        }

                return _Resp()

        runner.client = _FakeClient()
        result = runner.infer(
            [{"role": "user", "parts": [{"text": "q"}]}],
            system_instruction=None,
            cfg=InferenceConfig(max_tokens=128),
        )
        candidate = result["candidates"][0]
        self.assertEqual(candidate["content"]["parts"][0]["text"], "answer")
        self.assertEqual(candidate["finishReason"], "end_turn")
        self.assertEqual(result["usageMetadata"]["totalTokenCount"], 7)
        self.assertEqual(result["modelVersion"], "MiniMax-M3")

    def test_infer_surfaces_api_error(self):
        runner = build_runner("minimax-m3")

        class _FakeClient:
            def call(self, *args, **kwargs):
                class _Resp:
                    def json(self_inner):
                        return {"type": "error", "error": {"message": "bad request"}}

                return _Resp()

        runner.client = _FakeClient()
        result = runner.infer([{"role": "user", "parts": [{"text": "q"}]}])
        self.assertEqual(result["candidates"][0]["finishReason"], "ERROR")
        self.assertIn("bad request", result["candidates"][0]["content"]["parts"][0]["text"])


if __name__ == "__main__":
    unittest.main()
